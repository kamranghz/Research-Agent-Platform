"""GRPO fine-tuning of Qwen3-4B answer generation, rewarded by the reviewer node.

Starts from the base model (fresh LoRA, no existing adapter). The reward for each
generated answer is the weighted reviewer score from agents.reviewer.reviewer_node.

Run from project root:
  python -m finetuning.grpo_answer_quality
"""
from dotenv import load_dotenv

load_dotenv()  # must run before importing modules that read env vars

import argparse  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

BASE_MODEL = 'Qwen/Qwen3-4B-Instruct-2507'
PILOT = 'finetuning/outputs/qwen3_4b_instruct_2507_tool_routing_pilot'
TEMPERATURE = 1.0  # sampling temperature; the log-prob shim must use the same value
MAX_TOOL_RESULT_CHARS = 6000  # keeps prompts inside the GPU budget
ANSWER_SYSTEM_PROMPT = (
    "You are a helpful research assistant. "
    "Answer the user's question clearly and concisely using the tool result provided. "
    "Cite sources when available. Do not request additional tools."
)


def load_rows(path):
    return [json.loads(line) for line in Path(path).open(encoding='utf-8') if line.strip()]


def split_rows(rows, seed=42):
    """Seed-42 shuffle; first 20% is the evaluation split."""
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    eval_count = max(1, round(len(shuffled) * 0.2))
    return shuffled[eval_count:], shuffled[:eval_count]


def build_messages(query, tool_result):
    return [
        {'role': 'system', 'content': ANSWER_SYSTEM_PROMPT},
        {'role': 'user', 'content': (
            'Answer the following question using the tool result below.\n\n'
            f'Question: {query}\n\n'
            f'Tool result:\n{tool_result[:MAX_TOOL_RESULT_CHARS]}')},
    ]


def review_overall(review):
    return (review['review_accuracy'] * 0.40 + review['review_relevance'] * 0.30 +
            review['review_clarity'] * 0.15 + review['review_completeness'] * 0.15)


def review_answer(query, draft, tool_result):
    """Call reviewer_node directly. Returns its review dict (raises on failure)."""
    from agents.reviewer import reviewer_node
    return reviewer_node({
        'query': query,
        'draft_answer': draft,
        'tool_result': tool_result,
        'retry_count': 1,
        'messages': [],
    })


def score_answer(query, draft, tool_result):
    try:
        review = review_answer(query, draft, tool_result)
        if review.get('review_status') == 'unavailable':
            print('[reward] judge unavailable; scoring 0.0', flush=True)
            return 0.0
        return float(review_overall(review))
    except Exception as exc:
        print(f'[reward] {type(exc).__name__}: {exc}', flush=True)
        return 0.0


def install_logprob_shim(model, temperature=1.0):
    """Serve TRL's `fused_lm_head=True` forward calls without Triton.

    TRL scores completions with model(..., labels=labels, fused_lm_head=True) and
    reads `.log_probs` / `.entropy`. Its fused kernel is None where Triton is
    unavailable (Windows). This wraps model.forward: calls carrying the keyword
    are run as a normal forward (LM head applied only to the completion
    positions) and the same per-token outputs are computed in plain PyTorch.
    Calls without the keyword (e.g. generation) pass through unchanged.
    """
    import torch
    from trl.trainer.utils import FusedCausalLMOutput

    original_forward = model.forward

    def forward(*args, fused_lm_head=False, labels=None, **kwargs):
        if not fused_lm_head:
            if labels is not None:
                kwargs['labels'] = labels
            return original_forward(*args, **kwargs)

        input_ids = kwargs.get('input_ids', args[0] if args else None)
        batch, length = input_ids.shape
        target = labels[:, 1:]  # label of the token each position predicts
        valid = target != -100
        first = int(valid.any(dim=0).nonzero()[0]) if valid.any() else length - 1
        keep = length - 1 - first  # trailing positions that carry a label
        kwargs['use_cache'] = False
        # logits_to_keep counts from the end; the last logit predicts no label
        logits = original_forward(*args, logits_to_keep=keep + 1, **kwargs).logits[:, :-1]
        log_probs = logits.new_zeros((batch, length - 1), dtype=torch.float32)
        entropy = logits.new_zeros((batch, length - 1), dtype=torch.float32)
        tail_labels = target[:, first:]
        tail_valid = valid[:, first:]
        for row in range(batch):
            logp = torch.log_softmax(logits[row].float() / temperature, dim=-1)
            token_logp = logp.gather(-1, tail_labels[row].clamp(min=0).unsqueeze(-1)).squeeze(-1)
            log_probs[row, first:] = token_logp * tail_valid[row]
            with torch.no_grad():
                row_entropy = -(logp.exp() * logp).sum(-1)
            entropy[row, first:] = row_entropy * tail_valid[row]
        return FusedCausalLMOutput(log_probs=log_probs, entropy=entropy)

    model.forward = forward
    return model


def reward_callable(prompts, completions, query, tool_result, **kwargs):
    rewards = []
    for q, tr, completion in zip(query, tool_result, completions):
        text = completion[0]['content'] if isinstance(completion, list) else completion
        rewards.append(score_answer(q, text, tr))
    return rewards


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='finetuning/data/real_world/answer_grpo_data.jsonl')
    p.add_argument('--output-dir', default=f'{PILOT}/final_adapter_answer_grpo')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--lr', type=float, default=1e-6)
    p.add_argument('--kl-coeff', type=float, default=0.05, help='KL penalty (GRPOConfig.beta)')
    p.add_argument('--group-size', type=int, default=4)
    p.add_argument('--max-length', type=int, default=1536)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    out = Path(args.output_dir)
    if out.exists():
        raise FileExistsError(f'{out} exists; refusing to overwrite')

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from finetuning.grpo_tool_routing import load_quantized_base

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Verify CUDA-enabled PyTorch installation.')

    rows = load_rows(args.data)
    train_split, eval_split = split_rows(rows, args.seed)
    print(f'train: {len(train_split)} | eval: {len(eval_split)}')

    train_ds = Dataset.from_list([
        {'prompt': build_messages(r['query'], r['tool_result']),
         'query': r['query'], 'tool_result': r['tool_result']}
        for r in train_split
    ])

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = prepare_model_for_kbit_training(load_quantized_base(BASE_MODEL))
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, target_modules=['q_proj', 'v_proj'],
        lora_dropout=0.05, bias='none', task_type='CAUSAL_LM'))
    model = install_logprob_shim(model, temperature=TEMPERATURE)
    model.print_trainable_parameters()

    config = GRPOConfig(
        output_dir=str(out), num_train_epochs=args.epochs, learning_rate=args.lr,
        beta=args.kl_coeff, temperature=TEMPERATURE, num_generations=args.group_size,
        max_completion_length=args.max_length, per_device_train_batch_size=args.group_size,
        gradient_accumulation_steps=1, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        bf16=True, logging_steps=1, save_strategy='epoch', save_total_limit=2,
        report_to='none', seed=args.seed, remove_unused_columns=False,
        use_vllm=False,
    )
    config.use_liger_kernel = False
    trainer = GRPOTrainer(model=model, args=config, train_dataset=train_ds,
                          reward_funcs=[reward_callable], processing_class=tokenizer)
    trainer.train()
    trainer.save_model(args.output_dir)

    train_loss = next((h['loss'] for h in reversed(trainer.state.log_history) if 'loss' in h), None)
    manifest = {
        'stage': 'grpo_answer_quality', 'base_model': BASE_MODEL, 'dataset': args.data,
        'train_examples': len(train_split), 'eval_examples': len(eval_split),
        'epochs': args.epochs, 'lr': args.lr, 'kl_coeff': args.kl_coeff,
        'group_size': args.group_size, 'reward': 'reviewer_node_weighted',
        'train_loss': train_loss, 'timestamp': datetime.now(timezone.utc).isoformat(),
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'GRPO answer-quality training complete -> {args.output_dir}')


if __name__ == '__main__':
    main()
