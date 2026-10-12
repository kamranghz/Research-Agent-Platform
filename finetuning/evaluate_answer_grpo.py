"""Compare baseline Qwen3-4B vs the answer-quality GRPO adapter, judged by reviewer_node.

Both models answer the held-out (query + tool_result) prompts; the reviewer node
scores each answer. Reviewer LLM calls are traced to LangSmith (tags: baseline / grpo).

Run from project root:
  python -m finetuning.evaluate_answer_grpo
"""
from dotenv import load_dotenv

load_dotenv()  # must run before importing modules that read env vars

import argparse  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

from finetuning.grpo_answer_quality import (BASE_MODEL, PILOT, build_messages,  # noqa: E402
                                            load_rows, review_answer, review_overall,
                                            split_rows)

DIMENSIONS = ('accuracy', 'relevance', 'clarity', 'completeness')


def generate_answers(model, tokenizer, rows, max_new_tokens=512):
    import torch
    answers = []
    for row in rows:
        prompt = tokenizer.apply_chat_template(
            build_messages(row['query'], row['tool_result']), tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        answers.append(tokenizer.decode(out[0][inputs['input_ids'].shape[1]:],
                                        skip_special_tokens=True).strip())
    return answers


def judge(tag, rows, answers):
    from langsmith import tracing_context
    results = []
    for row, answer in zip(rows, answers):
        entry = {'answer': answer, 'review_status': 'error', 'scores': None, 'overall': None}
        try:
            with tracing_context(tags=[tag]):
                review = review_answer(row['query'], answer, row['tool_result'])
            entry['review_status'] = review.get('review_status', '')
            if entry['review_status'] == 'ok':
                entry['scores'] = {d: review[f'review_{d}'] for d in DIMENSIONS}
                entry['overall'] = review_overall(review)
        except Exception as exc:
            entry['error'] = f'{type(exc).__name__}: {exc}'
        results.append(entry)
    return results


def averages(results):
    ok = [r for r in results if r['scores'] is not None]
    if not ok:
        return {'reviewed': 0, **{d: None for d in DIMENSIONS}, 'overall': None}
    avg = {d: sum(r['scores'][d] for r in ok) / len(ok) for d in DIMENSIONS}
    avg['overall'] = sum(r['overall'] for r in ok) / len(ok)
    avg['reviewed'] = len(ok)
    return avg


def fmt(v):
    return 'n/a' if v is None else f'{v:.3f}'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--grpo-adapter', default=f'{PILOT}/final_adapter_answer_grpo')
    p.add_argument('--data', default='finetuning/data/real_world/answer_grpo_data.jsonl')
    p.add_argument('--output-dir', default='finetuning/outputs/evaluations/answer_grpo_vs_baseline')
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    if (out_dir / 'comparison.json').exists():
        raise FileExistsError(f'{out_dir / "comparison.json"} exists; refusing to overwrite')
    if not (Path(args.grpo_adapter) / 'adapter_model.safetensors').is_file():
        raise FileNotFoundError(f'Missing GRPO adapter: {args.grpo_adapter}')

    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer

    from finetuning.grpo_tool_routing import load_quantized_base

    _, eval_rows = split_rows(load_rows(args.data))
    print(f'eval examples: {len(eval_rows)}')

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=False)
    base = load_quantized_base(BASE_MODEL)
    base.eval()
    base_answers = generate_answers(base, tokenizer, eval_rows)
    grpo_model = PeftModel.from_pretrained(base, args.grpo_adapter, is_trainable=False)
    grpo_model.eval()
    grpo_answers = generate_answers(grpo_model, tokenizer, eval_rows)
    del grpo_model, base
    gc.collect()
    torch.cuda.empty_cache()

    baseline = judge('baseline', eval_rows, base_answers)
    grpo = judge('grpo', eval_rows, grpo_answers)

    print(f"\n{'ex':>2} | {'query':<50} | baseline | grpo  | delta  | winner")
    wins = comparable = 0
    per_example = []
    for i, (row, b, g) in enumerate(zip(eval_rows, baseline, grpo)):
        delta = winner = None
        if b['overall'] is not None and g['overall'] is not None:
            delta = g['overall'] - b['overall']
            winner = 'grpo' if delta > 0 else 'baseline' if delta < 0 else 'tie'
            comparable += 1
            wins += winner == 'grpo'
        print(f"{i:>2} | {row['query'][:50]:<50} | {fmt(b['overall']):>8} | {fmt(g['overall']):>5} | "
              f"{'n/a' if delta is None else f'{delta:+.3f}':>6} | {winner or 'n/a'}")
        per_example.append({'example': i, 'query': row['query'], 'baseline': b, 'grpo': g,
                            'delta': delta, 'winner': winner})

    agg = {'baseline': averages(baseline), 'grpo': averages(grpo),
           'grpo_wins': wins, 'comparable_examples': comparable, 'total_examples': len(eval_rows)}
    print()
    for name in ('baseline', 'grpo'):
        a = agg[name]
        label = 'Baseline avg:' if name == 'baseline' else 'GRPO avg:    '
        print(f"{label} " + ', '.join(f'{d}={fmt(a[d])}' for d in DIMENSIONS) +
              f", overall={fmt(a['overall'])} ({a['reviewed']} reviewed)")
    print(f'GRPO wins: {wins}/{comparable} comparable examples ({len(eval_rows)} total)')

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'comparison.json').write_text(
        json.dumps({'grpo_adapter': args.grpo_adapter, 'examples': per_example, 'aggregate': agg},
                   indent=2, ensure_ascii=False), encoding='utf-8')
    print('Wrote', out_dir / 'comparison.json')
    project = os.getenv('LANGSMITH_PROJECT', 'research-agent-platform')
    print(f'LangSmith project: https://smith.langchain.com/projects/{project}')


if __name__ == '__main__':
    main()
