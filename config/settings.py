import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field


load_dotenv()


class ReviewThresholds(BaseModel):
    accuracy: float = Field(default=0.8, ge=0.0, le=1.0)
    relevance: float = Field(default=0.8, ge=0.0, le=1.0)
    clarity: float = Field(default=0.7, ge=0.0, le=1.0)
    completeness: float = Field(default=0.7, ge=0.0, le=1.0)


MAX_RETRY = 3
MAX_TOOL_ATTEMPTS = 3
REVIEW_THRESHOLDS = ReviewThresholds()

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen3:4b-instruct-2507-q4_K_M",
)

LLM_TEMPERATURE = float(
    os.getenv("LLM_TEMPERATURE", "0")
)

LANGSMITH_PROJECT = os.getenv(
    "LANGSMITH_PROJECT",
    "research-agent-platform",
)

TAVILY_API_KEY = os.getenv(
    "TAVILY_API_KEY",
    "",
)

NVIDIA_API_KEY = os.getenv(
    "NVIDIA_API_KEY",
    "",
)

NVIDIA_BASE_URL = os.getenv(
    "NVIDIA_BASE_URL",
    "https://integrate.api.nvidia.com/v1",
)

NVIDIA_REVIEWER_MODEL = os.getenv(
    "NVIDIA_REVIEWER_MODEL",
    "nvidia/nemotron-3.5-lightning-30b-a3b",
)

OLLAMA_EMBEDDING_MODEL = os.getenv(
    "OLLAMA_EMBEDDING_MODEL",
    "qwen3-embedding:0.6b",
)

CHROMA_PERSIST_DIR = os.getenv(
    "CHROMA_PERSIST_DIR",
    "data/chroma",
)

CHROMA_COLLECTION_NAME = os.getenv(
    "CHROMA_COLLECTION_NAME",
    "research_documents",
)

LLAMAINDEX_CHROMA_COLLECTION_NAME = os.getenv(
    "LLAMAINDEX_CHROMA_COLLECTION_NAME",
    "research_documents_llamaindex",
)

# Short-term conversational memory backend:
# - memory: InMemorySaver (process lifetime only)
# - sqlite: persistent SQLite checkpointer
MEMORY_BACKEND = os.getenv(
    "MEMORY_BACKEND",
    "memory",
).strip().lower()

SQLITE_CHECKPOINT_PATH = os.getenv(
    "SQLITE_CHECKPOINT_PATH",
    str(Path("data") / "checkpoints.sqlite"),
)

# How many recent conversation messages to include in prompts.
SHORT_TERM_MEMORY_MAX_MESSAGES = int(
    os.getenv("SHORT_TERM_MEMORY_MAX_MESSAGES", "12")
)

# Long-term durable memory (separate Chroma collection).
def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# Public demo mode: locks down risky tools and caps usage.
DEMO_MODE = _env_bool("DEMO_MODE", False)

ENABLE_LONG_TERM_MEMORY = (
    False
    if DEMO_MODE
    else _env_bool("ENABLE_LONG_TERM_MEMORY", True)
)

LONG_TERM_MEMORY_COLLECTION = os.getenv(
    "LONG_TERM_MEMORY_COLLECTION",
    "agent_long_term_memory",
)

LONG_TERM_MEMORY_TOP_K = int(
    os.getenv("LONG_TERM_MEMORY_TOP_K", "3")
)

LONG_TERM_MEMORY_DUPLICATE_THRESHOLD = float(
    os.getenv("LONG_TERM_MEMORY_DUPLICATE_THRESHOLD", "0.92")
)

MCP_CALL_TIMEOUT_SECONDS = float(
    os.getenv("MCP_CALL_TIMEOUT_SECONDS", "60")
)

DEFAULT_THREAD_ID = os.getenv(
    "DEFAULT_THREAD_ID",
    "default-session",
)


# --- LLM providers -------------------------------------------------
# researcher: ollama (local default) | nvidia | groq
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()

# Optional comma-separated fallback providers, tried in order.
LLM_FALLBACK_PROVIDERS = [
    name.strip().lower()
    for name in os.getenv("LLM_FALLBACK_PROVIDERS", "").split(",")
    if name.strip()
]

NVIDIA_RESEARCHER_MODEL = os.getenv(
    "NVIDIA_RESEARCHER_MODEL",
    "nvidia/nemotron-3-super-120b-a12b",
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = os.getenv(
    "GROQ_BASE_URL",
    "https://api.groq.com/openai/v1",
)
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# --- Embeddings ----------------------------------------------------
# ollama (local default) | nvidia (hosted, used by the public demo)
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "ollama").strip().lower()
NVIDIA_EMBEDDING_MODEL = os.getenv(
    "NVIDIA_EMBEDDING_MODEL",
    "nvidia/nemotron-3-embed-1b",
)

# --- Tool gating ---------------------------------------------------
# Tools that are off unless explicitly enabled.
ENABLE_BROWSER_TOOL = _env_bool("ENABLE_BROWSER_TOOL", not DEMO_MODE)
ENABLE_LLAMAINDEX_TOOL = _env_bool("ENABLE_LLAMAINDEX_TOOL", not DEMO_MODE)

# file_reader may only read below this directory.
FILE_READER_ROOT = Path(
    os.getenv("FILE_READER_ROOT", "datasets")
).resolve()

# --- Demo usage caps -----------------------------------------------
DEMO_MAX_QUESTIONS_PER_SESSION = int(
    os.getenv("DEMO_MAX_QUESTIONS_PER_SESSION", "5")
)
DEMO_MAX_QUESTIONS_TOTAL = int(
    os.getenv("DEMO_MAX_QUESTIONS_TOTAL", "300")
)
