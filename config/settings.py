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
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
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
ENABLE_LONG_TERM_MEMORY = os.getenv(
    "ENABLE_LONG_TERM_MEMORY",
    "true",
).strip().lower() in {"1", "true", "yes", "on"}

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
