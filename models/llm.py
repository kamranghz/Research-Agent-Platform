from langchain_ollama import ChatOllama

from config.settings import LLM_TEMPERATURE, OLLAMA_MODEL


llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=LLM_TEMPERATURE,
)