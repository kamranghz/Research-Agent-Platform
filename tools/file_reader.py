from pathlib import Path

from langchain_core.tools import tool


@tool
def file_reader(file_path: str) -> str:
    """Read a UTF-8 text or Markdown file and return its content."""

    path = Path(file_path)

    if not path.exists():
        return f"Error: file not found: {file_path}"

    if not path.is_file():
        return f"Error: path is not a file: {file_path}"

    if path.suffix.lower() not in {".txt", ".md"}:
        return "Error: only .txt and .md files are supported."

    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return "Error: file is not valid UTF-8 text."
    except OSError as exc:
        return f"Error: failed to read file: {exc}"