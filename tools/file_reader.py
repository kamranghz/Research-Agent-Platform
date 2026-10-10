from pathlib import Path

from langchain_core.tools import tool

from config.settings import FILE_READER_ROOT


MAX_FILE_CHARS = 20000


@tool
def file_reader(file_path: str) -> str:
    """Read a UTF-8 .txt or .md file from the datasets directory."""

    # Resolve symlinks and ".." before checking the sandbox root.
    path = Path(file_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()

    root = FILE_READER_ROOT.resolve()
    if not path.is_relative_to(root):
        return (
            "Error: access denied. Only files inside the "
            f"'{root.name}/' directory can be read."
        )

    if not path.exists():
        return f"Error: file not found: {file_path}"

    if not path.is_file():
        return f"Error: path is not a file: {file_path}"

    if path.suffix.lower() not in {".txt", ".md"}:
        return "Error: only .txt and .md files are supported."

    try:
        return path.read_text(encoding="utf-8")[:MAX_FILE_CHARS]
    except UnicodeDecodeError:
        return "Error: file is not valid UTF-8 text."
    except OSError as exc:
        return f"Error: failed to read file: {exc}"
