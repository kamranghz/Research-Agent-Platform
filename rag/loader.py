import hashlib
import re
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader
from pypdf.errors import PdfReadError


SUPPORTED_EXTENSIONS = {
    ".pdf",
}


def _create_document_id(file_path: Path) -> str:
    """
    Create a deterministic document ID from the file path
    and file content.

    When the PDF content changes, the document ID changes too.
    """

    file_hash = hashlib.sha256()

    file_hash.update(
        str(file_path.resolve()).encode("utf-8")
    )

    with file_path.open("rb") as pdf_file:
        while True:
            block = pdf_file.read(1024 * 1024)

            if not block:
                break

            file_hash.update(block)

    return file_hash.hexdigest()[:20]


def _clean_page_text(text: str) -> str:
    """
    Normalize text extracted from a PDF page while
    preserving paragraph boundaries.
    """

    normalized_text = text.replace("\x00", "")

    # Remove unnecessary spaces before line breaks.
    normalized_text = re.sub(
        r"[ \t]+\n",
        "\n",
        normalized_text,
    )

    # Replace three or more blank lines with two.
    normalized_text = re.sub(
        r"\n{3,}",
        "\n\n",
        normalized_text,
    )

    return normalized_text.strip()


def _open_pdf(file_path: Path) -> PdfReader:
    """
    Open a PDF and handle unreadable or encrypted files.
    """

    try:
        reader = PdfReader(str(file_path))

    except PdfReadError as error:
        raise ValueError(
            f"Invalid or corrupted PDF: {file_path}"
        ) from error

    except OSError as error:
        raise OSError(
            f"Failed to open PDF: {file_path}"
        ) from error

    if reader.is_encrypted:
        try:
            decrypt_result = reader.decrypt("")
        except Exception as error:
            raise ValueError(
                f"Encrypted PDF cannot be opened: {file_path}"
            ) from error

        if decrypt_result == 0:
            raise ValueError(
                f"PDF requires a password: {file_path}"
            )

    return reader


def _load_pdf(file_path: Path) -> list[Document]:
    """
    Load one PDF and return one LangChain Document
    for each page containing extractable text.
    """

    reader = _open_pdf(file_path)

    document_id = _create_document_id(file_path)
    resolved_path = str(file_path.resolve())
    total_pages = len(reader.pages)

    documents: list[Document] = []

    for page_index, page in enumerate(reader.pages):
        page_number = page_index + 1

        try:
            extracted_text = page.extract_text() or ""

        except Exception as error:
            raise ValueError(
                f"Failed to extract text from page "
                f"{page_number} of {file_path}"
            ) from error

        clean_text = _clean_page_text(extracted_text)

        # Ignore blank pages.
        if not clean_text:
            continue

        documents.append(
            Document(
                page_content=clean_text,
                metadata={
                    "document_id": document_id,
                    "source": resolved_path,
                    "file_name": file_path.name,
                    "file_type": "pdf",
                    "page_number": page_number,
                    "page_index": page_index,
                    "total_pages": total_pages,
                },
            )
        )

    if not documents:
        raise ValueError(
            f"No extractable text was found in {file_path}. "
            "The PDF may be scanned or image-only and may "
            "require OCR."
        )

    return documents


def load_documents(
    source_path: str | Path,
) -> list[Document]:
    """
    Load PDF documents from one file or directory.

    When a directory is provided, PDF files are discovered
    recursively.

    Each PDF page becomes one LangChain Document so page-level
    metadata and citations can be preserved.
    """

    path = Path(source_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Document source does not exist: {path}"
        )

    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported document type: {path.suffix}. "
                "Only PDF files are supported."
            )

        return _load_pdf(path)

    pdf_paths = sorted(
        file_path
        for file_path in path.rglob("*")
        if (
            file_path.is_file()
            and file_path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        )
    )

    documents: list[Document] = []

    for pdf_path in pdf_paths:
        documents.extend(
            _load_pdf(pdf_path)
        )

    return documents