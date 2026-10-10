#!/bin/sh
# Build the demo vector index on first start, then launch Streamlit.
# Indexing is best-effort: if it fails the app still starts and the
# RAG tool reports its own error when called.
set -u

_chroma_empty() {
    [ ! -d "${CHROMA_PERSIST_DIR}" ] \
        || [ -z "$(ls -A "${CHROMA_PERSIST_DIR}" 2>/dev/null)" ]
}

if _chroma_empty; then
    echo ">>> indexing demo documents..."
    python scripts/ingest_documents.py datasets/sample.pdf --reset \
        || echo ">>> indexing failed; continuing without an index"
fi

exec streamlit run ui/app.py \
    --server.port="${PORT:-8501}" \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false
