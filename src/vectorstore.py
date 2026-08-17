from pathlib import Path
import hashlib

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PERSIST_DIR = str(PROJECT_ROOT / "chroma_db")
COLLECTION_NAME = "documents"

EMBED_MODEL = "qwen3-embedding:0.6b"

# Fewer, slightly larger chunks = faster indexing for large PDFs.
CHUNK_SIZE = 900
CHUNK_OVERLAP = 100
EMBED_BATCH_SIZE = 32


def split_documents(
    documents,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"[CHUNKS] {len(documents)} docs -> {len(chunks)} chunks")
    return chunks


def get_embeddings():
    return OllamaEmbeddings(model=EMBED_MODEL)


def get_vectorstore():
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=PERSIST_DIR,
    )


def make_document_id(document):
    source = document.metadata.get("source", "unknown")
    page = document.metadata.get("page", "")
    raw = f"{source}::{page}::{document.page_content}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def add_documents(chunks):
    if not chunks:
        return 0

    store = get_vectorstore()
    ids = [make_document_id(c) for c in chunks]

    existing = store._collection.get(ids=ids, include=[])
    existing_ids = set(existing.get("ids", []))

    new_chunks = [
        (chunk, doc_id)
        for chunk, doc_id in zip(chunks, ids)
        if doc_id not in existing_ids
    ]

    if not new_chunks:
        print("[CHROMA] No new chunks to embed.")
        return 0

    docs = [x[0] for x in new_chunks]
    doc_ids = [x[1] for x in new_chunks]

    total = len(docs)
    for start in range(0, total, EMBED_BATCH_SIZE):
        end = min(start + EMBED_BATCH_SIZE, total)
        print(f"[EMBEDDING] {start + 1}-{end}/{total}")
        store.add_documents(docs[start:end], ids=doc_ids[start:end])

    print(f"[CHROMA] Added {total} new chunks.")
    return total


def create_vectorstore(chunks=None):
    store = get_vectorstore()
    if chunks:
        add_documents(chunks)
    return store


def count_documents():
    return get_vectorstore()._collection.count()