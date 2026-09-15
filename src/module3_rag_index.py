from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_MODEL_CACHE: dict[str, object] = {}


def split_guideline_document(
    document_text: str,
    min_tokens: int = 300,
    max_tokens: int = 500,
    overlap_tokens: int = 50,
    target_chunk_tokens: int = 400,
) -> list[str]:
    if not isinstance(document_text, str) or not document_text.strip():
        raise ValueError("document_text must be a non-empty string.")
    if min_tokens <= 0 or max_tokens < min_tokens:
        raise ValueError("Invalid token range: min_tokens must be >0 and <= max_tokens.")
    if overlap_tokens < 0 or overlap_tokens >= target_chunk_tokens:
        raise ValueError("overlap_tokens must be >=0 and smaller than target_chunk_tokens.")
    if not (min_tokens <= target_chunk_tokens <= max_tokens):
        raise ValueError("target_chunk_tokens must be within [min_tokens, max_tokens].")

    tokens = document_text.split()
    total_tokens = len(tokens)

    if total_tokens <= max_tokens:
        return [" ".join(tokens)]

    stride = target_chunk_tokens - overlap_tokens
    chunks: list[str] = []
    starts: list[int] = []
    start = 0

    while start < total_tokens:
        end = min(start + target_chunk_tokens, total_tokens)

        if chunks and (total_tokens - start) < min_tokens:
            start = max(starts[-1] + 1, total_tokens - min_tokens)
            end = total_tokens

        if starts and start <= starts[-1]:
            break

        chunks.append(" ".join(tokens[start:end]))
        starts.append(start)

        if end == total_tokens:
            break

        next_start = start + stride
        if (total_tokens - next_start) < min_tokens:
            next_start = max(start + 1, total_tokens - min_tokens)
        start = next_start

    return chunks


def _load_sentence_transformer(model_name: str):
    if model_name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer

        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def _embed_texts_with_model(texts: list[str], model_name: str) -> np.ndarray:
    model = _load_sentence_transformer(model_name)
    embeddings = model.encode(texts, normalize_embeddings=True)
    return np.asarray(embeddings, dtype=float)


def build_guideline_index(
    guideline_path: Path | str,
    persist_directory: Path | str = Path("outputs/chroma"),
    collection_name: str = "guideline_chunks",
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    embedding_function: Callable[[list[str]], np.ndarray] | None = None,
    min_tokens: int = 300,
    max_tokens: int = 500,
    overlap_tokens: int = 50,
    target_chunk_tokens: int = 400,
) -> dict[str, object]:
    guideline_path = Path(guideline_path)
    if not guideline_path.exists():
        raise FileNotFoundError(f"Guideline document not found: {guideline_path}")

    document_text = guideline_path.read_text(encoding="utf-8")
    chunks = split_guideline_document(
        document_text=document_text,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        target_chunk_tokens=target_chunk_tokens,
    )

    embedder = embedding_function or (lambda texts: _embed_texts_with_model(texts, model_name))
    embeddings = np.asarray(embedder(chunks), dtype=float)

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array.")
    if embeddings.shape[0] != len(chunks):
        raise ValueError("Number of embeddings must match number of chunks.")

    import chromadb

    persist_directory = Path(persist_directory)
    persist_directory.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_directory))
    collection = client.get_or_create_collection(
        name=collection_name, metadata={"hnsw:space": "cosine"}
    )

    ids = [f"chunk_{index}" for index in range(len(chunks))]
    metadatas = [{"chunk_index": index} for index in range(len(chunks))]
    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )

    return {
        "collection_name": collection_name,
        "persist_directory": persist_directory,
        "chunks": chunks,
        "embedding_count": embeddings.shape[0],
    }


def query_guidelines(
    question: str,
    persist_directory: Path | str = Path("outputs/chroma"),
    collection_name: str = "guideline_chunks",
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    top_k: int = 3,
    embedding_function: Callable[[list[str]], np.ndarray] | None = None,
    latency_log_path: Path | str | None = None,
) -> dict[str, object]:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string.")
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    import chromadb

    client = chromadb.PersistentClient(path=str(Path(persist_directory)))
    collection = client.get_collection(name=collection_name)

    if collection.count() < top_k:
        raise ValueError("Collection has fewer chunks than requested top_k.")

    embedder = embedding_function or (lambda texts: _embed_texts_with_model(texts, model_name))
    query_embedding = np.asarray(embedder([question]), dtype=float)
    if query_embedding.ndim != 2 or query_embedding.shape[0] != 1:
        raise ValueError("Query embedding function must return shape (1, embedding_dim).")

    start_time = perf_counter()
    result = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k,
        include=["documents", "distances"],
    )
    latency_seconds = max(perf_counter() - start_time, 1e-9)

    log_path = (
        Path(latency_log_path)
        if latency_log_path is not None
        else Path(persist_directory) / "query_latency_log.jsonl"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(
            json.dumps(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "question": question,
                    "collection_name": collection_name,
                    "top_k": top_k,
                    "latency_seconds": latency_seconds,
                }
            )
            + "\n"
        )

    chunks = result["documents"][0]
    distances = result["distances"][0]

    return {
        "chunks": chunks,
        "distances": distances,
        "latency_seconds": latency_seconds,
        "latency_log_path": log_path,
    }
