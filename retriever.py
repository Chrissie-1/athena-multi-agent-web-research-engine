"""Retrieval: embed the scraped chunks and keep only what answers the sub-question.

This is the RAG step. Feeding the Writer every scraped chunk would blow the context
window; ranking by embedding similarity keeps roughly 1.5k tokens of the best evidence.
"""

import os
import sys
from typing import List, Tuple

if sys.platform == "win32":
    # faiss-cpu and torch each bundle their own OpenBLAS. On Windows their thread
    # pools collide and encode() dies mid-batch with a bogus "not enough memory"
    # allocation failure (or a segfault). Capping BLAS threads before torch is
    # imported avoids it; override these in the environment if you know better.
    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_var, "1")

import numpy as np
from sentence_transformers import SentenceTransformer

MAX_CANDIDATES = 400  # embedding every chunk of five long pages is not worth the latency
BATCH_SIZE = 32

_model = None


def get_model() -> SentenceTransformer:
    """all-MiniLM-L6-v2: ~80MB, runs on CPU, free. Loaded once per process."""
    global _model
    if _model is None:
        print("[embed] loading all-MiniLM-L6-v2 (first call takes ~10s)")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def retrieve_relevant_chunks(
    chunks: List[Tuple[str, str]], query: str, top_k: int = 3
) -> List[Tuple[str, str]]:
    """Rank (chunk, url) pairs against the query and return the top_k."""
    if not chunks:
        return []
    if len(chunks) <= top_k:
        return chunks

    chunks = chunks[:MAX_CANDIDATES]
    texts = [c for c, _ in chunks]
    model = get_model()
    embeddings = np.asarray(
        model.encode(texts, normalize_embeddings=True, batch_size=BATCH_SIZE), dtype="float32"
    )
    query_embedding = np.asarray(model.encode([query], normalize_embeddings=True), dtype="float32")

    try:
        import faiss

        index = faiss.IndexFlatIP(embeddings.shape[1])  # vectors are normalized -> cosine
        index.add(embeddings)
        _, indices = index.search(query_embedding, min(top_k, len(chunks)))
        ranked = [int(i) for i in indices[0] if 0 <= i < len(chunks)]
    except ImportError:
        # faiss is a convenience at this scale; numpy gives the same ranking.
        scores = embeddings @ query_embedding[0]
        ranked = np.argsort(-scores)[:top_k].tolist()

    return [chunks[i] for i in ranked]
