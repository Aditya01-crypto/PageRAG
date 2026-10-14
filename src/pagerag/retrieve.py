from typing import Literal
from chromadb import Collection
from pagerag.hybrid import BM25State, bm25_search, rrf_fusion
from pagerag.reranker import rerank
from pagerag.schemas import Hit
from pagerag.vectorstore import vector_search

Mode = Literal["vector_only", "hybrid", "hybrid_rerank"]

def retrieve_hits(collection: Collection, state: BM25State, query: str,
                  fetch_limit: int = 20, top_k: int = 5,
                  mode: Mode = "vector_only") -> list[Hit]:
    vector_hits = vector_search(collection, query, fetch_limit)
    if mode == "vector_only" or not vector_hits:
        return vector_hits[:top_k]

    bm25_hits = bm25_search(state, query, fetch_limit)
    if mode == "hybrid":
        return rrf_fusion(vector_hits, bm25_hits, limit=top_k)

    candidates = rrf_fusion(vector_hits, bm25_hits, limit=fetch_limit)
    return rerank(query, candidates, top_k)

def retrieve_relevant_chunk(collection, state, query, fetch_limit=20, top_k=5, mode:Mode="vector_only"):
    hits = retrieve_hits(collection, state, query, fetch_limit, top_k, mode)
    context = [h.text for h in hits]
    sources = list(dict.fromkeys(h.source for h in hits))   # deduped, keeps rank order
    return context, sources