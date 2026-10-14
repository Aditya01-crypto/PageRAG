from sentence_transformers import CrossEncoder
from pagerag.schemas import Hit

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')



def rerank(query: str, hits: list[Hit], top_k: int = 5) -> list[Hit]:
    if not hits:
        return []
    scores = reranker.predict([(query, h.text) for h in hits])
    order = sorted(range(len(hits)), key=lambda i: scores[i], reverse=True)
    return [hits[i] for i in order[:top_k]]