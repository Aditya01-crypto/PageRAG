from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


def rerank(query:str,chunks:tuple|list,top_k:int=5):
    query_list = len(chunks)*[query]
    pairs = list(zip(query_list,chunks))
    scores = reranker.predict(pairs)
    rerank_pair = zip(chunks,scores)
    rerank_pair = sorted(rerank_pair,key=lambda x:x[1],reverse=True)
    reranked_chunks,reranked_scores = zip(*rerank_pair)

    return reranked_chunks[:top_k]
