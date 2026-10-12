from typing import Literal
from pagerag.hybrid import rrf_fusion,bm25_search,BM25State
from pagerag.reranker import rerank
from pagerag.vectorstore import vector_search,get_source
from chromadb import Collection
from fastapi import HTTPException

def retrieve_relevant_chunk(collection:Collection,bm25:BM25State,query:str,fetch_limit:int=20,top_k:int=5,mode:Literal["vector_only","hybrid","hybrid_rerank"] = "vector_only"):
    '''
    Provides Relevant Chunks based on given query and its source. 
    vector_only :  dense vector similarity search
    hybrid : vector + keyword search using RRF
    hybrid rerank :  (vector + BM25) refined with cross-encoder reranking
    '''

    vector_result= vector_search(collection,query,fetch_limit=fetch_limit)
    source = get_source(collection,vector_result)

    if mode == 'vector_only':
        if vector_result:
            return (vector_result[:top_k],source)#pass sliced result for vector search

    if vector_result and (mode=='hybrid' or mode =='hybrid_rerank'):

        if bm25.index is None or bm25.chunks == []:
            raise HTTPException(400, detail= "The search is empty. Please upload and process a document first")
        bm25_result = bm25_search(bm25=bm25.index,chunks=bm25.chunks,query=query,fetch_limit=fetch_limit)
        rrf_result = rrf_fusion(vector_result,bm25_result,limit=top_k)
        if mode == 'hybrid':
            source = get_source(collection,rrf_result)
            return (rrf_result,source) 
        
        if mode  == 'hybrid_rerank':
            rerank_result = rerank(query,rrf_result,top_k)
            source = get_source(collection,rerank_result)
            return (rerank_result,source) 
        
    return (None,[])