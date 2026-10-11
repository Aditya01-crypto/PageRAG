from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from httpx import AsyncClient
from pagerag.llm import ask_llm,event_generator
from pagerag.vectorstore import vector_search,get_source
from pagerag.config import limiter
from pagerag.schemas import QueryRequest, QueryResponse
from typing import Literal
from pagerag.hybrid import rrf_fusion,bm25_search
from pagerag.reranker import rerank

query_router = APIRouter(prefix='/query',tags=['query'])

@query_router.post('/',response_model=QueryResponse)
@limiter.limit('10/minute',)
async def ask_query(request:Request,query_req:QueryRequest):
    client: AsyncClient = request.app.state.http_client
    query=query_req.query.strip()
    limit = query_req.top_k
    context,sources = retrieve_relevant_chunk(request,query,fetch_limit=query_req.fetch_k,top_k=limit,mode=query_req.mode)
    if not context:
        raise HTTPException(400,"The search is empty. Please upload and process a document first")
    result = await ask_llm(context,sources,client,query,query_req.top_k,t_limit=query_req.token_limit)
    return result

@query_router.post('/stream')
@limiter.limit('10/minute')
async def ask_query_stream_rsp(request:Request,query_req:QueryRequest):
    client=request.app.state.http_client
    query = query_req.query
    limit = query_req.top_k
    context,sources = retrieve_relevant_chunk(request,query,fetch_limit=query_req.fetch_k,top_k=limit,mode=query_req.mode)
    if not context:
        raise HTTPException(400,"The search is empty. Please upload and process a document first")
    return StreamingResponse(event_generator(context,sources,client,query,query_req.top_k,query_req.token_limit),media_type="text/event-stream")

@query_router.post('/chunks')
def show_chunks(request:Request,query_req:QueryRequest):
    data = retrieve_relevant_chunk(request,query_req.query,fetch_limit=query_req.fetch_k,top_k=query_req.top_k,mode=query_req.mode)
    
    return {"data":str(data)}

def retrieve_relevant_chunk(request:Request,query:str,fetch_limit:int=20,top_k:int=5,mode:Literal["vector_only","hybrid","hybrid_rerank"] = "vector_only"):
    '''
    Provides Relevant Chunks based on given query and its source. 
    vector_only :  dense vector similarity search
    hybrid : vector + keyword search using RRF
    hybrid rerank :  (vector + BM25) refined with cross-encoder reranking
    '''
    collection = request.app.state.collection
    vector_result= vector_search(collection,query,fetch_limit=fetch_limit)
    source = get_source(collection,vector_result)

    if mode == 'vector_only':
        if vector_result:
            return (vector_result[:top_k],source)#pass sliced result for vector search

    if vector_result and (mode=='hybrid' or mode =='hybrid_rerank'):
        bm25,bm_chunks  = request.app.state.bm25obj,request.app.state.bm25chunk

        if bm25 is None or bm_chunks == []:
            raise HTTPException(400, detail= "The search is empty. Please upload and process a document first")
        bm25_result = bm25_search(bm25=bm25,chunks=bm_chunks,query=query,fetch_limit=fetch_limit)
        rrf_result = rrf_fusion(vector_result,bm25_result,limit=top_k)
        if mode == 'hybrid':
            source = get_source(collection,rrf_result)
            return (rrf_result,source) 
        
        if mode  == 'hybrid_rerank':
            rerank_result = rerank(query,rrf_result,top_k)
            source = get_source(collection,rerank_result)
            return (rerank_result,source) 
        
    return (None,[])