from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from httpx import AsyncClient
from pagerag.llm import ask_llm,event_generator
from pagerag.config import limiter
from pagerag.schemas import QueryRequest, QueryResponse
from pagerag.retrieve import retrieve_relevant_chunk
import os
import asyncio

query_router = APIRouter(prefix='/query',tags=['query'])
DEBUG = os.getenv('DEBUG',"False").lower() in ('true','y','t','1')

@query_router.post('/',response_model=QueryResponse)
@limiter.limit('10/minute',)
async def ask_query(request:Request,query_req:QueryRequest):
    client: AsyncClient = request.app.state.http_client
    collection = request.app.state.collection
    bm25state = request.app.state.bm25
    query=query_req.query.strip()
    limit = query_req.top_k
    context,sources = await asyncio.to_thread(retrieve_relevant_chunk,collection,bm25state,query,fetch_limit=query_req.fetch_k,top_k=limit,mode=query_req.mode)
    if not context:
        raise HTTPException(400,"The search is empty. Please upload and process a document first")
    result = await ask_llm(context,sources,client,query,query_req.top_k,t_limit=query_req.token_limit)
    return result

@query_router.post('/stream')
@limiter.limit('10/minute')
async def ask_query_stream_rsp(request:Request,query_req:QueryRequest):
    client=request.app.state.http_client
    collection = request.app.state.collection
    bm25state = request.app.state.bm25
    query = query_req.query
    limit = query_req.top_k
    context,sources = await asyncio.to_thread(retrieve_relevant_chunk,collection,bm25state,query,fetch_limit=query_req.fetch_k,top_k=limit,mode=query_req.mode)
    if not context:
        raise HTTPException(400,"The search is empty. Please upload and process a document first")
    return StreamingResponse(event_generator(context,sources,client,query,query_req.top_k,query_req.token_limit),media_type="text/event-stream")

if DEBUG:
    @query_router.post('/chunks')
    async def show_chunks(request:Request,query_req:QueryRequest):
        collection = request.app.state.collection
        bm25state = request.app.state.bm25
        data = await asyncio.to_thread(retrieve_relevant_chunk,collection,bm25state,query_req.query,fetch_limit=query_req.fetch_k,top_k=query_req.top_k,mode=query_req.mode)
        
        return {"data":str(data)}

