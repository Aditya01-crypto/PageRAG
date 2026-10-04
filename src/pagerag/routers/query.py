from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from httpx import AsyncClient
from pagerag.llm import ask_llm,event_generator
from pagerag.vectorstore import retrieve_relevant_chunk
from pagerag.config import limiter
from pagerag.schemas import QueryRequest, QueryResponse
query_router = APIRouter(prefix='/query',tags=['query'])

@query_router.post('/',response_model=QueryResponse)
@limiter.limit('10/minute',)
async def ask_query(request:Request,query_req:QueryRequest):
    client: AsyncClient = request.app.state.http_client
    query=query_req.query.strip()
    collection = request.app.state.collection
    result = await ask_llm(collection,client,query,query_req.chunk_limit,t_limit=query_req.token_limit)
    return result

@query_router.post('/stream')
@limiter.limit('10/minute')
async def ask_query_stream_rsp(request:Request,query_req:QueryRequest):
    client=request.app.state.http_client
    collection = request.app.state.collection
    query = query_req.query
    return StreamingResponse(event_generator(collection,client,query,query_req.chunk_limit,query_req.token_limit),media_type="text/event-stream")