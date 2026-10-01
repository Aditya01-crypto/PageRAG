from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from httpx import AsyncClient
from pagerag.llm import ask_llm,event_generator
from pagerag.vectorstore import retrieve_relevant_chunk
from pagerag.config import limiter
query_router = APIRouter(prefix='/query',tags=['query'])

@query_router.post('/')
@limiter.limit('10/minute')
async def ask_query(request:Request,query:str,chunk_limit:int=4,token_limit:int=2000):
    client: AsyncClient = request.app.state.http_client
    query=query.strip()
    result = await ask_llm(client,query,chunk_limit,t_limit=token_limit)
    return result

@query_router.post('/stream')
@limiter.limit('10/minute')
async def ask_query_stream_rsp(request:Request,query:str,chunk_limit:int=4,max_tokens:int=2000):
    client=request.app.state.http_client
    return StreamingResponse(event_generator(client,query,chunk_limit,max_tokens),media_type="text/event-stream")