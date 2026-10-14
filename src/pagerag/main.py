from fastapi import FastAPI,Request
from pagerag.routers.documents import router
import uvicorn
from httpx import AsyncClient
from contextlib import asynccontextmanager
from pagerag.routers.query import query_router
from pagerag.hybrid import build_bm25_index
import chromadb
import os
@asynccontextmanager
async def lifespan(app:FastAPI):
    app.state.http_client= AsyncClient(timeout=30.0)
    app.state.chroma_client= chromadb.PersistentClient(path="./doc_db")
    app.state.collection = app.state.chroma_client.get_or_create_collection("documents",metadata={"hnsw:space":"cosine"})
    app.state.bm25 = build_bm25_index(app.state.collection)

    yield
    await app.state.http_client.aclose()

app = FastAPI(lifespan=lifespan)
app.include_router(router)
app.include_router(query_router)

reload = os.getenv('RELOAD','False').lower() in ('true','t','y','1')

def start():
    uvicorn.run(app="pagerag.main:app",host='127.0.0.1',port=8000,reload=reload)

@app.get('/')
def index(): 
    endpoint ={
        "intro": "GET /",
        "docs" : 'GET /docs',
        "documents" : 'GET /documents',
        "delete" : "DELETE /documents/{name}",
        "upload" : "POST /documents/upload",
        "query" : "POST /query",
        "stream query" : "POST /query/stream"
    }
    return  endpoint


if __name__ == '__main__':
    start()