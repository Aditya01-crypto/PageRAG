from fastapi import FastAPI,Request
from pagerag.routers.documents import router
import uvicorn
from httpx import AsyncClient
from contextlib import asynccontextmanager
from pagerag.routers.query import query_router
@asynccontextmanager
async def lifespan(app:FastAPI):
    app.state.http_client= AsyncClient(timeout=30.0)
    yield
    await app.state.http_client.aclose()

app = FastAPI(lifespan=lifespan)
app.include_router(router)
app.include_router(query_router)

def start():
    uvicorn.run(app="pagerag.main:app",host='127.0.0.1',port=8000,reload=True)

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
    return 


if __name__ == '__main__':
    start()