from fastapi import FastAPI
from pagerag.routers.documents import router
import uvicorn
app = FastAPI()
app.include_router(router)

def start():
    uvicorn.run(app="pagerag.main:app",host='127.0.0.1',port=8000,reload=True)

@app.get('/')
def index():
    return "This is main endpoint"


if __name__ == '__main__':
    start()