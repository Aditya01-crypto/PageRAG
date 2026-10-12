from fastapi import APIRouter,File,UploadFile,HTTPException,Request
from pagerag.config import UPLOAD_DIR
import shutil
from pathlib import Path
from math import ceil
from pagerag.vectorstore import add_document,list_document,delete_document
from pagerag.hybrid import build_bm25_index
router = APIRouter(prefix='/documents',tags=['documents'])



UPLOAD_DIR.mkdir(exist_ok=True,parents=True)

async def file_validation(file:UploadFile):
    size_threshold = 20 # safety feature : files greater than threshold not entertained
    if not file.filename:
        raise HTTPException(404,'Got None')

    filename=Path(file.filename)
    valid_file_types = {".pdf",".docx"}

    if filename.suffix not in valid_file_types:
        raise HTTPException(400,"File format not supported")

    content= await file.read()
    size_inmb = ceil(len(content)/1048576) # 1048576 bytes = 1mb
    if size_inmb>size_threshold:
        raise HTTPException(400,f"File size should be of {size_threshold} or less")
    await file.seek(0)

@router.post('/upload')
async def upload_document(request:Request,file:UploadFile=File(...)):
    await file_validation(file)
    collection = request.app.state.collection
    filename=(file.filename)
    filepath=UPLOAD_DIR/filename #type: ignore 

    with filepath.open("wb") as buffer:
        shutil.copyfileobj(file.file,buffer)

    response = await add_document(collection,filepath)
    filepath.unlink() # to clean directory after processing
    request.app.state.bm25 = build_bm25_index(collection)
    if response:
        return response
    else:
        raise HTTPException(400,"File has no extractable content!")

@router.get('/')
def show_documents(request:Request,skip:int=0,limit:int=10):
    collection = request.app.state.collection
    result = list_document(collection) 
    if result:
        if len(result)>skip:
            return result[skip:limit+skip]
    return[] 

@router.delete('/{doc_name}')
def remove_document(request:Request,doc_name:str):
    collection = request.app.state.collection
    result=delete_document(collection,doc_name)
    request.app.state.bm25 = build_bm25_index(collection)
    if result:
        return {"Message":f"{doc_name} deleted succesfully","Chunks Removed":result}
    else:
        raise HTTPException(404,f"Document '{doc_name}' not found")
     