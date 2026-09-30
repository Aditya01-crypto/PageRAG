from fastapi import APIRouter,File,UploadFile,HTTPException
from pagerag.config import UPLOAD_DIR
import shutil
from pathlib import Path
from math import ceil
from pagerag.vectorstore import add_document,list_document,delete_document
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
async def upload_document(file:UploadFile=File(...)):
    await file_validation(file)

    filename=(file.filename)
    filepath=UPLOAD_DIR/filename #type: ignore 

    with filepath.open("wb") as buffer:
        shutil.copyfileobj(file.file,buffer)

    response = await add_document(filepath)
    filepath.unlink() # to clean directory after processing
    if response:
        return response
    else:
        raise HTTPException(400,"File has no extractable content!")

@router.get('/')
def show_documents(skip:int=0,limit:int=10):
    result = list_document() 
    if result:
        if len(result)>skip:
            return result[skip:limit+skip]
    return[] 

@router.delete('/{doc_name}')
def remove_document(doc_name:str):
    result=delete_document(doc_name)
    if result:
        return {"Message":f"{doc_name} deleted succesfully","Chunks Removed":result}
    else:
        raise HTTPException(404,f"Document '{doc_name}' not found")
    