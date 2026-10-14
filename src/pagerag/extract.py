import asyncio
import functools
import re
import time
from pathlib import Path
from docx import Document
from pypdf import PdfReader
import unicodedata

def clean_text(func): # decorator that cleans the text output
    @functools.wraps(func)
    async def wrapper(*args,**kwargs):
        text= await func(*args,**kwargs)
        if not isinstance(text,str):
            return text
        text = unicodedata.normalize('NFKC',text)
        text=re.sub(r'\s+'," ",text)
        return text
    return wrapper

def log(log_path):# decorator to calculate elapsed time for each file 
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args,**kwargs):
            print(args)
            filepath=args[0] if args else kwargs.get('f_path')
            filename=filepath.name #type:ignore 
            
            start=time.perf_counter()
            r=await func(*args,**kwargs)
            elapsed=f"{time.perf_counter()-start:.4f}"
            with open(f'{log_path}','a') as f:
                f.write(f"{filename} elapsed time => {elapsed}\n")
            return r
        return wrapper
    return decorator
    
def sync_pdf_extract(f_path:Path)->str|None:
    pages=[]
    try:
        reader=PdfReader(f_path)
        for page in reader.pages:
                text=page.extract_text()
                if text:
                    pages.append(text)
    except Exception as e:
        print(f"Could n't process file  : {f_path.name}")
        return 
    
    return "\n".join(pages)

def sync_docx_extract(f_path:Path)->str|None:
    try:
        doc=Document(f_path)#type: ignore
        paragraphs=[p.text for p in doc.paragraphs if p.text.strip()]
    except Exception as e :
        print(f"Could n't process file  : {f_path.name}")
        return 
    return "\n".join(paragraphs)
        

@log('meta.log')
@clean_text
async def extract_pdf_text(f_path:Path):
    loop=asyncio.get_running_loop()
    text_data=await loop.run_in_executor(None,sync_pdf_extract,f_path)
    return text_data

@log('meta.log')
@clean_text
async def extract_docx_text(f_path:Path):
    loop=asyncio.get_running_loop()
    text_data=await loop.run_in_executor(None,sync_docx_extract,f_path)
    return text_data    

EXTRACTORS ={
    ".pdf": extract_pdf_text,
    ".docx": extract_docx_text
}

def is_supported(name:str|Path)->bool:
    return Path(name).suffix.lower() in EXTRACTORS

def give_extractor(path:Path):
    return EXTRACTORS[path.suffix.lower()]

async def get_file_content(f_path:Path):
    text_data=None
    if is_supported(f_path):
        text_data = await give_extractor(f_path)(f_path)

    return text_data if type(text_data) is str and len(text_data)>0 else None