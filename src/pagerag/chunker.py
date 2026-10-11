'''
Provides utility function for chunking text data
'''
from pagerag.schemas import DocumentResult,Chunk
from datetime import datetime
from hashlib import md5

def generate_chunkid(chunk:str)->str:
    enc_chunk = chunk.encode('utf-8')
    hasher = md5(enc_chunk)
    return hasher.hexdigest()

def chunk_text(text:str,size:int,overlap:int):
    if overlap>=size:
        raise ValueError("Overlap must be less than chunk size")

    word_gen=(w for w in text.split())
    chunk=[]
    slice_start=-overlap if overlap>0 else size
    for word in word_gen:
        chunk.append(word)
        if len(chunk)>=size:
            yield chunk
            chunk=chunk[slice_start:]
    if chunk and len(chunk)>overlap:
        yield chunk

def get_count(chunk:list[str]):
    ''' A utility function that returns word_count:char_count for 
        given chunk'''
    
    word_count=len(chunk)
    char_count=len("".join(chunk))

    return (word_count,char_count)

def build_document_result(filename:str,file_type:str,text,chunk_size,overlap)->DocumentResult:
    if text and  isinstance(text,str):
        chunks_generator=chunk_text(text,chunk_size,overlap)
        id,total_char_count,total_word_count=1,0,0
        chunks,now=[],datetime.now()

        for chunk in chunks_generator:
            word_count,char_count=get_count(chunk)

            total_char_count+=char_count
            total_word_count+=word_count
            chunk_obj=Chunk(chunk_id=generate_chunkid(" ".join(chunk)),text=chunk,word_count=word_count,char_count=char_count)

            chunks.append(chunk_obj)
            

        return DocumentResult(filename=filename,file_type=file_type,total_chars=total_char_count,total_words=total_word_count,chunks=chunks,processed_at=now)
    else:
        now=datetime.now()
        return DocumentResult(filename=filename,file_type=file_type,total_chars=0,total_words=0,chunks=[],processed_at=now,error=f"Couldn't extract content from {filename},Reason: {str(text)}")
