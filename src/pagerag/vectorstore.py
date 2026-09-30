from pathlib import Path
from chromadb import PersistentClient,GetResult
from pagerag.extract import get_file_content
from pagerag.chunker import build_document_result
from pagerag.schemas import Chunk,CollectionData,DocumentResponse
from pagerag.embeddings import embed_text
import numpy as np



client = PersistentClient(path="./doc_db")
collection = client.get_or_create_collection("documents",metadata={"hnsw:space":"cosine"})



def process_chunks(filename:str,chunks:list[Chunk]):
    ids=[]
    embeddings=[]
    documents = []
    metadatas=[]
  
    for chunk in chunks:
        cid = chunk.chunk_id
        ids.append(f"{filename}_ch_"+str(cid))
        chunk_text = " ".join(chunk.text)
        embeddings.append(embed_text(chunk_text))
        documents.append(f"{filename} chunk {cid}")
        mdata= {"source":filename,"chunk_index":cid}
        metadatas.append(mdata)
    return CollectionData(ids=ids,embeddings=embeddings,documents=documents,mdata=metadatas)


async def add_document(filepath:Path):
    file_data = await get_file_content(filepath)
    if file_data:
        filename=filepath.name
        filetype=filepath.suffix
        result= build_document_result(filename=filename,file_type=filetype,text=file_data,chunk_size=367,overlap=70)

        chunks = result.chunks
        total_chunks = len(chunks)# for document response
        data = process_chunks(filename,chunks)

        collection.add(
            ids = data.ids,
            embeddings= data.embeddings,
            documents= data.documents,
            metadatas=data.mdata
        )
        return DocumentResponse(filename=filename,file_type=filetype,total_chunks=total_chunks)
    
    return None

def list_document():
    all_meta_data = collection.get(include=["metadatas"]).get("metadatas")
    if all_meta_data is not None:
        unique_keys={metadata.get("source") for metadata in all_meta_data}#type: ignore
        distinct_keys = list(unique_keys)
        return distinct_keys
    return None

if __name__ == "__main__":
    data =list_document()
    print(data)

def delete_document(source:str):
    count_before=collection.count()
    collection.delete(where={"source":source})
    count_after=collection.count()
    deleted_count =count_before -count_after
    if deleted_count >0:
        return deleted_count
    else:
        return None