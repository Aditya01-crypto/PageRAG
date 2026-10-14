from pathlib import Path
from pagerag.extract import get_file_content
from pagerag.chunker import build_document_result
from pagerag.schemas import Chunk,CollectionData,DocumentResponse,Hit
from pagerag.embeddings import embed_text
from chromadb import Collection
import asyncio

def process_chunks(filename:str,chunks:list[Chunk]):
    ids=[]
    embeddings=[]
    documents = []
    metadatas=[]
  
    for i, chunk in enumerate(chunks):
        cid = chunk.chunk_id
        ids.append(cid)
        chunk_text = " ".join(chunk.text)
        embeddings.append(embed_text(chunk_text))
        documents.append(chunk_text)
        mdata= {"source":filename,"chunk_index":i}
        metadatas.append(mdata)
    return CollectionData(ids=ids,embeddings=embeddings,documents=documents,mdata=metadatas)

def _ingest(collection, filename: str, filetype: str, file_data: str) -> int:
    result = build_document_result(
        filename=filename, file_type=filetype, text=file_data,
        chunk_size=165, overlap=33,
    )
    chunks = result.chunks
    data = process_chunks(filename, chunks)
    collection.delete(where={'source':filename})
    collection.add(
        ids=data.ids,
        embeddings=data.embeddings,
        documents=data.documents,
        metadatas=data.mdata,
    )
    return len(chunks)

async def add_document(collection,filepath:Path):
    file_data = await get_file_content(filepath)
    if file_data:
        filename=filepath.name
        filetype=filepath.suffix
        total_chunks = await asyncio.to_thread(_ingest,collection,filename,filetype,file_data)
        return DocumentResponse(filename=filename,file_type=filetype,total_chunks=total_chunks)
    
    return None

def list_document(collection):
    all_meta_data = collection.get(include=["metadatas"]).get("metadatas")
    if all_meta_data is not None:
        unique_keys={metadata.get("source") for metadata in all_meta_data}#type: ignore
        distinct_keys = list(unique_keys)
        return distinct_keys
    return None

def delete_document(collection,source:str):
    count_before=collection.count()
    collection.delete(where={"source":source})
    count_after=collection.count()
    deleted_count =count_before -count_after
    if deleted_count >0:
        return deleted_count
    else:
        return None


def vector_search(collection:Collection,query:str,fetch_limit:int=20,)->list[Hit]:
    query_emb=embed_text(query)
    result = collection.query(query_embeddings=query_emb,n_results=fetch_limit,include=['documents','metadatas'])
    ids = (result.get('ids') or [[]])[0]
    docs = (result.get('documents') or [[]])[0]
    metas = (result.get('metadatas') or [[]])[0]

    return [Hit(id=i, text=d, source=str((m or {}).get('source', '')))
            for i, d, m in zip(ids, docs, metas)]

