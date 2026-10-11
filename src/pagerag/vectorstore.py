from pathlib import Path
from pagerag.extract import get_file_content
from pagerag.chunker import build_document_result
from pagerag.schemas import Chunk,CollectionData,DocumentResponse
from pagerag.embeddings import embed_text
from chromadb import Collection
from pagerag.chunker import generate_chunkid


def process_chunks(filename:str,chunks:list[Chunk]):
    ids=[]
    embeddings=[]
    documents = []
    metadatas=[]
  
    for chunk in chunks:
        cid = chunk.chunk_id
        ids.append(cid)
        chunk_text = " ".join(chunk.text)
        embeddings.append(embed_text(chunk_text))
        documents.append(chunk_text)
        mdata= {"source":filename,"chunk_index":cid}
        metadatas.append(mdata)
    return CollectionData(ids=ids,embeddings=embeddings,documents=documents,mdata=metadatas)


async def add_document(collection,filepath:Path):
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


def vector_search(collection:Collection,query:str,fetch_limit:int=20,):
    query_emb=embed_text(query)
    result = collection.query(query_embeddings=query_emb,n_results=fetch_limit,include=['documents'])
    if result:
        documents  = result.get('documents') or [[]] # None proof 
        if documents :
            vector_result = documents[0]
            return vector_result
        
    return None

def get_source(collection:Collection,chunks:list[str]|tuple[str]|None)->list:
    if not chunks:
        return []
    
    chunk_ids = []
    for chunk in chunks:
        chunk_ids.append(generate_chunkid(chunk))
        collection_data = collection.get(ids=chunk_ids,include=['metadatas'])
        if collection_data:
            metadatas = collection_data.get('metadatas')
            if metadatas:
                sources = [mdata.get('source') for mdata in metadatas]
                return sources
    return []