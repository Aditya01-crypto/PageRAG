from pathlib import Path
from pagerag.extract import get_file_content
from pagerag.chunker import build_document_result
from pagerag.schemas import Chunk,CollectionData,DocumentResponse
from pagerag.embeddings import embed_text
import numpy as np
from typing import Literal
from pagerag.hybrid import rrf_fusion,bm25_search,build_bm25_index
from pagerag.reranker import rerank

def get_source(data:dict):
    ids = data.get('ids',[[]])[0]
    s=set()
    for i in ids:
        name=i.split('.')[0]
        s.add(name)
    return list(s)



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


def retrieve_relevant_chunk(collection,query:str,limit:int=3,mode:Literal["vector_only","hybrid","hybrid_rerank"] = "vector_only"):
    '''
    Provides Relevant Chunks based on given query. 
    vector_only :  dense vector similarity search
    hybrid : vector + keyword search using RRF
    hybrid rerank :  (vector + BM25) refined with cross-encoder reranking
    '''
    query_emb=embed_text(query)
    result = collection.query(query_embeddings=query_emb,n_results=limit,include=['documents'])
    source = get_source(result)
    vector_result  = result.get('documents',[[]])[0]
    if mode == 'vector_only':
        if vector_result:
            return (vector_result,source)
        return (None,None)
    
    bm25,bm_chunks = build_bm25_index(collection)

    bm25_result = bm25_search(bm25=bm25,chunks=bm_chunks,query=query,limit=limit)
    rrf_result = rrf_fusion(vector_result,bm25_result,limit)
    if mode == 'hybrid':
        return (rrf_result,None) #  build the  logic for getting source hybrid
    
    if mode  == 'hybrid_rerank':
        rerank_result = rerank(query,rrf_result,limit)
        return (rerank_result,None) 
        
    return (None,None)