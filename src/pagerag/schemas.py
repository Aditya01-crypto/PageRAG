'''Data Models for CLI Document Processor'''
from pydantic import BaseModel,Field,ConfigDict
from datetime import datetime
from pathlib import Path
from typing import Annotated,Literal
from chromadb import Metadata
import numpy as np

class Chunk(BaseModel):
    chunk_id:int
    text:list[str]
    word_count:Annotated[int,Field(ge=0)]
    char_count:Annotated[int,Field(ge=0)]

class CollectionData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    ids:list[str]
    embeddings:list[np.ndarray]
    documents:list[str]
    mdata:list[Metadata]


class DocumentResult(BaseModel):
    filename:str
    file_type:str
    total_chars:Annotated[int,Field(ge=0)]
    total_words:Annotated[int,Field(ge=0)]
    chunks:list[Chunk]
    processed_at:datetime
    error:None | str=None

class DocumentResponse(BaseModel):
    filename:str
    file_type:str
    total_chunks:int

class QueryRequest(BaseModel):
    query:str = Field(min_length=5)
    mode: Literal["vector_only","hybrid","hybrid_rerank"]
    fetch_k:int = Field(default=10,gt=0)
    top_k:int = Field(default=5,gt=0)
    token_limit:int = Field(default=2000,ge=500)

class QueryResponse(BaseModel):
    answer:str
    sources:list[str]|None
    question: str    