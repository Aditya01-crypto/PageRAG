'''Data Models for CLI Document Processor'''
from pydantic import BaseModel,Field,RootModel
from datetime import datetime
from pathlib import Path
from typing import Annotated

class Chunk(BaseModel):
    chunk_id:int
    text:list[str]
    word_count:Annotated[int,Field(ge=0)]
    char_count:Annotated[int,Field(ge=0)]


class DocumentResult(BaseModel):
    filename:str
    file_type:str
    total_chars:Annotated[int,Field(ge=0)]
    total_words:Annotated[int,Field(ge=0)]
    chunks:list[Chunk]
    processed_at:datetime
    error:None | str=None
