from rank_bm25 import BM25Okapi
from chromadb import Collection
from dataclasses import dataclass
import re
from pagerag.schemas import Hit

@dataclass(frozen=True)
class BM25State:
    index: BM25Okapi | None
    chunks : list[list[str]]
    hits : list[Hit]

def load_hits(collection:Collection) -> list[Hit]:
    data = collection.get(include=['documents', 'metadatas'])
    ids = data.get('ids') or []
    docs = data.get('documents') or []
    metas = data.get('metadatas') or []
    return [Hit(id=i, text=d, source=str((m or {}).get('source', '')))
            for i, d, m in zip(ids, docs, metas)]


def tokenize(text: str) -> list[str]:
    tokens = []
    for tok in text.lower().split():
        tok = re.sub(r"^\W+|\W+$", "", tok)   
        if tok:                                #
            tokens.append(tok)
    return tokens


def build_bm25_index(collection:Collection)->BM25State:
    '''
    Provides BM250kapi Object + Tokenized Corpus
    '''
    hits = load_hits(collection)
    tokenized = [tokenize(h.text) for h in hits]
    bm25 = BM25Okapi(tokenized) if tokenized else None
    return BM25State(index=bm25, chunks=tokenized, hits=hits)

def bm25_search(state:BM25State,query:str,fetch_limit:int):
    '''
        Provides Top  chunks as per limit value  
    '''
    if state.index is None:
        return []
    scores = state.index.get_scores(tokenize(query))
    ranked =sorted(range(len(scores)),key=lambda i : scores[i],reverse=True)
    return [state.hits[i] for i in ranked[:fetch_limit] if scores[i] > 0]

def rrf_fusion(vector_result:list[Hit],bm25_result:list[Hit],limit:int,k:int=60):
    '''
        Provides Reciprocal Rank Fusion based order of result
    '''
    scores : dict[str,float] ={}
    by_id : dict [str,Hit] = {}
    for hits in (vector_result,bm25_result):
        for rank, hit  in enumerate(hits,start=1):
            by_id[hit.id] = hit
            scores[hit.id] = scores.get(hit.id,0.0)+1/(k+rank)
    best = sorted(scores,key=scores.__getitem__,reverse=True)
    return [by_id[i] for i in best[:limit]]

    
