from rank_bm25 import BM25Okapi
from chromadb import Collection
from hashlib import md5

def generate_chunkid(chunk:str):
    enc_chunk = chunk.encode('utf-8')
    hasher = md5(enc_chunk)
    return hasher.hexdigest()

def get_chunks(collection:Collection):
    data = collection.get()
    chunks=[]
    if data:
        chunks = data.get('documents',[])
    return chunks

def build_tokenized_corpus(collection:Collection):
    corpus  = get_chunks(collection)
    if isinstance(corpus,list):
        return [chunk.split() for chunk in corpus ]
    return []

def build_bm25_index(collection:Collection):
    '''
    Provides BM250kapi Object + Tokenized Corpus
    '''
    tokenized_corpus = build_tokenized_corpus(collection)# a list of chunks
    bm25 = BM25Okapi(tokenized_corpus)

    return bm25,tokenized_corpus

def bm25_search(bm25:BM25Okapi,chunks:list[list],query:str,fetch_limit:int):
    '''
        Provides Top  chunks as per limit value  along with their  BM25 score
    '''
    tokenized_query =query.split()
    top_n_chunk = bm25.get_top_n(tokenized_query,chunks,n=fetch_limit)
    return [" ".join(chunk)for chunk in top_n_chunk]

def rrf_fusion(vector_result:list,bm25_result:list,limit:int,k:int=60):
    '''
        Provides Reciprocal Rank Fusion based order of result
    '''
    rrf_rank=[]
    vector_scores ={}
    bm25_scores = {}
    chunk_map ={}
    # vector score
    for i in range(len(vector_result)):
        chunk = vector_result[i]
        chunk_id = generate_chunkid(chunk)
        chunk_map[chunk_id]=chunk
        vector_score = 1/(k+i+1)
        vector_scores[chunk_id]=vector_score

    # bm25 score
    for i in range(len(bm25_result)):
        chunk = bm25_result[i]
        chunk_id = generate_chunkid(bm25_result[i])
        chunk_map[chunk_id] = chunk
        bm25_score = 1/(k+i+1)
        bm25_scores[chunk_id]=bm25_score

    for id in chunk_map:
        rrf_score = vector_scores.get(id,0)+ bm25_scores.get(id,0)
        rank = (chunk_map[id],rrf_score)
        rrf_rank.append(rank)

    pair =  sorted(rrf_rank,key=lambda x : x[1],reverse=True)
    if pair :
        chunks,score = zip(*pair)
        return chunks[:limit]
    else :
        return []


