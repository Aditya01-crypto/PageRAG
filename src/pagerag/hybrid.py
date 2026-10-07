from rank_bm25 import BM25Okapi
from chromadb import Collection

def get_chunks(collection:Collection):
    data = collection.get()
    chunks=[]
    if data:
        chunks = data.get('documents',[])
    return chunks

def build_tokenized_corpus(collection:Collection):
    corpus = get_chunks(collection)
    return [chunk.split() for chunk in corpus]

def build_bm25_index(collection:Collection):
    '''
    Provides BM250kapi Object + Tokenized Corpus
    '''
    tokenized_corpus = build_tokenized_corpus(collection)# a list of chunks
    bm25 = BM25Okapi(tokenized_corpus)

    return bm25,tokenized_corpus

def bm25_search(bm25:BM25Okapi,chunks:list[list],query:str,limit:int):
    '''
        Provides Top  chunks as per limit value  along with their  BM25 score
    '''
    tokenized_query =query.split()
    scores = bm25.get_scores(tokenized_query)
    pairs = list(zip(chunks,scores))
    top_20_pairs = sorted(pairs,key=lambda x:x[1],reverse=True)[:limit]
    return top_20_pairs


if __name__ == '__main__':
    import chromadb
    client = chromadb.PersistentClient(path='./doc_db')
    collection = client.get_or_create_collection('documents',metadata={"hnsw:space":"cosine"})

    bmobj,chunks = build_bm25_index(collection)
    top_chunks = bm25_search(bm25=bmobj,chunks=chunks,query='What is mind Mapping ?',limit=10)
    print(top_chunks[0])