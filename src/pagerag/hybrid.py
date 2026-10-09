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

def bm25_search(bm25:BM25Okapi,chunks:list[list],query:str,limit:int):
    '''
        Provides Top  chunks as per limit value  along with their  BM25 score
    '''
    tokenized_query =query.split()
    top_n_chunk = bm25.get_top_n(tokenized_query,chunks,n=limit)
    return [" ".join(chunk)for chunk in top_n_chunk]

def rrf_fusion(vector_result:list,bm25_result:list,k:int=60):
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
        return chunks
    else :
        return []



if __name__ == '__main__':
    import chromadb
    client = chromadb.PersistentClient(path='./doc_db')
    collection = client.get_or_create_collection('documents',metadata={"hnsw:space":"cosine"})

    bmobj,chunks = build_bm25_index(collection)
    bm25_result = bm25_search(bm25=bmobj,chunks=chunks,query='What is mind Mapping ?',limit=10)
    v_r= ['Mapping the Swiss army knife for the brain. It is not only a method for expanding your memory, but a way to improve your thinking skills. Mind Mapping can be used for: memorizing, learning, presenting, communicating, organizing, planning, meetings, negotiating, and all types of thinking. A Mind Map is a multi-sensory way of transferring your thoughts to paper. It is incredibly easy and simple to use. At first it may take a bit of practice, but then your brain will remember how to have fun, and your life and learning will never be the same again. Mind Maps are a wonderful way of structuring information, so that you can see the big picture and the details. With linear notes, which are lists and lines, you will never have the flexibility that you have with Mind Maps. To be a successful Mind Mapper all you need is the following: 1. Your brain; 2. A blank piece of paper, the bigger the better, and turn it to landscape; 3. Lots of colored pens and pencils. The best way to explain a Mind Map is to map something out. The Mind Map that I will be creating here is about all the systems that I have shared with you in this book. With every Mind Map you start in the centre of a blank page with a central image. This central image is what the whole Mind Map is about; therefore I will call this central image Systems . As we now know images are memorable and stimulate more creativity. Step 1: Step 2: Once you have your central image, then you connect branches to the central image and start branching out the headings. The main branches are all the memory systems we have covered. Step 3: Once we have our main branches, then we can connect second and third level branches to give more detail to each main branch. We can add even more branches to the existing branches to help clarify ideas or give more detail. Remember to use only one word per branch. This aids your associating mind to bloom freely, and remember to add lots of images. Each main branch will also have one color, this', 'our main branches, then we can connect second and third level branches to give more detail to each main branch. We can add even more branches to the existing branches to help clarify ideas or give more detail. Remember to use only one word per branch. This aids your associating mind to bloom freely, and remember to add lots of images. Each main branch will also have one color, this helps visually to distinguish between different branches or content. A Mind Map can never end because your associating mind can always find just one more memory. Mind Maps are entertaining; they are fun and make use of your creative brain. If you choose to stick with it, you will take your mind to a new level. You will improve your creativity, planning power, develop more of your brain, and increase your powers of memory and observation. You can use Mind Maps for a whole range of learning areas; they can be used very effectively to summarize large amounts of information, and to get the gist of what is being communicated. Here is a Mind Map that I made of the book The Seven Habits of Highly Effective People. You will notice that each main branch shows a concept that we remembered on The Car List. This Mind Map summarizes the key content from Stephen Covey s whole book. I created this Mind Map with iMindMap software; you can go to www.imindmap.co.za and download your free version of iMindMap basic today. There are many Mind Mapping computer programs out there, but nothing comes close to the flexibility and usability of iMindMap. Play with it, you will be surprised at how much you can achieve with this one thinking tool. C HAPTER 14. U SING THE METHODS Success is neither magical nor mysterious. Success is a natural consequence of consistently applying basic fundamentals. ~ Jim Rohn Now that you know the fundamentals of the memory methods you can succeed with any information. By being more creative with information it helps increase your involvement with the content, and makes it part of your reality, therefore improving your memory. In this chapter I want to share with you how you can', "you will find thousands and thousands of places just waiting to be used in your mind. There are no limits to this system, only limits in your own thinking. The important thing is that you practise. The more you practise the better you will get. C HAPTER 10. L INKING THOUGHTS No memory is ever alone; it's at the end of a trail of memories, a dozen trails that each have their own associations. ~ Louis L Amour In the previous chapters we learned to bring information to life, and to store it in a long-term memory compartment system. Now in this chapter we are going to learn to link more thoughts together. It is a way to direct attention and to strengthen your imagination and your ability to associate concepts. Your mind is an associating machine and it has no limits. I often hear people say, Oh, do you learn by association? The answer to that is we only learn by association. Learning is connecting new information to old information, it doesn t happen any other way. It is creating a relationship between the known and the unknown and the more you know, the easier it is to connect more information and get to know more. Now let s memorize a list together to experience this method. It will seem silly, but stick with it and I will make a point. This story takes longer for me to explain than what happens in your mind. Read it and remember to use the SEE principle. I want you to imagine that you are washing a tin; really see it in your mind. As you wash the tin, it suddenly begins developing a huge Adams apple. A chef and her son grab the Adams apple and rip it out. The Chef and her son then decide to make some medicine, which they give to Marilyn Monroe and she starts to develop a massive Adams apple too. Michael Jackson sees her Adams apple throbbing and runs away screaming and jumps into a van with beer in it. The van is being driven by a big yellow hairy sun really see it, make it silly, hairy, and let it stick.", 'one for you to look at or a drawing to represent it, it would be really difficult to understand. The more we turn information into images or mind movies, the more we will remember and comprehend. We can learn to make all our learning more creative and memorable if we use our unlimited imagination. You can learn to enhance your memory imagination system by making your mind movies exciting and sticky. The way to do this is with the SEE principle. T HE SEE P RINCIPLE Use your S Senses: there are only five ways to get anything into your brain, and that is through sight, sound, smell, touch, and taste. When you utilize your senses you experience more of life and you remember more. Our senses help us mentally recreate our world. If you train your senses you will be using more of your brain, and if you learn to engage as many of your senses as you can then you will automatically improve your memory. Think of a horse: see it in your mind, touch it, smell it, hear it, and even taste it. You didn t see the letters H.O.R.S.E in your mind; you saw a multisensory picture of what the word represents. Your senses make mind movies real and memorable. Use them! E Exaggeration: what is easier to remember: a strawberry that is normal size or one the size of a house? Make your images larger or smaller than life. What is more memorable: an elephant or an elephant wearing a pink bikini? Exaggerate with Humor; tickle your mind. There is no scientific evidence to prove that learning should be serious. Make your images illogical. Have fun; create some positive exaggerated learning memories. E Energize: give your pictures action. Would you rather watch a movie of your holiday or a slide show? What creates more feeling in your imagination: a horse standing still or a horse that is running and moving? Make your information vivid, colorful, and not boring, flat and black and white. Use action; it brings life to your memories. Make your images act in illogical ways: you can weave, crash, stick, or wrap things together. We can make things talk,', 'in the book and you battle to put it down because you don t want to turn the movie off. Your mind is like an internal movie screen on which you can ask it to produce information. This is how we think and learn effectively. Your brain creates miracles everyday by converting lifeless information into pictures and ideas. When you become aware of this, every word becomes a picture drawn with letters because words are only symbols of three-dimensional images. Arthur Gordon said, Isn t it amazing how we take them for granted? Those little black marks on paper. Twenty-six different shapes known as letters, arranged in endless combinations, known as words, lifeless, until someone s eye falls on them. If your brain was unable to make images out of symbols, all learning and reading would be worthless and incredibly boring. Your brain likes pictures and we are really good at remembering them. As neuroscientist John Medina says, Hear a piece of information and three days later you ll remember 10% of it. Add a picture and you ll remember 65%. Some people say, I can t make pictures in my mind. We all make pictures in our mind. If you were unable to create or remember visual images, you would be severely handicapped. Learn to use your imagination; it is a learned skill and not a natural talent. Reading and understanding is also a creative imagination process. It is a power that can be compared to magic. We succeed in this area when we produce images in our mind. When we don t, we feel confused or ignorant. If I tried to explain to you how a car engine works but you don t know what an engine looks like or if I didn t have one for you to look at or a drawing to represent it, it would be really difficult to understand. The more we turn information into images or mind movies, the more we will remember and comprehend. We can learn to make all our learning more creative and memorable if we use our unlimited imagination. You can learn to enhance your memory imagination system by making your mind movies exciting and']

    r=rrf_fusion(v_r,bm25_result)
    print((r))