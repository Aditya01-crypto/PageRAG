from pagerag.vectorstore import retrieve_relevant_chunk
from httpx import AsyncClient
import os
from fastapi import HTTPException
from dotenv import load_dotenv
import json
from pagerag.schemas import QueryResponse
load_dotenv()
MODEL = os.getenv('MODEL')
_API_KEY = os.getenv('API_KEY')

BASE_URL=f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"

STREAM_URL=f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:streamGenerateContent?alt=sse"

def initialize_prompt(context,question):
    prompt_template = f"""You are a helpful assistant. Answer the question based ONLY on the context provided below.
    If the context does not contain enough information to answer the question, say "I don't have enough information in the provided documents to answer this."
    Do not use any prior knowledge outside of the context.

    Context:
    {context}

    Question: {question}

    Answer:"""

    return prompt_template

headers={
        'x-goog-api-key':_API_KEY,
        "Content-Type":"application/json"
    }


def extract_gemini_response(data:dict)->str:
    candidates=data.get("candidates",[])
    if not candidates:
        return ""
    contents=candidates[0].get('content',{})
    parts=contents.get('parts',[])

    return "".join(part.get("text","")for part in parts if "text" in part)

async def ask_llm(collection,client:AsyncClient,query:str,c_limit:int,t_limit:int=2000):
    context,sources = retrieve_relevant_chunk(collection,query=query,limit=c_limit)
    prompt = initialize_prompt(context=context,question=query)
    payload={
    "contents": [{"parts": [{"text":prompt}]}],
    "generationConfig": {"maxOutputTokens": t_limit}
    }
    response =await client.post(url=BASE_URL,headers=headers,json=payload)
    if response.status_code ==200:
        response_data=response.json()
        prompt_result = extract_gemini_response(response_data)
        return QueryResponse(answer=prompt_result,sources=sources,question=query)
    else:
        raise HTTPException(status_code=response.status_code,detail=f"{response.text}")

async def event_generator(collection,client:AsyncClient,query:str,chunk_limit:int=4,token_limit:int=2000):
    accumulated_text=""

    context,sources = retrieve_relevant_chunk(collection,query=query,limit=chunk_limit)
    prompt=initialize_prompt(context,query)
    payload={
        "contents": [{"parts": [{"text":prompt}]}],
        "generationConfig": {"maxOutputTokens": token_limit}
    }
    async with client.stream("POST",STREAM_URL,headers=headers,json=payload)as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                raw_data =  line[6:].strip()
                if not raw_data or raw_data == '[DONE]':
                    continue

                try:
                    data=json.loads(raw_data)

                    chunk_text=extract_gemini_response(data)

                    if chunk_text:
                        accumulated_text+=chunk_text
                        yield f"data: {json.dumps({'chunk':chunk_text})}\n\n"

                except json.JSONDecodeError:
                    continue
    response_payload= QueryResponse(answer=accumulated_text,sources=sources,question=query)
    yield f"data: {response_payload.model_dump_json()}\n\n"