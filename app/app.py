# venv\Scripts\Activate.ps1


from typing import Annotated
from pydantic import BaseModel

from fastapi import FastAPI, Header, HTTPException, Query, BackgroundTasks

from rank_bm25 import BM25Okapi
import asyncio
import time
from fastapi.concurrency import run_in_threadpool

app = FastAPI()

corpus = [
    "Artificial Intelligence is transforming industries.",
    "Machine learning is a subset of AI.",
    "Natural language processing helps computers understand human language."
]

tokenized_corpus = []
for doc in corpus:
    tokenized_corpus.append(doc.split())
bm25 = BM25Okapi(tokenized_corpus)

MAX_CONCURRENT_REQUESTS = 5  
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

async def update_bm25():
    global bm25
    if not corpus:  
        bm25 = None
        return

    tokenized_corpus = []
    for doc in corpus:
        tokenized_corpus.append(doc.split())
    bm25 = BM25Okapi(tokenized_corpus)
    await asyncio.sleep(0.1)


# class ResetRequest(BaseModel):
#     delete_all: str 

# class QueryRequest(BaseModel):
#     query: str

# class AddDocRequest(BaseModel):
#     new_doc: str

# def TokenizedCorpus():
#     tokenized_corpus = []
#     for doc in corpus:
#         doc_tokens = doc.split()
#         tokenized_corpus.append(doc_tokens)
#     return tokenized_corpus

@app.post("/add-doc/")
async def add_doc(new_doc: str = Query(..., description="New document to be added to the corpus"), background_tasks: BackgroundTasks = None):
    corpus.append(new_doc)
    background_tasks.add_task(update_bm25)
    return {"Corpus added": corpus}

@app.post("/reset-corpus/")
async def reset_corpus(delete_all: str = Query(..., description="Pass 'Y' to reset corpus"), background_tasks: BackgroundTasks = None):
    global corpus
    if delete_all.upper() == "Y":
        corpus.clear()
        background_tasks.add_task(update_bm25)
        return {"message": "Corpus reset"}
    return {"message": "Corpus not reset"}

# Are the idf values what I was supposed to be returning?
@app.get("/query/")
async def get_query():
    global bm25
    if not corpus:
        raise HTTPException(status_code=400, detail="Corpus is empty. Add documents first.")
    return {"bm25 vectors":bm25}

@app.post("/find-similar/")
async def find_similar(query: str = Query(..., description="Search query to find similar documents")):
    global bm25
    if not corpus:
        raise HTTPException(status_code=400, detail="Corpus is empty. Add documents first.")
    
    async with semaphore:
        query_tokens = query.lower().split()
        relevant_docs = await run_in_threadpool(bm25.get_top_n,query_tokens,corpus,n=1)
    return {"most similar result": relevant_docs}

