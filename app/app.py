"""
AI Vectorizer - BM25 Search API

This module provides a FastAPI application with endpoints for document management
and BM25-based similarity search functionality.
"""

from typing import List, Dict, Any
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from rank_bm25 import BM25Okapi
import asyncio
from fastapi.concurrency import run_in_threadpool

app = FastAPI(
    title="AI Vectorizer",
    description="API for document management and BM25-based similarity search",
    version="1.0.0"
)

# Limit concurrent requests to prevent overloading
MAX_CONCURRENT_REQUESTS = 5  
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

def tokenize_text(text: str) -> List[str]:
    """
    Tokenize a text string into words.
    
    Args:
        text: The input text to tokenize
        
    Returns:
        A list of tokens (words)
    """
    return text.lower().split()

def tokenize_corpus(documents: List[str]) -> List[List[str]]:
    """
    Tokenize a corpus of documents.
    
    Args:
        documents: List of document strings
        
    Returns:
        List of tokenized documents (list of lists of tokens)
    """
    return [tokenize_text(doc) for doc in documents]

# Initial corpus with example documents
corpus = [
    "Artificial Intelligence is transforming industries.",
    "Machine learning is a subset of AI.",
    "Natural language processing helps computers understand human language."
]

# Initialize BM25 with the initial corpus
bm25 = BM25Okapi(tokenize_corpus(corpus))

async def update_bm25():
    """
    Update the BM25 index with the current corpus.
    This is run as a background task when the corpus changes.
    """
    global bm25
    if not corpus:  
        bm25 = None
        return

    tokenized_corpus = tokenize_corpus(corpus)
    bm25 = BM25Okapi(tokenized_corpus)
    await asyncio.sleep(0.1)  # Small delay to ensure task completion

@app.post("/add-doc/", summary="Add a document to the corpus")
async def add_doc(
    new_doc: str = Query(..., description="New document to be added to the corpus"), 
    background_tasks: BackgroundTasks = None
) -> Dict[str, List[str]]:
    """
    Add a new document to the corpus and update the BM25 index.
    
    Args:
        new_doc: Text of the document to add
        background_tasks: FastAPI background tasks handler
        
    Returns:
        Dictionary with the updated corpus
    """
    corpus.append(new_doc)
    background_tasks.add_task(update_bm25)
    return {"Corpus added": corpus}

@app.post("/reset-corpus/", summary="Reset the document corpus")
async def reset_corpus(
    delete_all: str = Query(..., description="Pass 'Y' to reset corpus"), 
    background_tasks: BackgroundTasks = None
) -> Dict[str, str]:
    """
    Reset the corpus by removing all documents.
    
    Args:
        delete_all: Confirmation string, must be 'Y' to proceed
        background_tasks: FastAPI background tasks handler
        
    Returns:
        Status message
    """
    global corpus
    if delete_all.upper() == "Y":
        corpus.clear()
        background_tasks.add_task(update_bm25)
        return {"message": "Corpus reset"}
    return {"message": "Corpus not reset"}

@app.get("/query/", summary="Get corpus information")
async def get_query() -> Dict[str, Any]:
    """
    Get information about the current corpus.
    
    Returns:
        Dictionary with corpus information and statistics
    """
    if not corpus:
        raise HTTPException(status_code=400, detail="Corpus is empty. Add documents first.")
    
    # Return corpus information instead of the raw BM25 object
    return {
        "corpus_size": len(corpus),
        "documents": corpus,
        "has_bm25_index": bm25 is not None
    }

@app.post("/find-similar/", summary="Find similar documents")
async def find_similar(
    query: str = Query(..., description="Search query to find similar documents"),
    n: int = Query(1, description="Number of results to return", ge=1, le=10)
) -> Dict[str, List[str]]:
    """
    Find documents similar to the query using BM25 ranking.
    
    Args:
        query: Search query text
        n: Number of results to return (default: 1)
        
    Returns:
        Dictionary with the most similar documents
    """
    if not corpus:
        raise HTTPException(status_code=400, detail="Corpus is empty. Add documents first.")
    
    async with semaphore:
        query_tokens = tokenize_text(query)
        relevant_docs = await run_in_threadpool(
            lambda: bm25.get_top_n(query_tokens, corpus, n=n)
        )
    
    return {"most_similar_results": relevant_docs}
