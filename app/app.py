"""
AI Vectorizer - BM25 Search API

This module provides a FastAPI application with endpoints for document management
and BM25-based similarity search functionality. It also supports file uploads and
metadata management.
"""

import os
import uuid
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, UploadFile, File, Form, Depends
from fastapi.responses import FileResponse
from rank_bm25 import BM25Okapi
import asyncio
from fastapi.concurrency import run_in_threadpool

# Import document processing module
from app.document_processing import process_document, DocumentProcessingError

app = FastAPI(
    title="AI Vectorizer",
    description="API for document management, BM25-based similarity search, and file uploads",
    version="1.0.0"
)

# Create a temporary directory for file storage
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Document metadata model
class DocumentMetadata(BaseModel):
    doc_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="File type/extension")
    upload_timestamp: datetime = Field(..., description="Upload timestamp")
    file_size: int = Field(..., description="File size in bytes")
    processing_status: str = Field(..., description="Processing status (e.g., 'uploaded', 'processed')")
    title: Optional[str] = Field(None, description="Document title if extractable")
    page_count: Optional[int] = Field(None, description="Page count for PDFs/DOCX")
    word_count: Optional[int] = Field(None, description="Word count if applicable")
    session_id: Optional[str] = Field(None, description="Session identifier for grouping uploads")
    content_preview: Optional[str] = Field(None, description="Preview of document content")
    chunk_count: Optional[int] = Field(None, description="Number of chunks after processing")
    total_tokens: Optional[int] = Field(None, description="Total number of tokens in the document")

# In-memory storage for document metadata
documents_metadata: Dict[str, DocumentMetadata] = {}

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

def get_file_extension(filename: str) -> str:
    """
    Extract file extension from filename.
    
    Args:
        filename: Original filename
        
    Returns:
        File extension (lowercase, without dot)
    """
    return os.path.splitext(filename)[1].lower().lstrip(".")

def get_session_directory(session_id: Optional[str] = None) -> Path:
    """
    Get or create a session-specific directory for uploads.
    
    Args:
        session_id: Optional session identifier
        
    Returns:
        Path to the session directory
    """
    if session_id:
        session_dir = UPLOAD_DIR / session_id
    else:
        session_dir = UPLOAD_DIR / "default"
    
    session_dir.mkdir(exist_ok=True)
    return session_dir

def extract_text_content(file_path: Path, file_type: str) -> Optional[str]:
    """
    Extract text content from a file for preview.
    
    Args:
        file_path: Path to the file
        file_type: File type/extension
        
    Returns:
        Text content preview or None if extraction fails
    """
    try:
        # Use document processing module for supported file types
        if file_type in ["pdf", "docx", "doc", "txt", "md", "py", "js", "html", "css", "json"]:
            # Process the document to extract text
            result = process_document(file_path, max_tokens=512, overlap_tokens=50)
            
            # Return a preview of the extracted text
            if result and result["extracted_text"]:
                return result["extracted_text"][:1000]  # First 1000 chars for preview
        
        # For unsupported file types, return None
        return None
    except Exception:
        return None

def estimate_word_count(text: Optional[str]) -> Optional[int]:
    """
    Estimate word count from text content.
    
    Args:
        text: Text content
        
    Returns:
        Estimated word count or None if text is None
    """
    if text:
        return len(text.split())
    return None

async def process_document_task(file_path: Path, doc_id: str, file_type: str):
    """
    Background task to process a document.
    
    Args:
        file_path: Path to the document file
        doc_id: Document ID
        file_type: File type/extension
    """
    try:
        # Process the document
        result = process_document(file_path, max_tokens=512, overlap_tokens=50)
        
        if doc_id in documents_metadata:
            # Update metadata with processing results
            metadata = documents_metadata[doc_id]
            metadata.processing_status = "processed"
            metadata.word_count = len(result["extracted_text"].split())
            metadata.chunk_count = result["chunk_count"]
            metadata.total_tokens = result["total_tokens"]
            
            # For PDFs, update page count if available
            if file_type == "pdf" and "page_count" in result:
                metadata.page_count = result.get("page_count")
            
            # Update corpus with chunks instead of just preview
            for chunk in result["chunks"]:
                if chunk not in corpus:
                    corpus.append(chunk)
            
            # Update BM25 index
            await update_bm25()
    except Exception as e:
        # Log the error and update metadata
        if doc_id in documents_metadata:
            metadata = documents_metadata[doc_id]
            metadata.processing_status = "error"

@app.post("/upload/", summary="Upload a document")
async def upload_document(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
    title: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = None
) -> Dict[str, Any]:
    """
    Upload a document file or text content.
    
    Args:
        file: File to upload (optional)
        text: Text content (optional if file is provided)
        session_id: Optional session identifier for grouping uploads
        title: Optional document title
        background_tasks: FastAPI background tasks handler
        
    Returns:
        Document metadata
    """
    if not file and not text:
        raise HTTPException(
            status_code=400, 
            detail="Either file or text content must be provided"
        )
    
    # Generate a unique document ID
    doc_id = str(uuid.uuid4())
    
    # Get or create session directory
    session_dir = get_session_directory(session_id)
    
    metadata = None
    
    # Handle file upload
    if file:
        filename = file.filename
        file_type = get_file_extension(filename)
        file_path = session_dir / f"{doc_id}.{file_type}"
        
        # Save the file
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Extract content preview for indexing
        content_preview = extract_text_content(file_path, file_type)
        
        # Create metadata
        metadata = DocumentMetadata(
            doc_id=doc_id,
            filename=filename,
            file_type=file_type,
            upload_timestamp=datetime.now(),
            file_size=os.path.getsize(file_path),
            processing_status="processing",  # Changed from "uploaded" to "processing"
            title=title or filename,
            content_preview=content_preview,
            word_count=estimate_word_count(content_preview),
            session_id=session_id,
            chunk_count=None,
            total_tokens=None
        )
        
        # Add content to corpus if text was extracted
        if content_preview:
            corpus.append(content_preview)
            if background_tasks:
                background_tasks.add_task(update_bm25)
                
        # Add background task to process the document
        if background_tasks and file_type in ["pdf", "docx", "doc", "txt"]:
            background_tasks.add_task(process_document_task, file_path, doc_id, file_type)
    
    # Handle text content
    elif text:
        # Save text to a file
        filename = f"{title or 'document'}.txt"
        file_path = session_dir / f"{doc_id}.txt"
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
        
        # Process the text directly
        try:
            # Clean and chunk the text
            result = process_document(file_path, max_tokens=512, overlap_tokens=50)
            chunks = result["chunks"]
            total_tokens = result["total_tokens"]
            
            # Create metadata
            metadata = DocumentMetadata(
                doc_id=doc_id,
                filename=filename,
                file_type="txt",
                upload_timestamp=datetime.now(),
                file_size=len(text.encode('utf-8')),
                processing_status="processed",  # Already processed
                title=title or "Text Document",
                content_preview=text[:1000] if len(text) > 1000 else text,
                word_count=len(text.split()),
                session_id=session_id,
                chunk_count=len(chunks),
                total_tokens=total_tokens
            )
            
            # Add chunks to corpus
            for chunk in chunks:
                if chunk not in corpus:
                    corpus.append(chunk)
            
            if background_tasks:
                background_tasks.add_task(update_bm25)
                
        except Exception as e:
            # If processing fails, fall back to simple approach
            metadata = DocumentMetadata(
                doc_id=doc_id,
                filename=filename,
                file_type="txt",
                upload_timestamp=datetime.now(),
                file_size=len(text.encode('utf-8')),
                processing_status="error",
                title=title or "Text Document",
                content_preview=text[:1000] if len(text) > 1000 else text,
                word_count=len(text.split()),
                session_id=session_id
            )
            
            # Add content to corpus
            corpus.append(text)
            if background_tasks:
                background_tasks.add_task(update_bm25)
    
    # Store metadata
    documents_metadata[doc_id] = metadata
    
    return metadata.model_dump()

@app.get("/documents/", summary="List all documents")
async def list_documents(
    session_id: Optional[str] = Query(None, description="Filter by session ID")
) -> Dict[str, List[DocumentMetadata]]:
    """
    List all document metadata, optionally filtered by session ID.
    
    Args:
        session_id: Optional session identifier for filtering
        
    Returns:
        List of document metadata
    """
    if session_id:
        filtered_docs = [
            doc for doc in documents_metadata.values() 
            if doc.session_id == session_id
        ]
        return {"documents": filtered_docs}
    
    return {"documents": list(documents_metadata.values())}

@app.get("/documents/{doc_id}", summary="Get document metadata")
async def get_document(doc_id: str) -> DocumentMetadata:
    """
    Get metadata for a specific document.
    
    Args:
        doc_id: Document identifier
        
    Returns:
        Document metadata
    """
    if doc_id not in documents_metadata:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return documents_metadata[doc_id]

@app.delete("/documents/{doc_id}", summary="Delete a document")
async def delete_document(
    doc_id: str,
    background_tasks: BackgroundTasks = None
) -> Dict[str, str]:
    """
    Delete a document and its metadata.
    
    Args:
        doc_id: Document identifier
        background_tasks: FastAPI background tasks handler
        
    Returns:
        Status message
    """
    if doc_id not in documents_metadata:
        raise HTTPException(status_code=404, detail="Document not found")
    
    metadata = documents_metadata[doc_id]
    
    # Delete the file
    session_dir = get_session_directory(metadata.session_id)
    file_path = session_dir / f"{doc_id}.{metadata.file_type}"
    
    if file_path.exists():
        os.remove(file_path)
    
    # Remove from corpus if content preview exists
    if metadata.content_preview and metadata.content_preview in corpus:
        corpus.remove(metadata.content_preview)
        if background_tasks:
            background_tasks.add_task(update_bm25)
    
    # Delete metadata
    del documents_metadata[doc_id]
    
    return {"message": f"Document {doc_id} deleted successfully"}

@app.get("/documents/{doc_id}/download", summary="Download original document")
async def download_document(doc_id: str) -> FileResponse:
    """
    Download the original document file.
    
    Args:
        doc_id: Document identifier
        
    Returns:
        File response with the original document
    """
    if doc_id not in documents_metadata:
        raise HTTPException(status_code=404, detail="Document not found")
    
    metadata = documents_metadata[doc_id]
    
    # Get file path
    session_dir = get_session_directory(metadata.session_id)
    file_path = session_dir / f"{doc_id}.{metadata.file_type}"
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Document file not found")
    
    return FileResponse(
        path=str(file_path),
        filename=metadata.filename,
        media_type=f"application/{metadata.file_type}"
    )
