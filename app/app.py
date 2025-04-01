"""
AI Vectorizer - Search API

This module provides a FastAPI application with endpoints for document management
and similarity search functionality (BM25 and semantic search). It also supports 
file uploads and metadata management.
"""

import os
import uuid
import shutil
import enum
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Literal
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, UploadFile, File, Form, Depends
from fastapi.responses import FileResponse
from rank_bm25 import BM25Okapi
import asyncio
from fastapi.concurrency import run_in_threadpool

# Import document processing module
from app.document_processing import process_document, DocumentProcessingError

# Import semantic search module
from app.semantic_search import SemanticSearch

# Import visualization module
from app.visualization import VisualizationData, DimensionalityReductionMethod

# Define search types
class SearchType(str, enum.Enum):
    BM25 = "bm25"
    SEMANTIC = "semantic"

app = FastAPI(
    title="AI Vectorizer",
    description="API for document management, similarity search (BM25 and semantic), and file uploads",
    version="1.1.0"
)

# Create a temporary directory for file storage
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize semantic search
semantic_search = SemanticSearch()

# Search result model
class SearchResult(BaseModel):
    document: str = Field(..., description="Document text")
    score: float = Field(..., description="Relevance score")
    doc_id: Optional[str] = Field(None, description="Document ID if available")
    title: Optional[str] = Field(None, description="Document title if available")
    preview: str = Field(..., description="Preview snippet with highlighted matching text")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional document metadata")

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

# Map of document IDs to corpus indices
document_to_corpus_index: Dict[str, int] = {}

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

async def update_search_indices():
    """
    Update both BM25 and semantic search indices with the current corpus.
    This is run as a background task when the corpus changes.
    """
    global bm25
    
    if not corpus:  
        bm25 = None
        semantic_search.update_corpus([])
        return

    # Update BM25 index
    tokenized_corpus = tokenize_corpus(corpus)
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Update semantic search index
    semantic_search.update_corpus(corpus)
    
    await asyncio.sleep(0.1)  # Small delay to ensure task completion

@app.post("/add-doc/", summary="Add a document to the corpus")
async def add_doc(
    new_doc: str = Query(..., description="New document to be added to the corpus"), 
    doc_id: Optional[str] = Query(None, description="Optional document ID"),
    title: Optional[str] = Query(None, description="Optional document title"),
    background_tasks: BackgroundTasks = None
) -> Dict[str, Any]:
    """
    Add a new document to the corpus and update the search indices.
    
    Args:
        new_doc: Text of the document to add
        doc_id: Optional document ID
        title: Optional document title
        background_tasks: FastAPI background tasks handler
        
    Returns:
        Dictionary with the updated corpus and document info
    """
    # Generate a document ID if not provided
    if not doc_id:
        doc_id = str(uuid.uuid4())
    
    # Add to corpus
    corpus_index = len(corpus)
    corpus.append(new_doc)
    
    # Map document ID to corpus index
    document_to_corpus_index[doc_id] = corpus_index
    
    # Update search indices
    background_tasks.add_task(update_search_indices)
    
    return {
        "doc_id": doc_id,
        "title": title,
        "corpus_index": corpus_index,
        "document": new_doc
    }

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
    global corpus, document_to_corpus_index
    if delete_all.upper() == "Y":
        corpus.clear()
        document_to_corpus_index.clear()
        background_tasks.add_task(update_search_indices)
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

@app.post("/search/", summary="Search for documents")
async def search(
    query: str = Query(..., description="Search query"),
    search_type: SearchType = Query(SearchType.BM25, description="Search type (bm25 or semantic)"),
    n: int = Query(5, description="Number of results to return", ge=1, le=50),
    threshold: float = Query(0.5, description="Similarity threshold", ge=0.5, le=0.95)
) -> Dict[str, List[SearchResult]]:
    """
    Search for documents similar to the query using either BM25 or semantic search.
    
    Args:
        query: Search query text
        search_type: Type of search to perform (bm25 or semantic)
        n: Number of results to return (default: 5)
        threshold: Minimum similarity score threshold (default: 0.5)
        
    Returns:
        Dictionary with search results including document text, score, and preview
    """
    if not corpus:
        raise HTTPException(status_code=400, detail="Corpus is empty. Add documents first.")
    
    async with semaphore:
        results = []
        
        if search_type == SearchType.BM25:
            # BM25 search
            query_tokens = tokenize_text(query)
            bm25_scores = await run_in_threadpool(
                lambda: bm25.get_scores(query_tokens)
            )
            
            # Get indices of documents with scores above threshold
            # For BM25, normalize scores to 0-1 range for threshold comparison
            if len(bm25_scores) > 0:
                max_score = max(bm25_scores) if max(bm25_scores) > 0 else 1
                indices = [i for i, score in enumerate(bm25_scores) 
                          if score/max_score >= threshold]
                
                # Sort by score in descending order
                indices = sorted(indices, key=lambda i: bm25_scores[i], reverse=True)[:n]
                
                # Create results
                for idx in indices:
                    doc = corpus[idx]
                    score = float(bm25_scores[idx])
                    
                    # Find document ID and metadata if available
                    doc_id = None
                    title = None
                    metadata = None
                    
                    # Look up document ID from corpus index
                    for d_id, d_idx in document_to_corpus_index.items():
                        if d_idx == idx:
                            doc_id = d_id
                            if doc_id in documents_metadata:
                                metadata_obj = documents_metadata[doc_id]
                                title = metadata_obj.title
                                metadata = metadata_obj.model_dump()
                            break
                    
                    # Create preview with highlighted terms
                    preview = doc
                    for term in query_tokens:
                        if term in preview.lower():
                            # Simple highlighting with asterisks
                            start = preview.lower().find(term)
                            end = start + len(term)
                            term_in_doc = preview[start:end]
                            preview = preview.replace(term_in_doc, f"**{term_in_doc}**", 1)
                    
                    # Truncate preview if too long
                    if len(preview) > 300:
                        # Find a good breaking point
                        break_point = preview[:300].rfind(" ")
                        if break_point == -1:
                            break_point = 300
                        preview = preview[:break_point] + "..."
                    
                    results.append(SearchResult(
                        document=doc,
                        score=score / max_score,  # Normalize score to 0-1 range
                        doc_id=doc_id,
                        title=title,
                        preview=preview,
                        metadata=metadata
                    ))
        
        else:  # Semantic search
            # Perform semantic search
            semantic_results = await run_in_threadpool(
                lambda: semantic_search.search(query, n=n, threshold=threshold)
            )
            
            # Create results
            for result in semantic_results:
                doc = result["document"]
                idx = result["index"]
                score = result["score"]
                
                # Find document ID and metadata if available
                doc_id = None
                title = None
                metadata = None
                
                # Look up document ID from corpus index
                for d_id, d_idx in document_to_corpus_index.items():
                    if d_idx == idx:
                        doc_id = d_id
                        if doc_id in documents_metadata:
                            metadata_obj = documents_metadata[doc_id]
                            title = metadata_obj.title
                            metadata = metadata_obj.model_dump()
                        break
                
                # Create preview with highlighted matches
                preview = await run_in_threadpool(
                    lambda: semantic_search.highlight_matches(query, doc)
                )
                
                results.append(SearchResult(
                    document=doc,
                    score=score,
                    doc_id=doc_id,
                    title=title,
                    preview=preview,
                    metadata=metadata
                ))
    
    return {"results": results}

@app.post("/find-similar/", summary="Find similar documents (legacy endpoint)")
async def find_similar(
    query: str = Query(..., description="Search query to find similar documents"),
    n: int = Query(5, description="Number of results to return", ge=1, le=10)
) -> Dict[str, List[str]]:
    """
    Legacy endpoint for BM25 search. Use /search/ for more options.
    
    Args:
        query: Search query text
        n: Number of results to return (default: 5)
        
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
                    chunk_index = len(corpus)
                    corpus.append(chunk)
                    # Map this chunk to the same document ID
                    document_to_corpus_index[doc_id + f"_chunk_{chunk_index-len(corpus)+1}"] = chunk_index
            
            # Update search indices
            await update_search_indices()
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
            corpus_index = len(corpus)
            corpus.append(content_preview)
            document_to_corpus_index[doc_id] = corpus_index
            if background_tasks:
                background_tasks.add_task(update_search_indices)
                
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
            for i, chunk in enumerate(chunks):
                if chunk not in corpus:
                    chunk_index = len(corpus)
                    corpus.append(chunk)
                    # Map this chunk to the same document ID
                    document_to_corpus_index[doc_id + f"_chunk_{i+1}"] = chunk_index
            
            if background_tasks:
                background_tasks.add_task(update_search_indices)
                
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
            corpus_index = len(corpus)
            corpus.append(text)
            document_to_corpus_index[doc_id] = corpus_index
            if background_tasks:
                background_tasks.add_task(update_search_indices)
    
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
            background_tasks.add_task(update_search_indices)
    
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

@app.get("/visualization-data/", summary="Get visualization data for document embeddings")
async def get_visualization_data(
    method: DimensionalityReductionMethod = Query(
        DimensionalityReductionMethod.TSNE, 
        description="Dimensionality reduction method (tsne, umap, pca)"
    ),
    perplexity: int = Query(
        30, 
        description="Perplexity parameter for t-SNE (higher values consider more global structure)",
        ge=5, 
        le=50
    ),
    n_neighbors: int = Query(
        15, 
        description="Number of neighbors for UMAP (higher values consider more global structure)",
        ge=2, 
        le=100
    ),
    min_dist: float = Query(
        0.1, 
        description="Minimum distance for UMAP (lower values create tighter clusters)",
        ge=0.0, 
        le=0.99
    ),
    random_state: int = Query(
        42, 
        description="Random state for reproducibility",
        ge=0
    )
) -> Dict[str, Any]:
    """
    Generate visualization data for document embeddings using dimensionality reduction.
    
    Args:
        method: Dimensionality reduction method (t-SNE, UMAP, PCA)
        perplexity: Perplexity parameter for t-SNE (default: 30)
        n_neighbors: Number of neighbors for UMAP (default: 15)
        min_dist: Minimum distance for UMAP (default: 0.1)
        random_state: Random state for reproducibility (default: 42)
        
    Returns:
        Dictionary with visualization data including 2D coordinates, document IDs, and metadata
    """
    if not corpus:
        raise HTTPException(status_code=400, detail="Corpus is empty. Add documents first.")
    
    async with semaphore:
        # Get document embeddings from semantic search
        embeddings = await run_in_threadpool(
            lambda: semantic_search.model.encode(corpus, show_progress_bar=False)
        )
        
        # Prepare document IDs and metadata
        document_ids = []
        metadata_list = []
        
        for doc_idx, doc in enumerate(corpus):
            # Find document ID for this corpus index
            doc_id = None
            for d_id, d_idx in document_to_corpus_index.items():
                if d_idx == doc_idx:
                    doc_id = d_id
                    break
            
            # If no document ID found, generate one
            if not doc_id:
                doc_id = f"doc_{doc_idx}"
            
            document_ids.append(doc_id)
            
            # Get metadata if available
            meta = None
            if doc_id in documents_metadata:
                meta = documents_metadata[doc_id].model_dump()
            else:
                # Create minimal metadata
                meta = {
                    "document": doc[:100] + "..." if len(doc) > 100 else doc,
                    "corpus_index": doc_idx
                }
            
            metadata_list.append(meta)
        
        # Generate visualization data
        visualization_data = await run_in_threadpool(
            lambda: VisualizationData.generate_visualization_data(
                embeddings=embeddings,
                document_ids=document_ids,
                metadata=metadata_list,
                method=method,
                perplexity=perplexity,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                random_state=random_state
            )
        )
        
        return visualization_data
