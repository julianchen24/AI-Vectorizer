"""
Test module for the AI Vectorizer BM25 API endpoints.

This module contains tests for all the main endpoints of the BM25 API,
including adding documents, querying corpus information, resetting the corpus,
finding similar documents, and document upload and metadata management.
"""

import os
import io
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Import the FastAPI app
from app.app import app, UPLOAD_DIR

client = TestClient(app)

def test_add_doc():
    """Test adding a document to the corpus."""
    response = client.post("/add-doc/?new_doc=Deep learning is a powerful AI technique.")
    assert response.status_code == 200
    assert "Corpus added" in response.json()
    assert "Deep learning is a powerful AI technique." in response.json()["Corpus added"]


def test_get_query():
    """Test getting corpus information."""
    response = client.get("/query/")
    assert response.status_code == 200
    assert "corpus_size" in response.json()
    assert "documents" in response.json()
    assert "has_bm25_index" in response.json()
    assert response.json()["has_bm25_index"] is True


def test_reset_corpus():
    """Test resetting the corpus."""
    response = client.post("/reset-corpus/?delete_all=Y")
    assert response.status_code == 200
    assert response.json() == {"message": "Corpus reset"}
    
    # Verify corpus is empty after reset
    response = client.get("/query/")
    assert response.status_code == 400  # Should return 400 when corpus is empty
    

def test_find_similar():
    """Test finding similar documents."""
    # Add a document first
    client.post("/add-doc/", params={"new_doc": "Artificial Intelligence is transforming industries."})
    
    # Test finding similar documents
    response = client.post("/find-similar/", params={"query": "Artificial Intelligence"})
    assert response.status_code == 200
    assert "most_similar_results" in response.json()
    assert len(response.json()["most_similar_results"]) > 0
    
    # Test with n parameter
    response = client.post("/find-similar/", params={"query": "Artificial Intelligence", "n": 2})
    assert response.status_code == 200
    assert "most_similar_results" in response.json()

@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    """Setup and teardown for tests."""
    # Setup: ensure upload directory exists
    UPLOAD_DIR.mkdir(exist_ok=True)
    
    yield  # Run the tests
    
    # Teardown: clean up upload directory
    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR)
        UPLOAD_DIR.mkdir(exist_ok=True)

def test_upload_text_document():
    """Test uploading a text document."""
    response = client.post(
        "/upload/",
        data={
            "text": "This is a test document for upload.",
            "title": "Test Document"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Check metadata fields
    assert "doc_id" in data
    assert data["filename"] == "Test Document.txt"
    assert data["file_type"] == "txt"
    assert "upload_timestamp" in data
    assert data["file_size"] > 0
    assert data["processing_status"] == "uploaded"
    assert data["title"] == "Test Document"
    assert data["word_count"] == 7
    assert data["content_preview"] == "This is a test document for upload."
    
    # Store doc_id for later tests
    doc_id = data["doc_id"]
    
    # Test document retrieval
    response = client.get(f"/documents/{doc_id}")
    assert response.status_code == 200
    assert response.json()["doc_id"] == doc_id
    
    # Test document list
    response = client.get("/documents/")
    assert response.status_code == 200
    assert "documents" in response.json()
    assert any(doc["doc_id"] == doc_id for doc in response.json()["documents"])
    
    # Test document deletion
    response = client.delete(f"/documents/{doc_id}")
    assert response.status_code == 200
    assert response.json()["message"] == f"Document {doc_id} deleted successfully"
    
    # Verify document is deleted
    response = client.get(f"/documents/{doc_id}")
    assert response.status_code == 404

def test_upload_file_document():
    """Test uploading a file document."""
    # Create a test file
    test_content = b"This is a test file content for upload testing."
    test_file = io.BytesIO(test_content)
    
    response = client.post(
        "/upload/",
        files={"file": ("test_file.txt", test_file, "text/plain")},
        data={"title": "Test File Upload"}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Check metadata fields
    assert "doc_id" in data
    assert data["filename"] == "test_file.txt"
    assert data["file_type"] == "txt"
    assert "upload_timestamp" in data
    assert data["file_size"] > 0
    assert data["processing_status"] == "uploaded"
    assert data["title"] == "Test File Upload"
    
    # Store doc_id for download test
    doc_id = data["doc_id"]
    
    # Test file download
    response = client.get(f"/documents/{doc_id}/download")
    assert response.status_code == 200
    assert response.content == test_content
    
    # Clean up
    client.delete(f"/documents/{doc_id}")

def test_upload_with_session_id():
    """Test uploading documents with session ID."""
    # Upload two documents with the same session ID
    session_id = "test-session-123"
    
    # First document
    response1 = client.post(
        "/upload/",
        data={
            "text": "First document in session.",
            "title": "Session Doc 1",
            "session_id": session_id
        }
    )
    assert response1.status_code == 200
    doc_id1 = response1.json()["doc_id"]
    
    # Second document
    response2 = client.post(
        "/upload/",
        data={
            "text": "Second document in session.",
            "title": "Session Doc 2",
            "session_id": session_id
        }
    )
    assert response2.status_code == 200
    doc_id2 = response2.json()["doc_id"]
    
    # Test filtering by session ID
    response = client.get(f"/documents/?session_id={session_id}")
    assert response.status_code == 200
    session_docs = response.json()["documents"]
    assert len(session_docs) == 2
    assert any(doc["doc_id"] == doc_id1 for doc in session_docs)
    assert any(doc["doc_id"] == doc_id2 for doc in session_docs)
    
    # Verify session directory was created
    session_dir = UPLOAD_DIR / session_id
    assert session_dir.exists()
    assert session_dir.is_dir()
    
    # Clean up
    client.delete(f"/documents/{doc_id1}")
    client.delete(f"/documents/{doc_id2}")

def test_invalid_document_operations():
    """Test invalid document operations."""
    # Test upload with no file or text
    response = client.post("/upload/")
    assert response.status_code == 400
    
    # Test getting non-existent document
    response = client.get("/documents/non-existent-id")
    assert response.status_code == 404
    
    # Test deleting non-existent document
    response = client.delete("/documents/non-existent-id")
    assert response.status_code == 404
    
    # Test downloading non-existent document
    response = client.get("/documents/non-existent-id/download")
    assert response.status_code == 404
