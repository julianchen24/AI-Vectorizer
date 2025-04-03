"""
Integration tests for the AI Vectorizer application.

This module contains tests that verify the integration between different
components of the application, including document processing, search,
visualization, and insights.
"""

import os
import pytest
import tempfile
import json
from fastapi.testclient import TestClient
from app.app import app

client = TestClient(app)

@pytest.fixture
def sample_pdf():
    """Create a temporary PDF file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        # Write some dummy PDF content
        f.write(b"%PDF-1.5\n%Test PDF file for integration testing")
        file_path = f.name
    
    yield file_path
    
    # Clean up the file after the test
    if os.path.exists(file_path):
        os.remove(file_path)

@pytest.fixture
def sample_text():
    """Sample text content for testing."""
    return "This is a sample document about artificial intelligence and machine learning."

@pytest.fixture
def uploaded_document(sample_pdf):
    """Upload a document and return its metadata."""
    with open(sample_pdf, "rb") as f:
        response = client.post(
            "/upload/",
            files={"file": ("test.pdf", f, "application/pdf")}
        )
    
    assert response.status_code == 200
    return response.json()

@pytest.fixture
def uploaded_text(sample_text):
    """Upload text content and return its metadata."""
    response = client.post(
        "/upload/",
        data={"text": sample_text, "title": "Test Text Document"}
    )
    
    assert response.status_code == 200
    return response.json()

def test_document_upload_and_list(sample_pdf):
    """Test document upload and listing."""
    # Upload a document
    with open(sample_pdf, "rb") as f:
        response = client.post(
            "/upload/",
            files={"file": ("test.pdf", f, "application/pdf")}
        )
    
    assert response.status_code == 200
    doc_metadata = response.json()
    assert doc_metadata["filename"] == "test.pdf"
    assert doc_metadata["file_type"] == "pdf"
    assert "doc_id" in doc_metadata
    
    # List documents
    response = client.get("/documents/")
    assert response.status_code == 200
    documents = response.json()["documents"]
    assert len(documents) >= 1
    
    # Verify the uploaded document is in the list
    doc_ids = [doc["doc_id"] for doc in documents]
    assert doc_metadata["doc_id"] in doc_ids

def test_text_upload_and_processing(sample_text):
    """Test text upload and processing."""
    # Upload text content
    response = client.post(
        "/upload/",
        data={"text": sample_text, "title": "Test Text Document"}
    )
    
    assert response.status_code == 200
    doc_metadata = response.json()
    assert doc_metadata["filename"].endswith(".txt")
    assert doc_metadata["file_type"] == "txt"
    assert doc_metadata["processing_status"] == "processed"
    assert doc_metadata["title"] == "Test Text Document"
    assert doc_metadata["content_preview"] == sample_text
    
    # Get document by ID
    doc_id = doc_metadata["doc_id"]
    response = client.get(f"/documents/{doc_id}")
    assert response.status_code == 200
    retrieved_doc = response.json()
    assert retrieved_doc["doc_id"] == doc_id
    assert retrieved_doc["title"] == "Test Text Document"

def test_search_functionality(uploaded_text):
    """Test search functionality with uploaded text."""
    # Add the document to the corpus
    doc_id = uploaded_text["doc_id"]
    
    # Perform BM25 search
    response = client.post(
        "/search/",
        params={
            "query": "artificial intelligence",
            "search_type": "bm25",
            "n": 5,
            "threshold": 0.5
        }
    )
    
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) > 0
    
    # Check if our document is in the results
    result_doc_ids = [result["doc_id"] for result in results if result["doc_id"] is not None]
    assert doc_id in result_doc_ids
    
    # Perform semantic search
    response = client.post(
        "/search/",
        params={
            "query": "machine learning",
            "search_type": "semantic",
            "n": 5,
            "threshold": 0.5
        }
    )
    
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) > 0
    
    # Check if our document is in the results
    result_doc_ids = [result["doc_id"] for result in results if result["doc_id"] is not None]
    assert doc_id in result_doc_ids

def test_visualization_data(uploaded_text):
    """Test visualization data generation."""
    # Get visualization data with t-SNE
    response = client.get(
        "/visualization-data/",
        params={"method": "tsne", "perplexity": 30, "random_state": 42}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["method"] == "tsne"
    assert "points" in data
    assert len(data["points"]) > 0
    
    # Get visualization data with UMAP
    response = client.get(
        "/visualization-data/",
        params={"method": "umap", "n_neighbors": 15, "min_dist": 0.1, "random_state": 42}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["method"] == "umap"
    assert "points" in data
    assert len(data["points"]) > 0
    
    # Get visualization data with PCA
    response = client.get(
        "/visualization-data/",
        params={"method": "pca", "random_state": 42}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["method"] == "pca"
    assert "points" in data
    assert len(data["points"]) > 0

def test_insights_generation(uploaded_text):
    """Test insights generation."""
    # Get insights with K-Means clustering
    response = client.get(
        "/insights/",
        params={"clustering_method": "kmeans", "n_clusters": 2, "random_state": 42}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "clusters" in data
    assert len(data["clusters"]) > 0
    
    # Get similar document pairs
    response = client.get(
        "/insights/similar-pairs/",
        params={"threshold": 0.7, "max_pairs": 10}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "similar_pairs" in data
    # Note: We might not have similar pairs if there's only one document

def test_document_deletion(uploaded_text):
    """Test document deletion."""
    doc_id = uploaded_text["doc_id"]
    
    # Delete the document
    response = client.delete(f"/documents/{doc_id}")
    assert response.status_code == 200
    assert response.json()["message"] == f"Document {doc_id} deleted successfully"
    
    # Verify the document is no longer in the list
    response = client.get("/documents/")
    assert response.status_code == 200
    documents = response.json()["documents"]
    doc_ids = [doc["doc_id"] for doc in documents]
    assert doc_id not in doc_ids
    
    # Verify getting the document returns 404
    response = client.get(f"/documents/{doc_id}")
    assert response.status_code == 404

def test_end_to_end_workflow():
    """Test the complete end-to-end workflow."""
    # 1. Upload text content
    sample_text = "This document discusses artificial intelligence, machine learning, and natural language processing."
    response = client.post(
        "/upload/",
        data={"text": sample_text, "title": "AI Overview"}
    )
    
    assert response.status_code == 200
    doc_metadata = response.json()
    doc_id = doc_metadata["doc_id"]
    
    # 2. Verify document is in the list
    response = client.get("/documents/")
    assert response.status_code == 200
    documents = response.json()["documents"]
    doc_ids = [doc["doc_id"] for doc in documents]
    assert doc_id in doc_ids
    
    # 3. Perform semantic search
    response = client.post(
        "/search/",
        params={
            "query": "natural language processing",
            "search_type": "semantic",
            "n": 5,
            "threshold": 0.5
        }
    )
    
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) > 0
    result_doc_ids = [result["doc_id"] for result in results if result["doc_id"] is not None]
    assert doc_id in result_doc_ids
    
    # 4. Get visualization data
    response = client.get("/visualization-data/")
    assert response.status_code == 200
    data = response.json()
    assert "points" in data
    
    # 5. Get insights
    response = client.get("/insights/")
    assert response.status_code == 200
    data = response.json()
    assert "clusters" in data
    
    # 6. Delete the document
    response = client.delete(f"/documents/{doc_id}")
    assert response.status_code == 200
    
    # 7. Verify document is deleted
    response = client.get(f"/documents/{doc_id}")
    assert response.status_code == 404
