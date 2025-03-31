"""
Test module for the AI Vectorizer BM25 API endpoints.

This module contains tests for all the main endpoints of the BM25 API,
including adding documents, querying corpus information, resetting the corpus,
and finding similar documents.
"""

import pytest
from fastapi.testclient import TestClient

# Import the FastAPI app
from app.app import app 

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
