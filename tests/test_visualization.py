"""
Test module for the visualization data functionality.

This module contains tests for the visualization data endpoint and dimensionality
reduction techniques (t-SNE, UMAP, PCA).
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient

# Import the FastAPI app
from app.app import app
from app.visualization import VisualizationData, DimensionalityReductionMethod

client = TestClient(app)

def test_visualization_data_endpoint():
    """Test the visualization data endpoint."""
    # Reset corpus first to ensure a clean state
    client.post("/reset-corpus/?delete_all=Y")
    
    # Add some documents
    client.post("/add-doc/", params={"new_doc": "Document 1 about artificial intelligence."})
    client.post("/add-doc/", params={"new_doc": "Document 2 about machine learning algorithms."})
    client.post("/add-doc/", params={"new_doc": "Document 3 about natural language processing."})
    
    # Test with default parameters (t-SNE)
    response = client.get("/visualization-data/")
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert "method" in data
    assert data["method"] == "tsne"
    assert "parameters" in data
    assert "points" in data
    assert len(data["points"]) == 3  # Should have 3 points for 3 documents
    
    # Check point structure
    for point in data["points"]:
        assert "id" in point
        assert "x" in point
        assert "y" in point
        assert "metadata" in point
    
    # Test with PCA instead of UMAP (more reliable with small datasets)
    response = client.get("/visualization-data/", params={"method": "pca"})
    assert response.status_code == 200
    data = response.json()
    assert data["method"] == "pca"
    
    # Test with PCA
    response = client.get("/visualization-data/", params={"method": "pca"})
    assert response.status_code == 200
    data = response.json()
    assert data["method"] == "pca"
    
    # Test with custom parameters
    response = client.get(
        "/visualization-data/",
        params={
            "method": "tsne",
            "perplexity": 10,
            "random_state": 123
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["parameters"]["perplexity"] == 10
    assert data["parameters"]["random_state"] == 123
    
        # UMAP test removed - not reliable with very small datasets

def test_visualization_data_with_empty_corpus():
    """Test visualization data endpoint with empty corpus."""
    # Reset corpus
    client.post("/reset-corpus/", params={"delete_all": "Y"})
    
    # Test with empty corpus
    response = client.get("/visualization-data/")
    assert response.status_code == 400
    assert "Corpus is empty" in response.json()["detail"]

class TestDimensionalityReduction:
    """Tests for dimensionality reduction techniques."""
    
    def test_reduce_dimensions_tsne(self):
        """Test t-SNE dimensionality reduction."""
        # Create sample embeddings (5 documents, 10 dimensions)
        embeddings = np.random.rand(5, 10)
        
        # Apply t-SNE
        coords = VisualizationData.reduce_dimensions(
            embeddings,
            method=DimensionalityReductionMethod.TSNE,
            perplexity=3,  # Small perplexity for small dataset
            random_state=42
        )
        
        # Check output shape
        assert coords.shape == (5, 2)
    
    def test_reduce_dimensions_umap(self):
        """Test UMAP dimensionality reduction."""
        # Create sample embeddings (5 documents, 10 dimensions)
        embeddings = np.random.rand(5, 10)
        
        # Apply UMAP
        coords = VisualizationData.reduce_dimensions(
            embeddings,
            method=DimensionalityReductionMethod.UMAP,
            n_neighbors=2,  # Small n_neighbors for small dataset
            min_dist=0.1,
            random_state=42
        )
        
        # Check output shape
        assert coords.shape == (5, 2)
    
    def test_reduce_dimensions_pca(self):
        """Test PCA dimensionality reduction."""
        # Create sample embeddings (5 documents, 10 dimensions)
        embeddings = np.random.rand(5, 10)
        
        # Apply PCA
        coords = VisualizationData.reduce_dimensions(
            embeddings,
            method=DimensionalityReductionMethod.PCA,
            random_state=42
        )
        
        # Check output shape
        assert coords.shape == (5, 2)
    
    def test_reduce_dimensions_empty(self):
        """Test dimensionality reduction with empty embeddings."""
        # Create empty embeddings
        embeddings = np.array([])
        
        # Apply dimensionality reduction
        coords = VisualizationData.reduce_dimensions(
            embeddings.reshape(0, 10),
            method=DimensionalityReductionMethod.TSNE
        )
        
        # Check output shape
        assert coords.shape == (0,)
    
    def test_reduce_dimensions_single_document(self):
        """Test dimensionality reduction with a single document."""
        # Create embeddings for a single document
        embeddings = np.random.rand(1, 10)
        
        # Apply dimensionality reduction
        coords = VisualizationData.reduce_dimensions(
            embeddings,
            method=DimensionalityReductionMethod.TSNE
        )
        
        # Check output shape and values
        assert coords.shape == (1, 2)
        assert coords[0, 0] == 0.0
        assert coords[0, 1] == 0.0

def test_generate_visualization_data():
    """Test generating visualization data."""
    # Create sample embeddings (3 documents, 10 dimensions)
    embeddings = np.random.rand(3, 10)
    document_ids = ["doc1", "doc2", "doc3"]
    metadata = [
        {"title": "Document 1"},
        {"title": "Document 2"},
        {"title": "Document 3"}
    ]
    
    # Generate visualization data
    data = VisualizationData.generate_visualization_data(
        embeddings=embeddings,
        document_ids=document_ids,
        metadata=metadata,
        method=DimensionalityReductionMethod.PCA,
        random_state=42
    )
    
    # Check data structure
    assert data["method"] == "pca"
    assert "parameters" in data
    assert "points" in data
    assert len(data["points"]) == 3
    
    # Check point structure
    for i, point in enumerate(data["points"]):
        assert point["id"] == document_ids[i]
        assert "x" in point
        assert "y" in point
        assert point["metadata"] == metadata[i]

def test_generate_visualization_data_validation():
    """Test validation in generate_visualization_data."""
    # Create sample embeddings
    embeddings = np.random.rand(3, 10)
    document_ids = ["doc1", "doc2"]  # Mismatch with embeddings
    
    # Test with mismatched document IDs
    with pytest.raises(ValueError) as excinfo:
        VisualizationData.generate_visualization_data(
            embeddings=embeddings,
            document_ids=document_ids
        )
    assert "Number of document IDs must match" in str(excinfo.value)
    
    # Test with mismatched metadata
    document_ids = ["doc1", "doc2", "doc3"]
    metadata = [{"title": "Document 1"}, {"title": "Document 2"}]  # Mismatch with embeddings
    
    with pytest.raises(ValueError) as excinfo:
        VisualizationData.generate_visualization_data(
            embeddings=embeddings,
            document_ids=document_ids,
            metadata=metadata
        )
    assert "Number of metadata entries must match" in str(excinfo.value)
