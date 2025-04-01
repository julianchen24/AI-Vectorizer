"""
Test module for the insights generation functionality.

This module contains tests for the insights endpoints, clustering algorithms,
and similarity metrics.
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient

# Import the FastAPI app
from app.app import app
from app.insights import InsightsGenerator, ClusteringMethod

client = TestClient(app)

def test_insights_endpoint():
    """Test the insights endpoint."""
    # Reset corpus first to ensure a clean state
    client.post("/reset-corpus/?delete_all=Y")
    
    # Add some documents
    client.post("/add-doc/", params={"new_doc": "Document about artificial intelligence and machine learning."})
    client.post("/add-doc/", params={"new_doc": "Machine learning algorithms and neural networks."})
    client.post("/add-doc/", params={"new_doc": "Deep learning is a subset of machine learning."})
    client.post("/add-doc/", params={"new_doc": "Natural language processing helps computers understand human language."})
    client.post("/add-doc/", params={"new_doc": "Computer vision systems can recognize images and objects."})
    
    # Test with default parameters (KMeans)
    response = client.get("/insights/")
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert "document_count" in data
    assert data["document_count"] == 5
    assert "cluster_count" in data
    assert "clusters" in data
    assert "similarity_metrics" in data
    assert "documents" in data
    assert len(data["documents"]) == 5
    
    # Check cluster structure
    for cluster_id, cluster_info in data["clusters"].items():
        assert "label" in cluster_info
        assert "keywords" in cluster_info
        assert "size" in cluster_info
        assert "documents" in cluster_info
    
    # Check similarity metrics structure
    assert "average_similarity" in data["similarity_metrics"]
    assert "min_similarity" in data["similarity_metrics"]
    assert "max_similarity" in data["similarity_metrics"]
    assert "similar_pairs" in data["similarity_metrics"]
    assert "outliers" in data["similarity_metrics"]
    
    # Test with hierarchical clustering
    response = client.get("/insights/", params={"clustering_method": "hierarchical"})
    assert response.status_code == 200
    data = response.json()
    assert "cluster_count" in data
    
    # Test with DBSCAN clustering
    response = client.get("/insights/", params={"clustering_method": "dbscan", "eps": 0.5, "min_samples": 2})
    assert response.status_code == 200
    data = response.json()
    assert "cluster_count" in data

def test_similar_pairs_endpoint():
    """Test the similar pairs endpoint."""
    # Reset corpus first to ensure a clean state
    client.post("/reset-corpus/?delete_all=Y")
    
    # Add some documents
    client.post("/add-doc/", params={"new_doc": "Document about artificial intelligence and machine learning."})
    client.post("/add-doc/", params={"new_doc": "Machine learning algorithms and neural networks."})
    client.post("/add-doc/", params={"new_doc": "Deep learning is a subset of machine learning."})
    client.post("/add-doc/", params={"new_doc": "Natural language processing helps computers understand human language."})
    client.post("/add-doc/", params={"new_doc": "Computer vision systems can recognize images and objects."})
    
    # Test with default parameters
    response = client.get("/insights/similar-pairs/")
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert "similar_pairs" in data
    assert "total_pairs" in data
    assert "threshold" in data
    
    # Check similar pairs structure
    for pair in data["similar_pairs"]:
        assert "doc1" in pair
        assert "doc2" in pair
        assert "similarity" in pair
        assert "id" in pair["doc1"]
        assert "preview" in pair["doc1"]
        assert "id" in pair["doc2"]
        assert "preview" in pair["doc2"]
        assert pair["similarity"] >= data["threshold"]
    
    # Test with custom threshold
    response = client.get("/insights/similar-pairs/", params={"threshold": 0.8})
    assert response.status_code == 200
    data = response.json()
    assert data["threshold"] == 0.8
    
    # Test with custom max_pairs
    response = client.get("/insights/similar-pairs/", params={"max_pairs": 3})
    assert response.status_code == 200
    data = response.json()
    assert len(data["similar_pairs"]) <= 3

def test_insights_with_empty_corpus():
    """Test insights endpoint with empty corpus."""
    # Reset corpus
    client.post("/reset-corpus/", params={"delete_all": "Y"})
    
    # Test with empty corpus
    response = client.get("/insights/")
    assert response.status_code == 400
    assert "Corpus is empty" in response.json()["detail"]
    
    response = client.get("/insights/similar-pairs/")
    assert response.status_code == 400
    assert "Corpus is empty" in response.json()["detail"]

class TestInsightsGenerator:
    """Tests for the InsightsGenerator class."""
    
    def test_determine_optimal_clusters(self):
        """Test determining optimal number of clusters."""
        # Create sample embeddings (10 documents, 5 dimensions)
        embeddings = np.random.rand(10, 5)
        
        # Determine optimal number of clusters
        n_clusters = InsightsGenerator.determine_optimal_clusters(
            embeddings,
            min_clusters=2,
            max_clusters=5
        )
        
        # Check result
        assert 2 <= n_clusters <= 5
        
        # Test with small dataset
        small_embeddings = np.random.rand(3, 5)
        n_clusters = InsightsGenerator.determine_optimal_clusters(
            small_embeddings,
            min_clusters=2,
            max_clusters=5
        )
        assert n_clusters == 2
    
    def test_apply_clustering(self):
        """Test applying clustering algorithms."""
        # Create sample embeddings (10 documents, 5 dimensions)
        embeddings = np.random.rand(10, 5)
        
        # Test KMeans clustering
        kmeans_labels = InsightsGenerator.apply_clustering(
            embeddings,
            method=ClusteringMethod.KMEANS,
            n_clusters=3
        )
        assert kmeans_labels.shape == (10,)
        assert len(np.unique(kmeans_labels)) == 3
        
        # Test Hierarchical clustering
        hierarchical_labels = InsightsGenerator.apply_clustering(
            embeddings,
            method=ClusteringMethod.HIERARCHICAL,
            n_clusters=3
        )
        assert hierarchical_labels.shape == (10,)
        assert len(np.unique(hierarchical_labels)) == 3
        
        # Test DBSCAN clustering
        dbscan_labels = InsightsGenerator.apply_clustering(
            embeddings,
            method=ClusteringMethod.DBSCAN,
            eps=0.5,
            min_samples=2
        )
        assert dbscan_labels.shape == (10,)
        
        # Test with single document
        single_embedding = np.random.rand(1, 5)
        single_labels = InsightsGenerator.apply_clustering(
            single_embedding,
            method=ClusteringMethod.KMEANS
        )
        assert single_labels.shape == (1,)
        assert single_labels[0] == 0
    
    def test_extract_keywords(self):
        """Test extracting keywords from documents."""
        # Create sample documents
        documents = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing helps computers understand human language.",
            "Computer vision systems can recognize images and objects."
        ]
        
        # Extract keywords
        keywords = InsightsGenerator.extract_keywords(
            documents,
            n_keywords=5
        )
        
        # Check result
        assert len(keywords) <= 5
        assert all(isinstance(keyword, str) for keyword in keywords)
        
        # Test with empty documents
        empty_keywords = InsightsGenerator.extract_keywords([])
        assert empty_keywords == []
    
    def test_generate_cluster_labels(self):
        """Test generating cluster labels."""
        # Create sample documents
        documents = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Neural networks are inspired by the human brain.",
            "Natural language processing helps computers understand human language.",
            "Computer vision systems can recognize images and objects."
        ]
        
        # Create cluster labels
        cluster_labels = np.array([0, 0, 0, 1, 1])
        
        # Generate cluster labels
        cluster_info = InsightsGenerator.generate_cluster_labels(
            documents,
            cluster_labels,
            n_keywords=3
        )
        
        # Check result
        assert len(cluster_info) == 2  # Two clusters
        assert 0 in cluster_info
        assert 1 in cluster_info
        
        # Check cluster 0
        assert "label" in cluster_info[0]
        assert "keywords" in cluster_info[0]
        assert "size" in cluster_info[0]
        assert "documents" in cluster_info[0]
        assert cluster_info[0]["size"] == 3
        assert len(cluster_info[0]["keywords"]) <= 3
        
        # Check cluster 1
        assert "label" in cluster_info[1]
        assert "keywords" in cluster_info[1]
        assert "size" in cluster_info[1]
        assert "documents" in cluster_info[1]
        assert cluster_info[1]["size"] == 2
        assert len(cluster_info[1]["keywords"]) <= 3
        
        # Test with empty documents
        empty_cluster_info = InsightsGenerator.generate_cluster_labels([], np.array([]))
        assert empty_cluster_info == {}
    
    def test_calculate_similarity_metrics(self):
        """Test calculating similarity metrics."""
        # Create sample embeddings (5 documents, 3 dimensions)
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0]
        ])
        
        document_ids = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        
        # Calculate similarity metrics
        metrics = InsightsGenerator.calculate_similarity_metrics(
            embeddings,
            document_ids
        )
        
        # Check result
        assert "average_similarity" in metrics
        assert "min_similarity" in metrics
        assert "max_similarity" in metrics
        assert "similar_pairs" in metrics
        assert "outliers" in metrics
        
        # Check similar pairs
        assert len(metrics["similar_pairs"]) <= 5
        for pair in metrics["similar_pairs"]:
            assert "doc1" in pair
            assert "doc2" in pair
            assert "similarity" in pair
            assert pair["doc1"] in document_ids
            assert pair["doc2"] in document_ids
            assert 0.0 <= pair["similarity"] <= 1.0
        
        # Check outliers
        assert len(metrics["outliers"]) <= 3
        for outlier in metrics["outliers"]:
            assert "doc_id" in outlier
            assert "avg_similarity" in outlier
            assert outlier["doc_id"] in document_ids
            assert 0.0 <= outlier["avg_similarity"] <= 1.0
        
        # Test with single document
        single_metrics = InsightsGenerator.calculate_similarity_metrics(
            np.array([[1.0, 0.0, 0.0]]),
            ["doc1"]
        )
        assert single_metrics["average_similarity"] == 0.0
        assert single_metrics["min_similarity"] == 0.0
        assert single_metrics["max_similarity"] == 0.0
        assert single_metrics["similar_pairs"] == []
        assert single_metrics["outliers"] == []
    
    def test_generate_insights(self):
        """Test generating insights."""
        # Create sample embeddings (5 documents, 3 dimensions)
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0]
        ])
        
        documents = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Neural networks are inspired by the human brain.",
            "Natural language processing helps computers understand human language.",
            "Computer vision systems can recognize images and objects."
        ]
        
        document_ids = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        
        metadata = [
            {"title": "Document 1"},
            {"title": "Document 2"},
            {"title": "Document 3"},
            {"title": "Document 4"},
            {"title": "Document 5"}
        ]
        
        # Generate insights
        insights = InsightsGenerator.generate_insights(
            embeddings,
            documents,
            document_ids,
            metadata,
            clustering_method=ClusteringMethod.KMEANS,
            n_clusters=2
        )
        
        # Check result
        assert "document_count" in insights
        assert insights["document_count"] == 5
        assert "cluster_count" in insights
        assert "clusters" in insights
        assert "similarity_metrics" in insights
        assert "documents" in insights
        assert len(insights["documents"]) == 5
        
        # Check document structure
        for doc in insights["documents"]:
            assert "id" in doc
            assert "cluster" in doc
            assert "cluster_label" in doc
            assert "preview" in doc
            assert "metadata" in doc
            assert doc["id"] in document_ids
            assert doc["metadata"] in metadata
        
        # Test with mismatched inputs
        with pytest.raises(ValueError):
            InsightsGenerator.generate_insights(
                embeddings,
                documents[:-1],  # One less document
                document_ids,
                metadata
            )
        
        with pytest.raises(ValueError):
            InsightsGenerator.generate_insights(
                embeddings,
                documents,
                document_ids[:-1],  # One less document ID
                metadata
            )
        
        with pytest.raises(ValueError):
            InsightsGenerator.generate_insights(
                embeddings,
                documents,
                document_ids,
                metadata[:-1]  # One less metadata entry
            )
