"""
Insights Generation Module for AI Vectorizer

This module provides functionality for generating insights from document embeddings,
including clustering, cluster labeling, and similarity metrics.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from enum import Enum
import logging
from collections import Counter
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClusteringMethod(str, Enum):
    """Enum for clustering methods."""
    KMEANS = "kmeans"
    DBSCAN = "dbscan"
    HIERARCHICAL = "hierarchical"

class InsightsGenerator:
    """
    Class for generating insights from document embeddings.
    """
    
    @staticmethod
    def determine_optimal_clusters(
        embeddings: np.ndarray,
        min_clusters: int = 2,
        max_clusters: int = 10
    ) -> int:
        """
        Determine the optimal number of clusters using silhouette score.
        
        Args:
            embeddings: Document embeddings (n_samples, n_features)
            min_clusters: Minimum number of clusters to try
            max_clusters: Maximum number of clusters to try
            
        Returns:
            Optimal number of clusters
        """
        if embeddings.shape[0] <= min_clusters:
            return max(2, embeddings.shape[0] - 1)
        
        # Limit max_clusters to n_samples - 1
        max_clusters = min(max_clusters, embeddings.shape[0] - 1)
        
        # Try different numbers of clusters
        best_score = -1
        best_n_clusters = min_clusters
        
        for n_clusters in range(min_clusters, max_clusters + 1):
            # Skip if we have too few samples for this many clusters
            if embeddings.shape[0] <= n_clusters:
                continue
                
            try:
                # Apply KMeans clustering
                kmeans = KMeans(
                    n_clusters=n_clusters,
                    random_state=42,
                    n_init=10
                )
                cluster_labels = kmeans.fit_predict(embeddings)
                
                # Calculate silhouette score
                score = silhouette_score(embeddings, cluster_labels)
                
                # Update best score and number of clusters
                if score > best_score:
                    best_score = score
                    best_n_clusters = n_clusters
            except Exception as e:
                logger.warning(f"Error calculating silhouette score for {n_clusters} clusters: {str(e)}")
                continue
        
        return best_n_clusters
    
    @staticmethod
    def apply_clustering(
        embeddings: np.ndarray,
        method: ClusteringMethod = ClusteringMethod.KMEANS,
        n_clusters: Optional[int] = None,
        eps: float = 0.5,
        min_samples: int = 5
    ) -> np.ndarray:
        """
        Apply clustering to document embeddings.
        
        Args:
            embeddings: Document embeddings (n_samples, n_features)
            method: Clustering method
            n_clusters: Number of clusters (for KMeans and Hierarchical)
            eps: Maximum distance between samples (for DBSCAN)
            min_samples: Minimum number of samples in a cluster (for DBSCAN)
            
        Returns:
            Cluster labels for each document
        """
        if embeddings.shape[0] == 0:
            return np.array([])
        
        # If we only have one document, assign it to cluster 0
        if embeddings.shape[0] == 1:
            return np.array([0])
        
        try:
            if method == ClusteringMethod.KMEANS:
                # Determine optimal number of clusters if not provided
                if n_clusters is None:
                    n_clusters = InsightsGenerator.determine_optimal_clusters(embeddings)
                
                # Apply KMeans clustering
                kmeans = KMeans(
                    n_clusters=n_clusters,
                    random_state=42,
                    n_init=10
                )
                return kmeans.fit_predict(embeddings)
                
            elif method == ClusteringMethod.DBSCAN:
                # Apply DBSCAN clustering
                dbscan = DBSCAN(
                    eps=eps,
                    min_samples=min_samples
                )
                return dbscan.fit_predict(embeddings)
                
            elif method == ClusteringMethod.HIERARCHICAL:
                # Determine optimal number of clusters if not provided
                if n_clusters is None:
                    n_clusters = InsightsGenerator.determine_optimal_clusters(embeddings)
                
                # Apply Hierarchical clustering
                hierarchical = AgglomerativeClustering(
                    n_clusters=n_clusters
                )
                return hierarchical.fit_predict(embeddings)
                
            else:
                raise ValueError(f"Unknown clustering method: {method}")
                
        except Exception as e:
            logger.error(f"Error applying clustering: {str(e)}")
            raise ValueError(f"Failed to apply clustering: {str(e)}")
    
    @staticmethod
    def extract_keywords(
        documents: List[str],
        n_keywords: int = 5
    ) -> List[str]:
        """
        Extract keywords from a list of documents using TF-IDF.
        
        Args:
            documents: List of document strings
            n_keywords: Number of keywords to extract
            
        Returns:
            List of keywords
        """
        if not documents:
            return []
        
        try:
            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                max_df=0.9,
                min_df=1,
                stop_words='english',
                lowercase=True,
                use_idf=True,
                ngram_range=(1, 2)
            )
            
            # Fit and transform documents
            tfidf_matrix = vectorizer.fit_transform(documents)
            
            # Get feature names
            feature_names = vectorizer.get_feature_names_out()
            
            # Sum TF-IDF scores for each term across all documents
            tfidf_sums = np.array(tfidf_matrix.sum(axis=0)).flatten()
            
            # Get indices of top terms
            top_indices = tfidf_sums.argsort()[-n_keywords:][::-1]
            
            # Get top terms
            top_terms = [feature_names[i] for i in top_indices]
            
            return top_terms
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            return []
    
    @staticmethod
    def generate_cluster_labels(
        documents: List[str],
        cluster_labels: np.ndarray,
        n_keywords: int = 3
    ) -> Dict[int, Dict[str, Any]]:
        """
        Generate labels for clusters based on common keywords.
        
        Args:
            documents: List of document strings
            cluster_labels: Cluster label for each document
            n_keywords: Number of keywords to use for labeling
            
        Returns:
            Dictionary mapping cluster IDs to cluster information
        """
        if not documents or len(documents) == 0:
            return {}
        
        # Get unique cluster labels
        unique_clusters = np.unique(cluster_labels)
        
        # Initialize cluster information
        cluster_info = {}
        
        # Process each cluster
        for cluster_id in unique_clusters:
            # Skip noise points (cluster_id = -1 in DBSCAN)
            if cluster_id == -1:
                continue
                
            # Get documents in this cluster
            cluster_docs = [doc for i, doc in enumerate(documents) if cluster_labels[i] == cluster_id]
            
            # Extract keywords for this cluster
            keywords = InsightsGenerator.extract_keywords(cluster_docs, n_keywords)
            
            # Generate cluster label
            if keywords:
                label = " & ".join(keywords[:2])
            else:
                label = f"Cluster {cluster_id}"
            
            # Store cluster information
            cluster_info[int(cluster_id)] = {
                "label": label,
                "keywords": keywords,
                "size": len(cluster_docs),
                "documents": [i for i, label in enumerate(cluster_labels) if label == cluster_id]
            }
        
        return cluster_info
    
    @staticmethod
    def calculate_similarity_metrics(
        embeddings: np.ndarray,
        document_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Calculate similarity metrics for document embeddings.
        
        Args:
            embeddings: Document embeddings (n_samples, n_features)
            document_ids: List of document IDs
            
        Returns:
            Dictionary with similarity metrics
        """
        if embeddings.shape[0] <= 1:
            return {
                "average_similarity": 0.0,
                "min_similarity": 0.0,
                "max_similarity": 0.0,
                "similar_pairs": [],
                "outliers": []
            }
        
        try:
            # Calculate pairwise cosine similarities
            similarities = cosine_similarity(embeddings)
            
            # Set diagonal to 0 (self-similarity)
            np.fill_diagonal(similarities, 0)
            
            # Calculate metrics
            avg_similarity = np.mean(similarities)
            min_similarity = np.min(similarities)
            max_similarity = np.max(similarities)
            
            # Find highly similar document pairs (top 5)
            similar_pairs = []
            for i in range(embeddings.shape[0]):
                for j in range(i + 1, embeddings.shape[0]):
                    similar_pairs.append({
                        "doc1": document_ids[i],
                        "doc2": document_ids[j],
                        "similarity": float(similarities[i, j])
                    })
            
            # Sort by similarity (descending)
            similar_pairs.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Keep only top 5 pairs
            similar_pairs = similar_pairs[:5]
            
            # Find outliers (documents with low average similarity to others)
            avg_similarities = np.mean(similarities, axis=1)
            outlier_indices = np.argsort(avg_similarities)[:3]  # Bottom 3
            
            outliers = [
                {
                    "doc_id": document_ids[i],
                    "avg_similarity": float(avg_similarities[i])
                }
                for i in outlier_indices
            ]
            
            return {
                "average_similarity": float(avg_similarity),
                "min_similarity": float(min_similarity),
                "max_similarity": float(max_similarity),
                "similar_pairs": similar_pairs,
                "outliers": outliers
            }
            
        except Exception as e:
            logger.error(f"Error calculating similarity metrics: {str(e)}")
            return {
                "average_similarity": 0.0,
                "min_similarity": 0.0,
                "max_similarity": 0.0,
                "similar_pairs": [],
                "outliers": []
            }
    
    @staticmethod
    def generate_insights(
        embeddings: np.ndarray,
        documents: List[str],
        document_ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        clustering_method: ClusteringMethod = ClusteringMethod.KMEANS,
        n_clusters: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate insights from document embeddings.
        
        Args:
            embeddings: Document embeddings (n_samples, n_features)
            documents: List of document strings
            document_ids: List of document IDs
            metadata: Optional list of document metadata
            clustering_method: Clustering method
            n_clusters: Number of clusters (optional)
            
        Returns:
            Dictionary with insights
        """
        if len(documents) != embeddings.shape[0]:
            raise ValueError("Number of documents must match number of embeddings")
        
        if len(document_ids) != embeddings.shape[0]:
            raise ValueError("Number of document IDs must match number of embeddings")
        
        if metadata is not None and len(metadata) != embeddings.shape[0]:
            raise ValueError("Number of metadata entries must match number of embeddings")
        
        # Apply clustering
        cluster_labels = InsightsGenerator.apply_clustering(
            embeddings,
            method=clustering_method,
            n_clusters=n_clusters
        )
        
        # Generate cluster labels
        cluster_info = InsightsGenerator.generate_cluster_labels(
            documents,
            cluster_labels
        )
        
        # Calculate similarity metrics
        similarity_metrics = InsightsGenerator.calculate_similarity_metrics(
            embeddings,
            document_ids
        )
        
        # Prepare document information
        doc_info = []
        for i, (doc_id, doc, cluster) in enumerate(zip(document_ids, documents, cluster_labels)):
            info = {
                "id": doc_id,
                "cluster": int(cluster),
                "cluster_label": cluster_info.get(int(cluster), {}).get("label", f"Cluster {cluster}"),
                "preview": doc[:200] + "..." if len(doc) > 200 else doc
            }
            
            # Add metadata if available
            if metadata is not None:
                info["metadata"] = metadata[i]
            
            doc_info.append(info)
        
        # Return insights
        return {
            "document_count": len(documents),
            "cluster_count": len(cluster_info),
            "clusters": cluster_info,
            "similarity_metrics": similarity_metrics,
            "documents": doc_info
        }
