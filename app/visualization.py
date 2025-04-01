"""
Visualization Data Module for AI Vectorizer

This module provides functionality for generating data for vector space visualization.
It includes dimensionality reduction techniques (t-SNE, UMAP, PCA) to transform
document embeddings into 2D coordinates for visualization.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union, Literal
from enum import Enum
import logging

# Import dimensionality reduction techniques
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DimensionalityReductionMethod(str, Enum):
    """Enum for dimensionality reduction methods."""
    TSNE = "tsne"
    UMAP = "umap"
    PCA = "pca"

class VisualizationData:
    """
    Class for generating visualization data using dimensionality reduction techniques.
    """
    
    @staticmethod
    def reduce_dimensions(
        embeddings: np.ndarray,
        method: DimensionalityReductionMethod,
        perplexity: int = 30,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        random_state: int = 42
    ) -> np.ndarray:
        """
        Apply dimensionality reduction to embeddings.
        
        Args:
            embeddings: Document embeddings (n_samples, n_features)
            method: Dimensionality reduction method (t-SNE, UMAP, PCA)
            perplexity: Perplexity parameter for t-SNE (default: 30)
            n_neighbors: Number of neighbors for UMAP (default: 15)
            min_dist: Minimum distance for UMAP (default: 0.1)
            random_state: Random state for reproducibility (default: 42)
            
        Returns:
            2D coordinates (n_samples, 2)
        """
        if embeddings.shape[0] == 0:
            return np.array([])
        
        # If we only have one document, we can't reduce dimensions
        if embeddings.shape[0] == 1:
            return np.array([[0.0, 0.0]])
        
        try:
            if method == DimensionalityReductionMethod.TSNE:
                # Apply t-SNE
                tsne = TSNE(
                    n_components=2,
                    perplexity=min(perplexity, embeddings.shape[0] - 1),
                    random_state=random_state
                )
                return tsne.fit_transform(embeddings)
                
            elif method == DimensionalityReductionMethod.UMAP:
                # Apply UMAP
                reducer = umap.UMAP(
                    n_components=2,
                    n_neighbors=min(n_neighbors, embeddings.shape[0] - 1),
                    min_dist=min_dist,
                    random_state=random_state
                )
                return reducer.fit_transform(embeddings)
                
            elif method == DimensionalityReductionMethod.PCA:
                # Apply PCA
                pca = PCA(n_components=2, random_state=random_state)
                return pca.fit_transform(embeddings)
                
            else:
                raise ValueError(f"Unknown dimensionality reduction method: {method}")
                
        except Exception as e:
            logger.error(f"Error applying dimensionality reduction: {str(e)}")
            raise ValueError(f"Failed to apply dimensionality reduction: {str(e)}")
    
    @staticmethod
    def generate_visualization_data(
        embeddings: np.ndarray,
        document_ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        method: DimensionalityReductionMethod = DimensionalityReductionMethod.TSNE,
        perplexity: int = 30,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Generate visualization data from document embeddings.
        
        Args:
            embeddings: Document embeddings (n_samples, n_features)
            document_ids: List of document IDs
            metadata: Optional list of document metadata
            method: Dimensionality reduction method (t-SNE, UMAP, PCA)
            perplexity: Perplexity parameter for t-SNE
            n_neighbors: Number of neighbors for UMAP
            min_dist: Minimum distance for UMAP
            random_state: Random state for reproducibility
            
        Returns:
            Dictionary with visualization data
        """
        if len(document_ids) != embeddings.shape[0]:
            raise ValueError("Number of document IDs must match number of embeddings")
        
        if metadata is not None and len(metadata) != embeddings.shape[0]:
            raise ValueError("Number of metadata entries must match number of embeddings")
        
        # Apply dimensionality reduction
        coordinates = VisualizationData.reduce_dimensions(
            embeddings,
            method,
            perplexity,
            n_neighbors,
            min_dist,
            random_state
        )
        
        # Prepare visualization data
        points = []
        for i, (doc_id, coord) in enumerate(zip(document_ids, coordinates)):
            point = {
                "id": doc_id,
                "x": float(coord[0]),
                "y": float(coord[1])
            }
            
            # Add metadata if available
            if metadata is not None:
                point["metadata"] = metadata[i]
            
            points.append(point)
        
        # Return visualization data
        return {
            "method": method.value,
            "parameters": {
                "perplexity": perplexity if method == DimensionalityReductionMethod.TSNE else None,
                "n_neighbors": n_neighbors if method == DimensionalityReductionMethod.UMAP else None,
                "min_dist": min_dist if method == DimensionalityReductionMethod.UMAP else None,
                "random_state": random_state
            },
            "points": points
        }
