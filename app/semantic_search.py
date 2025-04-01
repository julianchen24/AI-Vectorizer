"""
Semantic Search Module for AI Vectorizer

This module provides functionality for semantic search using sentence-transformers
embeddings. It includes functions for generating embeddings, calculating similarity,
and performing semantic search on a corpus of documents.
"""

import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default model for sentence embeddings
DEFAULT_MODEL = "all-MiniLM-L6-v2"

class SemanticSearch:
    """
    Class for performing semantic search using sentence-transformers embeddings.
    """
    
    def __init__(self, model_name: str = DEFAULT_MODEL):
        """
        Initialize the semantic search with a sentence-transformers model.
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Loaded sentence-transformers model: {model_name}")
            self.embeddings = None
            self.corpus = None
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            raise ValueError(f"Failed to load model {model_name}: {str(e)}")
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to generate embeddings for
            
        Returns:
            NumPy array of embeddings
        """
        if not texts:
            return np.array([])
        
        try:
            embeddings = self.model.encode(texts, show_progress_bar=False)
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise ValueError(f"Failed to generate embeddings: {str(e)}")
    
    def update_corpus(self, corpus: List[str]) -> None:
        """
        Update the corpus and generate embeddings for all documents.
        
        Args:
            corpus: List of document strings
        """
        if not corpus:
            self.embeddings = None
            self.corpus = None
            return
        
        self.corpus = corpus
        self.embeddings = self.generate_embeddings(corpus)
        logger.info(f"Updated corpus with {len(corpus)} documents")
    
    def search(
        self, 
        query: str, 
        n: int = 5, 
        threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Search the corpus for documents similar to the query.
        
        Args:
            query: Search query text
            n: Maximum number of results to return
            threshold: Minimum similarity score threshold (0.0 to 1.0)
            
        Returns:
            List of dictionaries with document text and similarity score
        """
        if not self.corpus or self.embeddings is None:
            logger.warning("Corpus is empty. No search performed.")
            return []
        
        # Generate embedding for the query
        query_embedding = self.model.encode([query])[0]
        
        # Calculate similarity scores
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Filter by threshold and get top n results
        results = []
        for idx, score in enumerate(similarities):
            if score >= threshold:
                results.append({
                    "document": self.corpus[idx],
                    "score": float(score),
                    "index": idx
                })
        
        # Sort by score in descending order and limit to n results
        results = sorted(results, key=lambda x: x["score"], reverse=True)[:n]
        
        return results
    
    def highlight_matches(self, query: str, document: str) -> str:
        """
        Create a preview snippet with highlighted matching text.
        
        Args:
            query: Search query text
            document: Document text
            
        Returns:
            Document text with matching terms highlighted
        """
        # Simple highlighting by surrounding query terms with asterisks
        # In a real application, you might use more sophisticated highlighting
        
        # Split query into terms
        query_terms = query.lower().split()
        
        # Create a preview with highlighted terms
        preview = document
        
        # Find positions of query terms in the document
        positions = []
        for term in query_terms:
            start = 0
            while True:
                start = document.lower().find(term, start)
                if start == -1:
                    break
                end = start + len(term)
                positions.append((start, end))
                start = end
        
        # Sort positions by start index
        positions.sort()
        
        # Merge overlapping positions
        merged_positions = []
        if positions:
            current_start, current_end = positions[0]
            for start, end in positions[1:]:
                if start <= current_end:
                    current_end = max(current_end, end)
                else:
                    merged_positions.append((current_start, current_end))
                    current_start, current_end = start, end
            merged_positions.append((current_start, current_end))
        
        # Create preview with highlighted terms
        if merged_positions:
            # Extract a window around the first match
            first_match_start = merged_positions[0][0]
            first_match_end = merged_positions[0][1]
            
            # Create a window of ~200 characters around the first match
            window_start = max(0, first_match_start - 100)
            window_end = min(len(document), first_match_end + 100)
            
            # Adjust to word boundaries
            if window_start > 0:
                while window_start > 0 and document[window_start] != ' ':
                    window_start -= 1
                window_start += 1  # Skip the space
            
            if window_end < len(document):
                while window_end < len(document) and document[window_end] != ' ':
                    window_end += 1
            
            # Extract the preview window
            preview = document[window_start:window_end]
            
            # Add ellipsis if needed
            if window_start > 0:
                preview = "..." + preview
            if window_end < len(document):
                preview = preview + "..."
            
            # Highlight terms in the preview
            for start, end in merged_positions:
                if start >= window_start and end <= window_end:
                    # Adjust positions relative to the window
                    rel_start = start - window_start
                    rel_end = end - window_start
                    
                    # Only highlight if within the preview bounds
                    if 0 <= rel_start < len(preview) and 0 < rel_end <= len(preview):
                        term = preview[rel_start:rel_end]
                        preview = preview[:rel_start] + f"**{term}**" + preview[rel_end:]
                        
                        # Adjust subsequent positions for the added highlighting markers
                        for i in range(len(merged_positions)):
                            if merged_positions[i][0] > start:
                                merged_positions[i] = (merged_positions[i][0] + 4, merged_positions[i][1] + 4)
        
        return preview
