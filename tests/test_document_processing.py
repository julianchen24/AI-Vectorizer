"""
Test module for the document processing functionality.

This module contains tests for text extraction, cleaning, and chunking
functionality provided by the document_processing module.
"""

import os
import io
import tempfile
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

from app.document_processing import (
    clean_text,
    count_tokens,
    chunk_text,
    get_sentences,
    process_document,
    DocumentProcessingError
)

class TestTextCleaning:
    """Tests for text cleaning functionality."""
    
    def test_clean_text_removes_extra_whitespace(self):
        """Test that clean_text removes extra whitespace."""
        text = "This   has    extra   spaces."
        cleaned = clean_text(text)
        assert cleaned == "This has extra spaces."
    
    def test_clean_text_normalizes_newlines(self):
        """Test that clean_text normalizes multiple newlines."""
        text = "Line 1\n\n\nLine 2\n\n\nLine 3"
        cleaned = clean_text(text)
        assert cleaned == "Line 1\nLine 2\nLine 3"
    
    def test_clean_text_strips_lines(self):
        """Test that clean_text strips whitespace from each line."""
        text = "  Line 1  \n  Line 2  \n  Line 3  "
        cleaned = clean_text(text)
        assert cleaned == "Line 1\nLine 2\nLine 3"
    
    def test_clean_text_handles_empty_input(self):
        """Test that clean_text handles empty input."""
        assert clean_text("") == ""
        assert clean_text(None) == ""
    
    def test_clean_text_removes_non_printable_chars(self):
        """Test that clean_text removes non-printable characters."""
        text = "Text with \x00 non-printable \x1F characters."
        cleaned = clean_text(text)
        assert cleaned == "Text with  non-printable  characters."

class TestTokenCounting:
    """Tests for token counting functionality."""
    
    def test_count_tokens_returns_correct_count(self):
        """Test that count_tokens returns the correct token count."""
        text = "This is a simple test sentence."
        count = count_tokens(text)
        assert count > 0  # Exact count depends on the tokenizer
        
        # Test with longer text
        longer_text = "This is a longer test sentence with more tokens to count."
        longer_count = count_tokens(longer_text)
        assert longer_count > count
    
    def test_count_tokens_handles_empty_input(self):
        """Test that count_tokens handles empty input."""
        assert count_tokens("") == 0
        assert count_tokens(None) == 0

class TestSentenceTokenization:
    """Tests for sentence tokenization functionality."""
    
    def test_get_sentences_splits_text_correctly(self):
        """Test that get_sentences correctly splits text into sentences."""
        text = "This is sentence one. This is sentence two! Is this sentence three?"
        sentences = get_sentences(text)
        assert len(sentences) == 3
        assert sentences[0] == "This is sentence one."
        assert sentences[1] == "This is sentence two!"
        assert sentences[2] == "Is this sentence three?"
    
    def test_get_sentences_handles_empty_input(self):
        """Test that get_sentences handles empty input."""
        assert get_sentences("") == []
        assert get_sentences(None) == []

class TestTextChunking:
    """Tests for text chunking functionality."""
    
    def test_chunk_text_respects_max_tokens(self):
        """Test that chunk_text respects the maximum token limit."""
        # Create a long text with multiple paragraphs
        paragraphs = ["Paragraph " + str(i) + ": " + "word " * 50 for i in range(10)]
        text = "\n".join(paragraphs)
        
        # Chunk with a small max_tokens value
        max_tokens = 20
        chunks = chunk_text(text, max_tokens=max_tokens, overlap_tokens=0)
        
        # Verify each chunk is within the token limit
        for chunk in chunks:
            assert count_tokens(chunk) <= max_tokens
    
    def test_chunk_text_creates_overlap(self):
        """Test that chunk_text creates the specified overlap between chunks."""
        # Create text with distinct sentences
        sentences = [f"This is test sentence number {i}." for i in range(20)]
        text = " ".join(sentences)
        
        # Chunk with overlap
        max_tokens = 30
        overlap_tokens = 10
        chunks = chunk_text(text, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
        
        # Need at least 2 chunks to test overlap
        if len(chunks) >= 2:
            # Check for overlap between consecutive chunks
            for i in range(len(chunks) - 1):
                # Get the end of the current chunk and start of the next chunk
                current_chunk_end = chunks[i][-50:]  # Take last 50 chars as a sample
                next_chunk_start = chunks[i+1][:50]  # Take first 50 chars as a sample
                
                # There should be some common text (overlap)
                # We don't check exact token count but verify there is some overlap
                assert any(word in next_chunk_start for word in current_chunk_end.split())
    
    def test_chunk_text_handles_long_sentences(self):
        """Test that chunk_text correctly handles sentences longer than max_tokens."""
        # Create a very long sentence
        long_sentence = "word " * 100
        
        # Chunk with a small max_tokens value
        max_tokens = 20
        chunks = chunk_text(long_sentence, max_tokens=max_tokens, overlap_tokens=0)
        
        # Verify the sentence was split into multiple chunks
        assert len(chunks) > 1
        
        # Verify each chunk is within the token limit
        for chunk in chunks:
            assert count_tokens(chunk) <= max_tokens
    
    def test_chunk_text_handles_empty_input(self):
        """Test that chunk_text handles empty input."""
        assert chunk_text("") == []
        assert chunk_text(None) == []

class TestDocumentProcessing:
    """Tests for the document processing functionality."""
    
    @pytest.fixture
    def sample_text_file(self):
        """Create a temporary text file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", encoding="utf-8") as f:
            f.write("This is a sample text file.\n")
            f.write("It has multiple lines.\n")
            f.write("Each line is a separate paragraph.\n")
            f.write("This is used for testing document processing.")
            file_path = f.name
        
        yield file_path
        
        # Cleanup
        if os.path.exists(file_path):
            os.remove(file_path)
    
    def test_process_document_with_text_file(self, sample_text_file):
        """Test processing a text file."""
        result = process_document(sample_text_file, max_tokens=100, overlap_tokens=10)
        
        # Verify the result contains the expected keys
        assert "file_path" in result
        assert "file_name" in result
        assert "file_type" in result
        assert "file_size" in result
        assert "modified_time" in result
        assert "total_tokens" in result
        assert "chunk_count" in result
        assert "chunks" in result
        assert "extracted_text" in result
        assert "processing_time" in result
        
        # Verify the extracted text contains the content of the file
        assert "sample text file" in result["extracted_text"]
        assert "multiple lines" in result["extracted_text"]
        
        # Verify chunks were created
        assert len(result["chunks"]) > 0
    
    def test_process_document_handles_nonexistent_file(self):
        """Test that process_document raises an error for nonexistent files."""
        with pytest.raises(DocumentProcessingError) as excinfo:
            process_document("nonexistent_file.txt")
        assert "File not found" in str(excinfo.value)
    
    @patch("app.document_processing.extract_text_from_pdf")
    def test_process_document_with_pdf(self, mock_extract_pdf):
        """Test processing a PDF file."""
        # Mock the PDF extraction function
        mock_extract_pdf.return_value = "This is extracted PDF text."
        
        # Create a temporary file with .pdf extension
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            file_path = f.name
        
        try:
            result = process_document(file_path)
            
            # Verify the PDF extraction function was called
            mock_extract_pdf.assert_called_once_with(Path(file_path))
            
            # Verify the result contains the extracted text
            assert result["extracted_text"] == "This is extracted PDF text."
            assert result["file_type"] == "pdf"
            
        finally:
            # Cleanup
            if os.path.exists(file_path):
                os.remove(file_path)
    
    @patch("app.document_processing.extract_text_from_docx")
    def test_process_document_with_docx(self, mock_extract_docx):
        """Test processing a DOCX file."""
        # Mock the DOCX extraction function
        mock_extract_docx.return_value = "This is extracted DOCX text."
        
        # Create a temporary file with .docx extension
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
            file_path = f.name
        
        try:
            result = process_document(file_path)
            
            # Verify the DOCX extraction function was called
            mock_extract_docx.assert_called_once_with(Path(file_path))
            
            # Verify the result contains the extracted text
            assert result["extracted_text"] == "This is extracted DOCX text."
            assert result["file_type"] == "docx"
            
        finally:
            # Cleanup
            if os.path.exists(file_path):
                os.remove(file_path)
    
    def test_process_document_with_unsupported_file_type(self):
        """Test that process_document raises an error for unsupported file types."""
        # Create a temporary file with an unsupported extension
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            file_path = f.name
        
        try:
            with pytest.raises(DocumentProcessingError) as excinfo:
                process_document(file_path)
            
            assert "Unsupported file type" in str(excinfo.value)
            
        finally:
            # Cleanup
            if os.path.exists(file_path):
                os.remove(file_path)
