import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import SearchInterface from './SearchInterface';
import apiService from '../services/api';

// Mock the API service
jest.mock('../services/api', () => ({
  searchDocuments: jest.fn(),
}));

describe('SearchInterface Component', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });
  
  test('renders search form correctly', () => {
    render(<SearchInterface />);
    
    expect(screen.getByText('Search Documents')).toBeInTheDocument();
    expect(screen.getByLabelText('Search Query')).toBeInTheDocument();
    expect(screen.getByText('BM25 (Keyword)')).toBeInTheDocument();
    expect(screen.getByText('Semantic')).toBeInTheDocument();
    expect(screen.getByLabelText(/Result Count/)).toBeInTheDocument();
    expect(screen.getByLabelText(/Similarity Threshold/)).toBeInTheDocument();
  });
  
  test('performs search when form is submitted', async () => {
    const mockResults = [
      {
        document: 'This is a test document',
        score: 0.85,
        doc_id: '123',
        title: 'Test Document',
        preview: 'This is a **test** document',
        metadata: {
          file_type: 'txt',
          upload_timestamp: '2025-04-01T12:00:00Z',
          word_count: 5
        }
      }
    ];
    
    apiService.searchDocuments.mockResolvedValue(mockResults);
    
    render(<SearchInterface />);
    
    // Fill in the search query
    fireEvent.change(screen.getByLabelText('Search Query'), {
      target: { value: 'test query' },
    });
    
    // Change search type to semantic
    fireEvent.click(screen.getByLabelText('Semantic'));
    
    // Change result count
    fireEvent.change(screen.getByLabelText(/Result Count/), {
      target: { value: '10' },
    });
    
    // Change threshold
    fireEvent.change(screen.getByLabelText(/Similarity Threshold/), {
      target: { value: '0.7' },
    });
    
    // Submit the form
    fireEvent.click(screen.getByText('Search'));
    
    // Check if API was called with correct parameters
    await waitFor(() => {
      expect(apiService.searchDocuments).toHaveBeenCalledWith(
        'test query',
        'semantic',
        10,
        0.7
      );
    });
    
    // Check if results are displayed
    await waitFor(() => {
      expect(screen.getByText('Search Results (1)')).toBeInTheDocument();
    });
    
    expect(screen.getByText('Test Document')).toBeInTheDocument();
    expect(screen.getByText('Score: 0.85')).toBeInTheDocument();
  });
  
  test('shows error message when search fails', async () => {
    apiService.searchDocuments.mockRejectedValue({
      response: { data: { detail: 'Search failed' } },
    });
    
    render(<SearchInterface />);
    
    // Fill in the search query
    fireEvent.change(screen.getByLabelText('Search Query'), {
      target: { value: 'test query' },
    });
    
    // Submit the form
    fireEvent.click(screen.getByText('Search'));
    
    // Check if error message is displayed
    await waitFor(() => {
      expect(screen.getByText('Search failed')).toBeInTheDocument();
    });
  });
  
  test('shows no results message when search returns empty results', async () => {
    apiService.searchDocuments.mockResolvedValue([]);
    
    render(<SearchInterface />);
    
    // Fill in the search query
    fireEvent.change(screen.getByLabelText('Search Query'), {
      target: { value: 'test query' },
    });
    
    // Submit the form
    fireEvent.click(screen.getByText('Search'));
    
    // Check if no results message is displayed
    await waitFor(() => {
      expect(screen.getByText('No results found for your query.')).toBeInTheDocument();
    });
  });
});
