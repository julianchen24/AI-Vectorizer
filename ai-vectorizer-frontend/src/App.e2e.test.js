import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App from './App';
import apiService from './services/api';

// Mock the API service
jest.mock('./services/api');

describe('AI Vectorizer End-to-End Tests', () => {
  // Sample test data
  const sampleDocuments = [
    {
      doc_id: '1',
      filename: 'sample1.pdf',
      file_type: 'pdf',
      upload_timestamp: new Date().toISOString(),
      file_size: 1024,
      processing_status: 'processed',
      title: 'Sample Document 1',
      content_preview: 'This is a sample document about artificial intelligence.',
      word_count: 100,
      chunk_count: 2,
      total_tokens: 150
    },
    {
      doc_id: '2',
      filename: 'sample2.txt',
      file_type: 'txt',
      upload_timestamp: new Date().toISOString(),
      file_size: 512,
      processing_status: 'processed',
      title: 'Sample Document 2',
      content_preview: 'This document discusses machine learning concepts.',
      word_count: 75,
      chunk_count: 1,
      total_tokens: 100
    }
  ];

  const sampleSearchResults = [
    {
      document: 'This is a sample document about artificial intelligence.',
      score: 0.85,
      doc_id: '1',
      title: 'Sample Document 1',
      preview: 'This is a sample document about **artificial intelligence**.',
      metadata: sampleDocuments[0]
    },
    {
      document: 'This document discusses machine learning concepts.',
      score: 0.75,
      doc_id: '2',
      title: 'Sample Document 2',
      preview: 'This document discusses **machine learning** concepts.',
      metadata: sampleDocuments[1]
    }
  ];

  const sampleVisualizationData = {
    method: 'tsne',
    points: [
      {
        id: '1',
        x: 0.5,
        y: 0.3,
        metadata: sampleDocuments[0]
      },
      {
        id: '2',
        x: -0.2,
        y: 0.7,
        metadata: sampleDocuments[1]
      }
    ]
  };

  const sampleInsightsData = {
    clusters: [
      {
        label: 'AI Documents',
        keywords: ['artificial', 'intelligence', 'learning'],
        documents: [
          { id: '1', metadata: sampleDocuments[0] },
          { id: '2', metadata: sampleDocuments[1] }
        ]
      }
    ]
  };

  const sampleSimilarPairs = {
    similar_pairs: [
      {
        doc1: { id: '1', preview: 'Sample Document 1' },
        doc2: { id: '2', preview: 'Sample Document 2' },
        similarity: 0.8
      }
    ],
    total_pairs: 1,
    threshold: 0.7
  };

  // Setup API mocks
  beforeEach(() => {
    // Reset all mocks
    jest.clearAllMocks();

    // Mock API responses
    apiService.listDocuments.mockResolvedValue(sampleDocuments);
    apiService.uploadDocument.mockResolvedValue(sampleDocuments[0]);
    apiService.uploadText.mockResolvedValue(sampleDocuments[1]);
    apiService.searchDocuments.mockResolvedValue(sampleSearchResults);
    apiService.getVisualizationData.mockResolvedValue(sampleVisualizationData);
    apiService.getInsights.mockResolvedValue(sampleInsightsData);
    apiService.getSimilarPairs.mockResolvedValue(sampleSimilarPairs);
    apiService.getDocument.mockImplementation((docId) => 
      Promise.resolve(sampleDocuments.find(doc => doc.doc_id === docId))
    );
    apiService.deleteDocument.mockResolvedValue({ message: 'Document deleted successfully' });
  });

  test('Full user journey through the application', async () => {
    // Render the app
    render(<App />);

    // 1. Initial state - Upload tab should be active
    expect(screen.getByText('Upload & Manage')).toBeInTheDocument();
    expect(screen.getByText('Your Documents')).toBeInTheDocument();

    // 2. Document list should be populated
    await waitFor(() => {
      expect(apiService.listDocuments).toHaveBeenCalled();
    });
    
    expect(screen.getByText('Sample Document 1')).toBeInTheDocument();
    expect(screen.getByText('Sample Document 2')).toBeInTheDocument();

    // 3. Upload a text document
    const textInput = screen.getByPlaceholderText('Enter text content here...');
    const titleInput = screen.getByLabelText('Title (optional)');
    const uploadButton = screen.getByText('Upload Text');

    userEvent.type(textInput, 'This is a test document about natural language processing.');
    userEvent.type(titleInput, 'Test NLP Document');
    fireEvent.click(uploadButton);

    await waitFor(() => {
      expect(apiService.uploadText).toHaveBeenCalledWith(
        'This is a test document about natural language processing.',
        'Test NLP Document'
      );
    });
    
    await waitFor(() => {
      expect(apiService.listDocuments).toHaveBeenCalledTimes(2); // Initial + after upload
    });

    // 4. Switch to Search tab
    fireEvent.click(screen.getByText('Search'));
    
    expect(screen.getByText('Document Search')).toBeInTheDocument();
    
    // 5. Perform a search
    const searchInput = screen.getByPlaceholderText('Enter your search query...');
    const searchButton = screen.getByText('Search');
    
    userEvent.type(searchInput, 'artificial intelligence');
    fireEvent.click(searchButton);
    
    await waitFor(() => {
      expect(apiService.searchDocuments).toHaveBeenCalledWith(
        'artificial intelligence',
        'bm25',
        5,
        0.5
      );
    });
    
    expect(screen.getByText('Sample Document 1')).toBeInTheDocument();
    expect(screen.getByText('This is a sample document about artificial intelligence.')).toBeInTheDocument();
    
    // 6. Switch to Visualization tab
    fireEvent.click(screen.getByText('Visualization'));
    
    expect(screen.getByText('Vector Space Visualization')).toBeInTheDocument();
    
    await waitFor(() => {
      expect(apiService.getVisualizationData).toHaveBeenCalled();
    });
    
    // Check for SVG element using Testing Library
    const svgElements = screen.getAllByRole('img', { hidden: true });
    expect(svgElements.length).toBeGreaterThan(0);
    
    // 7. Switch to Insights tab
    fireEvent.click(screen.getByText('Insights'));
    
    expect(screen.getByText('Document Insights')).toBeInTheDocument();
    
    await waitFor(() => {
      expect(apiService.getInsights).toHaveBeenCalled();
    });
    
    // Check for cluster information
    expect(screen.getByText('AI Documents')).toBeInTheDocument();
    
    // 8. Switch to Similar Pairs tab within Insights
    fireEvent.click(screen.getByText('Similar Document Pairs'));
    
    await waitFor(() => {
      expect(apiService.getSimilarPairs).toHaveBeenCalled();
    });
    
    // Check for similar pairs table
    expect(screen.getByText('Document 1')).toBeInTheDocument();
    expect(screen.getByText('Document 2')).toBeInTheDocument();
    expect(screen.getByText('Similarity')).toBeInTheDocument();
    
    // 9. Return to Upload tab and delete a document
    fireEvent.click(screen.getByText('Upload & Manage'));
    
    // Find delete button for first document
    const deleteButtons = screen.getAllByText('Delete');
    fireEvent.click(deleteButtons[0]);
    
    await waitFor(() => {
      expect(apiService.deleteDocument).toHaveBeenCalledWith('1');
    });
    
    await waitFor(() => {
      expect(apiService.listDocuments).toHaveBeenCalledTimes(3); // Initial + after upload + after delete
    });
  });

  test('Error handling during document upload', async () => {
    // Mock API error
    apiService.uploadDocument.mockRejectedValue(new Error('Upload failed'));
    
    // Render the app
    render(<App />);
    
    // Try to upload a file (this will fail)
    const file = new File(['dummy content'], 'test.pdf', { type: 'application/pdf' });
    const fileInput = screen.getByLabelText('Choose a file');
    
    Object.defineProperty(fileInput, 'files', {
      value: [file]
    });
    
    fireEvent.change(fileInput);
    
    const uploadButton = screen.getByText('Upload File');
    fireEvent.click(uploadButton);
    
    await waitFor(() => {
      expect(apiService.uploadDocument).toHaveBeenCalled();
    });
    
    // Check for error message
    expect(screen.getByText('Error: Upload failed')).toBeInTheDocument();
  });

  test('Error handling during search', async () => {
    // Mock API error
    apiService.searchDocuments.mockRejectedValue(new Error('Search failed'));
    
    // Render the app
    render(<App />);
    
    // Switch to Search tab
    fireEvent.click(screen.getByText('Search'));
    
    // Try to search (this will fail)
    const searchInput = screen.getByPlaceholderText('Enter your search query...');
    const searchButton = screen.getByText('Search');
    
    userEvent.type(searchInput, 'test query');
    fireEvent.click(searchButton);
    
    await waitFor(() => {
      expect(apiService.searchDocuments).toHaveBeenCalled();
    });
    
    // Check for error message
    expect(screen.getByText('Error: Search failed')).toBeInTheDocument();
  });
});
