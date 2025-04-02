import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import FileUpload from './FileUpload';
import apiService from '../services/api';

// Mock the API service
jest.mock('../services/api', () => ({
  uploadDocument: jest.fn(),
  uploadText: jest.fn(),
}));

describe('FileUpload Component', () => {
  const mockOnUploadSuccess = jest.fn();
  
  beforeEach(() => {
    jest.clearAllMocks();
  });
  
  test('renders file upload tab by default', () => {
    render(<FileUpload onUploadSuccess={mockOnUploadSuccess} />);
    
    expect(screen.getByText('Upload File')).toHaveClass('border-indigo-500');
    expect(screen.getByText('Drag & drop a file here, or click to select a file')).toBeInTheDocument();
  });
  
  test('switches to text upload tab when clicked', () => {
    render(<FileUpload onUploadSuccess={mockOnUploadSuccess} />);
    
    fireEvent.click(screen.getByText('Enter Text'));
    
    expect(screen.getByText('Enter Text')).toHaveClass('border-indigo-500');
    expect(screen.getByLabelText('Text Content')).toBeInTheDocument();
  });
  
  test('submits text content when form is submitted', async () => {
    const mockResult = { doc_id: '123', title: 'Test Document' };
    apiService.uploadText.mockResolvedValue(mockResult);
    
    render(<FileUpload onUploadSuccess={mockOnUploadSuccess} />);
    
    // Switch to text tab
    fireEvent.click(screen.getByText('Enter Text'));
    
    // Fill in the form
    fireEvent.change(screen.getByLabelText('Title (Optional)'), {
      target: { value: 'Test Title' },
    });
    
    fireEvent.change(screen.getByLabelText('Text Content'), {
      target: { value: 'This is a test document content.' },
    });
    
    // Submit the form
    fireEvent.click(screen.getByText('Upload Text'));
    
    // Check if API was called with correct parameters
    await waitFor(() => {
      expect(apiService.uploadText).toHaveBeenCalledWith(
        'This is a test document content.',
        'Test Title'
      );
    });
    
    // Check if success callback was called
    await waitFor(() => {
      expect(mockOnUploadSuccess).toHaveBeenCalledWith(mockResult);
    });
  });
  
  test('shows error message when text submission fails', async () => {
    apiService.uploadText.mockRejectedValue({
      response: { data: { detail: 'Upload failed' } },
    });
    
    render(<FileUpload onUploadSuccess={mockOnUploadSuccess} />);
    
    // Switch to text tab
    fireEvent.click(screen.getByText('Enter Text'));
    
    // Fill in the form
    fireEvent.change(screen.getByLabelText('Text Content'), {
      target: { value: 'This is a test document content.' },
    });
    
    // Submit the form
    fireEvent.click(screen.getByText('Upload Text'));
    
    // Check if error message is displayed
    await waitFor(() => {
      expect(screen.getByText('Upload failed')).toBeInTheDocument();
    });
    
    // Check that success callback was not called
    expect(mockOnUploadSuccess).not.toHaveBeenCalled();
  });
});
