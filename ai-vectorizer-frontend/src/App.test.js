import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import App from './App';

// Mock the components to avoid issues with API calls
jest.mock('./components/FileUpload', () => () => <div data-testid="file-upload">File Upload Component</div>);
jest.mock('./components/DocumentList', () => () => <div data-testid="document-list">Document List Component</div>);
jest.mock('./components/SearchInterface', () => () => <div data-testid="search-interface">Search Interface Component</div>);
jest.mock('./components/VectorVisualization', () => () => <div data-testid="vector-visualization">Vector Visualization Component</div>);

// Mock useState for activeTab
const mockSetActiveTab = jest.fn();
jest.mock('react', () => ({
  ...jest.requireActual('react'),
  useState: jest.fn()
}));

describe('App Component', () => {
  beforeEach(() => {
    // Reset mocks
    React.useState.mockImplementation((initialValue) => {
      // Only mock the activeTab state, let other useState calls work normally
      if (initialValue === 'upload') {
        return ['upload', mockSetActiveTab];
      }
      return [initialValue, jest.fn()];
    });
  });

  test('renders header with app title', () => {
    render(<App />);
    // Use a more specific selector to target the h1 element
    const headerElement = screen.getByRole('heading', { level: 1, name: /AI Vectorizer/i });
    expect(headerElement).toBeInTheDocument();
  });

  test('renders upload tab by default', () => {
    // Mock the activeTab state to return 'upload'
    React.useState.mockImplementation((initialValue) => {
      if (initialValue === 'upload') {
        return ['upload', mockSetActiveTab];
      }
      return [initialValue, jest.fn()];
    });
    
    render(<App />);
    expect(screen.getByTestId('file-upload')).toBeInTheDocument();
    expect(screen.getByTestId('document-list')).toBeInTheDocument();
    expect(screen.queryByTestId('search-interface')).not.toBeInTheDocument();
  });

  test('renders search tab when activeTab is search', () => {
    // Mock the activeTab state to return 'search'
    React.useState.mockImplementation((initialValue) => {
      if (initialValue === 'upload') {
        return ['search', mockSetActiveTab];
      }
      return [initialValue, jest.fn()];
    });
    
    render(<App />);
    
    // Check that search interface is visible
    expect(screen.queryByTestId('file-upload')).not.toBeInTheDocument();
    expect(screen.queryByTestId('document-list')).not.toBeInTheDocument();
    expect(screen.getByTestId('search-interface')).toBeInTheDocument();
    expect(screen.queryByTestId('vector-visualization')).not.toBeInTheDocument();
  });
  
  test('renders visualization tab when activeTab is visualization', () => {
    // Mock the activeTab state to return 'visualization'
    React.useState.mockImplementation((initialValue) => {
      if (initialValue === 'upload') {
        return ['visualization', mockSetActiveTab];
      }
      return [initialValue, jest.fn()];
    });
    
    render(<App />);
    
    // Check that visualization component is visible
    expect(screen.queryByTestId('file-upload')).not.toBeInTheDocument();
    expect(screen.queryByTestId('document-list')).not.toBeInTheDocument();
    expect(screen.queryByTestId('search-interface')).not.toBeInTheDocument();
    expect(screen.getByTestId('vector-visualization')).toBeInTheDocument();
  });

  test('calls setActiveTab when search tab is clicked', () => {
    render(<App />);
    
    // Click on the Search tab
    const searchTab = screen.getByRole('button', { name: /^Search$/i });
    fireEvent.click(searchTab);
    
    // Check that setActiveTab was called with 'search'
    expect(mockSetActiveTab).toHaveBeenCalledWith('search');
  });
  
  test('calls setActiveTab when visualization tab is clicked', () => {
    render(<App />);
    
    // Click on the Visualization tab
    const visualizationTab = screen.getByRole('button', { name: /^Visualization$/i });
    fireEvent.click(visualizationTab);
    
    // Check that setActiveTab was called with 'visualization'
    expect(mockSetActiveTab).toHaveBeenCalledWith('visualization');
  });
});
