import axios from 'axios';

// Base URL for the API
const API_BASE_URL = 'http://localhost:8000';

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// API service functions
const apiService = {
  // Visualization data
  getVisualizationData: async (params = {}) => {
    const response = await api.get('/visualization-data/', { params });
    return response.data;
  },
  
  // Document upload
  uploadDocument: async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await api.post('/upload/', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },
  
  // Upload text content
  uploadText: async (text, title) => {
    const formData = new FormData();
    formData.append('text', text);
    if (title) {
      formData.append('title', title);
    }
    
    const response = await api.post('/upload/', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },
  
  // List documents
  listDocuments: async () => {
    const response = await api.get('/documents/');
    return response.data.documents;
  },
  
  // Search documents
  searchDocuments: async (query, searchType = 'bm25', n = 5, threshold = 0.5) => {
    const response = await api.post('/search/', null, {
      params: {
        query,
        search_type: searchType,
        n,
        threshold,
      },
    });
    return response.data.results;
  },
  
  // Get document by ID
  getDocument: async (docId) => {
    const response = await api.get(`/documents/${docId}`);
    return response.data;
  },
  
  // Delete document
  deleteDocument: async (docId) => {
    const response = await api.delete(`/documents/${docId}`);
    return response.data;
  },
};

export default apiService;
