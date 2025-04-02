import React, { useState, useEffect } from 'react';
import apiService from '../services/api';

const DocumentList = ({ refreshTrigger }) => {
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchDocuments = async () => {
      setLoading(true);
      try {
        const docs = await apiService.listDocuments();
        setDocuments(docs);
        setError(null);
      } catch (err) {
        console.error('Error fetching documents:', err);
        let errorMessage = 'Failed to load documents. Please try again later.';
        
        if (err.response) {
          // The request was made and the server responded with a status code
          // that falls out of the range of 2xx
          errorMessage = err.response.data?.detail || `Server error: ${err.response.status}`;
          console.error('Response data:', err.response.data);
          console.error('Response status:', err.response.status);
        } else if (err.request) {
          // The request was made but no response was received
          errorMessage = 'No response from server. Please check if the backend is running.';
          console.error('Request:', err.request);
        } else {
          // Something happened in setting up the request that triggered an Error
          errorMessage = err.message || 'Failed to load documents. Please try again later.';
          console.error('Error message:', err.message);
        }
        
        setError(errorMessage);
      } finally {
        setLoading(false);
      }
    };

    fetchDocuments();
  }, [refreshTrigger]);

  const handleDelete = async (docId) => {
    if (!window.confirm('Are you sure you want to delete this document?')) {
      return;
    }

    try {
      await apiService.deleteDocument(docId);
      setDocuments(documents.filter(doc => doc.doc_id !== docId));
    } catch (err) {
      console.error('Error deleting document:', err);
      
      let errorMessage = 'Failed to delete document. Please try again.';
      
      if (err.response) {
        errorMessage = err.response.data?.detail || `Server error: ${err.response.status}`;
        console.error('Response data:', err.response.data);
        console.error('Response status:', err.response.status);
      } else if (err.request) {
        errorMessage = 'No response from server. Please check if the backend is running.';
        console.error('Request:', err.request);
      } else {
        errorMessage = err.message || 'Failed to delete document. Please try again.';
        console.error('Error message:', err.message);
      }
      
      alert(errorMessage);
    }
  };

  if (loading) {
    return (
      <div className="animate-pulse p-4">
        <div className="h-4 bg-gray-200 rounded w-1/4 mb-4"></div>
        {[...Array(3)].map((_, i) => (
          <div key={i} className="h-24 bg-gray-100 rounded mb-4 p-4">
            <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
            <div className="h-4 bg-gray-200 rounded w-1/2"></div>
          </div>
        ))}
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border-l-4 border-red-500 p-4 text-red-700 mb-4">
        <p>{error}</p>
      </div>
    );
  }

  if (documents.length === 0) {
    return (
      <div className="text-center py-8 bg-gray-50 rounded-lg">
        <svg 
          className="mx-auto h-12 w-12 text-gray-400" 
          fill="none" 
          viewBox="0 0 24 24" 
          stroke="currentColor" 
          aria-hidden="true"
        >
          <path 
            strokeLinecap="round" 
            strokeLinejoin="round" 
            strokeWidth="2" 
            d="M9 13h6m-3-3v6m-9 1V7a2 2 0 012-2h6l2 2h6a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2z" 
          />
        </svg>
        <h3 className="mt-2 text-sm font-medium text-gray-900">No documents</h3>
        <p className="mt-1 text-sm text-gray-500">Upload a document to get started.</p>
      </div>
    );
  }

  return (
    <div className="bg-white shadow overflow-hidden sm:rounded-md">
      <ul className="divide-y divide-gray-200">
        {documents.map((doc) => (
          <li key={doc.doc_id} className="px-6 py-4 hover:bg-gray-50">
            <div className="flex items-center justify-between">
              <div className="flex-1 min-w-0">
                <div className="flex items-center">
                  {/* File type icon */}
                  <div className="flex-shrink-0 h-10 w-10 bg-indigo-100 rounded-full flex items-center justify-center">
                    <span className="text-indigo-700 font-medium text-xs uppercase">
                      {doc.file_type}
                    </span>
                  </div>
                  
                  <div className="ml-4">
                    <h3 className="text-sm font-medium text-gray-900 truncate">
                      {doc.title || doc.filename}
                    </h3>
                    <div className="mt-1 flex items-center text-xs text-gray-500">
                      <span className="truncate">
                        {new Date(doc.upload_timestamp).toLocaleString()} • 
                        {' '}{(doc.file_size / 1024).toFixed(1)} KB
                        {doc.word_count && ` • ${doc.word_count} words`}
                        {doc.page_count && ` • ${doc.page_count} pages`}
                      </span>
                    </div>
                    {doc.content_preview && (
                      <p className="mt-1 text-xs text-gray-500 line-clamp-2">
                        {doc.content_preview.substring(0, 150)}
                        {doc.content_preview.length > 150 ? '...' : ''}
                      </p>
                    )}
                  </div>
                </div>
              </div>
              
              <div className="ml-4 flex-shrink-0 flex">
                <button
                  onClick={() => handleDelete(doc.doc_id)}
                  className="ml-2 text-red-600 hover:text-red-900 text-sm font-medium"
                >
                  Delete
                </button>
              </div>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default DocumentList;
