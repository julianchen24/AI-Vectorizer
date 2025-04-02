import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import apiService from '../services/api';

const FileUpload = ({ onUploadSuccess }) => {
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const [textInput, setTextInput] = useState('');
  const [textTitle, setTextTitle] = useState('');
  const [uploadType, setUploadType] = useState('file'); // 'file' or 'text'

  const onDrop = useCallback(async (acceptedFiles) => {
    if (acceptedFiles.length === 0) return;
    
    setUploading(true);
    setError(null);
    
    try {
      const file = acceptedFiles[0];
      const result = await apiService.uploadDocument(file);
      
      if (onUploadSuccess) {
        onUploadSuccess(result);
      }
    } catch (err) {
      console.error('Upload error:', err);
      let errorMessage = 'Error uploading file';
      
      if (err.response) {
        // The request was made and the server responded with a status code
        // that falls out of the range of 2xx
        errorMessage = err.response.data?.detail || `Server error: ${err.response.status}`;
        console.error('Response data:', err.response.data);
        console.error('Response status:', err.response.status);
        console.error('Response headers:', err.response.headers);
      } else if (err.request) {
        // The request was made but no response was received
        errorMessage = 'No response from server. Please check if the backend is running.';
        console.error('Request:', err.request);
      } else {
        // Something happened in setting up the request that triggered an Error
        errorMessage = err.message || 'Error uploading file';
        console.error('Error message:', err.message);
      }
      
      setError(errorMessage);
    } finally {
      setUploading(false);
    }
  }, [onUploadSuccess]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ 
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/msword': ['.doc'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'text/plain': ['.txt'],
      'text/markdown': ['.md'],
    }
  });

  const handleTextUpload = async (e) => {
    e.preventDefault();
    
    if (!textInput.trim()) {
      setError('Please enter some text');
      return;
    }
    
    setUploading(true);
    setError(null);
    
    try {
      const result = await apiService.uploadText(textInput, textTitle);
      
      if (onUploadSuccess) {
        onUploadSuccess(result);
      }
      
      // Clear form
      setTextInput('');
      setTextTitle('');
    } catch (err) {
      console.error('Text upload error:', err);
      let errorMessage = 'Error uploading text';
      
      if (err.response) {
        // The request was made and the server responded with a status code
        // that falls out of the range of 2xx
        errorMessage = err.response.data?.detail || `Server error: ${err.response.status}`;
        console.error('Response data:', err.response.data);
        console.error('Response status:', err.response.status);
        console.error('Response headers:', err.response.headers);
      } else if (err.request) {
        // The request was made but no response was received
        errorMessage = 'No response from server. Please check if the backend is running.';
        console.error('Request:', err.request);
      } else {
        // Something happened in setting up the request that triggered an Error
        errorMessage = err.message || 'Error uploading text';
        console.error('Error message:', err.message);
      }
      
      setError(errorMessage);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="mb-8">
      <div className="bg-white shadow-md rounded-lg overflow-hidden">
        <div className="border-b border-gray-200">
          <nav className="flex -mb-px">
            <button 
              className={`py-4 px-6 text-center border-b-2 font-medium text-sm ${
                uploadType === 'file'
                  ? 'border-indigo-500 text-indigo-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
              onClick={() => setUploadType('file')}
            >
              Upload File
            </button>
            <button 
              className={`py-4 px-6 text-center border-b-2 font-medium text-sm ${
                uploadType === 'text'
                  ? 'border-indigo-500 text-indigo-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
              onClick={() => setUploadType('text')}
            >
              Enter Text
            </button>
          </nav>
        </div>
        
        <div className="p-6">
          {error && (
            <div className="mb-4 bg-red-50 border-l-4 border-red-500 p-4 text-red-700">
              <p>{error}</p>
            </div>
          )}
          
          {uploadType === 'file' ? (
            <div 
              {...getRootProps()} 
              className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer ${
                isDragActive 
                  ? 'border-indigo-500 bg-indigo-50' 
                  : 'border-gray-300 hover:border-indigo-500 hover:bg-indigo-50'
              }`}
            >
              <input {...getInputProps()} />
              {isDragActive ? (
                <p className="text-indigo-600">Drop the file here...</p>
              ) : (
                <div>
                  <svg 
                    className="mx-auto h-12 w-12 text-gray-400" 
                    stroke="currentColor" 
                    fill="none" 
                    viewBox="0 0 48 48" 
                    aria-hidden="true"
                  >
                    <path 
                      d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" 
                      strokeWidth="2" 
                      strokeLinecap="round" 
                      strokeLinejoin="round" 
                    />
                  </svg>
                  <p className="mt-2 text-gray-700">Drag & drop a file here, or click to select a file</p>
                  <p className="mt-1 text-sm text-gray-500">Supported formats: PDF, DOCX, DOC, TXT, MD</p>
                </div>
              )}
              {uploading && (
                <div className="mt-4">
                  <div className="animate-pulse flex space-x-4">
                    <div className="flex-1 space-y-4 py-1">
                      <div className="h-2 bg-indigo-200 rounded"></div>
                      <div className="space-y-2">
                        <div className="h-2 bg-indigo-200 rounded"></div>
                      </div>
                    </div>
                  </div>
                  <p className="mt-2 text-indigo-600">Uploading...</p>
                </div>
              )}
            </div>
          ) : (
            <form onSubmit={handleTextUpload}>
              <div className="mb-4">
                <label htmlFor="textTitle" className="block text-sm font-medium text-gray-700 mb-1">
                  Title (Optional)
                </label>
                <input
                  type="text"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                  id="textTitle"
                  value={textTitle}
                  onChange={(e) => setTextTitle(e.target.value)}
                  placeholder="Enter a title for your text"
                />
              </div>
              
              <div className="mb-4">
                <label htmlFor="textInput" className="block text-sm font-medium text-gray-700 mb-1">
                  Text Content
                </label>
                <textarea
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                  id="textInput"
                  rows="5"
                  value={textInput}
                  onChange={(e) => setTextInput(e.target.value)}
                  placeholder="Enter or paste text content here"
                  required
                ></textarea>
              </div>
              
              <button 
                type="submit" 
                className={`px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 ${
                  uploading 
                    ? 'opacity-70 cursor-not-allowed' 
                    : 'hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500'
                }`}
                disabled={uploading}
              >
                {uploading ? 'Uploading...' : 'Upload Text'}
              </button>
            </form>
          )}
        </div>
      </div>
    </div>
  );
};

export default FileUpload;
