import React, { useState } from 'react';
import FileUpload from './components/FileUpload';
import DocumentList from './components/DocumentList';
import SearchInterface from './components/SearchInterface';

function App() {
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  const [activeTab, setActiveTab] = useState('upload'); // 'upload' or 'search'

  const handleUploadSuccess = () => {
    // Trigger a refresh of the document list
    setRefreshTrigger(prev => prev + 1);
  };

  return (
    <div className="min-h-screen bg-gray-100">
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold text-gray-900">AI Vectorizer</h1>
        </div>
      </header>
      
      <main>
        <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
          {/* Tabs */}
          <div className="border-b border-gray-200 mb-6">
            <nav className="-mb-px flex space-x-8">
              <button
                className={`${
                  activeTab === 'upload'
                    ? 'border-indigo-500 text-indigo-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`}
                onClick={() => setActiveTab('upload')}
              >
                Upload & Manage
              </button>
              <button
                className={`${
                  activeTab === 'search'
                    ? 'border-indigo-500 text-indigo-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`}
                onClick={() => setActiveTab('search')}
              >
                Search
              </button>
            </nav>
          </div>
          
          {/* Content */}
          <div className="bg-white shadow overflow-hidden sm:rounded-lg p-6">
            {activeTab === 'upload' ? (
              <div>
                <FileUpload onUploadSuccess={handleUploadSuccess} />
                <h2 className="text-lg font-medium text-gray-900 mb-4">Your Documents</h2>
                <DocumentList refreshTrigger={refreshTrigger} />
              </div>
            ) : (
              <SearchInterface />
            )}
          </div>
        </div>
      </main>
      
      <footer className="bg-white">
        <div className="max-w-7xl mx-auto py-4 px-4 sm:px-6 lg:px-8">
          <p className="text-center text-sm text-gray-500">
            AI Vectorizer - Document Management and Semantic Search
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
