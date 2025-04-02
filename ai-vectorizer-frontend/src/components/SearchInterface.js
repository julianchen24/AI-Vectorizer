import React, { useState } from 'react';
import apiService from '../services/api';

const SearchInterface = () => {
  const [query, setQuery] = useState('');
  const [searchType, setSearchType] = useState('bm25');
  const [resultCount, setResultCount] = useState(5);
  const [threshold, setThreshold] = useState(0.5);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [hasSearched, setHasSearched] = useState(false);

  const handleSearch = async (e) => {
    e.preventDefault();
    
    if (!query.trim()) {
      setError('Please enter a search query');
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      const searchResults = await apiService.searchDocuments(
        query,
        searchType,
        resultCount,
        threshold
      );
      
      setResults(searchResults);
      setHasSearched(true);
    } catch (err) {
      console.error('Search error:', err);
      setError(err.response?.data?.detail || 'Error performing search');
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mb-8">
      <div className="bg-white shadow-md rounded-lg overflow-hidden">
        <div className="p-6">
          <h2 className="text-lg font-medium text-gray-900 mb-4">Search Documents</h2>
          
          {error && (
            <div className="mb-4 bg-red-50 border-l-4 border-red-500 p-4 text-red-700">
              <p>{error}</p>
            </div>
          )}
          
          <form onSubmit={handleSearch}>
            <div className="mb-4">
              <label htmlFor="query" className="block text-sm font-medium text-gray-700 mb-1">
                Search Query
              </label>
              <div className="mt-1 flex rounded-md shadow-sm">
                <input
                  type="text"
                  name="query"
                  id="query"
                  className="focus:ring-indigo-500 focus:border-indigo-500 flex-1 block w-full rounded-md sm:text-sm border-gray-300 px-3 py-2 border"
                  placeholder="Enter your search query"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  required
                />
                <button
                  type="submit"
                  className="ml-3 inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                  disabled={loading}
                >
                  {loading ? 'Searching...' : 'Search'}
                </button>
              </div>
            </div>
            
            <div className="grid grid-cols-1 gap-y-6 gap-x-4 sm:grid-cols-3">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Search Type
                </label>
                <div className="flex items-center space-x-4">
                  <div className="flex items-center">
                    <input
                      id="bm25"
                      name="searchType"
                      type="radio"
                      className="focus:ring-indigo-500 h-4 w-4 text-indigo-600 border-gray-300"
                      checked={searchType === 'bm25'}
                      onChange={() => setSearchType('bm25')}
                    />
                    <label htmlFor="bm25" className="ml-2 block text-sm text-gray-700">
                      BM25 (Keyword)
                    </label>
                  </div>
                  <div className="flex items-center">
                    <input
                      id="semantic"
                      name="searchType"
                      type="radio"
                      className="focus:ring-indigo-500 h-4 w-4 text-indigo-600 border-gray-300"
                      checked={searchType === 'semantic'}
                      onChange={() => setSearchType('semantic')}
                    />
                    <label htmlFor="semantic" className="ml-2 block text-sm text-gray-700">
                      Semantic
                    </label>
                  </div>
                </div>
              </div>
              
              <div>
                <label htmlFor="resultCount" className="block text-sm font-medium text-gray-700 mb-1">
                  Result Count
                </label>
                <select
                  id="resultCount"
                  name="resultCount"
                  className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
                  value={resultCount}
                  onChange={(e) => setResultCount(Number(e.target.value))}
                >
                  <option value="3">3 results</option>
                  <option value="5">5 results</option>
                  <option value="10">10 results</option>
                  <option value="20">20 results</option>
                  <option value="50">50 results</option>
                </select>
              </div>
              
              <div>
                <label htmlFor="threshold" className="block text-sm font-medium text-gray-700 mb-1">
                  Similarity Threshold: {threshold.toFixed(2)}
                </label>
                <input
                  type="range"
                  id="threshold"
                  name="threshold"
                  min="0.5"
                  max="0.95"
                  step="0.05"
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                  value={threshold}
                  onChange={(e) => setThreshold(Number(e.target.value))}
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>More Results</span>
                  <span>Higher Relevance</span>
                </div>
              </div>
            </div>
          </form>
        </div>
      </div>
      
      {/* Search Results */}
      {loading ? (
        <div className="mt-6 animate-pulse">
          <div className="h-4 bg-gray-200 rounded w-1/4 mb-4"></div>
          {[...Array(3)].map((_, i) => (
            <div key={i} className="h-24 bg-gray-100 rounded mb-4 p-4">
              <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
              <div className="h-4 bg-gray-200 rounded w-1/2"></div>
            </div>
          ))}
        </div>
      ) : hasSearched && (
        <div className="mt-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">
            Search Results {results.length > 0 && `(${results.length})`}
          </h3>
          
          {results.length === 0 ? (
            <div className="bg-white shadow overflow-hidden sm:rounded-md p-6 text-center">
              <p className="text-gray-500">No results found for your query.</p>
              <p className="text-gray-500 text-sm mt-1">
                Try adjusting your search terms or lowering the similarity threshold.
              </p>
            </div>
          ) : (
            <div className="bg-white shadow overflow-hidden sm:rounded-md">
              <ul className="divide-y divide-gray-200">
                {results.map((result, index) => (
                  <li key={index} className="px-6 py-4">
                    <div className="flex flex-col">
                      <div className="flex items-center justify-between">
                        <h4 className="text-sm font-medium text-gray-900">
                          {result.title || `Document ${index + 1}`}
                        </h4>
                        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                          Score: {result.score.toFixed(2)}
                        </span>
                      </div>
                      
                      <div className="mt-2 text-sm text-gray-700">
                        <div 
                          className="prose prose-sm max-w-none"
                          dangerouslySetInnerHTML={{ 
                            __html: result.preview.replace(/\*\*(.*?)\*\*/g, '<mark class="bg-yellow-200">$1</mark>') 
                          }}
                        />
                      </div>
                      
                      {result.metadata && (
                        <div className="mt-2 flex items-center text-xs text-gray-500">
                          <span className="truncate">
                            {result.metadata.file_type && (
                              <span className="mr-2 uppercase">{result.metadata.file_type}</span>
                            )}
                            {result.metadata.upload_timestamp && (
                              <span className="mr-2">
                                {new Date(result.metadata.upload_timestamp).toLocaleDateString()}
                              </span>
                            )}
                            {result.metadata.word_count && (
                              <span className="mr-2">{result.metadata.word_count} words</span>
                            )}
                          </span>
                        </div>
                      )}
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default SearchInterface;
