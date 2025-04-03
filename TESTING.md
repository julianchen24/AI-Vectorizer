# AI Vectorizer Testing Documentation

This document outlines the testing procedures for the AI Vectorizer application, including integration testing, end-to-end testing, and manual testing procedures.

## Integration Overview

The AI Vectorizer application consists of the following integrated components:

1. **Backend (Flask/FastAPI)**
   - Document Management (upload, list, delete)
   - Document Processing (extraction, chunking)
   - Search (BM25 and semantic search)
   - Vector Visualization
   - Document Insights

2. **Frontend (React)**
   - File Upload component
   - Document List component
   - Search Interface component
   - Vector Visualization component
   - Insights component

All backend endpoints are exposed through the API and consumed by the frontend through the API service.

## Automated Tests

### Backend Tests

1. **Unit Tests**
   - `tests/test_document_processing.py`: Tests for document processing functionality
   - `tests/test_visualization.py`: Tests for visualization functionality
   - `tests/test_insights.py`: Tests for insights functionality
   - `tests/test_main.py`: Tests for main application functionality

2. **Integration Tests**
   - `tests/test_integration.py`: Tests the integration between different components of the backend

To run backend tests:
```bash
pytest
```

### Frontend Tests

1. **Component Tests**
   - `src/components/FileUpload.test.js`: Tests for the FileUpload component
   - `src/components/SearchInterface.test.js`: Tests for the SearchInterface component
   - `src/components/VectorVisualization.test.js`: Tests for the VectorVisualization component

2. **End-to-End Tests**
   - `src/App.e2e.test.js`: Tests the complete user journey through the application

To run frontend tests:
```bash
cd ai-vectorizer-frontend
npm test
```

## Manual Testing Procedures

### Setup

1. Start the backend server:
```bash
uvicorn app.app:app --reload
```

2. Start the frontend development server:
```bash
cd ai-vectorizer-frontend
npm start
```

3. Open a web browser and navigate to `http://localhost:3000`

### Test Case 1: Document Upload and Management

1. **Upload a PDF Document**
   - Click on the "Upload & Manage" tab
   - Click "Choose a file" and select a PDF document
   - Click "Upload File"
   - Verify that the document appears in the document list
   - Verify that the document metadata is displayed correctly

2. **Upload Text Content**
   - Enter text in the "Text Content" field
   - Enter a title in the "Title" field
   - Click "Upload Text"
   - Verify that the document appears in the document list
   - Verify that the document metadata is displayed correctly

3. **Delete a Document**
   - Click the "Delete" button for a document in the list
   - Verify that the document is removed from the list

### Test Case 2: Search Functionality

1. **BM25 Search**
   - Click on the "Search" tab
   - Enter a search query related to the content of an uploaded document
   - Select "BM25" as the search type
   - Click "Search"
   - Verify that relevant documents are displayed in the results
   - Verify that the search results include document metadata and previews

2. **Semantic Search**
   - Enter a search query related to the content of an uploaded document
   - Select "Semantic" as the search type
   - Click "Search"
   - Verify that relevant documents are displayed in the results
   - Verify that the search results include document metadata and previews

### Test Case 3: Vector Visualization

1. **View Vector Visualization**
   - Click on the "Visualization" tab
   - Verify that the visualization is displayed
   - Verify that document points are shown in the visualization

2. **Change Visualization Parameters**
   - Change the dimensionality reduction method (t-SNE, UMAP, PCA)
   - Adjust the parameters (perplexity, n_neighbors, min_dist)
   - Click "Apply Parameters"
   - Verify that the visualization updates accordingly

3. **Interact with Visualization**
   - Hover over points to see document details
   - Click on a point to select it
   - Verify that document details are displayed for the selected point
   - Use zoom and pan functionality to navigate the visualization

### Test Case 4: Document Insights

1. **View Document Clusters**
   - Click on the "Insights" tab
   - Verify that document clusters are displayed
   - Verify that cluster information is shown

2. **Change Clustering Parameters**
   - Change the clustering method (K-Means, DBSCAN, Hierarchical)
   - Adjust the parameters (number of clusters)
   - Click "Apply Parameters"
   - Verify that the clustering updates accordingly

3. **View Similar Document Pairs**
   - Click on the "Similar Document Pairs" tab within Insights
   - Verify that similar document pairs are displayed
   - Verify that similarity scores are shown

### Error Handling Tests

1. **Upload Invalid File**
   - Try to upload a file with an unsupported format
   - Verify that an appropriate error message is displayed

2. **Search with Empty Corpus**
   - Delete all documents
   - Perform a search
   - Verify that an appropriate error message is displayed

3. **Invalid API Requests**
   - Manually test API endpoints with invalid parameters
   - Verify that appropriate error responses are returned

## Troubleshooting

### Common Issues

1. **Backend Server Not Running**
   - Error: "Failed to fetch" or "Network Error" in the frontend
   - Solution: Ensure the backend server is running on port 8000

2. **Missing Dependencies**
   - Error: Import errors or module not found
   - Solution: Install required dependencies using `pip install -r requirements.txt` for backend and `npm install` for frontend

3. **File Permission Issues**
   - Error: Permission denied when uploading or processing files
   - Solution: Ensure the uploads directory has appropriate write permissions

### Debugging Tips

1. Check the browser console for frontend errors
2. Check the backend server logs for API errors
3. Use the Network tab in browser developer tools to inspect API requests and responses
4. For visualization issues, check that the document corpus is not empty

## Conclusion

The AI Vectorizer application has been thoroughly tested through a combination of automated tests and manual testing procedures. All components are properly integrated and working together as expected.

The application successfully demonstrates the following capabilities:
- Document upload and management
- Text extraction and processing
- BM25 and semantic search
- Vector visualization with different dimensionality reduction methods
- Document clustering and insights generation

Future enhancements could include:
- More advanced document processing capabilities
- Additional visualization options
- Improved clustering algorithms
- User authentication and document sharing
