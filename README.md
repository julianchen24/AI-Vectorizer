# AI Vectorizer Application

## Overview
The AI Vectorizer is a full-stack application that combines a FastAPI backend with a React frontend to provide document management, semantic search, vector visualization, and document insights capabilities. The application allows users to upload documents, process them into vector representations, search for similar content, visualize document relationships, and gain insights through clustering and similarity analysis.

![image](https://github.com/user-attachments/assets/4c6d2f98-641b-4905-a795-99195e96af7f)

![image](https://github.com/user-attachments/assets/3dc92d43-fde6-4552-9e61-3551070b1e16)

![image](https://github.com/user-attachments/assets/17826272-5af1-4f5e-a4a2-e3642921c505)

![image](https://github.com/user-attachments/assets/cf99361b-ff37-4ea7-9ea5-436b1cfd1747)


## Features
- **Document Management:** Upload, list, and delete documents (PDF, DOCX, TXT, etc.)
- **Document Processing:** Extract text, chunk documents, and generate vector embeddings
- **Search Capabilities:** 
  - **BM25 Search:** Traditional keyword-based search using BM25Okapi algorithm
  - **Semantic Search:** Meaning-based search using sentence transformers
- **Vector Visualization:** Visualize document embeddings in 2D space using various dimensionality reduction techniques (t-SNE, UMAP, PCA)
- **Document Insights:** 
  - **Clustering:** Group similar documents using various clustering algorithms (K-Means, DBSCAN, Hierarchical)
  - **Similar Pairs:** Identify and visualize pairs of similar documents
- **Asynchronous Processing:** Efficiently handles multiple requests using Async IO
- **Comprehensive Testing:** Includes unit tests, integration tests, and end-to-end tests

## Architecture

### Backend (FastAPI)
- **Document Management Module:** Handles file uploads, metadata storage, and document retrieval
- **Document Processing Module:** Extracts text, chunks documents, and processes content
- **Semantic Search Module:** Implements BM25 and semantic search using sentence transformers
- **Visualization Module:** Generates 2D visualizations of document embeddings
- **Insights Module:** Provides document clustering and similarity analysis

### Frontend (React)
- **File Upload Component:** Allows users to upload documents and text content
- **Document List Component:** Displays and manages uploaded documents
- **Search Interface Component:** Provides search functionality with result display
- **Vector Visualization Component:** Interactive visualization of document embeddings
- **Insights Component:** Displays document clusters and similar document pairs

## Installation

### Prerequisites
- Python 3.8+
- Node.js 14+
- npm 6+
- Virtual environment (recommended)

### Backend Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/julianchen24/AI-Vectorizer.git
   cd AI-Vectorizer
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install backend dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Frontend Setup
1. Navigate to the frontend directory:
   ```bash
   cd ai-vectorizer-frontend
   ```
2. Install frontend dependencies:
   ```bash
   npm install
   ```

## Usage

### Running the Application

1. Start the backend server:
   ```bash
   uvicorn app.app:app --reload --host 0.0.0.0 --port 8000
   ```

2. Start the frontend development server:
   ```bash
   cd ai-vectorizer-frontend
   npm start
   ```

3. Open your browser and navigate to `http://localhost:3000`

### Using the Application

1. **Upload Documents:**
   - Navigate to the "Upload & Manage" tab
   - Upload PDF, DOCX, or TXT files, or enter text content directly
   - View and manage your uploaded documents

2. **Search Documents:**
   - Navigate to the "Search" tab
   - Enter a search query
   - Choose between BM25 and semantic search
   - View search results with relevance scores

3. **Visualize Document Vectors:**
   - Navigate to the "Visualization" tab
   - View document embeddings in 2D space
   - Change visualization parameters (method, perplexity, etc.)
   - Interact with the visualization (zoom, pan, select)

4. **Explore Document Insights:**
   - Navigate to the "Insights" tab
   - View document clusters and their keywords
   - Change clustering parameters
   - Explore similar document pairs

### API Endpoints

The backend provides a comprehensive API for document management, search, visualization, and insights. Key endpoints include:

#### Document Management
- `POST /upload/`: Upload a document or text content
- `GET /documents/`: List all documents
- `GET /documents/{doc_id}`: Get document metadata
- `DELETE /documents/{doc_id}`: Delete a document

#### Search
- `POST /search/`: Search for documents using BM25 or semantic search
- `POST /find-similar/`: Legacy endpoint for BM25 search

#### Visualization
- `GET /visualization-data/`: Get visualization data for document embeddings

#### Insights
- `GET /insights/`: Get insights from document corpus
- `GET /insights/similar-pairs/`: Get similar document pairs

## Testing

The application includes comprehensive testing at multiple levels:

### Backend Tests

1. Run backend tests:
   ```bash
   pytest
   ```

These tests include:
- Unit tests for individual components
- Integration tests for component interactions
- API tests using FastAPI's TestClient

### Frontend Tests

1. Run frontend tests:
   ```bash
   cd ai-vectorizer-frontend
   npm test
   ```

These tests include:
- Component tests using React Testing Library
- End-to-end tests simulating user journeys

For detailed testing procedures, see [TESTING.md](TESTING.md).

## Performance Considerations

- **Asynchronous Processing:** The backend uses asyncio and semaphores to manage concurrent requests efficiently
- **Chunking Strategy:** Documents are chunked to optimize vector representation and search relevance
- **Caching:** Search indices are updated asynchronously to improve response times
- **Lazy Loading:** The frontend implements lazy loading for improved performance

## Future Enhancements

- Advanced document processing capabilities
- Additional visualization options
- Improved clustering algorithms
- User authentication and document sharing
- Deployment options (Docker, cloud services)

## Contributions

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.
