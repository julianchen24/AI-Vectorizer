# AI Vectorizer Service

## Overview
This AI Vectorizer Service is a FastAPI-based application utilizing the BM25 algorithm for text similarity and ranking. The service allows users to add documents, query the stored corpus, and retrieve the most similar documents based on a given search query. It is designed for high concurrency, leveraging Async IO and semaphores for efficient request handling.

## Features
- **FastAPI-based API:** Provides high-performance endpoints for document management and similarity search.
- **BM25 Algorithm:** Uses BM25Okapi for ranking and retrieval of relevant documents.
- **Asynchronous Processing:** Efficiently handles multiple requests using Async IO.
- **Semaphore-based Request Control:** Limits the number of concurrent requests to prevent overload.
- **Background Task Handling:** Updates the BM25 index asynchronously when adding or resetting documents.
- **Automated Testing:** Includes test cases using FastAPI's TestClient and pytest.

## Installation

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Setup
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
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Service
Start the FastAPI application:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### API Endpoints

#### 1. Add Document
- **Endpoint:** `POST /add-doc/`
- **Description:** Adds a new document to the corpus.
- **Query Parameters:** `new_doc` (string) - The document to be added.
- **Example Request:**
  ```bash
  curl -X POST "http://localhost:8000/add-doc/?new_doc=Deep%20learning%20is%20a%20powerful%20AI%20technique."
  ```

#### 2. Reset Corpus
- **Endpoint:** `POST /reset-corpus/`
- **Description:** Clears the document corpus.
- **Query Parameters:** `delete_all` (string) - Must be set to "Y" to confirm deletion.
- **Example Request:**
  ```bash
  curl -X POST "http://localhost:8000/reset-corpus/?delete_all=Y"
  ```

#### 3. Query BM25 Vectors
- **Endpoint:** `GET /query/`
- **Description:** Retrieves BM25 vectors from the stored corpus.
- **Example Request:**
  ```bash
  curl -X GET "http://localhost:8000/query/"
  ```

#### 4. Find Similar Document
- **Endpoint:** `POST /find-similar/`
- **Description:** Finds the most similar document based on the given query.
- **Query Parameters:** `query` (string) - The search query.
- **Example Request:**
  ```bash
  curl -X POST "http://localhost:8000/find-similar/?query=Artificial%20Intelligence"
  ```

## Asynchronous Request Handling
This service uses **asyncio.Semaphore** to manage concurrent requests, ensuring that no more than a specified number of requests are processed simultaneously. This prevents resource exhaustion and improves system stability.

## Running Tests
Automated tests are provided using pytest and FastAPI's TestClient.

1. Install test dependencies:
   ```bash
   pip install pytest
   ```
2. Run the tests:
   ```bash
   pytest
   ```

## Contributions

All contributions welcome.



## License
This project is licensed under the MIT License.

