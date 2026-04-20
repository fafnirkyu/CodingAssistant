# Backend Implementation

This directory contains the FastAPI application and RAG pipeline implementation.

## Key Components:

### 1. `main.py`
- Main FastAPI application entry point
- Defines API endpoints for querying documents
- Initializes the RAG pipeline on startup

### 2. `rag_pipeline.py` 
- Core logic for document processing, embedding generation, and query handling
- Handles loading PDFs from data directory using unstructured.io (via PyPDFLoader)
- Implements chunking with metadata preservation
- Uses FAISS for vector storage and similarity search
- Integrates HuggingFace sentence transformers for embeddings

### 3. `requirements.txt`
- Lists all Python dependencies needed for the backend service

## How It Works:

1. **Document Loading**: PDFs are loaded recursively from the data directory using LangChain's DirectoryLoader
2. **Metadata Extraction**: File names and paths are parsed to extract document type, title, and date information  
3. **Chunking**: Documents are split into smaller chunks (1000 characters) with overlap for better context retention
4. **Embedding Generation**: Uses sentence-transformers/all-MiniLM-L6-v2 model to create vector representations of text chunks
5. **Vector Storage**: FAISS index stores embeddings for fast similarity search during queries
6. **Query Processing**: When a user asks a question:
   - The query is embedded using the same model
   - Top-k most similar document chunks are retrieved from FAISS
   - These chunks form context for an LLM prompt template
   - Answer is generated and source documents are returned for citation

## API Endpoints:

- `POST /query`: Accepts a question string and returns answer with sources