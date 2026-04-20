"""
Main FastAPI application for the Enterprise Document Q&A Assistant.
This file sets up the API endpoints and integrates with the RAG pipeline.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

# Import our custom modules
from rag_pipeline import initialize_rag_pipeline, query_documents

app = FastAPI(title="Enterprise Document Q&A Assistant", version="1.0.0")

# Initialize the RAG pipeline at startup
rag_pipeline = None

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG pipeline when the application starts."""
    global rag_pipeline
    try:
        rag_pipeline = initialize_rag_pipeline()
        print("RAG pipeline initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize RAG pipeline: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during initialization")

class QueryRequest(BaseModel):
    """Pydantic model for query request payload."""
    question: str

class QueryResponse(BaseModel):
    """Pydantic model for query response payload."""
    answer: str
    sources: list

@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    """
    Handle user queries by retrieving relevant documents and generating answers.
    
    Args:
        request (QueryRequest): The incoming query from the client
        
    Returns:
        QueryResponse: The generated answer along with source citations
    """
    if not rag_pipeline:
        raise HTTPException(status_code=500, detail="RAG pipeline not initialized")
        
    try:
        # Process the user's question through our RAG system
        result = query_documents(rag_pipeline, request.question)
        return QueryResponse(answer=result["answer"], sources=result["sources"])
    except Exception as e:
        print(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail="Failed to process query")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)