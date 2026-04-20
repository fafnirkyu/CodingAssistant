# Frontend Implementation

This directory contains the Streamlit-based user interface for interacting with the RAG system.

## Key Components:

### 1. `streamlit_app.py`
- Main application file that creates a chat-like UI using Streamlit
- Communicates with the FastAPI backend via HTTP requests
- Displays conversation history and source citations

### 2. How It Works:
- Users type questions in the chat input area
- When submitted, the question is sent to `/query` endpoint of the backend API  
- Backend processes the query using RAG pipeline and returns answer + sources
- Frontend displays both the answer and a collapsible section showing source documents

## Features:

1. **Chat Interface**: Natural conversation flow with message history
2. **Source Citations**: Shows which documents were used to generate answers
3. **Error Handling**: Graceful handling of network issues or backend errors  
4. **Responsive Design**: Works well on different screen sizes

## Usage:
Run the Streamlit app after starting the FastAPI backend: