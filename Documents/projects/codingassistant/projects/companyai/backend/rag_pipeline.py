"""
RAG (Retrieval-Augmented Generation) pipeline implementation.
This module handles document ingestion, embedding generation, and query processing.
"""

import os
from typing import Dict, List, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
import sqlite3

# Constants for file paths and configurations
DATA_DIR = "data"
DB_PATH = "metadata.db"

def initialize_database():
    """
    Initialize SQLite database to store document metadata.
    
    This creates a table 'documents' with columns:
    - id: unique identifier (primary key)
    - filename: name of the original file
    - category: type of document (e.g., Invoices, PurchaseOrders)
    - title: extracted from filename or content
    - date: timestamp if available in filename
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create table for storing document metadata
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            category TEXT NOT NULL,
            title TEXT,
            date TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

def extract_metadata_from_filename(filename: str) -> Dict[str, str]:
    """
    Extract metadata from file name based on naming convention.
    
    Args:
        filename (str): Name of the document file
        
    Returns:
        dict: Dictionary containing extracted metadata
    """
    # Remove extension and split by underscores
    base_name = os.path.splitext(filename)[0]
    parts = base_name.split('_')
    
    # Extract category from directory name, not filename
    # This would be handled during loading process
    
    # Try to extract date if present in format like StockReport_2016-08.pdf
    date = None
    for part in parts:
        if '-' in part and len(part) == 10:  # Format YYYY-MM-DD or similar
            date = part
            break
    
    return {
        "title": base_name,
        "date": date
    }

def load_documents():
    """
    Load documents from data directory using LangChain's DirectoryLoader.
    
    This function:
    - Loads all PDF files recursively from the data directory
    - Extracts metadata from filenames
    - Stores document information in SQLite database
    
    Returns:
        list: List of loaded Document objects
    """
    # Initialize database first
    initialize_database()
    
    # Load documents using DirectoryLoader with PyPDFLoader for PDF files
    loader = DirectoryLoader(
        DATA_DIR,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        load_dir=True  # This ensures directory structure is preserved in metadata
    )
    
    documents = loader.load()
    
    # Store document metadata in database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    for doc in documents:
        filename = os.path.basename(doc.metadata.get('source', 'unknown'))
        
        # Extract category from directory path
        source_path = doc.metadata.get('source', '')
        category = None
        
        # Find which subdirectory this file came from
        for root, dirs, files in os.walk(DATA_DIR):
            if filename in files:
                relative_path = os.path.relpath(root, DATA_DIR)
                category = relative_path  # This gives us the directory name (e.g., "Invoices")
                break
        
        metadata = extract_metadata_from_filename(filename)
        
        cursor.execute('''
            INSERT INTO documents (filename, category, title, date)
            VALUES (?, ?, ?, ?)
        ''', (
            filename,
            category or 'unknown',
            metadata['title'],
            metadata['date']
        ))
    
    conn.commit()
    conn.close()
    
    return documents

def create_chunks(documents):
    """
    Split loaded documents into smaller chunks for better retrieval.
    
    Args:
        documents (list): List of Document objects
        
    Returns:
        list: List of chunked Document objects
    """
    # Initialize text splitter with parameters suitable for technical documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Add metadata to each chunk for better context
    for i, chunk in enumerate(chunks):
        chunk.metadata['chunk_id'] = f"chunk_{i}"
        
    return chunks

def create_vector_store(chunks):
    """
    Create FAISS vector store from document chunks.
    
    Args:
        chunks (list): List of Document objects to embed and store
        
    Returns:
        FAISS: Vector store instance for similarity search
    """
    # Initialize embeddings model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Create vector store from chunks and embeddings
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    return vector_store

def setup_prompt_template():
    """
    Set up the prompt template for LLM queries.
    
    The template instructs the model to answer only based on provided context,
    which is crucial for preventing hallucinations in RAG systems.
    
    Returns:
        PromptTemplate: Formatted prompt template
    """
    # Template that guides the LLM to use only given context
    template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
Context:
{context}

Question: {question}
Helpful Answer:"""
    
    return PromptTemplate(template=template, input_variables=["context", "question"])

def initialize_rag_pipeline():
    """
    Initialize and return the complete RAG pipeline.
    
    This function orchestrates all steps of the RAG process:
    1. Load documents from data directory
    2. Split into chunks with metadata
    3. Generate embeddings and store in FAISS vector store
    4. Set up prompt template for LLM queries
    
    Returns:
        dict: Dictionary containing initialized components (vector_store, qa_chain)
    """
    print("Loading documents...")
    documents = load_documents()
    
    print(f"Loaded {len(documents)} documents")
    
    print("Creating chunks...")
    chunks = create_chunks(documents)
    print(f"Created {len(chunks)} chunks")
    
    print("Creating vector store...")
    vector_store = create_vector_store(chunks)
    print("Vector store created successfully")
    
    # Set up prompt template
    prompt_template = setup_prompt_template()
    
    # Create the QA chain with retrieval and LLM components
    qa_chain = RetrievalQA.from_chain_type(
        llm=None,  # We'll use default or specify later if needed
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template}
    )
    
    print("RAG pipeline initialized successfully")
    
    return {
        "vector_store": vector_store,
        "qa_chain": qa_chain
    }

def query_documents(pipeline, question: str):
    """
    Process a user's question through the RAG pipeline.
    
    Args:
        pipeline (dict): Initialized RAG pipeline components
        question (str): User's natural language question
        
    Returns:
        dict: Dictionary containing answer and source citations
    """
    # Use the QA chain to get results
    result = pipeline["qa_chain"]({"query": question})
    
    # Extract answer from response
    answer = result.get("result", "No answer generated")
    
    # Extract source documents for citation purposes
    sources = []
    if "source_documents" in result:
        for doc in result["source_documents"]:
            # Get metadata about the document (filename, category)
            filename = doc.metadata.get('source', 'unknown')
            title = doc.metadata.get('title', 'Unknown Title')
            
            # Extract just the filename from full path
            short_filename = os.path.basename(filename)
            
            sources.append({
                "filename": short_filename,
                "category": doc.metadata.get('category', 'unknown'),
                "title": title
            })
    
    return {
        "answer": answer,
        "sources": sources
    }