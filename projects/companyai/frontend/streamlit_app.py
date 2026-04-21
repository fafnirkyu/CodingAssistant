"""
Streamlit-based frontend for the Enterprise Document Q&A Assistant.
This provides a simple web interface to interact with the RAG system.
"""

import streamlit as st
import requests
import os

# Configuration
BACKEND_URL = "http://localhost:8000"  # Adjust if running on different host/port

def main():
    """Main Streamlit application function."""
    
    # Page configuration
    st.set_page_config(
        page_title="Enterprise Document Q&A Assistant",
        page_icon="🔍",
        layout="centered"
    )
    
    # Title and description
    st.title("🏢 Enterprise Document Q&A Assistant")
    st.markdown("""
    Ask questions about your company's documents (Invoices, Purchase Orders, Shipping Orders, etc.)
    
    This system uses Retrieval-Augmented Generation to find relevant information from your document repository.
    """)
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display existing messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available (only for assistant responses)
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📄 Sources"):
                    for source in message["sources"]:
                        st.write(f"- **{source['filename']}** ({source['category']})")
    
    # Input area
    prompt = st.chat_input("Ask a question about your company documents...")
    
    if prompt:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Show assistant response placeholder while processing
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                # Send query to backend API
                response = requests.post(
                    f"{BACKEND_URL}/query",
                    json={"question": prompt},
                    timeout=30  # Timeout after 30 seconds
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Update the assistant's message with full response and sources
                    full_response = result["answer"]
                    message_placeholder.markdown(full_response)
                    
                    # Add assistant message to chat history including sources
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": full_response,
                        "sources": result.get("sources", [])
                    })
                else:
                    error_msg = f"Error: {response.status_code} - {response.text}"
                    message_placeholder.markdown(f"❌ **Error**: {error_msg}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Error processing query: {error_msg}"
                    })
                    
            except requests.exceptions.RequestException as e:
                error_msg = f"Network error connecting to backend API: {str(e)}"
                message_placeholder.markdown(f"❌ **Error**: {error_msg}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Error processing query: {error_msg}"
                })
            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                message_placeholder.markdown(f"❌ **Error**: {error_msg}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Error processing query: {error_msg}"
                })

if __name__ == "__main__":
    main()