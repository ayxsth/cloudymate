"""
Streamlit frontend for CloudyMate - AWS Learning Assistant.
"""
import streamlit as st
import requests
from typing import Dict, Any


# API configuration
API_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="CloudyMate",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False


def upload_pdf(file) -> Dict[str, Any]:
    """
    Upload PDF file to backend API.

    Args:
        file: Uploaded file object from Streamlit

    Returns:
        Response dictionary from API
    """
    try:
        files = {"file": (file.name, file, "application/pdf")}
        response = requests.post(f"{API_URL}/upload_pdf", files=files, timeout=600)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to upload PDF: {str(e)}")
        return None


def ask_question(query: str, k: int = 4) -> Dict[str, Any]:
    """
    Send query to backend API for RAG processing.

    Args:
        query: User's question
        k: Number of documents to retrieve

    Returns:
        Response dictionary with answer and sources
    """
    try:
        payload = {"query": query, "k": k}
        response = requests.post(f"{API_URL}/ask", json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to process query: {str(e)}")
        return None


# Main title
st.title("🌤️ CloudyMate (AWS Learning Assistant)")
st.markdown("Ask questions about your uploaded documents using AWS Bedrock AI")

# Sidebar for PDF upload
with st.sidebar:
    st.header("📄 Document Upload")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload a PDF document to process and query"
    )

    if uploaded_file is not None:
        st.info(f"**Selected:** {uploaded_file.name}")

        if st.button("🚀 Upload & Ingest", type="primary", use_container_width=True):
            with st.spinner("Processing PDF..."):
                result = upload_pdf(uploaded_file)

                if result:
                    st.success("✅ PDF successfully processed!")
                    st.session_state.pdf_uploaded = True
                    st.write(f"**Chunks created:** {result['num_chunks']}")
                    st.write(f"**Filename:** {result['filename']}")
                    st.balloons()

    st.markdown("---")
    st.markdown("### 📊 Status")
    if st.session_state.pdf_uploaded:
        st.success("✓ Documents loaded")
    else:
        st.warning("⚠ No documents uploaded yet")

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    CloudyMate uses:
    - **AWS Bedrock** for AI
    - **ChromaDB** for vectors
    - **RAG** for context-aware answers
    """)

    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main chat interface
st.markdown("---")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show sources if available
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 View Sources"):
                for source in message["sources"]:
                    st.markdown(f"**Source {source['id']}:**")
                    st.text(source["content"])
                    if source.get("metadata"):
                        st.caption(f"Metadata: {source['metadata']}")
                    st.markdown("---")

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Check if documents are uploaded
    if not st.session_state.pdf_uploaded:
        st.warning("⚠️ Please upload a PDF document first!")
    else:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = ask_question(prompt)

                if response:
                    st.markdown(response["answer"])

                    # Store assistant message with sources
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "sources": response.get("sources", [])
                    })

                    # Show sources in expander
                    if response.get("sources"):
                        with st.expander("📚 View Sources"):
                            for source in response["sources"]:
                                st.markdown(f"**Source {source['id']}:**")
                                st.text(source["content"])
                                if source.get("metadata"):
                                    st.caption(f"Metadata: {source['metadata']}")
                                st.markdown("---")
                else:
                    error_msg = "Sorry, I encountered an error processing your question. Please try again."
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Powered by AWS Bedrock | Built with Streamlit"
    "</div>",
    unsafe_allow_html=True
)
