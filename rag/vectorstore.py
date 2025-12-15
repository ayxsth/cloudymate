"""
Vector store management using ChromaDB for document storage and retrieval.
"""
import os
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from services.bedrock.models import get_bedrock_embeddings
from config import config


# Global variable to hold the vector store instance
_vector_store = None


def get_chroma():
    """
    Get or create a persistent ChromaDB vector store instance.

    Returns:
        Chroma: LangChain Chroma vector store instance

    Raises:
        Exception: If ChromaDB initialization fails
    """
    global _vector_store

    if _vector_store is not None:
        return _vector_store

    try:
        # Ensure the ChromaDB directory exists
        os.makedirs(config.CHROMA_DB_DIR, exist_ok=True)

        # Get embeddings from Bedrock
        print(f"Initializing Bedrock embeddings with model: {config.EMBEDDING_MODEL_ID}")
        print(f"Region: {config.AWS_REGION}")
        embeddings = get_bedrock_embeddings()
        print("Bedrock embeddings initialized successfully")

        # Create persistent ChromaDB client
        print(f"Creating ChromaDB client at: {config.CHROMA_DB_DIR}")
        client = chromadb.PersistentClient(
            path=config.CHROMA_DB_DIR,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Initialize LangChain Chroma vector store
        _vector_store = Chroma(
            client=client,
            collection_name="cloudymate_docs",
            embedding_function=embeddings,
            persist_directory=config.CHROMA_DB_DIR
        )
        print("ChromaDB vector store initialized successfully")

        return _vector_store

    except Exception as e:
        import traceback
        print(f"ERROR in get_chroma: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise Exception(f"Failed to initialize ChromaDB: {str(e)}")


def add_documents_to_chroma(docs: List[Dict[str, Any]]) -> List[str]:
    """
    Add documents to the ChromaDB vector store.

    Args:
        docs: List of document dictionaries with 'text' and 'metadata' keys

    Returns:
        List of document IDs

    Raises:
        Exception: If document addition fails
    """
    try:
        print(f"Starting to add {len(docs)} documents to ChromaDB...")
        vector_store = get_chroma()

        # Convert dict documents to LangChain Document objects
        documents = [
            Document(
                page_content=doc["text"],
                metadata=doc.get("metadata", {})
            )
            for doc in docs
        ]
        print(f"Converted {len(documents)} documents to LangChain format")

        # Add documents to vector store in batches to avoid timeout
        batch_size = 10
        all_ids = []

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            print(f"Adding batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1} ({len(batch)} documents)...")
            batch_ids = vector_store.add_documents(batch)
            all_ids.extend(batch_ids)

        print(f"Successfully added all {len(all_ids)} documents to ChromaDB")
        return all_ids

    except Exception as e:
        raise Exception(f"Failed to add documents to ChromaDB: {str(e)}")


def similarity_search(query: str, k: int = 4) -> List[Document]:
    """
    Perform similarity search in the vector store.

    Args:
        query: Search query string
        k: Number of documents to retrieve

    Returns:
        List of most similar Document objects

    Raises:
        Exception: If similarity search fails
    """
    try:
        vector_store = get_chroma()

        # Perform similarity search
        results = vector_store.similarity_search(query, k=k)

        return results

    except Exception as e:
        raise Exception(f"Failed to perform similarity search: {str(e)}")
