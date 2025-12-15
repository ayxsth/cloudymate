"""
RAG pipeline for end-to-end question answering with document retrieval.
"""
from typing import Dict, List, Any
from langchain_core.documents import Document
from rag.vectorstore import get_chroma, similarity_search
from services.bedrock.models import get_bedrock_llm
from prompts.rag_prompt import format_context, get_rag_prompt


def run_rag(query: str, k: int = 4) -> Dict[str, Any]:
    """
    Execute the complete RAG pipeline for a given query.

    Steps:
    1. Embed query (handled internally by Chroma)
    2. Retrieve relevant chunks from Chroma
    3. Build prompt using RAG template
    4. Call Bedrock LLM with strict AWS-only instructions
    5. Return structured response with answer and sources

    Args:
        query: User's question or query
        k: Number of documents to retrieve (default: 4)

    Returns:
        Dictionary containing:
            - answer: LLM's response
            - sources: List of source documents used
            - query: Original query
            - num_sources: Number of sources retrieved

    Raises:
        Exception: If any step in the pipeline fails
    """
    try:
        # Step 1 & 2: Embed query and retrieve relevant chunks from Chroma
        retrieved_docs = similarity_search(query, k=k)

        if not retrieved_docs:
            return {
                "answer": "No documents have been uploaded yet. Please upload a PDF document first, then ask questions about its content.",
                "sources": [],
                "query": query,
                "num_sources": 0
            }

        # Step 3: Build prompt using RAG template
        # Note: We rely on the strict prompt instructions and guardrails (if configured)
        # to enforce AWS-only responses. The vector search ensures we retrieve relevant docs.
        context = format_context(retrieved_docs)
        prompt = get_rag_prompt(context=context, question=query)

        # Step 5: Call Bedrock LLM
        llm = get_bedrock_llm()
        response = llm.invoke(prompt)

        # Extract the text content from the response
        answer = response.content if hasattr(response, 'content') else str(response)

        # Step 6: Prepare sources information
        sources = _extract_sources(retrieved_docs)

        # Return structured response
        return {
            "answer": answer,
            "sources": sources,
            "query": query,
            "num_sources": len(retrieved_docs)
        }

    except Exception as e:
        raise Exception(f"RAG pipeline failed: {str(e)}")


def _extract_sources(documents: List[Document]) -> List[Dict[str, Any]]:
    """
    Extract source information from retrieved documents.

    Args:
        documents: List of Document objects from vector store

    Returns:
        List of dictionaries containing source information
    """
    sources = []

    for idx, doc in enumerate(documents, 1):
        source = {
            "id": idx,
            "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            "metadata": doc.metadata if hasattr(doc, 'metadata') else {}
        }
        sources.append(source)

    return sources


def run_rag_with_history(query: str, conversation_history: List[Dict[str, str]] = None, k: int = 4) -> Dict[str, Any]:
    """
    Execute RAG pipeline with conversation history for context-aware responses.

    Args:
        query: User's question or query
        conversation_history: List of previous exchanges [{"role": "user/assistant", "content": "..."}]
        k: Number of documents to retrieve

    Returns:
        Dictionary containing answer, sources, and updated conversation history
    """
    if conversation_history is None:
        conversation_history = []

    # Run standard RAG pipeline
    result = run_rag(query, k=k)

    # Append to conversation history
    conversation_history.append({"role": "user", "content": query})
    conversation_history.append({"role": "assistant", "content": result["answer"]})

    # Add conversation history to result
    result["conversation_history"] = conversation_history

    return result
