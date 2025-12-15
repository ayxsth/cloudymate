"""
RAG prompt templates for context-aware question answering.
"""
from langchain_core.prompts import PromptTemplate


# RAG prompt template with persona, directives, and output structure
RAG_TEMPLATE = """You are CloudyMate, an AI assistant that helps users understand AWS documentation. Your role is to answer questions based on the provided AWS documentation context.

## CONTEXT
The following context has been retrieved from relevant AWS documents:

{context}

## INSTRUCTIONS
1. Answer the user's question using the information provided in the context above
2. If the context contains relevant information, provide a helpful answer
3. If the context doesn't contain enough information, say: "I don't have enough information in the uploaded documents to fully answer this question."
4. Stay focused on the information in the context - don't add external knowledge
5. Be helpful and conversational while staying accurate to the source material

## QUESTION
{question}

## RESPONSE
Please provide a clear, helpful answer based on the context above."""


# Create LangChain PromptTemplate
rag_prompt = PromptTemplate(
    template=RAG_TEMPLATE,
    input_variables=["context", "question"]
)


def format_context(documents: list) -> str:
    """
    Format retrieved documents into a single context string.

    Args:
        documents: List of Document objects from similarity search

    Returns:
        Formatted context string
    """
    if not documents:
        return "No relevant context found."

    context_parts = []
    for idx, doc in enumerate(documents, 1):
        content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
        context_parts.append(f"[Document {idx}]\n{content}\n")

    return "\n".join(context_parts)


def get_rag_prompt(context: str, question: str) -> str:
    """
    Generate a formatted RAG prompt with context and question.

    Args:
        context: Retrieved context from vector store
        question: User's question

    Returns:
        Formatted prompt string
    """
    return rag_prompt.format(context=context, question=question)
