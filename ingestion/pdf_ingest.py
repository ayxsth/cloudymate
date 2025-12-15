"""
PDF ingestion module for extracting and chunking text from PDF documents.
"""
import re
from typing import List, Dict
import pdfplumber


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file using pdfplumber.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Extracted text as a single string

    Raises:
        FileNotFoundError: If PDF file doesn't exist
        Exception: If PDF extraction fails
    """
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        # Clean up the extracted text
        text = _cleanup_text(text)
        return text

    except FileNotFoundError:
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    except Exception as e:
        raise Exception(f"Failed to extract text from PDF: {str(e)}")


def _cleanup_text(text: str) -> str:
    """
    Clean up extracted text by removing extra whitespace and formatting issues.

    Args:
        text: Raw extracted text

    Returns:
        Cleaned text
    """
    # Remove multiple consecutive blank lines
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

    # Remove excessive spaces
    text = re.sub(r' +', ' ', text)

    # Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    # Remove any remaining leading/trailing whitespace
    text = text.strip()

    return text


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks for better context preservation.

    Args:
        text: Input text to chunk
        chunk_size: Maximum size of each chunk in characters
        overlap: Number of characters to overlap between chunks

    Returns:
        List of text chunks
    """
    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # If this is not the last chunk, try to break at a sentence boundary
        if end < text_length:
            # Look for sentence endings within the last 100 characters of the chunk
            chunk_text = text[start:end]
            last_period = chunk_text.rfind('.')
            last_newline = chunk_text.rfind('\n')
            last_question = chunk_text.rfind('?')
            last_exclamation = chunk_text.rfind('!')

            # Find the best breaking point
            break_point = max(last_period, last_newline, last_question, last_exclamation)

            # Only use the break point if it's in the last portion of the chunk
            if break_point > chunk_size * 0.7:
                end = start + break_point + 1

        # Extract chunk
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position, accounting for overlap
        start = end - overlap if end < text_length else text_length

    return chunks


def prepare_documents(chunks: List[str]) -> List[Dict[str, str]]:
    """
    Prepare document chunks for vector store ingestion.

    Args:
        chunks: List of text chunks

    Returns:
        List of document dictionaries with text and metadata
    """
    documents = []

    for idx, chunk in enumerate(chunks):
        doc = {
            "text": chunk,
            "metadata": {
                "chunk_id": idx,
                "chunk_size": len(chunk)
            }
        }
        documents.append(doc)

    return documents
