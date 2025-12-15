"""
FastAPI backend for CloudyMate - handles PDF uploads and RAG queries.
"""
import os
import shutil
from typing import Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ingestion.pdf_ingest import extract_text_from_pdf, chunk_text, prepare_documents
from rag.vectorstore import add_documents_to_chroma
from rag.rag_pipeline import run_rag
from utils.content_validator import validate_aws_content


# Initialize FastAPI app
app = FastAPI(
    title="CloudyMate API",
    description="RAG-based document Q&A system powered by AWS Bedrock",
    version="1.0.0"
)

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (adjust for production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Create uploads directory
UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# Request/Response models
class QueryRequest(BaseModel):
    """Request model for asking questions."""
    query: str
    k: int = 4


class QueryResponse(BaseModel):
    """Response model for RAG queries."""
    answer: str
    sources: list
    query: str
    num_sources: int


class UploadResponse(BaseModel):
    """Response model for PDF uploads."""
    filename: str
    num_chunks: int
    message: str


@app.get("/")
async def root():
    """Root endpoint - health check."""
    return {
        "service": "CloudyMate API",
        "status": "running",
        "version": "1.0.0"
    }


@app.post("/upload_pdf", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload and process a PDF file.

    Steps:
    1. Save uploaded PDF to disk
    2. Extract text from PDF
    3. Chunk the text
    4. Store chunks in ChromaDB

    Args:
        file: PDF file upload

    Returns:
        UploadResponse with processing details

    Raises:
        HTTPException: If processing fails
    """
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )

    try:
        # Save uploaded file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract text from PDF
        text = extract_text_from_pdf(file_path)

        if not text or len(text.strip()) == 0:
            raise HTTPException(
                status_code=400,
                detail="No text could be extracted from the PDF"
            )

        # Validate that content is AWS-related
        print("Validating document content...")
        is_valid, validation_message = validate_aws_content(text, strict=True)

        if not is_valid:
            # Remove the uploaded file since it's not valid
            os.remove(file_path)
            raise HTTPException(
                status_code=400,
                detail=validation_message
            )

        print(f"✓ Content validation passed: {validation_message}")

        # Chunk the text (using larger chunks for faster processing)
        chunks = chunk_text(text, chunk_size=1500, overlap=300)

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="Failed to create text chunks from PDF"
            )

        print(f"Created {len(chunks)} chunks from PDF")

        # Prepare documents with metadata
        documents = prepare_documents(chunks)

        # Add filename to metadata
        for doc in documents:
            doc["metadata"]["source"] = file.filename

        # Store in ChromaDB
        print(f"Storing {len(documents)} documents in ChromaDB...")
        add_documents_to_chroma(documents)

        return UploadResponse(
            filename=file.filename,
            num_chunks=len(chunks),
            message=f"Successfully processed and stored {len(chunks)} chunks from {file.filename}"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process PDF: {str(e)}"
        )


@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """
    Ask a question using RAG pipeline.

    Steps:
    1. Run RAG pipeline with the query
    2. Return answer with sources

    Args:
        request: QueryRequest with question and optional k parameter

    Returns:
        QueryResponse with answer and sources

    Raises:
        HTTPException: If query processing fails
    """
    if not request.query or len(request.query.strip()) == 0:
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty"
        )

    try:
        # Run RAG pipeline
        result = run_rag(query=request.query, k=request.k)

        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            query=result["query"],
            num_sources=result["num_sources"]
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process query: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "CloudyMate API"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
