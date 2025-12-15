# CloudyMate 🌤️

An AWS-focused RAG (Retrieval-Augmented Generation) application powered by AWS Bedrock, ChromaDB, and Streamlit. CloudyMate allows you to upload AWS documentation PDFs and ask questions about AWS services and best practices using advanced AI.

**🎯 AWS-Only Focus**: CloudyMate only accepts AWS-related documentation and responds exclusively to AWS-related questions.

## 🏗️ Architecture Overview

### Backend Architecture

```
┌─────────────────────────────────────────────────────┐
│                 FastAPI Backend                      │
│  (backend/api.py)                                    │
│                                                       │
│  ┌─────────────────┐      ┌──────────────────┐     │
│  │  POST /upload_pdf│──────│  PDF Ingestion   │     │
│  │                  │      │  Pipeline        │     │
│  └─────────────────┘      └──────────────────┘     │
│                                    │                 │
│  ┌─────────────────┐              ▼                 │
│  │  POST /ask      │      ┌──────────────────┐     │
│  │                  │──────│  RAG Pipeline    │     │
│  └─────────────────┘      └──────────────────┘     │
└─────────────────────────────────────────────────────┘
                   │                    │
                   ▼                    ▼
        ┌──────────────────┐  ┌─────────────────┐
        │  ChromaDB        │  │  AWS Bedrock    │
        │  (Vector Store)  │  │  (LLM + Embed)  │
        └──────────────────┘  └─────────────────┘
```

**Components:**

- **FastAPI Backend** (`backend/api.py`): RESTful API with CORS enabled
- **PDF Ingestion** (`ingestion/pdf_ingest.py`): Extract, clean, and chunk PDF text
- **RAG Pipeline** (`rag/rag_pipeline.py`): Orchestrates retrieval and generation
- **Vector Store** (`rag/vectorstore.py`): ChromaDB for persistent embeddings
- **Bedrock Services** (`services/bedrock/models.py`): LLM and embedding interfaces

### Frontend Architecture

```
┌─────────────────────────────────────────────────────┐
│              Streamlit Frontend                      │
│           (frontend/app.py)                          │
│                                                       │
│  ┌──────────────────┐      ┌──────────────────┐    │
│  │   Sidebar        │      │   Main Chat      │    │
│  │   - PDF Upload   │      │   - Messages     │    │
│  │   - Status       │      │   - Input Box    │    │
│  │   - Clear Chat   │      │   - Sources      │    │
│  └──────────────────┘      └──────────────────┘    │
│                                                       │
│           Session State Management                   │
│           - messages[]                               │
│           - pdf_uploaded                             │
└─────────────────────────────────────────────────────┘
                       │
                       ▼ (HTTP Requests)
              FastAPI Backend API
```

**Features:**

- Real-time chat interface with message history
- PDF upload with visual feedback
- Expandable source citations
- Session state persistence

### RAG Flow Diagram

```
User Query → RAG Pipeline
    │
    ├─ Step 1: Query Embedding
    │     │
    │     └─ AWS Bedrock (Titan Embed Text v2)
    │
    ├─ Step 2: Similarity Search
    │     │
    │     └─ ChromaDB retrieves top-k relevant chunks
    │
    ├─ Step 3: Context Building
    │     │
    │     └─ Format retrieved documents into context
    │
    ├─ Step 4: Prompt Construction
    │     │
    │     └─ Use RAG prompt template (prompts/rag_prompt.py)
    │           - Persona: CloudyMate AI Assistant
    │           - Directives: Context-grounded responses
    │           - Output Structure: Answer + Details + Confidence
    │
    ├─ Step 5: LLM Generation
    │     │
    │     └─ AWS Bedrock (Amazon Nova Lite)
    │
    └─ Step 6: Response Formatting
          │
          └─ Return { answer, sources, query, num_sources }
```

## 📁 Project Structure

```
/cloudymate
    /backend            - FastAPI REST API
        api.py          - API endpoints (/upload_pdf, /ask)
    /ingestion          - PDF processing pipeline
        pdf_ingest.py   - Text extraction and chunking
    /rag                - RAG implementation
        vectorstore.py  - ChromaDB integration
        rag_pipeline.py - End-to-end RAG orchestration
    /services/bedrock   - AWS Bedrock interfaces
        models.py       - LLM and embeddings clients
    /prompts            - Prompt templates
        rag_prompt.py   - RAG prompt with PDO structure
    /utils              - Utility functions
    /frontend           - Streamlit UI
        app.py          - Chat interface
    /tests              - Test suites
    /uploads            - Uploaded PDF storage (created automatically)
    /chroma_store       - ChromaDB persistent storage
    config.py           - Configuration management
    requirements.txt    - Python dependencies
    .env.example        - Environment variable template
```

## 🚀 Setup & Run Instructions

### Prerequisites

- Python 3.8+
- AWS account with Bedrock access
- AWS CLI configured or credentials ready

### 1. Install Dependencies

```bash
# Clone or navigate to the project directory
cd cloudymate

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python packages
pip install -r requirements.txt
```

### 2. Configure AWS Bedrock Access

**Enable the required Bedrock models** in your AWS account:

- `amazon.nova-lite-v1:0` (LLM for question answering)
- `amazon.titan-embed-text-v2:0` (Embeddings for vector search)

Go to AWS Bedrock Console → Model access → Request access for these models.

### 3. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your AWS credentials
nano .env  # or use your preferred editor
```

**Required `.env` variables:**

```bash
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1
# Optional: AWS_SESSION_TOKEN=your_session_token

CHROMA_DB_DIR=./chroma_store
BEDROCK_MODEL_ID=amazon.nova-lite-v1:0
EMBEDDING_MODEL_ID=amazon.titan-embed-text-v2:0

# Optional: AWS Bedrock Guardrails (Recommended for production)
# BEDROCK_GUARDRAIL_ID=your_guardrail_id
# BEDROCK_GUARDRAIL_VERSION=DRAFT
```

**⚠️ Important: Set up AWS Bedrock Guardrails** (Recommended)

For production deployments, it's highly recommended to configure AWS Bedrock Guardrails to enforce that CloudyMate only responds to questions about uploaded PDF documents. See [docs/GUARDRAILS_SETUP.md](docs/GUARDRAILS_SETUP.md) for detailed setup instructions.

### 4. Start the Backend (FastAPI)

```bash
# Start FastAPI server on port 8000
python backend/api.py

# Or use uvicorn directly:
uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at: `http://localhost:8000`

**API Documentation**: `http://localhost:8000/docs` (Swagger UI)

### 5. Start the Frontend (Streamlit)

In a new terminal:

```bash
# Start Streamlit UI
streamlit run frontend/app.py
```

The UI will open automatically at: `http://localhost:8501`

### 6. Ingest PDFs and Query

1. **Upload an AWS PDF:**

   - Use the sidebar in Streamlit
   - Click "Choose a PDF file"
   - Select your AWS documentation (e.g., AWS whitepapers, service guides, best practices)
   - Click "🚀 Upload & Ingest"
   - Wait for validation and success message

   **⚠️ Important**: Only AWS-related documentation is accepted. The system will automatically validate the content and reject non-AWS documents.

2. **Ask AWS Questions:**

   - Type your AWS-related question in the chat input
   - Press Enter
   - View AI response with source citations from your AWS documentation

   **Note**: Questions must be about AWS topics covered in your uploaded documents.

## 🔧 Developer Notes

### Dependencies Overview

**Current Dependencies** (see `requirements.txt`):

```bash
# Core LangChain packages (modern architecture)
langchain-aws>=0.2.0      # AWS Bedrock integration
langchain-core>=0.3.0     # Core abstractions
langchain-chroma>=0.1.0   # ChromaDB integration

# Vector Database
chromadb>=1.3.0           # Persistent vector storage

# AWS Services
boto3>=1.42.0             # AWS SDK

# PDF Processing
pdfplumber>=0.11.0        # Text extraction

# Configuration
python-dotenv>=1.0.0      # Environment management

# FastAPI Backend
fastapi>=0.124.0          # REST API framework
uvicorn[standard]>=0.38.0 # ASGI server
python-multipart>=0.0.20 # File upload support
pydantic>=2.0.0           # Data validation

# Streamlit Frontend
streamlit>=1.50.0         # UI framework
requests>=2.32.0          # HTTP client
```

**Note:** This optimized requirements.txt removes unused packages and uses modern LangChain architecture (no `langchain-community` needed).

### AWS Bedrock Guardrails (Recommended)

CloudyMate supports **AWS Bedrock Guardrails** to enforce that responses are strictly based on uploaded PDF content.

**Why Use Guardrails?**

- ✅ **Stronger enforcement** - Model cannot bypass guardrails (unlike prompt engineering)
- ✅ **Cost savings** - Off-topic queries blocked before LLM invocation
- ✅ **Better security** - Native AWS content filtering
- ✅ **Audit trails** - Track what's being blocked in CloudWatch

**Quick Setup:**

1. Create a guardrail in AWS Bedrock Console with:

   - **Denied topics**: General knowledge, programming, current events
   - **Contextual grounding**: Enable with threshold 0.7-0.8
   - **Relevance filtering**: Ensure responses use only provided context

2. Add to `.env`:

   ```bash
   BEDROCK_GUARDRAIL_ID=your_guardrail_id
   BEDROCK_GUARDRAIL_VERSION=DRAFT
   ```

3. Restart the application

**Detailed Guide:** See [docs/GUARDRAILS_SETUP.md](docs/GUARDRAILS_SETUP.md) for complete setup instructions.

**Alternative Approaches:**

- **Prompt engineering** (currently implemented as fallback)
- **Guardrails AI** (open-source, requires separate setup)
- **LangChain validators** (custom validation logic)

AWS Bedrock Guardrails are recommended as they provide the most robust protection with minimal code changes.

### AWS-Only Content Validation

CloudyMate enforces that only AWS-related documentation can be uploaded and queried.

**Upload Validation:**

When you upload a PDF, the system:

1. **Extracts text** from the PDF
2. **Scans for AWS keywords** (EC2, S3, Lambda, CloudFormation, etc.)
3. **Uses LLM analysis** for borderline cases to determine if content is AWS-related
4. **Rejects non-AWS documents** with a clear error message

**Accepted Documents:**

- ✅ AWS whitepapers and technical guides
- ✅ AWS service documentation
- ✅ AWS architecture best practices
- ✅ AWS certification study materials
- ✅ AWS blog posts and case studies

**Rejected Documents:**

- ❌ Non-AWS cloud provider documentation (Azure, GCP)
- ❌ General programming tutorials
- ❌ Non-technical documents
- ❌ Documents with minimal AWS content

**Query Validation:**

Questions must be:

- About AWS services or concepts
- Answerable from uploaded AWS documentation
- Not about general programming or other topics

**Implementation Details:**

The validation uses a two-tier approach:

1. **Fast keyword filter** - Checks for AWS-related terms (instant)
2. **LLM classifier** - For borderline cases, uses Claude to determine relevance

See [utils/content_validator.py](utils/content_validator.py) for implementation details.

**Customization:**

To adjust strictness, modify the `validate_aws_content()` function:

```python
# In backend/api.py
is_valid, validation_message = validate_aws_content(text, strict=False)  # More lenient
```

### Embedding Configuration

**Current Setup:**

- Uses AWS Bedrock Titan Embed Text v2
- Embedding dimension: 1024 (Titan default)
- Automatically handled by `get_bedrock_embeddings()`

**Fallback Options:**
If Bedrock embeddings are unavailable, you can switch to HuggingFace embeddings.

To implement fallback, first add `sentence-transformers` to requirements.txt:

```bash
pip install sentence-transformers>=5.0.0
```

Then modify services/bedrock/models.py:

```python
# Add fallback import
from langchain_community.embeddings import HuggingFaceEmbeddings

def get_fallback_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
```

### ChromaDB Persistence

**Storage Location:** `./chroma_store` (configurable via `CHROMA_DB_DIR`)

**Key Features:**

- Persistent storage across sessions
- Collection name: `cloudymate_docs`
- Automatic directory creation
- No manual database management needed

**Reset Vector Store:**

```bash
# Delete ChromaDB directory to reset
rm -rf ./chroma_store

# Restart application and re-ingest documents
```

**Production Considerations:**

- Consider using ChromaDB server mode for multi-user scenarios
- Implement backup strategy for `chroma_store/` directory
- Monitor storage size as documents accumulate

### AWS Bedrock TODOs

**Current Limitations & Future Enhancements:**

1. **Model Configuration:**

   - [ ] Add support for Amazon Nova Pro for higher quality
   - [ ] Implement model selection in UI (Nova Micro/Lite/Pro)
   - [ ] Add temperature/top_p controls in frontend

2. **Cost Optimization:**

   - [ ] Implement token usage tracking
   - [ ] Add caching for repeated queries
   - [ ] Consider Amazon Nova Micro for simpler queries

3. **Error Handling:**

   - [ ] Implement exponential backoff for rate limits
   - [ ] Add better error messages for quota exceeded
   - [ ] Handle model unavailability gracefully

4. **Security:**

   - [ ] Use AWS IAM roles instead of access keys (production)
   - [ ] Implement request authentication
   - [ ] Add input validation and sanitization

5. **Features:**
   - [ ] Support for streaming responses
   - [ ] Multi-modal support (images in PDFs)
   - [ ] Conversation memory across sessions
   - [ ] Export chat history

### Chunking Strategy

**Current Settings:**

- Chunk size: 800 characters
- Overlap: 200 characters
- Smart boundary detection (sentences)

**Optimization Tips:**

- Increase chunk size for technical documents (1000-1500)
- Reduce overlap for memory efficiency (100-150)
- Adjust based on document structure

### Testing

```bash
# Run tests (when implemented)
pytest tests/

# Test API directly
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is CloudyMate?", "k": 4}'
```

### Debugging

**Enable debug logging:**

```python
# In config.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Common Issues:**

- **ChromaDB errors**: Delete `chroma_store/` and restart
- **Bedrock access denied**: Check AWS credentials and model access
- **CORS errors**: Verify FastAPI CORS middleware settings
- **Empty responses**: Ensure PDFs were ingested successfully

## 📚 API Reference

### POST /upload_pdf

Upload and process a PDF document.

**Request:** `multipart/form-data`

- `file`: PDF file

**Response:**

```json
{
  "filename": "document.pdf",
  "num_chunks": 42,
  "message": "Successfully processed..."
}
```

### POST /ask

Query the RAG system.

**Request:**

```json
{
  "query": "Your question here",
  "k": 4
}
```

**Response:**

```json
{
  "answer": "AI generated answer",
  "sources": [...],
  "query": "Your question here",
  "num_sources": 4
}
```

## 🔐 Security Considerations

- Never commit `.env` file to version control
- Use IAM roles in production (not access keys)
- Implement rate limiting on API endpoints
- Add authentication for production deployments
- Sanitize user inputs to prevent injection attacks

## 📝 License

TBD

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a pull request.

## 📧 Support

For issues or questions, please open a GitHub issue or contact the maintainers.
