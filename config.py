"""
Configuration settings for CloudyMate application.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Application configuration."""

    # AWS Configuration
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")
    AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

    # Bedrock Configuration
    BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "amazon.nova-lite-v1:0")
    EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")

    # ChromaDB Configuration
    CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_store")

    # Bedrock Guardrails Configuration (optional)
    BEDROCK_GUARDRAIL_ID = os.getenv("BEDROCK_GUARDRAIL_ID")  # e.g., "abc123xyz"
    BEDROCK_GUARDRAIL_VERSION = os.getenv("BEDROCK_GUARDRAIL_VERSION", "DRAFT")  # or specific version number

    @classmethod
    def get_aws_credentials(cls):
        """Get AWS credentials as a dictionary."""
        creds = {
            "aws_access_key_id": cls.AWS_ACCESS_KEY_ID,
            "aws_secret_access_key": cls.AWS_SECRET_ACCESS_KEY,
            "region_name": cls.AWS_REGION
        }
        if cls.AWS_SESSION_TOKEN:
            creds["aws_session_token"] = cls.AWS_SESSION_TOKEN
        return creds

    @classmethod
    def validate(cls):
        """Validate required configuration."""
        required_vars = [
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_REGION"
        ]
        missing = [var for var in required_vars if not getattr(cls, var)]
        if missing:
            raise ValueError(f"Missing required configuration: {', '.join(missing)}")


# Create a singleton config instance
config = Config()
