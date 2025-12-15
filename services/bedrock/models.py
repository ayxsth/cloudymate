"""
AWS Bedrock model interfaces for LLM and embeddings.
"""
import boto3
from langchain_aws import ChatBedrock, BedrockEmbeddings
from config import config


def get_bedrock_llm(use_guardrails: bool = True):
    """
    Get a LangChain Bedrock LLM instance with optional Guardrails.

    Args:
        use_guardrails: Whether to apply Bedrock Guardrails (default: True)

    Returns:
        ChatBedrock: Configured Bedrock LLM client

    Raises:
        Exception: If Bedrock client initialization fails
    """
    try:
        # Create bedrock-runtime client
        client_kwargs = {
            "service_name": "bedrock-runtime",
            "region_name": config.AWS_REGION,
            "aws_access_key_id": config.AWS_ACCESS_KEY_ID,
            "aws_secret_access_key": config.AWS_SECRET_ACCESS_KEY
        }
        if config.AWS_SESSION_TOKEN:
            client_kwargs["aws_session_token"] = config.AWS_SESSION_TOKEN
        bedrock_runtime = boto3.client(**client_kwargs)

        # Prepare model kwargs
        model_kwargs = {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 2048
        }

        # Add guardrails configuration if enabled and configured
        guardrails_config = None
        if use_guardrails and config.BEDROCK_GUARDRAIL_ID:
            guardrails_config = {
                "guardrailIdentifier": config.BEDROCK_GUARDRAIL_ID,
                "guardrailVersion": config.BEDROCK_GUARDRAIL_VERSION,
                "trace": "enabled"  # Enable tracing for debugging
            }
            print(f"✓ Bedrock Guardrails enabled: {config.BEDROCK_GUARDRAIL_ID}")

        # Initialize LangChain ChatBedrock
        llm_kwargs = {
            "client": bedrock_runtime,
            "model_id": config.BEDROCK_MODEL_ID,
            "model_kwargs": model_kwargs
        }

        if guardrails_config:
            llm_kwargs["guardrails"] = guardrails_config

        llm = ChatBedrock(**llm_kwargs)

        return llm

    except Exception as e:
        raise Exception(f"Failed to initialize Bedrock LLM: {str(e)}")


def get_bedrock_embeddings():
    """
    Get a LangChain BedrockEmbeddings instance.

    Returns:
        BedrockEmbeddings: Configured Bedrock embeddings client

    Raises:
        Exception: If Bedrock embeddings initialization fails
    """
    try:
        # Create bedrock-runtime client
        client_kwargs = {
            "service_name": "bedrock-runtime",
            "region_name": config.AWS_REGION,
            "aws_access_key_id": config.AWS_ACCESS_KEY_ID,
            "aws_secret_access_key": config.AWS_SECRET_ACCESS_KEY
        }
        if config.AWS_SESSION_TOKEN:
            client_kwargs["aws_session_token"] = config.AWS_SESSION_TOKEN

        print(f"Creating Bedrock client with region: {config.AWS_REGION}")
        bedrock_runtime = boto3.client(**client_kwargs)

        # Initialize LangChain BedrockEmbeddings
        print(f"Initializing BedrockEmbeddings with model: {config.EMBEDDING_MODEL_ID}")
        embeddings = BedrockEmbeddings(
            client=bedrock_runtime,
            model_id=config.EMBEDDING_MODEL_ID
        )
        print("BedrockEmbeddings created successfully")

        return embeddings

    except Exception as e:
        import traceback
        print(f"ERROR in get_bedrock_embeddings: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise Exception(f"Failed to initialize Bedrock embeddings: {str(e)}")
