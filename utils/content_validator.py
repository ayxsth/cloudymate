"""
Content validation utilities for AWS-specific document filtering.
"""
from typing import Tuple
from services.bedrock.models import get_bedrock_llm


AWS_VALIDATION_PROMPT = """You are a content classifier. Your task is to determine if a document is related to Amazon Web Services (AWS).

Analyze the following text excerpt and determine if it discusses AWS topics such as:
- AWS services (EC2, S3, Lambda, RDS, CloudFront, etc.)
- Cloud computing concepts in AWS context
- AWS architecture, best practices, or configurations
- AWS pricing, billing, or cost management
- AWS security, IAM, or compliance
- AWS development tools, SDKs, or APIs
- AWS certifications or training materials
- Any other AWS-related technical content

TEXT EXCERPT:
{text_sample}

INSTRUCTIONS:
1. Respond with ONLY "YES" if the document is clearly about AWS topics
2. Respond with ONLY "NO" if the document is not about AWS
3. Be strict - the document should have substantial AWS-related content

Your response (YES or NO):"""


def is_aws_related_content(text: str, sample_length: int = 3000) -> Tuple[bool, str]:
    """
    Check if document content is AWS-related using LLM classification.

    Args:
        text: Full text content to validate
        sample_length: Number of characters to sample for analysis (default: 3000)

    Returns:
        Tuple of (is_valid, reason):
            - is_valid: True if content is AWS-related, False otherwise
            - reason: Explanation of the decision
    """
    try:
        # Sample the text (beginning and middle portions)
        text_length = len(text)

        if text_length <= sample_length:
            text_sample = text
        else:
            # Take samples from beginning and middle
            beginning = text[:sample_length // 2]
            middle_start = text_length // 2 - sample_length // 4
            middle_end = middle_start + sample_length // 2
            middle = text[middle_start:middle_end]
            text_sample = beginning + "\n...\n" + middle

        # Quick keyword check first (fast filter)
        aws_keywords = [
            'aws', 'amazon web services', 'ec2', 's3', 'lambda', 'cloudformation',
            'cloudfront', 'rds', 'dynamodb', 'ecs', 'eks', 'fargate', 'elasticache',
            'route53', 'vpc', 'iam', 'cloudwatch', 'sns', 'sqs', 'kinesis',
            'redshift', 'aurora', 'bedrock', 'sagemaker', 'elastic beanstalk'
        ]

        text_lower = text_sample.lower()
        keyword_matches = sum(1 for keyword in aws_keywords if keyword in text_lower)

        # If no AWS keywords found, reject immediately
        if keyword_matches == 0:
            return False, "Document does not contain any AWS-related keywords or terminology."

        # If many keywords found, likely AWS-related
        if keyword_matches >= 5:
            return True, f"Document contains substantial AWS content ({keyword_matches} AWS-related terms found)."

        # For borderline cases (1-4 keywords), use LLM for deeper analysis
        llm = get_bedrock_llm(use_guardrails=False)  # Don't use guardrails for validation

        prompt = AWS_VALIDATION_PROMPT.format(text_sample=text_sample[:2000])
        response = llm.invoke(prompt)

        answer = response.content.strip().upper() if hasattr(response, 'content') else str(response).strip().upper()

        if 'YES' in answer:
            return True, "LLM analysis confirms document is AWS-related."
        else:
            return False, "LLM analysis determined document is not sufficiently AWS-focused."

    except Exception as e:
        # On error, be permissive (allow the upload)
        print(f"Warning: Content validation failed: {str(e)}")
        return True, f"Validation check failed, document allowed by default: {str(e)}"


def validate_aws_content(text: str, strict: bool = True) -> Tuple[bool, str]:
    """
    Validate if text content is AWS-related.

    Args:
        text: Document text to validate
        strict: If True, requires substantial AWS content. If False, more lenient.

    Returns:
        Tuple of (is_valid, message)
    """
    if not text or len(text.strip()) < 100:
        return False, "Document is too short or empty to validate."

    is_valid, reason = is_aws_related_content(text)

    if not is_valid:
        message = (
            "⚠️ This document does not appear to be AWS-related. "
            "CloudyMate only accepts AWS documentation and technical content. "
            f"Reason: {reason}"
        )
        return False, message

    return True, f"✓ Document validated as AWS-related. {reason}"
