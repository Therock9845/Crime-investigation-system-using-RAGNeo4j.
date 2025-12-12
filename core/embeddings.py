"""
Text embedding utilities using OpenAI API or Hugging Face.
"""

import asyncio
import logging
import random
import time
from typing import List

import httpx
from openai import OpenAI
import requests

from config.settings import settings

logger = logging.getLogger(__name__)


def retry_with_exponential_backoff(max_retries=3, base_delay=1.0, max_delay=60.0):
    """
    Decorator for retrying API calls with exponential backoff on rate limiting errors.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Check for rate limiting error (429) or connection errors
                    if attempt == max_retries:
                        logger.error(
                            f"Max retries ({max_retries}) exceeded for {func.__name__}"
                        )
                        raise

                    # Check if this is a retryable error
                    is_retryable = False
                    if (
                        hasattr(e, "status_code")
                        and getattr(e, "status_code", None) == 429
                    ):
                        is_retryable = True
                        logger.warning(
                            f"Rate limit hit in {func.__name__}, attempt {attempt + 1}/{max_retries}"
                        )
                    elif "Too Many Requests" in str(e) or "429" in str(e):
                        is_retryable = True
                        logger.warning(
                            f"Rate limit detected in {func.__name__}, attempt {attempt + 1}/{max_retries}"
                        )
                    elif "Connection" in str(e) or "Timeout" in str(e):
                        is_retryable = True
                        logger.warning(
                            f"Connection error in {func.__name__}, attempt {attempt + 1}/{max_retries}"
                        )

                    if not is_retryable:
                        raise

                    # Calculate delay with exponential backoff and jitter
                    delay = min(base_delay * (2**attempt), max_delay)
                    jitter = random.uniform(0.1, 0.3) * delay  # Add 10-30% jitter
                    total_delay = delay + jitter

                    logger.info(f"Retrying in {total_delay:.2f} seconds...")
                    time.sleep(total_delay)

            return None  # Should never reach here

        return wrapper

    return decorator


def async_retry_with_exponential_backoff(max_retries=3, base_delay=1.0, max_delay=60.0):
    """
    Async decorator for retrying API calls with exponential backoff on rate limiting errors.
    """

    def decorator(func):
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    # Check for rate limiting error (429) or connection errors
                    if attempt == max_retries:
                        logger.error(
                            f"Max retries ({max_retries}) exceeded for {func.__name__}"
                        )
                        raise

                    # Check if this is a retryable error
                    is_retryable = False
                    if (
                        hasattr(e, "status_code")
                        and getattr(e, "status_code", None) == 429
                    ):
                        is_retryable = True
                        logger.warning(
                            f"Rate limit hit in {func.__name__}, attempt {attempt + 1}/{max_retries}"
                        )
                    elif "Too Many Requests" in str(e) or "429" in str(e):
                        is_retryable = True
                        logger.warning(
                            f"Rate limit detected in {func.__name__}, attempt {attempt + 1}/{max_retries}"
                        )
                    elif "Connection" in str(e) or "Timeout" in str(e):
                        is_retryable = True
                        logger.warning(
                            f"Connection error in {func.__name__}, attempt {attempt + 1}/{max_retries}"
                        )

                    if not is_retryable:
                        raise

                    # Calculate delay with exponential backoff and jitter
                    delay = min(base_delay * (2**attempt), max_delay)
                    jitter = random.uniform(0.1, 0.3) * delay  # Add 10-30% jitter
                    total_delay = delay + jitter

                    logger.info(f"Retrying in {total_delay:.2f} seconds...")
                    await asyncio.sleep(total_delay)

            return None  # Should never reach here

        return wrapper

    return decorator


class EmbeddingManager:
    """Manages text embeddings using OpenAI API, Hugging Face, or Ollama."""

    def __init__(self):
        """Initialize the embedding manager."""
        self.provider = getattr(settings, "llm_provider", "openai").lower()
        
        # Priority 1: Check if we have Ollama embedding model configured (LOCAL & FREE)
        ollama_embedding_model = getattr(settings, "ollama_embedding_model", None)
        if ollama_embedding_model:
            self.embedding_provider = "ollama"
            self.model = ollama_embedding_model
            self.ollama_base_url = getattr(settings, "ollama_base_url", "http://localhost:11434")
            self.client = None
            logger.info(f"Using Ollama for LOCAL embeddings with model: {self.model}")
            return
        
        # Priority 2: Check if Hugging Face API key is available (CLOUD & FREE)
        self.huggingface_api_key = getattr(settings, "huggingface_api_key", None)
        if self.huggingface_api_key:
            self.embedding_provider = "huggingface"
            self.model = getattr(settings, "embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
            # Use the direct model endpoint without any prefixes
            self.hf_api_url = f"https://api-inference.huggingface.co/models/{self.model}"
            self.client = None
            logger.info(f"Using Hugging Face for embeddings with model: {self.model}")
            return
            
        # Priority 3: Fall back to OpenAI embeddings (PAID)
        if self.provider == "openai":
            self.embedding_provider = "openai"
            self.model = getattr(settings, "embedding_model", "text-embedding-3-small")
            
            # Initialize OpenAI client with proper configuration
            client_kwargs = {
                "api_key": settings.openai_api_key,
            }
            
            # Add base_url if provided
            if settings.openai_base_url:
                client_kwargs["base_url"] = settings.openai_base_url
            
            # Add proxy if provided
            if settings.openai_proxy and settings.openai_proxy.strip():
                try:
                    client_kwargs["http_client"] = httpx.Client(
                        verify=False,
                        proxy=settings.openai_proxy
                    )
                except Exception as e:
                    logger.warning(f"Failed to configure proxy for embeddings: {e}")
            
            self.client = OpenAI(**client_kwargs)
            logger.info(f"Using OpenAI for embeddings with model: {self.model}")
            
        else:
            # Should not reach here, but default to ollama
            self.embedding_provider = "ollama"
            self.model = "nomic-embed-text"
            self.ollama_base_url = "http://localhost:11434"
            self.client = None
            logger.warning("No embedding provider configured, defaulting to Ollama")

    @retry_with_exponential_backoff(max_retries=3, base_delay=1.0, max_delay=60.0)
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text with retry logic."""
        try:
            if self.embedding_provider == "huggingface":
                return self._get_huggingface_embedding(text)
            elif self.embedding_provider == "ollama":
                return self._get_ollama_embedding(text)
            else:  # openai
                # Use the properly configured client instance
                response = self.client.embeddings.create(
                    input=text, 
                    model=self.model
                )
                
                # Extract embedding safely
                if response and hasattr(response, 'data') and len(response.data) > 0:
                    embedding = response.data[0].embedding
                    return embedding
                else:
                    logger.error("Invalid response structure from OpenAI embeddings API")
                    logger.debug(f"Response: {response}")
                    raise ValueError("Invalid embedding response structure")
                    
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    @retry_with_exponential_backoff(max_retries=5, base_delay=2.0, max_delay=60.0)
    def _get_huggingface_embedding(self, text: str) -> List[float]:
        """Generate embedding using Hugging Face Inference API (FREE!)."""
        headers = {
            "Authorization": f"Bearer {self.huggingface_api_key}",
            "Content-Type": "application/json"
        }
        
        # Truncate text if too long (HF has limits)
        max_length = 512  # Most models support up to 512 tokens
        if len(text) > max_length * 4:  # Rough estimate: 1 token ≈ 4 chars
            text = text[:max_length * 4]
        
        # Try with simple string input first
        payload = {"inputs": text}
        
        try:
            response = requests.post(
                self.hf_api_url,
                headers=headers,
                json=payload,
                timeout=30,
            )
            
            # If we get 410, the model might be archived. Try with wait_for_model
            if response.status_code == 410:
                logger.warning(f"Model {self.model} returned 410. Trying alternative payload...")
                payload = {
                    "inputs": text,
                    "options": {"wait_for_model": True, "use_cache": True}
                }
                response = requests.post(
                    self.hf_api_url,
                    headers=headers,
                    json=payload,
                    timeout=60,
                )
            
            response.raise_for_status()
            result = response.json()
            
            # Handle error responses
            if isinstance(result, dict) and "error" in result:
                error_msg = result.get("error", "Unknown error")
                # Check if model is loading
                if "loading" in error_msg.lower():
                    estimated_time = result.get("estimated_time", 20)
                    logger.info(f"Model is loading, waiting {estimated_time} seconds...")
                    time.sleep(min(estimated_time, 30))
                    # Retry the request
                    response = requests.post(
                        self.hf_api_url,
                        headers=headers,
                        json=payload,
                        timeout=60,
                    )
                    response.raise_for_status()
                    result = response.json()
                else:
                    raise ValueError(f"HuggingFace API error: {error_msg}")
            
            # Handle different response formats
            if isinstance(result, list):
                # If it's a nested list (batch format), take the first element
                if len(result) > 0 and isinstance(result[0], list):
                    # Could be [[embedding]] or [embedding]
                    if isinstance(result[0][0], list):
                        return result[0][0]  # [[[embedding]]]
                    return result[0]  # [[embedding]]
                return result  # [embedding]
            elif isinstance(result, dict) and "embeddings" in result:
                # Some models return {"embeddings": [...]}
                return result["embeddings"]
            else:
                raise ValueError(f"Unexpected Hugging Face response format: {type(result)}, content: {str(result)[:200]}")
                
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 410:
                logger.error(f"Model {self.model} is no longer available (410 Gone). Please use a different model.")
                logger.error("Try one of these models in your .env file:")
                logger.error("  - BAAI/bge-small-en-v1.5")
                logger.error("  - intfloat/e5-small-v2")
                logger.error("  - thenlper/gte-small")
            raise

    @retry_with_exponential_backoff(max_retries=3, base_delay=1.0, max_delay=60.0)
    def _get_ollama_embedding(self, text: str) -> List[float]:
        """Generate embedding using Ollama with retry logic."""
        try:
            # Truncate extremely long text to avoid timeouts
            max_chars = 8000  # Ollama can handle longer text than cloud APIs
            if len(text) > max_chars:
                logger.warning(f"Text too long ({len(text)} chars), truncating to {max_chars}")
                text = text[:max_chars]
            
            # Ollama expects 'prompt' for embedding generation
            response = requests.post(
                f"{self.ollama_base_url.rstrip('/')}/api/embeddings",
                json={
                    "model": self.model,  # Works with tags like "nomic-embed-text:latest"
                    "prompt": text
                },
                timeout=120,
            )
            response.raise_for_status()
            
            result = response.json()
            embedding = result.get("embedding", [])
            
            if not embedding:
                raise ValueError(f"Empty embedding returned from Ollama. Response: {result}")
            
            return embedding
            
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Cannot connect to Ollama at {self.ollama_base_url}. Is Ollama running?")
            logger.error(f"Start Ollama with: ollama serve")
            raise
        except requests.exceptions.Timeout as e:
            logger.error(f"Ollama embedding request timed out. Text length: {len(text)}")
            raise
        except Exception as e:
            logger.error(f"Ollama embedding failed: {e}")
            logger.error(f"Model: {self.model}, Base URL: {self.ollama_base_url}")
            raise

    async def aget_embedding(self, text: str) -> List[float]:
        """Asynchronously generate embedding for a single text using httpx.AsyncClient with retry logic."""

        # Apply async retry decorator
        @async_retry_with_exponential_backoff(
            max_retries=3, base_delay=1.0, max_delay=60.0
        )
        async def _get_embedding_with_retry():
            async with httpx.AsyncClient(
                verify=False if settings.openai_proxy else True
            ) as client:
                if self.embedding_provider == "huggingface":
                    headers = {"Authorization": f"Bearer {self.huggingface_api_key}"}
                    
                    # Truncate text if too long
                    max_length = 512
                    truncated_text = text if len(text) <= max_length * 4 else text[:max_length * 4]
                    
                    resp = await client.post(
                        self.hf_api_url,
                        headers=headers,
                        json={"inputs": truncated_text, "options": {"wait_for_model": True}},
                        timeout=120.0,
                    )
                    resp.raise_for_status()
                    result = resp.json()
                    
                    # Handle different response formats
                    if isinstance(result, list):
                        if len(result) > 0 and isinstance(result[0], list):
                            if isinstance(result[0][0], list):
                                return result[0][0]
                            return result[0]
                        return result
                    elif isinstance(result, dict) and "embeddings" in result:
                        return result["embeddings"]
                    else:
                        raise ValueError(f"Unexpected HF response format: {type(result)}")
                        
                elif self.embedding_provider == "ollama":
                    url = f"{self.ollama_base_url.rstrip('/')}/api/embeddings"
                    
                    # Truncate extremely long text
                    max_chars = 8000
                    truncated_text = text if len(text) <= max_chars else text[:max_chars]
                    
                    resp = await client.post(
                        url, 
                        json={
                            "model": self.model,  # Handles tags like "nomic-embed-text:latest"
                            "prompt": truncated_text
                        }, 
                        timeout=120.0
                    )
                    resp.raise_for_status()
                    result = resp.json()
                    embedding = result.get("embedding", [])
                    
                    if not embedding:
                        raise ValueError(f"Empty embedding from Ollama. Response: {result}")
                    
                    return embedding
                    
                else:  # openai
                    # Handle None base_url
                    base_url = settings.openai_base_url or "https://api.openai.com/v1"
                    base = base_url.rstrip("/")
                    url = f"{base}/embeddings"
                    headers = {"Authorization": f"Bearer {settings.openai_api_key}"}
                    resp = await client.post(
                        url,
                        json={"input": text, "model": self.model},
                        headers=headers,
                        timeout=120.0,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    
                    # Safely extract embedding from response
                    if "data" in data and len(data["data"]) > 0:
                        return data["data"][0].get("embedding", [])
                    else:
                        logger.error(f"Invalid async embedding response structure: {data}")
                        raise ValueError("Invalid embedding response structure")

        try:
            return await _get_embedding_with_retry()
        except Exception as e:
            logger.error(f"Failed to generate async embedding: {e}")
            raise


# Global embedding manager instance
embedding_manager = EmbeddingManager()