"""
Gemini API text generation utilities.

This module provides functions for generating text using Google's Gemini API
with proper error handling, logging, and validation.
"""

import logging
from typing import Optional, List, Any, Union
from dataclasses import dataclass

import genai_functions.gemini_usage_logging as gemini_log_functs

# Constants
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.1

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for text generation parameters."""
    max_tokens: int = 2024
    temperature: float = 0.7
    model: str = "gemini-2.5-flash"
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if not 1 <= self.max_tokens <= 32768:
            raise ValueError("max_tokens must be between 1 and 32768")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a non-empty string")


class GeminiTextGenerator:
    """
    A production-ready wrapper for Gemini API text generation.
    
    This class provides methods to generate text using Google's Gemini API
    with comprehensive error handling, logging, and input validation.
    """
    
    def __init__(self, client: Any, default_config: Optional[GenerationConfig] = None):
        """
        Initialize the Gemini text generator.
        
        Args:
            client: The Gemini API client instance
            default_config: Default configuration for text generation
            
        Raises:
            ValueError: If client is None or invalid
        """
        if client is None:
            raise ValueError("Client cannot be None")
        
        self.client = client
        self.default_config = default_config or GenerationConfig()
        logger.info("GeminiTextGenerator initialized with model: %s", 
                   self.default_config.model)
    
    def extract_text_from_response(self, response: Any) -> str:
        """
        Extract text content from Gemini API response.
        
        This function safely extracts text from the complex nested structure
        of Gemini API responses, handling missing attributes gracefully.
        
        Args:
            response: The response object from Gemini API
            
        Returns:
            str: Extracted text content, empty string if no text found
            
        Raises:
            TypeError: If response is None
        """
        if response is None:
            raise TypeError("Response cannot be None")
        
        extracted_texts: List[str] = []
        
        try:
            # Extract from candidates
            candidates = getattr(response, "candidates", []) or []
            
            for candidate in candidates:
                # Log finish reason for debugging (optional)
                finish_reason = getattr(candidate, "finish_reason", None)
                if finish_reason:
                    logger.debug("Candidate finish reason: %s", finish_reason)
                
                content = getattr(candidate, "content", None)
                if not content:
                    continue
                    
                parts = getattr(content, "parts", []) or []
                for part in parts:
                    text = getattr(part, "text", None)
                    if text and isinstance(text, str):
                        extracted_texts.append(text.strip())
            
            # Join all extracted texts or fallback to response.text
            result = "\n".join(extracted_texts)
            if not result:
                fallback_text = getattr(response, "text", "")
                result = fallback_text if isinstance(fallback_text, str) else ""
            
            logger.debug("Extracted %d characters of text", len(result))
            return result
            
        except Exception as e:
            logger.error("Error extracting text from response: %s", str(e))
            # Try fallback to response.text
            fallback_text = getattr(response, "text", "")
            return fallback_text if isinstance(fallback_text, str) else ""
    
    def generate_text(self, 
                     prompt: str, 
                     config: Optional[GenerationConfig] = None,
                     verbose: bool = False) -> Optional[str]:
        """
        Generate text using Gemini API with comprehensive error handling.
        
        Args:
            prompt: The input prompt for text generation
            config: Generation configuration (uses default if None)
            verbose: Whether to log the generated response
            
        Returns:
            Optional[str]: Generated text or None if generation fails
            
        Raises:
            ValueError: If prompt is empty or invalid
            TypeError: If prompt is not a string
        """
        # Input validation
        if not isinstance(prompt, str):
            raise TypeError("Prompt must be a string")
        
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty or whitespace only")
        
        # Use provided config or default
        gen_config = config or self.default_config
        
        logger.info("Generating text with model: %s, max_tokens: %d, temperature: %.2f",
                   gen_config.model, gen_config.max_tokens, gen_config.temperature)
        
        try:
            # Import types here to avoid import issues if not available
            try:
                from google.genai import types
            except ImportError as import_err:
                logger.error("Failed to import google.generativeai.types: %s", str(import_err))
                return None
            
            # Make API request
            response = self.client.models.generate_content(
                model=gen_config.model,
                contents=[{
                    "role": "user", 
                    "parts": [{"text": prompt}]
                }],
                config=types.GenerateContentConfig(
                    max_output_tokens=gen_config.max_tokens,
                    temperature=gen_config.temperature,
                )
            )
            
            # Extract text from response
            generated_text = self.extract_text_from_response(response)
            # Logging usage to csv
            gemini_logger = gemini_log_functs.GeminiUsageLogger(log_path="logs/gemini_usage.csv")  # auto-load if exists
            gemini_logger.add_log_entry(prompt, response=response, uploaded_file=None)  # auto-saves every call 
            
            if verbose and generated_text:
                logger.info("Generated response: %s", generated_text[:200] + "..." 
                           if len(generated_text) > 200 else generated_text)
            
            if not generated_text:
                logger.warning("No text generated from API response")
                return None, response
            
            logger.info("Successfully generated %d characters of text", len(generated_text))
            return generated_text, response
            
        except Exception as e:
            logger.error("Error generating text: %s", str(e), exc_info=True)
            return None, None
