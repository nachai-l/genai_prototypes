"""
Gemini API text evaluation utilities.

This module provides functions for evaluating text quality using Google's Gemini API
with proper error handling, logging, and validation.
"""

import re
import logging
from typing import Optional, List, Pattern
from dataclasses import dataclass

import genai_functions.gemini_usage_logging as gemini_log_functs

# Constants
DEFAULT_EVALUATION_MAX_TOKENS = 2048
DEFAULT_EVALUATION_TEMPERATURE = 0.1
MIN_SCORE = 0.0
MAX_SCORE = 1.0
FAILURE_SCORE = -1.0

DEFAULT_SYSTEM_PROMPT = """
Please evaluate the quality of the given answer to the question on a scale of 0.0 to 1.0, where:
- 0.0 = Completely incorrect, irrelevant, or nonsensical
- 0.5 = Partially correct but missing key information or has some errors
- 1.0 = Excellent, accurate, and comprehensive answer

Consider accuracy, completeness, clarity, and relevance. Respond with just the numerical score (e.g., 0.75) followed by a brief explanation.
If Reference answer is provided, answer according to its information only.
"""

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for text evaluation parameters."""
    max_tokens: int = DEFAULT_EVALUATION_MAX_TOKENS
    temperature: float = DEFAULT_EVALUATION_TEMPERATURE
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if not 1 <= self.max_tokens <= 32768:
            raise ValueError("max_tokens must be between 1 and 32768")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        if not isinstance(self.system_prompt, str) or not self.system_prompt.strip():
            raise ValueError("system_prompt must be a non-empty string")


class GeminiTextEvaluator:
    """
    A production-ready wrapper for Gemini API text evaluation.
    
    This class provides methods to evaluate text quality using Google's Gemini API
    with comprehensive error handling, logging, and input validation.
    """
    
    # Compiled regex patterns for better performance
    _SCORE_PATTERNS: List[Pattern] = [
        re.compile(r'\b(0\.\d+|1\.0+|0\.0+)\b'),  # Decimal format (0.75, 1.0, 0.0)
        re.compile(r'\b([0-9](?:\.[0-9]+)?)/10\b'),  # X/10 format
        re.compile(r'\b(\d+)%\b'),  # Percentage format
    ]
    
    def __init__(self, client, default_config: Optional[EvaluationConfig] = None):
        """
        Initialize the Gemini text evaluator.
        
        Args:
            client: The Gemini API client instance
            default_config: Default configuration for text evaluation
            
        Raises:
            ValueError: If client is None or invalid
        """
        if client is None:
            raise ValueError("Client cannot be None")
        
        self._client = client
        self._default_config = default_config or EvaluationConfig()
        logger.info("GeminiTextEvaluator initialized")
    
    def extract_score_from_text(self, evaluation_text: str) -> float:
        """
        Extract numerical score from evaluation response text.
        
        This method uses multiple regex patterns to find scores in various formats:
        - Decimal format: 0.75, 1.0, 0.0
        - Fraction format: 8/10, 7.5/10
        - Percentage format: 75%, 80%
        
        Args:
            evaluation_text: The evaluation response text containing a score
            
        Returns:
            float: Extracted score between 0.0 and 1.0, or 0.0 if extraction fails
            
        Raises:
            TypeError: If evaluation_text is not a string
        """
        if not isinstance(evaluation_text, str):
            raise TypeError("evaluation_text must be a string")
        
        if not evaluation_text.strip():
            logger.warning("Empty evaluation text provided")
            return 0.0
        
        logger.debug("Extracting score from text: %s", evaluation_text[:100])
        
        for i, pattern in enumerate(self._SCORE_PATTERNS):
            matches = pattern.findall(evaluation_text)
            if matches:
                try:
                    score_str = matches[0]
                    logger.debug("Found match with pattern %d: %s", i, score_str)
                    
                    if '/' in score_str:
                        # Handle X/10 format
                        numerator = float(score_str.split('/')[0])
                        score = numerator / 10.0
                    elif '%' in score_str:
                        # Handle percentage format
                        score = float(score_str.replace('%', '')) / 100.0
                    else:
                        # Handle decimal format
                        score = float(score_str)
                    
                    # Ensure score is in valid range
                    if MIN_SCORE <= score <= MAX_SCORE:
                        logger.debug("Successfully extracted score: %.3f", score)
                        return score
                    else:
                        logger.warning("Score %f is outside valid range [%.1f, %.1f]", 
                                     score, MIN_SCORE, MAX_SCORE)
                        
                except (ValueError, IndexError) as e:
                    logger.debug("Failed to parse score '%s': %s", score_str, str(e))
                    continue
        
        logger.warning("Could not extract score from evaluation text: %s", 
                      evaluation_text[:200])
        return 0.0
    
    def _construct_evaluation_prompt(self, 
                                   question: str, 
                                   answer: str, 
                                   reference_answer: Optional[str],
                                   system_prompt: str) -> str:
        """
        Construct the evaluation prompt from components.
        
        Args:
            question: The original question
            answer: The answer to evaluate
            reference_answer: Optional reference answer for comparison
            system_prompt: The system prompt with evaluation instructions
            
        Returns:
            str: The constructed evaluation prompt
        """
        prompt_parts = [
            system_prompt,
            f"\nQuestion: {question}",
            f"\nAnswer to evaluate: {answer}"
        ]
        
        if reference_answer and reference_answer.strip():
            prompt_parts.append(f"\nReference answer: {reference_answer}")
        
        return "".join(prompt_parts)
    
    def evaluate_answer_quality(self,
                              question: str,
                              answer: str,
                              reference_answer: Optional[str] = None,
                              config: Optional[EvaluationConfig] = None,
                              verbose: bool = False) -> float:
        """
        Evaluate the quality of an answer using Gemini API.
        
        This method sends a structured prompt to Gemini to evaluate how well
        an answer addresses a given question, optionally using a reference answer
        for comparison.
        
        Args:
            question: The original question that was asked
            answer: The answer to be evaluated
            reference_answer: Optional reference answer for comparison
            config: Evaluation configuration (uses default if None)
            verbose: Whether to log detailed evaluation information
            
        Returns:
            float: Evaluation score between 0.0 and 1.0, or -1.0 if evaluation fails
            
        Raises:
            ValueError: If question or answer is empty
            TypeError: If question or answer is not a string
        """
        # Input validation
        if not isinstance(question, str) or not isinstance(answer, str):
            raise TypeError("question and answer must be strings")
        
        if not question.strip() or not answer.strip():
            raise ValueError("question and answer cannot be empty or whitespace only")
        
        # Use provided config or default
        eval_config = config or self._default_config
        
        logger.info("Evaluating answer quality for question length: %d, answer length: %d",
                   len(question), len(answer))
        
        try:
            # Import types here to avoid import issues if not available
            try:
                from google.genai import types
            except ImportError as import_err:
                logger.error("Failed to import google.genai.types: %s", str(import_err))
                return FAILURE_SCORE
            
            # Construct evaluation prompt
            evaluation_prompt = self._construct_evaluation_prompt(
                question=question,
                answer=answer,
                reference_answer=reference_answer,
                system_prompt=eval_config.system_prompt
            )
            
            if verbose:
                logger.info("Evaluation prompt: %s", evaluation_prompt[:300] + "...\n")
            
            # Make API request
            response = self._client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[{
                    "role": "user", 
                    "parts": [{"text": evaluation_prompt}]
                }],
                config=types.GenerateContentConfig(
                    max_output_tokens=eval_config.max_tokens,
                    temperature=eval_config.temperature,
                )
            )

            # Logging usage to csv
            gemini_logger = gemini_log_functs.GeminiUsageLogger(log_path="logs/gemini_usage.csv")  # auto-load if exists
            gemini_logger.add_log_entry(evaluation_prompt, response=response, uploaded_file=None)  # auto-saves every call 
            
            # Extract text from response (assuming you have this method from the text generation module)
            evaluation_response = self._extract_text_from_response(response)
            
            if not evaluation_response:
                logger.error("Failed to get evaluation response from API")
                return FAILURE_SCORE
            
            if verbose:
                logger.info("Evaluation response: %s", evaluation_response)
            
            # Extract numerical score from response
            score = self.extract_score_from_text(evaluation_response)
            
            # Ensure score is in valid range (extra safety check)
            final_score = max(MIN_SCORE, min(MAX_SCORE, score))
            
            if verbose:
                logger.info("Final evaluation score: %.3f", final_score)
            
            return final_score
            
        except Exception as e:
            logger.error("Error during answer evaluation: %s", str(e), exc_info=True)
            return FAILURE_SCORE
    
    def _extract_text_from_response(self, response) -> str:
        """
        Extract text content from Gemini API response.
        
        This is a simplified version of the text extraction logic.
        In production, you might want to import this from your text generation module.
        
        Args:
            response: The response object from Gemini API
            
        Returns:
            str: Extracted text content
        """
        if response is None:
            return ""
        
        extracted_texts = []
        
        try:
            candidates = getattr(response, "candidates", []) or []
            
            for candidate in candidates:
                content = getattr(candidate, "content", None)
                if not content:
                    continue
                    
                parts = getattr(content, "parts", []) or []
                for part in parts:
                    text = getattr(part, "text", None)
                    if text and isinstance(text, str):
                        extracted_texts.append(text.strip())
            
            result = "\n".join(extracted_texts)
            if not result:
                fallback_text = getattr(response, "text", "")
                result = fallback_text if isinstance(fallback_text, str) else ""
            
            return result
            
        except Exception as e:
            logger.error("Error extracting text from response: %s", str(e))
            fallback_text = getattr(response, "text", "")
            return fallback_text if isinstance(fallback_text, str) else ""
