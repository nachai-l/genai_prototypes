"""
Gemini API usage logging utilities.

This module provides functions for logging API usage, token consumption, and responses
with proper error handling, validation, DataFrame management, and CSV persistence.
"""

import logging
import os
import pandas as pd
from datetime import datetime, timezone
from typing import Optional, Any, Dict
from dataclasses import dataclass, field
import tempfile
import shutil

import genai_functions.helper_functions as helper_functs

# Constants
LOG_COLUMNS = [
    "timestamp", "query", "uploaded_file", "response_text", "finish_reason",
    "cached_content_token_count", "candidates_token_count",
    "prompt_token_count", "thoughts_token_count", "total_token_count",
]

ISO_UTC_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"
MAX_RESPONSE_TEXT_LENGTH = 10000  # Limit stored response text to prevent memory issues

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class UsageMetrics:
    """Container for API usage metrics."""
    cached_content_token_count: Optional[int] = None
    candidates_token_count: Optional[int] = None
    prompt_token_count: Optional[int] = None
    thoughts_token_count: Optional[int] = None
    total_token_count: Optional[int] = None
    
    def __post_init__(self):
        for field_name, value in self.__dict__.items():
            if value is not None and not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer or None")
            if value is not None and value < 0:
                raise ValueError(f"{field_name} cannot be negative: {value}")

@dataclass
class LogEntry:
    """Container for a single log entry."""
    timestamp: datetime
    query: str
    uploaded_file: Optional[str] = None
    response_text: Optional[str] = None
    finish_reason: Optional[str] = None
    usage_metrics: UsageMetrics = field(default_factory=UsageMetrics)
    
    def __post_init__(self):
        if not isinstance(self.timestamp, datetime):
            raise TypeError("timestamp must be a datetime object")
        if not isinstance(self.query, str):
            raise TypeError("query must be a string")
        if self.query.strip() == "":
            raise ValueError("query cannot be empty or whitespace only")

class GeminiUsageLogger:
    """
    A production-ready logger for Gemini API usage tracking with CSV persistence.
    """
    def __init__(self, initial_df: Optional[pd.DataFrame] = None, log_path: Optional[str] = None):
        """
        Initialize the usage logger.

        Args:
            initial_df: Optional existing DataFrame to start with
            log_path: Optional path to a CSV file. If provided, will be loaded and used for persistence.
        """
        self.log_path = log_path
        df_from_disk = helper_functs._load_csv_if_exists(log_path) if log_path else None
        base_df = df_from_disk if df_from_disk is not None else initial_df
        self._logs_df = helper_functs._ensure_valid_logs_dataframe(base_df, log_cols = LOG_COLUMNS)
        logger.info(
            "GeminiUsageLogger initialized with %d existing logs (persist=%s)",
            len(self._logs_df), bool(self.log_path)
        )

    # ---------- extractors ----------
    def _extract_usage_metrics(self, response: Any) -> UsageMetrics:
        if response is None:
            return UsageMetrics()
        try:
            usage = getattr(response, "usage_metadata", None)
            if usage is None:
                return UsageMetrics()
            return UsageMetrics(
                cached_content_token_count=getattr(usage, "cached_content_token_count", None),
                candidates_token_count=getattr(usage, "candidates_token_count", None),
                prompt_token_count=getattr(usage, "prompt_token_count", None),
                thoughts_token_count=getattr(usage, "thoughts_token_count", None),
                total_token_count=getattr(usage, "total_token_count", None),
            )
        except Exception as e:
            logger.warning("Error extracting usage metrics: %s", e)
            return UsageMetrics()

    def _extract_finish_reason(self, response: Any) -> Optional[str]:
        if response is None:
            return None
        try:
            candidates = getattr(response, "candidates", None)
            if candidates and len(candidates) > 0:
                finish_reason = getattr(candidates[0], "finish_reason", None)
                if finish_reason:
                    return getattr(finish_reason, "name", str(finish_reason))
        except Exception as e:
            logger.debug("Could not extract finish_reason: %s", e)
        return None

    def _extract_response_text(self, response: Any) -> Optional[str]:
        if response is None:
            return None
        try:
            response_text = getattr(response, "text", None)
            if response_text and isinstance(response_text, str):
                if len(response_text) > MAX_RESPONSE_TEXT_LENGTH:
                    return response_text[:MAX_RESPONSE_TEXT_LENGTH] + "... [truncated]"
                return response_text
        except Exception as e:
            logger.debug("Could not extract response text: %s", e)
        return None

    # ---------- core ops ----------
    def add_log_entry(
        self,
        query_text: str,
        response: Any = None,
        uploaded_file: Optional[str] = None,
        custom_timestamp: Optional[datetime] = None,
        save: bool = True,
    ) -> None:
        """
        Add a new usage log entry to the DataFrame (and persist to CSV if configured).
        """
        if not isinstance(query_text, str):
            raise TypeError("query_text must be a string")
        if not query_text.strip():
            raise ValueError("query_text cannot be empty or whitespace only")
        if uploaded_file is not None and not isinstance(uploaded_file, str):
            raise TypeError("uploaded_file must be a string or None")
    
        timestamp = custom_timestamp or datetime.now(timezone.utc)
    
        usage_metrics = self._extract_usage_metrics(response)
        finish_reason = self._extract_finish_reason(response)
        response_text = self._extract_response_text(response)
    
        log_entry = LogEntry(
            timestamp=timestamp,
            query=query_text,
            uploaded_file=uploaded_file,
            response_text=response_text,
            finish_reason=finish_reason,
            usage_metrics=usage_metrics,
        )
    
        new_row = {
            "timestamp": log_entry.timestamp,
            "query": log_entry.query,
            "uploaded_file": log_entry.uploaded_file,
            "response_text": log_entry.response_text,
            "finish_reason": log_entry.finish_reason,
            "cached_content_token_count": log_entry.usage_metrics.cached_content_token_count,
            "candidates_token_count": log_entry.usage_metrics.candidates_token_count,
            "prompt_token_count": log_entry.usage_metrics.prompt_token_count,
            "thoughts_token_count": log_entry.usage_metrics.thoughts_token_count,
            "total_token_count": log_entry.usage_metrics.total_token_count,
        }
    
        # 🔑 Normalize: replace None → pd.NA
        new_row = {col: (val if val is not None else pd.NA) for col, val in new_row.items()}
    
        # Append to DataFrame
        self._logs_df.loc[len(self._logs_df)] = [new_row.get(col, pd.NA) for col in LOG_COLUMNS]
    
        logger.info(
            "Added log entry: query_len=%d, total_tokens=%s",
            len(query_text), usage_metrics.total_token_count
        )
    
        if save and self.log_path:
            try:
                helper_functs._write_csv_atomic(self._logs_df, self.log_path)
            except Exception as e:
                logger.error("Failed to persist logs to %s: %s", self.log_path, e)

    def get_logs_dataframe(self) -> pd.DataFrame:
        return self._logs_df.copy()

    def get_usage_summary(self) -> Dict[str, Any]:
        df = self._logs_df
        if df.empty:
            return {
                "total_requests": 0,
                "total_tokens": 0,
                "avg_tokens_per_request": 0,
                "date_range": None,
                "finish_reasons": {},
            }
        # Coerce tokens to numeric for safety
        toks = pd.to_numeric(df["total_token_count"], errors="coerce")
        total_requests = len(df)
        total_tokens = int(toks.fillna(0).sum())
        avg_tokens = float(toks.fillna(0).mean()) if total_requests else 0.0

        ts = df["timestamp"]
        # If timestamp already datetime, ok; if strings, attempt parse for range only
        if ts.dtype == "O" and isinstance(ts.dropna().iloc[0], str) if not ts.dropna().empty else False:
            ts = helper_functs._parse_timestamp(ts)
        date_range = (ts.min(), ts.max()) if not ts.isna().all() else None

        return {
            "total_requests": int(total_requests),
            "total_tokens": total_tokens,
            "avg_tokens_per_request": avg_tokens,
            "date_range": date_range,
            "finish_reasons": df["finish_reason"].value_counts(dropna=True).to_dict(),
        }

    def save_to_csv(self, filepath: Optional[str] = None) -> None:
        target = filepath or self.log_path
        if not target or not isinstance(target, str) or not target.strip():
            raise ValueError("filepath must be a non-empty string (or logger must have log_path set)")
        helper_functs._write_csv_atomic(self._logs_df, target)

    @classmethod
    def load_from_csv(cls, filepath: str) -> 'GeminiUsageLogger':
        if not isinstance(filepath, str) or not filepath.strip():
            raise ValueError("filepath must be a non-empty string")
        df = helper_functs._load_csv_if_exists(filepath)
        if df is None:
            raise IOError(f"Failed to load logs from {filepath}")
        return cls(initial_df=df, log_path=filepath)
