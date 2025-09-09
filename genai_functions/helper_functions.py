import logging
import os
import pandas as pd
from datetime import datetime, timezone
from typing import Optional, Any, Dict
import tempfile
import shutil
import logging, sys
import regex as re

ISO_UTC_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"

# Configure logging
logger = logging.getLogger(__name__)

# -------------------------
# Logger nb text enable
# -------------------------
def enable_notebook_logging(
    logger_name: str | None = None,
    level: int = logging.INFO,
    fmt: str = "%(levelname)s:%(name)s:%(message)s",
    datefmt: str = None,
    use_stderr: bool = True,
):
    """
    Configure logging so INFO/WARN/ERROR messages show directly in Jupyter cells
    with the same 'pink box' style as printouts from stderr.

    Parameters
    ----------
    logger_name : str | None
        If provided, bump this module logger to the chosen level.
    level : int
        Logging level (default: logging.INFO).
    fmt : str
        Log format string.
    datefmt : str | None
        Optional date format (default: None).
    use_stderr : bool
        If True, send logs to stderr (pink box in Jupyter). If False, send to stdout.
    """
    stream = sys.stderr if use_stderr else sys.stdout

    # Reset existing handlers to avoid duplicate logs
    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        handlers=[logging.StreamHandler(stream)],
        force=True,   # Python >=3.8
    )

    if logger_name:
        logging.getLogger(logger_name).setLevel(level)
        logging.getLogger(logger_name).propagate = True

# -------------------------
# Helpers (file + time)
# -------------------------
def _ensure_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def _to_iso_utc(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime(ISO_UTC_FMT)

def _parse_timestamp(series: pd.Series) -> pd.Series:
    # Be tolerant: try parse, coerce errors to NaT
    return pd.to_datetime(series, errors="coerce", utc=True)

def _load_csv_if_exists(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, dtype="object")  # keep flexible; we'll coerce types as needed
        # Ensure expected columns and parse timestamp
        df = _ensure_valid_logs_dataframe(df)
        if not df.empty:
            df["timestamp"] = _parse_timestamp(df["timestamp"])
        return df
    except Exception as e:
        logger.error("Error loading logs from CSV %s: %s", path, e)
        return None

def _write_csv_atomic(df: pd.DataFrame, path: str) -> None:
    _ensure_dir(path)
    # Convert timestamps to ISO strings for stable CSV
    df_to_write = df.copy()
    if "timestamp" in df_to_write.columns:
        df_to_write["timestamp"] = df_to_write["timestamp"].apply(
            lambda x: _to_iso_utc(x) if isinstance(x, datetime) else (str(x) if pd.notna(x) else "")
        )
    # Atomic write
    dirpath = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".logs_tmp_", suffix=".csv", dir=dirpath)
    os.close(fd)
    try:
        df_to_write.to_csv(tmp_path, index=False)
        shutil.move(tmp_path, path)
    finally:
        # If something fails before move, try to clean up
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

# -------------------------
# Data validation helpers
# -------------------------
def _ensure_valid_logs_dataframe(
    logs_df: Optional[pd.DataFrame],
    log_cols:list = [
        "timestamp", "query", "uploaded_file", "response_text", "finish_reason",
        "cached_content_token_count", "candidates_token_count",
        "prompt_token_count", "thoughts_token_count", "total_token_count",
    ]
) -> pd.DataFrame:
    """
    Public-style helper (also used by legacy function) that guarantees
    presence and ordering of LOG_COLUMNS.
    """
    if logs_df is None or logs_df.empty:
        logs_df = pd.DataFrame({col: pd.Series(dtype="object") for col in LOG_COLUMNS})
    else:
        if not isinstance(logs_df, pd.DataFrame):
            raise TypeError("logs_df must be a pandas DataFrame or None")
        missing = set(log_cols) - set(logs_df.columns)
        for col in missing:
            logs_df[col] = pd.Series(dtype="object")
        logs_df = logs_df[log_cols].copy()
    return logs_df

# -------------------------
# String helpers
# -------------------------
# --- 1) Ensure any literal backslash-n become real newlines and tidy blanks
def normalize_newlines(s: str) -> str:
    if not isinstance(s, str):
        return ""
    # if the text literally contains backslash + n, turn into newline
    s = s.replace("\\n", "\n")
    # collapse 3+ blanks → 2
    import re
    s = re.sub(r"\n{3,}", "\n\n", s.strip())
    return s

# --- 2) Safe printer for plain console
def print_block(text: str) -> None:
    print(normalize_newlines(text))

# --- 3) Pretty render inside Jupyter as Markdown (if available)
def display_markdown(text: str) -> None:
    try:
        from IPython.display import Markdown, display
        display(Markdown(normalize_newlines(text)))
    except Exception:
        # fallback to console printing
        print_block(text)
        
def simple_token_count(text: str) -> int:
    """Very rough token estimate (whitespace split).
    Replace with your tokenizer if needed."""
    return len(text.split())

def simple_tokenize(txt: str):
    return re.findall(r"[A-Za-z0-9\-]+", (txt or "").lower())
