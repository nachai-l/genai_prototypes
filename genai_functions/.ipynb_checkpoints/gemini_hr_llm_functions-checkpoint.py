"""
gemini_hr_llm_functions.py

Utilities for HR-style summarization with Gemini:

- CV (PDF) upload ➜ structured summary generation ➜ CSV usage logging
- JD (plain text) summarization ➜ formatted output ➜ CSV usage logging
- Consistent, atomic CSV persistence via `GeminiUsageLogger`

Features
--------
- Wraps Gemini API calls (upload + generate_content) with robust error handling
- Normalizes missing values (None → pd.NA) for consistent CSV logs
- Captures token usage, finish reasons, and response size
- Provides Markdown-friendly formatting for job descriptions
- Exposes both OOP (`CVSummarizer`) and functional wrappers for drop-in use

Dependencies
------------
- pandas
- google.genai (client + types)
- genai_functions.gemini_usage_logging.GeminiUsageLogger

Notes
-----
- Timestamps stored in UTC
- Logging is best-effort and will not interrupt generation
- Intended for HR/recruitment workflows (resume and JD parsing, matching)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from textwrap import dedent
from typing import Any, Dict, Optional, Tuple
import re
import pickle
from copy import deepcopy
import pandas as pd
import numpy as np

import google.genai as genai
from google.genai import types
import faiss
from rank_bm25 import BM25Okapi

from genai_functions.gemini_usage_logging import GeminiUsageLogger
from genai_functions.hybrid_vectordb_functions import FaissVectorStore, BM25Index, embed_texts_with_gemini
import genai_functions.helper_functions as helper_functs

logger = logging.getLogger(__name__)

# -------------------------
# Small utility
# -------------------------
def calc_resp_text_size(resp: Any) -> Dict[str, int]:
    """
    Calculate the size of resp.text in characters and rough token estimate.

    Parameters
    ----------
    resp : object
        Gemini response object (must expose `.text` for size calc)

    Returns
    -------
    dict : {"char_length": int, "token_est": int}
    """
    try:
        txt = getattr(resp, "text", None)
        if not txt:
            return {"char_length": 0, "token_est": 0}
        char_length = len(txt)
        token_est = len(txt.split())  # rough approximation
        return {"char_length": char_length, "token_est": token_est}
    except Exception as e:
        logger.debug("calc_resp_text_size failed: %s", e)
        return {"char_length": 0, "token_est": 0}


# -------------------------
# Data Contracts
# -------------------------
@dataclass(frozen=True)
class GenerateConfig:
    model: str = "gemini-2.5-flash"
    max_output_tokens: int = 12288
    temperature: float = 0.1
    top_p: float = 0.9 # At each step, the model sorts possible next tokens by probability, then keeps only the smallest set whose 
    top_k: int   = 40  # At each step, the model only considers the top_k most likely tokens.

@dataclass(frozen=True)
class GenerateResult:
    response: Any
    logs_df: pd.DataFrame
    size_info: Dict[str, int]


# -------------------------
# Core helper (OOP)
# -------------------------
class CVSummarizer:
    """
    Encapsulates CV upload + summary + usage logging.
    Dependency-inject your Gemini client and logger for testability.
    """

    def __init__(
        self,
        client: Any,
        logger_instance: GeminiUsageLogger,
        *,
        default_model: str = "gemini-2.5-flash",
    ) -> None:
        """
        Parameters
        ----------
        client : Any
            Gemini client with `files.upload()` and `models.generate_content()`.
        logger_instance : GeminiUsageLogger
            Usage logger that persists to CSV on every entry (if configured with log_path).
        default_model : str
            Default model to use if not specified in GenerateConfig.
        """
        self._client = client
        self._usage_logger = logger_instance
        self._default_model = default_model

    def summarize_cv(
        self,
        uploaded_filename: str,
        prompt_text: str,
        *,
        system_text: Optional[str] = None,
        gen_cfg: Optional[GenerateConfig] = None,
    ) -> GenerateResult:
        """
        Upload a CV PDF and request a summary from Gemini.

        Returns
        -------
        GenerateResult
            response: raw Gemini response
            logs_df : up-to-date usage log DataFrame
            size_info: {"char_length": int, "token_est": int}

        Raises
        ------
        FileNotFoundError
            If uploaded_filename does not exist or can't be read.
        RuntimeError
            If upload or generation fail unexpectedly.
        """
        if not uploaded_filename or not isinstance(uploaded_filename, str):
            raise ValueError("uploaded_filename must be a non-empty string")

        # Default system text includes a stable 'today' marker.
        if system_text is None:
            today_str = datetime.now(timezone.utc).date().isoformat()
            system_text = (
                "You are an expert HR assistant. Summarize the uploaded CV clearly. "
                f"Note that today is {today_str}."
            )

        cfg = gen_cfg or GenerateConfig(model=self._default_model)

        # ---- Step 1: Upload file ----
        try:
            uploaded_file = self._client.files.upload(
                file=uploaded_filename,
                config={"display_name": "CV"},
            )
            file_uri = getattr(uploaded_file, "uri", None)
            if not file_uri:
                raise RuntimeError("Upload succeeded but no file URI returned.")
            logger.info("Uploaded the CV file to Gemini")
        except FileNotFoundError:
            logger.exception("CV file not found: %s", uploaded_filename)
            raise
        except Exception as e:
            logger.exception("Failed to upload CV to Gemini: %s", e)
            raise RuntimeError(f"File upload failed: {e}") from e

        # ---- Step 2: Generate content ----
        try:
            resp = self._client.models.generate_content(
                model=cfg.model,
                contents=[
                    {
                        "role": "user",
                        "parts": [
                            {"text": prompt_text},
                            {"file_data": {"file_uri": file_uri}},
                        ],
                    }
                ],
                config=types.GenerateContentConfig(  # type: ignore[attr-defined]
                    system_instruction=[system_text],
                    max_output_tokens=cfg.max_output_tokens,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    top_k=cfg.top_k,
                ),
            )
            logger.info("Generation succeeded with model=%s", cfg.model)
        except Exception as e:
            logger.exception("Gemini generation failed: %s", e)
            # Still attempt to log the failed call (with resp=None)
            try:
                self._usage_logger.add_log_entry(
                    query_text=prompt_text,
                    response=None,
                    uploaded_file=uploaded_filename,
                    save=True,
                )
            except Exception as log_e:
                logger.error("Failed to log failed generation attempt: %s", log_e)
            raise RuntimeError(f"Content generation failed: {e}") from e

        # ---- Size info (safe) ----
        size_info = calc_resp_text_size(resp)
        logger.debug(
            "Response size: chars=%s tokens≈%s",
            size_info.get("char_length"), size_info.get("token_est")
        )

        # ---- Append usage log (success path) ----
        try:
            self._usage_logger.add_log_entry(
                query_text=prompt_text,
                response=resp,
                uploaded_file=uploaded_filename,
                save=True,
            )
        except Exception as e:
            # Do NOT fail the main flow due to logging issues
            logger.error("Failed to append usage log: %s", e)

        # ---- Return result ----
        try:
            logs_df = self._usage_logger.get_logs_dataframe()
        except Exception as e:
            logger.warning("Could not fetch logs DataFrame: %s", e)
            logs_df = pd.DataFrame()

        return GenerateResult(response=resp, logs_df=logs_df, size_info=size_info)

# ==========================================================================================

# -------------------------
# JD Formatting Utility
# -------------------------
def format_job_description(raw_text: str) -> str:
    """
    Clean and format a job description string into a more readable format.
    Produces Markdown-friendly text. Safe and idempotent (repeated calls are fine).

    Parameters
    ----------
    raw_text : str

    Returns
    -------
    str
    """
    if not isinstance(raw_text, str) or not raw_text:
        return ""

    import re
    from textwrap import dedent

    # Replace literal "\n" with actual line breaks (in case text came escaped)
    formatted = raw_text.replace("\\n", "\n")

    # Trim outer quotes if present
    formatted = formatted.strip().strip('"').strip("'")

    # Collapse runs of 3+ blank lines → 2
    formatted = re.sub(r"\n{3,}", "\n\n", formatted)

    # Dedent to align nested blocks
    formatted = dedent(formatted)

    # Final trim
    return formatted.strip()

# -------------------------
# CVSummarizer: JD generation method
# -------------------------
class CVSummarizer(CVSummarizer):  # type: ignore[misc]
    """
    Extends the helper with a JD summarization method (no file upload).
    """

    def summarize_jd(
        self,
        prompt_text: str,
        *,
        system_text: Optional[str] = None,
        gen_cfg: Optional[GenerateConfig] = None,
    ) -> GenerateResult:
        """
        Ask Gemini to summarize a job description (pure text prompt).

        Returns
        -------
        GenerateResult
            response: raw Gemini response
            logs_df : updated usage DataFrame
            size_info: {"char_length": int, "token_est": int}

        Notes
        -----
        - Uses the same CSV-persisted logger.
        - Does not upload files (JD comes from prompt_text).
        """
        if not isinstance(prompt_text, str) or not prompt_text.strip():
            raise ValueError("prompt_text must be a non-empty string")

        if system_text is None:
            system_text = (
                "You are an expert HR assistant. Summarize the job description into:\n"
                "- Key responsibilities\n"
                "- Mandatory experiences & skills\n"
                "- Preferred experiences & skills"
            )

        cfg = gen_cfg or GenerateConfig(model=self._default_model)

        # ---- Generate content ----
        try:
            resp = self._client.models.generate_content(
                model=cfg.model,
                contents=[
                    {
                        "role": "user",
                        "parts": [{"text": prompt_text}],
                    }
                ],
                config=types.GenerateContentConfig(  # type: ignore[attr-defined]
                    system_instruction=[system_text],
                    max_output_tokens=cfg.max_output_tokens,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    top_k=cfg.top_k,
                ),
            )
            logger.info("JD generation succeeded with model=%s", cfg.model)
        except Exception as e:
            logger.exception("Gemini JD generation failed: %s", e)
            # Attempt to log the failed call
            try:
                self._usage_logger.add_log_entry(
                    query_text=prompt_text,
                    response=None,
                    uploaded_file=None,
                    save=True,
                )
            except Exception as log_e:
                logger.error("Failed to log failed JD attempt: %s", log_e)
            raise RuntimeError(f"JD content generation failed: {e}") from e

        # ---- Size info ----
        size_info = calc_resp_text_size(resp)
        logger.debug(
            "JD response size: chars=%s tokens≈%s",
            size_info.get("char_length"), size_info.get("token_est")
        )

        # ---- Append usage log ----
        try:
            self._usage_logger.add_log_entry(
                query_text=prompt_text,
                response=resp,
                uploaded_file=None,
                save=True,
            )
        except Exception as e:
            logger.error("Failed to append JD usage log: %s", e)

        # ---- Get logs df ----
        try:
            logs_df = self._usage_logger.get_logs_dataframe()
        except Exception as e:
            logger.warning("Could not fetch logs DataFrame (JD): %s", e)
            import pandas as pd  # local import to avoid top-level dependency if unused
            logs_df = pd.DataFrame()

        return GenerateResult(response=resp, logs_df=logs_df, size_info=size_info)

# ==========================================================================================

def _normalize_checkmarks(line: str, CHECK_OK:str = "✓", CHECK_BAD:str = "✗") -> str:
    """
    Normalize various pass/miss notations to a consistent '— ✓' / '— ✗'
    Examples handled: '- ✓', ' - ✓', '— [✓/✗]', '[✓]', '[x]', '(pass)', '(missing)', etc.
    """
    s = line
    # common ' - ✓' or ' — ✓' endings
    s = re.sub(r"\s*[-–—]\s*([✓xX✗])\s*$", r" — \1", s)
    # bracketed marks like [✓] / [x] / [✗]
    s = re.sub(r"\s*\[\s*([✓xX✗])\s*\]\s*$", r" — \1", s)
    # explicit words
    s = re.sub(r"\s*\(\s*pass\s*\)\s*$", f" — {CHECK_OK}", s, flags=re.I)
    s = re.sub(r"\s*\(\s*(miss|missing|gap)\s*\)\s*$", f" — {CHECK_BAD}", s, flags=re.I)
    # normalize x/X to ✗
    s = re.sub(r" — [xX]\b", f" — {CHECK_BAD}", s)
    return s

def format_cvxjd_output(raw_text: str) -> str:
    """
    Clean & normalize CV×JD output (Markdown-friendly).
    - fixes escaped newlines, quotes, tuple/pprint artifacts
    - dedents, collapses excessive blanks
    - normalizes checkmark notations
    """
    if not isinstance(raw_text, str):
        return ""

    s = raw_text

    # Handle repr/tuple artifacts like: ('...text...\n',)
    # strip surrounding parentheses if the whole thing looks like a single-elem tuple repr
    m = re.fullmatch(r"\(\s*(['\"])(.*)\1\s*,\s*\)", s, flags=re.S)
    if m:
        s = m.group(2)

    # Replace literal "\n" with real newlines
    s = s.replace("\\n", "\n")

    # Trim outer quotes if present
    s = s.strip().strip('"').strip("'")

    # Dedent + collapse 3+ blank lines -> 2
    s = dedent(s)
    s = re.sub(r"\n{3,}", "\n\n", s)

    # Normalize bullet checkmarks at line ends
    lines = []
    for line in s.splitlines():
        # normalize different dashes
        line = re.sub(r"[–—]+", "—", line)  # use em-dash visually
        # tidy spaces around em-dash in our patterns
        line = re.sub(r"\s*—\s*", " — ", line)
        line = _normalize_checkmarks(line.rstrip())
        lines.append(line)
    s = "\n".join(lines)

    # Final whitespace tidy
    s = re.sub(r"[ \t]+\n", "\n", s)  # trailing spaces at EOL
    return s.strip()

def parse_cvxjd_output(clean_text: str) -> Dict[str, Any]:
    """
    Parse cleaned CV×JD markdown into a structured dict (best-effort).
    Keys:
      - jd_mandatory: list[str]
      - jd_preferred: list[str]
      - strengths: list[str]
      - weaknesses: list[str]
      - score: float | None
      - reasoning: list[str]
    """
    sections = {
        "jd_mandatory": r"^#+\s*JD\s+Mandatory\s+Requirements.*$",
        "jd_preferred": r"^#+\s*JD\s+Preferred\s+Requirements.*$",
        "strengths": r"^#+\s*Candidate\s+Strengths.*$",
        "weaknesses": r"^#+\s*Candidate\s+Weaknesses.*$",
        "score": r"^#+\s*JD\s+vs\s+CV\s+Matching\s+Score.*$",
        "reasoning": r"^\*\*Reasoning.*$|^Reasoning.*$",
    }

    # Build line index
    lines = clean_text.splitlines()
    idx_map: Dict[str, int] = {}
    for i, ln in enumerate(lines):
        for key, pat in sections.items():
            if re.match(pat, ln.strip(), flags=re.I):
                idx_map.setdefault(key, i)

    def _collect_until(next_idx: int) -> list:
        out = []
        for j in range(start+1, next_idx):
            t = lines[j].strip()
            if not t:
                continue
            # list item patterns
            t = re.sub(r"^\s*(?:[-*]|\d+[.)])\s+", "", t)
            out.append(t)
        return out

    result: Dict[str, Any] = {
        "jd_mandatory": [],
        "jd_preferred": [],
        "strengths": [],
        "weaknesses": [],
        "score": None,
        "reasoning": [],
    }

    # Ordered keys to slice sections
    ordered = [k for k in ["jd_mandatory","jd_preferred","strengths","weaknesses","score","reasoning"] if k in idx_map]
    for pos, key in enumerate(ordered):
        start = idx_map[key]
        end = idx_map[ordered[pos+1]] if pos+1 < len(ordered) else len(lines)

        if key == "score":
            # Search score value in the next few lines
            block = "\n".join(lines[start:end])
            m = re.search(r"Score\D+([0-9]+(?:\.[0-9]+)?)\s*/\s*10", block, flags=re.I)
            if m:
                try:
                    result["score"] = float(m.group(1))
                except ValueError:
                    result["score"] = None
            continue

        items = _collect_until(end)
        if key == "jd_mandatory":
            result["jd_mandatory"] = items
        elif key == "jd_preferred":
            result["jd_preferred"] = items
        elif key == "strengths":
            result["strengths"] = items
        elif key == "weaknesses":
            result["weaknesses"] = items
        elif key == "reasoning":
            result["reasoning"] = items

    return result

def postprocess_cvxjd_output(raw_text: str) -> Tuple[str, Dict[str, Any]]:
    """
    End-to-end: clean text and parse it.
    Returns (clean_markdown, parsed_dict).
    """
    cleaned = format_cvxjd_output(raw_text)
    parsed  = parse_cvxjd_output(cleaned)
    cleaned = helper_functs.normalize_newlines(cleaned)
    return cleaned, parsed

# ==========================================================================================

# -------------------------
# Functional API (drop-in)
# -------------------------
def master_gemini_upload_cv_prompt_w_log(
    uploaded_filename: str,
    prompt_text: str,
    *,
    logs_df: Optional[pd.DataFrame] = None,
    client: Any,
    usage_logger: Optional[GeminiUsageLogger] = None,
    log_path: Optional[str] = None,
    system_text: Optional[str] = None,
    max_output_tokens: int = 4096,
    temperature: float = 0.1,
    top_p: float = 0.9,
    top_k: int = 40,
) -> Tuple[Any, pd.DataFrame]:
    """
    Backward-compatible functional wrapper.

    Parameters
    ----------
    uploaded_filename : str
        Path to the CV PDF to upload.
    prompt_text : str
        Prompt used to summarize the CV.
    logs_df : Optional[pd.DataFrame]
        Existing logs in memory (can be None).
    client : Any
        Gemini client instance.
    usage_logger : Optional[GeminiUsageLogger]
        If provided, will be used. Otherwise a new one is created
        (preferring `log_path` for CSV persistence).
    log_path : Optional[str]
        CSV path for persistent logging. Ignored if `usage_logger` is supplied.
    system_text : Optional[str]
        System instruction; defaults to a safe HR summary instruction with UTC date.
    max_output_tokens, temperature, top_p, top_k : generation config.

    Returns
    -------
    (response, logs_df) : Tuple[Any, pd.DataFrame]
        Raw Gemini response and the updated logs DataFrame.
    """
    # Build/resolve logger with persistence:
    if usage_logger is None:
        # If log_path is supplied, auto-load from CSV if exists.
        usage_logger = GeminiUsageLogger(initial_df=logs_df, log_path=log_path)
    else:
        # Ensure initial_df is respected only when no CSV is present in usage_logger.
        pass

    summarizer = CVSummarizer(client=client, logger_instance=usage_logger)

    gen_cfg = GenerateConfig(
        model="gemini-2.5-flash",
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )

    result = summarizer.summarize_cv(
        uploaded_filename=uploaded_filename,
        prompt_text=prompt_text,
        system_text=system_text,
        gen_cfg=gen_cfg,
    )

    return result.response, result.logs_df

# -------------------------
# Functional wrapper (JD)
# -------------------------
def master_gemini_jd_prompt_w_log(
    prompt_text: str,
    *,
    logs_df: Optional["pd.DataFrame"] = None,
    client: Any,
    usage_logger: Optional["GeminiUsageLogger"] = None,
    log_path: Optional[str] = None,
    system_text: Optional[str] = None,
    max_output_tokens: int = 4096,
    temperature: float = 0.1,
    top_p: float = 0.9,
    top_k: int = 40,
):
    """
    Backward-compatible functional API for JD summarization with CSV logging.

    Parameters
    ----------
    prompt_text : str
        The JD text or prompt containing the JD.
    logs_df : Optional[pd.DataFrame]
        Existing logs (may be None).
    client : Any
        Gemini client instance (must expose .models.generate_content and .types.GenerateContentConfig).
    usage_logger : Optional[GeminiUsageLogger]
        Inject an existing logger instance; otherwise one is created (preferring `log_path`).
    log_path : Optional[str]
        CSV persistence path (used only if `usage_logger` is not provided).
    system_text, max_output_tokens, temperature, top_p, top_k : generation config.

    Returns
    -------
    (response, logs_df_formatted_text)
        response : raw Gemini response
        logs_df  : updated logs DataFrame
        formatted_text : Markdown-friendly JD summary string (best-effort)
    """
    # Lazy import to avoid hard dependency if caller only uses class API
    import pandas as pd  # type: ignore

    # Build/resolve logger (CSV-persistent)
    if usage_logger is None:
        usage_logger = GeminiUsageLogger(initial_df=logs_df, log_path=log_path)

    summarizer = CVSummarizer(client=client, logger_instance=usage_logger)

    gen_cfg = GenerateConfig(
        model="gemini-2.5-flash",
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )

    result = summarizer.summarize_jd(
        prompt_text=prompt_text,
        system_text=system_text,
        gen_cfg=gen_cfg,
    )

    # Best-effort formatted markdown text for immediate display
    raw_text = getattr(result.response, "text", "") or ""
    formatted_text = format_job_description(raw_text)

    return result.response, result.logs_df, formatted_text

# -------------------------------------
# Functional wrapper (CV × JD matching)
# -------------------------------------
def master_gemini_cvxjd_matching_w_log(
    resp_jd: Any = None,
    resp_cv: Any = None,
    *,
    jd_text: Optional[str] = None,
    cv_text: Optional[str] = None,
    logs_df: Optional["pd.DataFrame"] = None,
    client: Any,
    usage_logger: Optional["GeminiUsageLogger"] = None,
    log_path: Optional[str] = None,
    output_template: Optional[str] = None,
    system_text: Optional[str] = None,
    max_output_tokens: int = 4096,
    temperature: float = 0.1,
    top_p: float = 0.9,
    top_k: int = 40,
):
    """
    Compare a JD and a summarized CV using a fixed output template, and persist usage logs.

    Inputs can be provided as:
      - resp_jd / resp_cv (Gemini responses with `.text`) OR
      - jd_text / cv_text (plain strings)

    Returns
    -------
    (response, logs_df, formatted_output)
        response          : raw Gemini response
        logs_df           : updated logs DataFrame
        formatted_output  : model's text (filled output_template)
    """
    import pandas as pd  # local import to avoid hard dependency at import time

    # Resolve logger (CSV-persistent)
    if usage_logger is None:
        usage_logger = GeminiUsageLogger(initial_df=logs_df, log_path=log_path)

    # Resolve texts from either resp_* or *_text
    if jd_text is None:
        jd_text = getattr(resp_jd, "text", None)
    if cv_text is None:
        cv_text = getattr(resp_cv, "text", None)

    jd_text = format_job_description(jd_text or "")
    cv_text = (cv_text or "").strip()

    if not jd_text:
        raise ValueError("JD text is empty. Provide `jd_text` or a response with `.text`.")
    if not cv_text:
        raise ValueError("CV text is empty. Provide `cv_text` or a response with `.text`.")

    # Default output template
    if output_template is None:
        output_template = """
        ### JD Mandatory Requirements [✓: Pass | ✗: Missing]
        1. Requirement 1 — [✓/✗]
        2. Requirement 2 — [✓/✗]
        
        ### JD Preferred Requirements [✓: Pass | ✗: Missing]
        1. Requirement 1 — [✓/✗]
        2. Requirement 2 — [✓/✗]
        
        ### Candidate Strengths (1–5 bullets)
        1. Strength 1
        2. Strength 2
        
        ### Candidate Weaknesses (1–5 bullets)
        1. Weakness 1
        2. Weakness 2
        
        ### JD vs CV Matching Score
        **Score:** X.X / 10.0
        
        **Reasoning (3–5 bullets):**
        1. Reason 1
        2. Reason 2
        3. Reason 3
        """.strip()

    # Default system prompt (injects the template)
    if system_text is None:
        system_text = (
            "You are an expert HR assistant.\n"
            "Fill in the output_template below based on the given job description (JD) "
            "and the summarized resume (CV). Only answer in the given output_template.\n\n"
            f"# output_template:\n{output_template}"
        )

    # Build prompt_text from inputs
    prompt_text = (
        f"# job description (JD):\n{jd_text}\n\n"
        f"# summarized resume (CV):\n{cv_text}\n"
    )

    # Prepare generation config (reuse the same pattern as other wrappers)
    gen_cfg = GenerateConfig(
        model="gemini-2.5-flash",
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )

    # Perform generation
    try:
        resp = client.models.generate_content(
            model=gen_cfg.model,
            contents=[{"role": "user", "parts": [{"text": prompt_text}]}],
            config=types.GenerateContentConfig(
                system_instruction=[system_text],
                max_output_tokens=gen_cfg.max_output_tokens,
                temperature=gen_cfg.temperature,
                top_p=gen_cfg.top_p,
                top_k=gen_cfg.top_k,
            ),
        )
        logger.info("CV×JD matching generation succeeded with model=%s", gen_cfg.model)
    except Exception as e:
        logger.exception("Gemini CV×JD matching generation failed: %s", e)
        # Attempt to log the failed attempt
        try:
            usage_logger.add_log_entry(
                query_text=prompt_text,
                response=None,
                uploaded_file=None,
                save=True,
            )
        except Exception as log_e:
            logger.error("Failed to log failed CV×JD attempt: %s", log_e)
        raise RuntimeError(f"CV×JD content generation failed: {e}") from e

    # Log success
    try:
        usage_logger.add_log_entry(
            query_text=prompt_text,
            response=resp,
            uploaded_file=None,
            save=True,
        )
    except Exception as e:
        logger.error("Failed to append CV×JD usage log: %s", e)

    # Fetch latest logs
    try:
        updated_logs_df = usage_logger.get_logs_dataframe()
    except Exception as e:
        logger.warning("Could not fetch logs DataFrame (CV×JD): %s", e)
        updated_logs_df = pd.DataFrame()

    # Best-effort formatted output (the model already adheres to the template)
    formatted_output = getattr(resp, "text", "") or ""
    clean_output_md, parsed_output = postprocess_cvxjd_output(formatted_output)

    return resp, updated_logs_df, formatted_output, clean_output_md, parsed_output

# --------------------------------------------------------------------
# Single-CV worker: summarize -> embed -> index (no disk save here)
# --------------------------------------------------------------------
def process_single_cv(
    client: Any,
    uploaded_filename: str,
    applied_position: str,
    *,
    cv_prompt_text: str,
    cv_system_text: str,
    log_path: str,
    max_output_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    embedding_modelname: str,
    store: Optional[FaissVectorStore],
    bm25_index: BM25Index,
    doc_id: int,
) -> tuple[Optional[FaissVectorStore], BM25Index, dict, pd.DataFrame]:
    """
    Summarize one CV, embed its summary, and add it to FAISS + BM25 (in-memory only).

    Parameters
    ----------
    client : Any
        Gemini client
    uploaded_filename : str
        Path to the CV (PDF)
    applied_position : str
        Role applied for (stored in metadata)
    cv_prompt_text, cv_system_text : str
        Generation prompts
    log_path : str
        CSV path for persistence (used by GeminiUsageLogger inside summarize call)
    max_output_tokens, temperature, top_p, top_k : generation params
    embedding_modelname : str
        Embedding model name
    store : Optional[FaissVectorStore]
        Existing FAISS store (None = will be created on first embedding)
    bm25_index : BM25Index
        In-memory BM25 index to mutate
    doc_id : int
        Integer id to put in metadata (e.g., loop index - 1)

    Returns
    -------
    (store, bm25_index, result, logs_df)
        store         : Possibly newly-initialized FAISS store (or the same)
        bm25_index    : Mutated BM25 index
        result        : {'success': bool, 'uploaded_filename': str, 'meta': dict, 'error': Optional[str]}
        logs_df       : Latest logs dataframe from the summarize step
    """
    # 1) Summarize CV (also persists usage to CSV via master_gemini_upload_cv_prompt_w_log)
    try:
        resp_cv, logs_df = master_gemini_upload_cv_prompt_w_log(
            uploaded_filename=uploaded_filename,
            prompt_text=cv_prompt_text,
            system_text=cv_system_text,
            client=client,
            log_path=log_path,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
    except Exception as e:
        logger.exception("Summarization failed for %s: %s", uploaded_filename, e)
        return (
            store,
            bm25_index,
            {"success": False, "uploaded_filename": uploaded_filename, "meta": {}, "error": str(e)},
            pd.DataFrame(),
        )

    doc_text = (getattr(resp_cv, "text", "") or "").strip()
    if not doc_text:
        err = "Gemini returned empty text for this CV."
        logger.error("%s: %s", uploaded_filename, err)
        return (
            store,
            bm25_index,
            {"success": False, "uploaded_filename": uploaded_filename, "meta": {}, "error": err},
            logs_df,
        )

    # 2) Embed summary
    try:
        embeddings = embed_texts_with_gemini(client, [doc_text], embedding_modelname=embedding_modelname)
        if not embeddings or embeddings[0] is None:
            raise RuntimeError("Embedding failed (empty vector).")
        vector = embeddings[0]
        dim = len(vector)
    except Exception as e:
        logger.exception("Embedding failed for %s: %s", uploaded_filename, e)
        return (
            store,
            bm25_index,
            {"success": False, "uploaded_filename": uploaded_filename, "meta": {}, "error": str(e)},
            logs_df,
        )

    # 3) Ensure FAISS store exists (infer dim from first embedding)
    if store is None:
        logger.info("Initializing FAISS store with inferred dim=%d", dim)
        store = FaissVectorStore(dim=dim)

    # 4) Build metadata and index in FAISS & BM25 (in-memory)
    meta_item = {
        "id": doc_id,
        "uploaded_filename": uploaded_filename,
        "applied_position": applied_position,
        "embedding_model": embedding_modelname,
        "project": "cv-summarization",
    }

    try:
        store.add_texts_and_metadata(
            texts=[doc_text],
            embeddings=[vector],
            metadata_list=[meta_item],
            default_metadata={},
        )
        bm25_index.add_or_replace({**meta_item, "text": doc_text})
    except Exception as e:
        logger.exception("Indexing failed for %s: %s", uploaded_filename, e)
        return (
            store,
            bm25_index,
            {"success": False, "uploaded_filename": uploaded_filename, "meta": meta_item, "error": str(e)},
            logs_df,
        )

    return (
        store,
        bm25_index,
        {"success": True, "uploaded_filename": uploaded_filename, "meta": meta_item, "error": None},
        logs_df,
    )

# --------------------------------------------------------------------
# Functional wrapper to create FAISS vector db and BM25 index for CVs
# --------------------------------------------------------------------
def master_loading_cvs_to_faiss_and_bm25(
    client: Any,
    uploaded_filenames_list: list,
    applied_position: str,
    *,
    embedding_modelname = "text-embedding-004",
    vector_dbname:str   = "vector_and_bm25_dbs/vector_index.faiss",
    vector_dbmeta:str   = "vector_and_bm25_dbs/vector_metadata.jsonl",
    bm25_dbmeta:str     = "vector_and_bm25_dbs/bm25_metadata.jsonl" ,
    cv_system_text:str  =  None,
    cv_prompt_text:str  = """
    Analyze the uploaded CV and provide the following:
    
    1. Identify the individual (full name).
    2. Summarize education and work experience in **10–15 concise bullet points**, including years of experience for each role.
    3. Provide a breakdown of total experience (in years) aggregated by **position title** across the job history.
    4. Extract the list of skills and output them as a valid Python list (e.g., ["skill1", "skill2", "skill3"]).
    """,
    log_path: str         = "logs/gemini_usage.csv",
    max_output_tokens:int = 12288,
    temperature: float    = 0.1,
    top_p: float          = 0.9, 
    top_k:int             = 40,  # At each step, the model only considers the top_k most likely tokens.
    clear_initial_faiss   = False,
    clear_initial_bm25    = False,
) -> dict:
    """
    Summarize a batch of CV PDFs with Gemini and index the summaries into FAISS + BM25.
    
    This function:
    - Generates a structured summary for each CV (via `master_gemini_upload_cv_prompt_w_log`)
    - Embeds each summary and adds it to a FAISS vector index
    - Adds the same summary to a BM25 index for lexical retrieval
    - Persists both indexes once at the end (atomic save patterns handled by the stores)
    - Logs Gemini usage to CSV on every request
    
    Parameters
    ----------
    client : Any
        Initialized Gemini client.
    uploaded_filenames_list : list[str]
        Paths to CV PDF files to process.
    applied_position : str
        Role/position to attach in per-document metadata.
    embedding_modelname : str, optional
        Embedding model to use (default: "text-embedding-004").
    vector_dbname : str, optional
        Path to FAISS index file (default: "vector_and_bm25_dbs/vector_index.faiss").
    vector_dbmeta : str, optional
        Path to FAISS metadata JSONL (default: "vector_and_bm25_dbs/vector_metadata.jsonl").
    bm25_dbmeta : str, optional
        Path to BM25 metadata JSONL (default: "vector_and_bm25_dbs/bm25_metadata.jsonl").
    cv_system_text : str | None, optional
        System instruction for CV summarization. If None, a default with today’s UTC date is used.
    cv_prompt_text : str, optional
        User prompt for CV summarization (see default for required outputs).
    log_path : str, optional
        CSV path for usage logs (default: "logs/gemini_usage.csv").
    max_output_tokens : int, optional
        Max tokens for Gemini generation (default: 12288).
    temperature : float, optional
        Sampling temperature (default: 0.1).
    top_p : float, optional
        Nucleus sampling parameter (default: 0.9).
    top_k : int, optional
        Top-k sampling parameter (default: 40).

    Returns
    -------
    dict[str, Any]
        {
            "num_inputs": int,       # total CVs requested
            "num_success": int,      # successfully summarized & indexed
            "num_failed": int,       # failed during summarize/embed/index
            "failed_files": list[str],  # file paths that failed
            "faiss_path": str,       # FAISS index path
            "faiss_meta": str,       # FAISS metadata path
            "bm25_meta": str,        # BM25 metadata path
            "last_logs_df": pd.DataFrame,  # most recent usage logs snapshot
        }
    """

    if not uploaded_filenames_list:
        raise ValueError("uploaded_filenames_list must be a non-empty list of file paths.")
    if not isinstance(applied_position, str) or not applied_position.strip():
        raise ValueError("applied_position must be a non-empty string.")

    # Build default system text with UTC date if not provided
    if cv_system_text is None:
        today_str = datetime.now(timezone.utc).date().isoformat()
        cv_system_text = (
            "You are an expert HR assistant, make a summarization of the uploaded CV. "
            f"Note that today is {today_str}."
        )

    # ---- Initialize usage logger (CSV persistence) ----
    usage_logger = GeminiUsageLogger(initial_df=None, log_path=log_path)
    
    # ---- Load or create FAISS store (keep in memory during the loop) ----
    try:
        store = FaissVectorStore.load(vector_dbname, vector_dbmeta)
        if clear_initial_faiss:
            store.clear()
            logger.info(">>> Loaded & cleared FAISS store: %s", vector_dbname)
    except Exception:
        logger.warning(">>> FAISS load failed, creating new store at: %s", vector_dbname)
        # We'll infer dimension from first embedding below if possible
        store = None  # placeholder until we embed first doc

    # ---- Load or create BM25 index (keep in memory during the loop) ----
    try:
        bm25_index = BM25Index.load(bm25_dbmeta)
        if clear_initial_bm25: 
            bm25_index.clear()
            logger.info(">>> Loaded & cleared BM25 index: %s", bm25_dbmeta)
    except Exception:
        logger.warning(">>> BM25 load failed, creating new index at: %s", bm25_dbmeta)
        bm25_index = BM25Index()

    num_success, num_failed = 0, 0
    failed_files: list[str] = []
    last_logs_df: Optional[pd.DataFrame] = None

    # ==================================
    # Ingest loop (load -> add -> save)
    # ==================================
    for i, uploaded_filename in enumerate(uploaded_filenames_list, start=1):
        logger.info("\n==== Processing CV %d/%d: %s ====", i, len(uploaded_filenames_list), uploaded_filename)
        store, bm25_index, result, logs_df = process_single_cv(
            client,
            uploaded_filename, applied_position,
            cv_prompt_text      = cv_prompt_text,
            cv_system_text      = cv_system_text,
            log_path            = log_path,
            max_output_tokens   = max_output_tokens,
            temperature         = temperature,
            top_p               = top_p,
            top_k               = top_k,
            embedding_modelname = embedding_modelname,
            bm25_index = bm25_index,
            store  = store,
            doc_id = i - 1,
        )
    
        last_logs_df = logs_df if not logs_df.empty else last_logs_df
        if result["success"]:
            num_success += 1
        else:
            num_failed += 1
            failed_files.append(uploaded_filename)
            logger.warning("CV failed: %s | error=%s", uploaded_filename, result["error"])
            continue
        logger.info("==== Completed CV %d/%d: %s ====\n", i, len(uploaded_filenames_list), uploaded_filename)

    # ---- Persist both indexes once at the end ----
    if store is None:
        # Nothing succeeded
        logger.warning("No FAISS store to save (no successful documents).")
    else:
        store.save(vector_dbname, vector_dbmeta)
        logger.info("Saved FAISS store & metadata: %s | %s", vector_dbname, vector_dbmeta)

    bm25_index.save(bm25_dbmeta)
    logger.info("Saved BM25 metadata: %s", bm25_dbmeta)

    return {
        "num_inputs": len(uploaded_filenames_list),
        "num_success": num_success,
        "num_failed": num_failed,
        "failed_files": failed_files,
        "faiss_path": vector_dbname,
        "faiss_meta": vector_dbmeta,
        "bm25_meta": bm25_dbmeta,
        "last_logs_df": last_logs_df if last_logs_df is not None else pd.DataFrame(),
    }