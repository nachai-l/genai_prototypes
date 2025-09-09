from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Tuple

from google.genai import types

from genai_functions.gemini_usage_logging import GeminiUsageLogger
import genai_functions.hybrid_vectordb_functions as vectordb_functs

logger = logging.getLogger(__name__)

# -------------------------
# Small text helpers
# -------------------------
def _doc_id(meta: Mapping[str, Any]) -> str:
    """Return a stable identifier for a metadata dict."""
    return str(
        meta.get("id")
        or meta.get("uploaded_filename")
        or meta.get("doc_id")
        or meta.get("uuid")
        or id(meta)
    )


def _safe(txt: Optional[str]) -> str:
    return (txt or "").strip()


def _truncate_chars(txt: Optional[str], max_chars: int) -> str:
    t = _safe(txt)
    return t if len(t) <= max_chars else (t[: max(0, max_chars)].rstrip() + "…")


# -------------------------
# Fusion
# -------------------------
def _rrf_fuse(
    vec_hits: List[Tuple[Dict[str, Any], float]],
    bm25_hits: List[Tuple[Dict[str, Any], float]],
    *,
    k_final: int = 5,
    rrf_k: int = 60,
    w_vec: float = 1.0,
    w_bm: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Weighted Reciprocal Rank Fusion (RRF) over vector and BM25 results.

    Parameters
    ----------
    vec_hits : list[(meta, score)]
        Results from vector search (higher score = better).
    bm25_hits : list[(meta, score)]
        Results from BM25 search (higher score = better).
    k_final : int
        Number of fused items to return.
    rrf_k : int
        RRF damping constant (>0). 60 is a common default.
    w_vec : float
        Weight for the vector ranking.
    w_bm : float
        Weight for the BM25 ranking.

    Returns
    -------
    list[dict]
        Shallow copies of the input metadata for the top-k fused items with:
            - "_score_vec"
            - "_score_bm25"
            - "_score_fused"
    """
    if rrf_k <= 0:
        raise ValueError("rrf_k must be > 0")

    # Assign ranks (1 = best) based on descending score
    vec_sorted = sorted(vec_hits, key=lambda x: x[1], reverse=True)
    bm_sorted = sorted(bm25_hits, key=lambda x: x[1], reverse=True)

    vec_ranks = {_doc_id(m): r for r, (m, _) in enumerate(vec_sorted, start=1)}
    bm_ranks = {_doc_id(m): r for r, (m, _) in enumerate(bm_sorted, start=1)}

    vec_scores = {_doc_id(m): float(s) for m, s in vec_hits}
    bm_scores = {_doc_id(m): float(s) for m, s in bm25_hits}

    all_ids = set(vec_ranks) | set(bm_ranks)

    fused_pairs: List[Tuple[str, float]] = []
    for did in all_ids:
        r_vec = vec_ranks.get(did, 10**9)
        r_bm = bm_ranks.get(did, 10**9)
        fused_score = (w_vec * (1.0 / (rrf_k + r_vec))) + (w_bm * (1.0 / (rrf_k + r_bm)))
        fused_pairs.append((did, fused_score))

    fused_pairs.sort(key=lambda x: x[1], reverse=True)

    # First metadata instance wins if duplicates
    by_id_meta: Dict[str, Dict[str, Any]] = {}
    for m, _ in vec_hits + bm25_hits:
        did = _doc_id(m)
        if did not in by_id_meta:
            by_id_meta[did] = m

    out: List[Dict[str, Any]] = []
    for did, f in fused_pairs[: max(0, k_final)]:
        m = dict(by_id_meta[did])  # shallow copy
        m["_score_vec"] = vec_scores.get(did, 0.0)
        m["_score_bm25"] = bm_scores.get(did, 0.0)
        m["_score_fused"] = float(f)
        out.append(m)

    return out


# -------------------------
# Context block builder
# -------------------------
def _build_context_block(
    ranked_ctx: List[Mapping[str, Any]],
    *,
    max_chars_per_ctx: int = 900,
    show_scores: bool = True,
) -> str:
    """
    Build a compact, numbered context block with optional score diagnostics.
    """
    if not ranked_ctx:
        return "(no relevant context found)"

    lines: List[str] = []
    for i, m in enumerate(ranked_ctx, start=1):
        uf = _safe(str(m.get("uploaded_filename", m.get("source", "doc"))))
        preview = _truncate_chars(_safe(str(m.get("text", ""))), max_chars_per_ctx)
        score_str = (
            f" | vec={m.get('_score_vec', 0):.3f} | bm25={m.get('_score_bm25', 0):.3f} | fused={m.get('_score_fused', 0):.3f}"
            if show_scores
            else ""
        )
        lines.append(f"[{i}] file: {uf}{score_str}\n{preview}")

    return "\n\n".join(lines)

# -------------------------
# Main: Hybrid RAG + Gemini
# -------------------------
def master_gemini_rag_bm25_answer_w_log(
    client: Any,
    query: str,
    vector_store: Any, # expects .search(embedding, k) -> List[(meta: dict, score: float)]
    bm25_index: Any,   # expects .search(query, k) -> List[(meta: dict, score: float)]
    *,
    usage_logger: Optional[GeminiUsageLogger] = None,
    log_path: Optional[str] = None,
    # retrieval knobs
    k_vec: int = 4,
    k_bm25: int = 6,
    k_final: int = 5,
    rrf_k: int = 60,
    w_vec: float = 1.0,
    w_bm: float = 1.0,
    max_chars_per_ctx: int = 4096,
    embedding_modelname: str = "text-embedding-004",
    # model knobs
    system_text: str = (
        "You are an expert HR assistant.\n"
        "Use only the provided CONTEXT to answer. If not found, say so.\n"
        "Cite sources with [n] where n matches the context block number.\n"
        "Respond concisely for recruiters."
    ),
    model_name: str = "gemini-2.5-flash",
    max_output_tokens: int = 12288,
    temperature: float = 0.1,
    top_p: float = 0.9,
    top_k: int = 40,
    verbose: int = 0,
) -> Tuple[Any, Optional["pd.DataFrame"], List[Dict[str, Any]]]:
    """
    Hybrid retrieval (vector + BM25) with RRF fusion, then Gemini answer generation.

    Returns
    -------
    (response, logs_df, fused_ctx)
        response : Gemini response object
        logs_df  : latest usage logs DataFrame (if a logger was provided/created), else None
        fused_ctx: fused list of context metadata dicts with score diagnostics
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    # Create a usage logger on the fly if only a path is provided
    if usage_logger is None and log_path:
        usage_logger = GeminiUsageLogger(initial_df=None, log_path=log_path)

    # 1) Vector + BM25 retrieval
    try:
        q_emb, _ = vectordb_functs.embed_query_with_gemini(client, query, embedding_modelname="text-embedding-004")
        if q_emb is None:
            raise RuntimeError("Query embedding returned None.")
    except Exception as e:
        logger.exception("Failed to embed query: %s", e)
        raise

    try:
        vec_hits = vector_store.search(q_emb, k=k_vec) if k_vec > 0 else []
        bm_hits = bm25_index.search(query, k=k_bm25) if k_bm25 > 0 else []
    except Exception as e:
        logger.exception("Vector/BM25 search failed: %s", e)
        raise

    # 2) Fusion
    fused_ctx = _rrf_fuse(
        vec_hits=vec_hits,
        bm25_hits=bm_hits,
        k_final=k_final,
        rrf_k=rrf_k,
        w_vec=w_vec,
        w_bm=w_bm,
    )
    context_block = _build_context_block(fused_ctx, max_chars_per_ctx=max_chars_per_ctx, show_scores=True)

    # 3) Build final prompt
    user_prompt = (
        f"QUESTION:\n{query}\n\n"
        "CONTEXT (numbered; cite as [n]):\n"
        f"{context_block}\n\n"
        "INSTRUCTIONS:\n"
        "- Answer the QUESTION based only on the CONTEXT.\n"
        '- If multiple candidates match, list top matches with 1–2 line justifications.\n'
        "- Use [n] citations after each claim referencing a specific context block.\n"
        '- If the answer is not present, say "Not found in provided documents."\n'
    )

    # 4) Call Gemini
    try:
        resp = client.models.generate_content(
            model=model_name,
            contents=[{"role": "user", "parts": [{"text": user_prompt}]}],
            config=types.GenerateContentConfig(
                system_instruction=[system_text],
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            ),
        )
    except Exception as e:
        logger.exception("Gemini generation failed: %s", e)
        # Best-effort usage log on failure
        if usage_logger:
            try:
                usage_logger.add_log_entry(query_text=query, response=None, uploaded_file=None, save=True)
            except Exception as log_e:
                logger.error("Failed to log failed generation attempt: %s", log_e)
        raise

    if verbose:
        try:
            print("LLM answer:\n", getattr(resp, "text", "<no text>"))
        except Exception:
            print("LLM answer:\n <no text>")

    # 5) Persist usage log (success path)
    logs_df = None
    if usage_logger:
        try:
            usage_logger.add_log_entry(query_text=query, response=resp, uploaded_file=None, save=True)
            logs_df = usage_logger.get_logs_dataframe()
        except Exception as e:
            logger.error("Failed to append usage log or fetch logs DF: %s", e)

    return resp, logs_df, fused_ctx
