import gradio as gr
import logging
import re
from typing import Any, Dict, List, Tuple, Optional

import genai_functions.hybrid_rag_functions as rag_functs

# --- helpers -------------------------------------------------------------

def _format_sources(used_ctx: Optional[List[Dict[str, Any]]]) -> str:
    if not used_ctx:
        return "No sources."
    lines = []
    for i, m in enumerate(used_ctx, 1):
        name = m.get("uploaded_filename") or m.get("source") or "doc"
        lines.append(
            f"[{i}] {name} | "
            f"vec={m.get('_score_vec',0):.3f} "
            f"bm25={m.get('_score_bm25',0):.3f} "
            f"fused={m.get('_score_fused',0):.3f}"
        )
    return "\n".join(lines)

def _clean(s: str) -> str:
    # collapse many blank lines and trim
    s = s.replace("\\n", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def _history_to_text(history: List[Dict[str, str]], max_turns: int = 5) -> str:
    """Format past turns into a string for the LLM context."""
    if not history:
        return ""
    # keep only the last N turns
    truncated = history[-max_turns*2:]  
    lines = []
    for msg in truncated:
        role = msg["role"].capitalize()
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)

# ---- factory -------------------------------------------------------------

def make_rag_chat_w_gradio(
    *,
    client: Any,
    vector_store: Any,
    bm25_index: Any,
    log_path: str = "logs/gemini_usage.csv",
):
    """
    Returns a configured gr.Blocks app. Call .launch() from your notebook.
    """

    def chat_handler(
        user_msg: str,
        history: List[Dict[str, str]],
        k_vec: float = 3,  k_bm25: float = 3, k_final: float = 2,
        rrf_k: float = 60, w_vec: float = 1.0, w_bm: float = 1.0,
        temperature: float = 0.2,
        max_tokens: float  = 8192,
        logs_df_state=None,  # gr.State(pd.DataFrame or None)
    ):
        history = history or []
        if not isinstance(user_msg, str) or not user_msg.strip():
            return history, (logs_df_state or None), "Please enter a question."

        try:
            # Build history string
            history_text = _history_to_text(history)
        
            # Combine history + new user message
            query_with_history = (
                f"Conversation so far:\n{history_text}\n\n"
                f"User's new question:\n{user_msg.strip()}"
            )
            
            resp, logs_df, used_ctx = rag_functs.master_gemini_rag_bm25_answer_w_log(
                client=client,
                query=query_with_history.strip(),
                vector_store=vector_store,
                bm25_index=bm25_index,
                log_path=log_path,
                k_vec=int(k_vec),
                k_bm25=int(k_bm25),
                k_final=int(k_final),
                rrf_k=int(rrf_k),
                w_vec=float(w_vec),
                w_bm=float(w_bm),
                max_output_tokens=int(max_tokens),
                temperature=float(temperature),
            )

            answer = _clean(getattr(resp, "text", "") or "(no text)")
            sources_block = "**Sources**\n" + _format_sources(used_ctx)
            assistant_msg = f"{answer}\n\n{sources_block}"

            history.append({"role": "user", "content": user_msg})
            history.append({"role": "assistant", "content": assistant_msg})

            logs_df_state = logs_df if logs_df is not None else logs_df_state
            return history, logs_df_state, _format_sources(used_ctx)

        except Exception as e:
            logging.getLogger(__name__).exception("Chat handler failed: %s", e)
            history.append({"role": "user", "content": user_msg})
            history.append({"role": "assistant", "content": f"Error: {e}"})
            return history, logs_df_state, f"Error: {e}"

    with gr.Blocks() as demo:
        gr.Markdown("### 🔎 RAG + BM25 Chat")
        logs_df_state = gr.State(None)

        with gr.Row():
            with gr.Column(scale=4):
                chat = gr.Chatbot(label="Conversation", type="messages", height=520, show_copy_button=True)
                user_box = gr.Textbox(label="Your question", placeholder="e.g., Which candidates worked in Singapore?")
                send_btn = gr.Button("Send", variant="primary")
            with gr.Column(scale=3):
                with gr.Accordion("Settings", open=False):
                    k_vec   = gr.Slider(1, 20, value=4, step=1, label="Top-k (Vector)")
                    k_bm25  = gr.Slider(1, 20, value=6, step=1, label="Top-k (BM25)")
                    k_final = gr.Slider(1, 20, value=5, step=1, label="Top-k (Fused)")
                    rrf_k   = gr.Slider(1, 200, value=60, step=1, label="RRF k (damping)")
                    w_vec   = gr.Slider(0.0, 3.0, value=1.0, step=0.1, label="Weight: Vector")
                    w_bm    = gr.Slider(0.0, 3.0, value=1.0, step=0.1, label="Weight: BM25")
                    temperature = gr.Slider(0.0, 1.5, value=0.2, step=0.05, label="Temperature")
                    max_tokens  = gr.Slider(256, 4096*3, value=8192, step=64, label="Max output tokens")
                sources = gr.Textbox(label="Sources (debug)", lines=8, show_copy_button=True)

        user_box.submit(
            fn=chat_handler,
            inputs=[user_box, chat, k_vec, k_bm25, k_final, rrf_k, w_vec, w_bm, temperature, max_tokens, logs_df_state],
            outputs=[chat, logs_df_state, sources],
        ).then(lambda: "", None, user_box)

        send_btn.click(
            fn=chat_handler,
            inputs=[user_box, chat, k_vec, k_bm25, k_final, rrf_k, w_vec, w_bm, temperature, max_tokens, logs_df_state],
            outputs=[chat, logs_df_state, sources],
        ).then(lambda: "", None, user_box)

    return demo
