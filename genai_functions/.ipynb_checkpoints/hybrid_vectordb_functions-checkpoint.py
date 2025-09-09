import os
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
import faiss
from rank_bm25 import BM25Okapi
import regex as re
import numpy as np
import pickle
from copy import deepcopy
import google.genai as genai # For embedding

import genai_functions.helper_functions as helper_functs

class FaissVectorStore:
    """
    Minimal FAISS + metadata store with cosine similarity search.
    - Stores vectors in FAISS (IndexFlatIP) with L2-normalized vectors (so IP == cosine sim)
    - Stores metadata separately (JSONL or pickle)
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim) # inner product (use with normalized vectors)
        self._vectors = []                  # keep in RAM until saved (optional)
        self._metadata: List[Dict] = []     # [{"text":..., "timestamp":..., "token_length":...}, ...]

    # ---------- Embedding helpers (plug your embedding client here) ----------
    @staticmethod
    def _to_float32(arr) -> np.ndarray:
        return np.array(arr, dtype="float32")

    def add_texts_and_metadata(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadata_list: Optional[List[Dict]] = None,
        default_metadata: Optional[Dict] = None,
    ):
        """
        Add texts with precomputed embeddings + flexible per-item metadata.
    
        Parameters
        ----------
        texts : List[str]
            Plaintext chunks to index.
        embeddings : List[List[float]]
            Precomputed embeddings matching each text (dim == self.dim).
            These will be L2-normalized for cosine similarity with IndexFlatIP.
        metadata_list : Optional[List[Dict]]
            A list of metadata dicts (one per text). Each dict can contain any keys
            (JSONL-friendly). Missing keys are allowed and will be auto-filled where applicable.
            If None, minimal metadata will be generated.
        default_metadata : Optional[Dict]
            Default metadata merged into each item *before* item-specific overrides.
            Example: {"source": "my_corpus", "tag": "v1"}.
    
        Behavior
        --------
        - For each item i, final metadata = {**default_metadata, **metadata_list[i]} (if provided)
        - Auto-fill:
            * "timestamp" (UTC ISO 8601) if not present
            * "token_length" via simple_token_count(text) if not present
        - All keys are preserved as-is to support JSONL dumps without schema constraints.
    
        Raises
        ------
        ValueError
            If lengths of texts, embeddings, and metadata_list (when provided) mismatch.
        """
        n = len(texts)
        if len(embeddings) != n:
            raise ValueError("embeddings length must match texts length")
    
        if metadata_list is not None and len(metadata_list) != n:
            raise ValueError("metadata_list length must match texts length when provided")
    
        # Normalize and add vectors
        vecs = np.asarray(embeddings, dtype="float32")
        faiss.normalize_L2(vecs)
        self.index.add(vecs)
        self._vectors.extend(vecs)
    
        # Prepare defaults
        default_metadata = default_metadata or {}
    
        # Build metadata rows JSONL-friendly (arbitrary keys preserved)
        now_iso = datetime.utcnow().isoformat()
        for i, t in enumerate(texts):
            item_meta = metadata_list[i] if metadata_list is not None else {}
            # Merge: defaults first, then item-specific override
            meta = {**default_metadata, **item_meta}
    
            # Auto-fill timestamp if missing
            meta.setdefault("timestamp", now_iso)
            # Auto-fill token_length if missing (won't overwrite if provided)
            meta.setdefault("token_length", helper_functs.simple_token_count(t))
            # Always store the raw text (if you want to make it optional, remove this line)
            meta.setdefault("text", t)
    
            self._metadata.append(meta)
            
    def clear(self):
        """
        Clear all vectors and metadata from the store.
        This reinitializes the FAISS index and empties metadata.
        """
        # Recreate a fresh FAISS index with the same dimension
        self.index = faiss.IndexFlatIP(self.dim)  # cosine sim (normalized IP)
        
        # Reset vectors and metadata
        self._vectors = []
        self._metadata = []         

    def search(self, query_embedding: List[float], k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Search the index by embedding. Returns top-k as [(metadata, similarity), ...].
        Similarity is cosine similarity (since we normalized and use IP index).
        """
        q = self._to_float32([query_embedding])
        faiss.normalize_L2(q)  # normalize query for cosine/IP
        D, I = self.index.search(q, k)
        hits = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:  # no result
                continue
            hits.append((self._metadata[idx], float(score)))  # score ∈ [-1, 1]
        return hits

    # ---------- Persistence ----------
    def save(self, index_path: str, metadata_path: str):
        """Save FAISS index and metadata (JSONL if .jsonl else pickle)."""
        faiss.write_index(self.index, index_path)

        # Save metadata (choose JSONL by default for readability)
        if metadata_path.endswith(".jsonl"):
            with open(metadata_path, "w", encoding="utf-8") as f:
                for row in self._metadata:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            with open(metadata_path, "wb") as f:
                pickle.dump(self._metadata, f)

    @classmethod
    def load(cls, index_path: str, metadata_path: str):
        """Load FAISS index and metadata; returns an initialized store."""
        index = faiss.read_index(index_path)

        # Load metadata
        if metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, "r", encoding="utf-8") as f:
                for line in f:
                    metadata.append(json.loads(line))
        else:
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)

        dim = index.d  # read dimension from index
        store = cls(dim)
        store.index = index
        store._metadata = metadata
        return store
        
class BM25Index:
    """
    Persistent BM25 index with UNIQUE docs keyed by doc_id.
    - doc_id := doc['id'] if present, else doc['uploaded_filename']
    - add_or_replace(): replaces existing doc with same doc_id
    - load(): dedupes by doc_id (last one wins)
    """
    def __init__(self, by_id=None):
        self.by_id = by_id or {}  # doc_id -> meta (includes 'text')
        self._bm25 = None
        self._ids = []     # order aligned with _tokens
        self._tokens = []  # tokenized texts in same order
        self._docs = {} 

    # ---------- persistence ----------
    @classmethod
    def load(cls, meta_path: str):
        by_id = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    doc_id = d.get("id") or d.get("uploaded_filename")
                    if not doc_id:
                        # skip malformed rows
                        continue
                    # normalize stored id & text
                    d["id"] = doc_id
                    d["text"] = (d.get("text") or "").strip()
                    # last one wins for same doc_id
                    by_id[doc_id] = d
        return cls(by_id)

    def save(self, meta_path: str):
        with open(meta_path, "w", encoding="utf-8") as f:
            for doc_id, d in self.by_id.items():
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    def clear(self):
        self.by_id.clear()
        self._bm25 = None
        self._ids = []
        self._tokens = []

    # ---------- core ops ----------
    def _rebuild(self):
        # build arrays in a stable order
        self._ids = list(self.by_id.keys())
        self._tokens = [helper_functs.simple_tokenize(self.by_id[i].get("text", "")) for i in self._ids]
        self._bm25 = BM25Okapi(self._tokens)

    def add_or_replace(self, doc_meta: dict):
        doc_id = doc_meta.get("id") or doc_meta.get("uploaded_filename")
        if not doc_id:
            raise ValueError("doc_meta must include 'id' or 'uploaded_filename'")
        # normalize
        doc_meta = {**doc_meta, "id": doc_id, "text": (doc_meta.get("text") or "").strip()}
        self.by_id[doc_id] = doc_meta
        self._bm25 = None  # mark dirty (lazy rebuild)

    def search(self, query: str, k: int = 5):
        if not self.by_id:
            return []
        if self._bm25 is None:
            self._rebuild()
        q_tokens = helper_functs.simple_tokenize(query or "")
        scores = self._bm25.get_scores(q_tokens)
        order = np.argsort(scores)[::-1]
        hits = []
        for i in order:
            doc_id = self._ids[i]
            meta = self.by_id[doc_id]
            hits.append((meta, float(scores[i])))
            if len(hits) >= k:
                break
        return hits

    def __len__(self):
        return len(self._docs)

    def count(self):
        return len(self._docs)

# -------------------------
# Text and query embedding
# -------------------------
def embed_texts_with_gemini(client, texts: List[str], embedding_modelname:str ="text-embedding-004") -> List[List[float]]:
    emb_list = []
    for t in texts:
        emb = client.models.embed_content(
            model=embedding_modelname,
            contents=[{"role": "user", "parts": [{"text": t}]}]
        ).embeddings[0].values
        emb_list.append(emb)
    return emb_list

def embed_query_with_gemini(client, query: str, embedding_modelname:str ="text-embedding-004") -> List[float]:
    try:
        resp = client.models.embed_content(
            model=embedding_modelname,
            contents=[{"role": "user", "parts": [{"text": query}]}]
        )
        embed_query = resp.embeddings[0].values
        return embed_query, resp
    except:
        return None, None

def show_hits(hits, title):
    print(f"\n=== {title} ===")
    for meta, score in hits:
        print(f"[score={score:.4f}] {meta['uploaded_filename']}")
