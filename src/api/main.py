from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
import skfuzzy as fuzz
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# ── Paths ──────────────────────────────────────────────────────────────
# This file is at src/api/main.py
# parents[2] goes up two levels to reach the project root
BASE_DIR   = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"
DATA_DIR   = BASE_DIR / "data" / "processed"


# ── Request / Response schemas ─────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class QueryResponse(BaseModel):
    query           : str
    cache_hit       : bool
    matched_query   : Optional[str]
    similarity_score: Optional[float]
    dominant_cluster: int
    result          : List[Dict[str, Any]]
    latency_ms      : float


class CacheStats(BaseModel):
    total_entries  : int
    total_lookups  : int
    total_hits     : int
    hit_rate       : float
    threshold      : float
    clusters_active: List[int]


# ── Cache entry dataclass ──────────────────────────────────────────────

@dataclass
class CacheEntry:
    query          : str
    query_embedding: np.ndarray
    result         : Any
    cluster_id     : int
    timestamp      : float = field(default_factory=time.time)
    hit_count      : int   = 0


# ── Cosine similarity helper ───────────────────────────────────────────

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-9 else 0.0


# ── Semantic Cache class ───────────────────────────────────────────────

class SemanticCache:
    """
    Cluster-aware semantic cache.

    Structure:
        cache = { cluster_id: [CacheEntry, ...] }

    Only searches entries inside the same cluster as the query.
    Uses LFU (Least Frequently Used) eviction when a cluster bucket is full.
    """

    def __init__(self, similarity_threshold: float = 0.85, max_per_cluster: int = 300):
        self.similarity_threshold = similarity_threshold
        self.max_per_cluster      = max_per_cluster
        self._cache               : Dict[int, List[CacheEntry]] = {}
        self._total_lookups       = 0
        self._total_hits          = 0
        self._total_stores        = 0

    def lookup(self, query_embedding: np.ndarray, cluster_id: int) -> Optional[dict]:
        self._total_lookups += 1
        best_score, best_entry = -1.0, None

        for entry in self._cache.get(cluster_id, []):
            score = _cosine(query_embedding, entry.query_embedding)
            if score > best_score:
                best_score, best_entry = score, entry

        if best_entry and best_score >= self.similarity_threshold:
            self._total_hits      += 1
            best_entry.hit_count  += 1
            return {
                "hit"             : True,
                "matched_query"   : best_entry.query,
                "similarity_score": float(best_score),
                "result"          : best_entry.result,
                "cluster_id"      : cluster_id,
            }
        return None

    def store(self, query: str, query_embedding: np.ndarray,
              result: Any, cluster_id: int) -> None:
        self._total_stores += 1
        self._cache.setdefault(cluster_id, [])

        # Evict least-frequently-used entry if bucket is full
        if len(self._cache[cluster_id]) >= self.max_per_cluster:
            self._cache[cluster_id].sort(key=lambda e: (e.hit_count, e.timestamp))
            self._cache[cluster_id].pop(0)

        self._cache[cluster_id].append(
            CacheEntry(
                query           = query,
                query_embedding = query_embedding.copy(),
                result          = result,
                cluster_id      = cluster_id
            )
        )

    def stats(self) -> dict:
        total = sum(len(v) for v in self._cache.values())
        hr    = self._total_hits / self._total_lookups if self._total_lookups else 0.0
        return {
            "total_entries"  : total,
            "total_lookups"  : self._total_lookups,
            "total_hits"     : self._total_hits,
            "hit_rate"       : round(hr, 4),
            "threshold"      : self.similarity_threshold,
            "clusters_active": list(self._cache.keys()),
        }

    def reset(self) -> None:
        self._cache.clear()
        self._total_lookups = 0
        self._total_hits    = 0
        self._total_stores  = 0


# ── FastAPI app ────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Semantic Search API",
    description = "Embedding + FAISS + Fuzzy Clustering + Semantic Cache",
    version     = "1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_methods = ["*"],
    allow_headers = ["*"],
)


# ── Global variables (loaded once at startup) ──────────────────────────

_embed_model  : SentenceTransformer = None
_faiss_index  : faiss.Index         = None
_cluster_model: dict                = None
_corpus       : dict                = None
_cache        : SemanticCache       = SemanticCache(similarity_threshold=0.85)


# ── Startup: load all artifacts from disk ─────────────────────────────

@app.on_event("startup")
def load_artifacts():
    global _embed_model, _faiss_index, _cluster_model, _corpus

    print("Loading ML artifacts...")

    _embed_model   = SentenceTransformer("BAAI/bge-base-en-v1.5")
    _faiss_index   = faiss.read_index(str(MODELS_DIR / "faiss.index"))
    _cluster_model = pickle.loads((MODELS_DIR / "cluster_model.pkl").read_bytes())
    _corpus        = pickle.loads((DATA_DIR   / "clean_corpus.pkl").read_bytes())

    print(f"Embedding model : BAAI/bge-base-en-v1.5")
    print(f"FAISS index     : {_faiss_index.ntotal} vectors")
    print(f"Clusters        : {_cluster_model['n_clusters']}")
    print(f"Corpus          : {len(_corpus['texts'])} documents")
    print("Server ready")


# ── Internal helper functions ──────────────────────────────────────────

def _embed(text: str) -> np.ndarray:
    """Convert text to L2-normalised embedding vector."""
    return _embed_model.encode(
        [text], normalize_embeddings=True
    )[0].astype(np.float32)


def _get_cluster(q_emb: np.ndarray) -> int:
    """Assign query to its dominant fuzzy cluster."""
    q_nd = _cluster_model["reducer_nd"].transform(q_emb.reshape(1, -1))

    # cmeans_predict returns 6 values: u, u0, d, jm, p, fpc
    mem, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
        q_nd.T,
        _cluster_model["cntr"],
        m=2.0,
        error=0.005,
        maxiter=1000
    )
    return int(np.argmax(mem[:, 0]))


def _faiss_search(q_emb: np.ndarray, top_k: int) -> List[dict]:
    """Run FAISS nearest-neighbour search and return top-k results."""
    scores, idxs = _faiss_index.search(q_emb.reshape(1, -1), k=top_k)
    return [
        {
            "doc_id"      : int(idx),
            "score"       : round(float(score), 4),
            "category"    : _corpus["categories"][idx],
            "text_preview": _corpus["texts"][idx][:300],
        }
        for idx, score in zip(idxs[0], scores[0])
    ]


# ── API Endpoints ──────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "message"  : "Semantic Search API is running",
        "docs"     : "http://localhost:8000/docs",
        "endpoints": [
            "POST /query",
            "GET  /cache/stats",
            "DELETE /cache",
            "GET  /health"
        ]
    }


@app.get("/health")
def health():
    return {
        "status"    : "ok",
        "index_size": _faiss_index.ntotal if _faiss_index else 0
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """
    Main search endpoint.

    Flow:
        1. Embed the query
        2. Assign to a fuzzy cluster
        3. Check semantic cache (fast path)
        4. If cache miss → FAISS retrieval (slow path)
        5. Store result in cache
        6. Return JSON response
    """
    t0 = time.time()

    # Step 1 — Embed
    q_emb = _embed(req.query)

    # Step 2 — Cluster assignment
    cluster_id = _get_cluster(q_emb)

    # Step 3 — Cache lookup
    cached = _cache.lookup(q_emb, cluster_id)
    if cached:
        return QueryResponse(
            query            = req.query,
            cache_hit        = True,
            matched_query    = cached["matched_query"],
            similarity_score = round(cached["similarity_score"], 4),
            dominant_cluster = cluster_id,
            result           = cached["result"],
            latency_ms       = round((time.time() - t0) * 1000, 2),
        )

    # Step 4 — FAISS retrieval
    results = _faiss_search(q_emb, req.top_k)

    # Step 5 — Store in cache
    _cache.store(req.query, q_emb, results, cluster_id)

    return QueryResponse(
        query            = req.query,
        cache_hit        = False,
        matched_query    = None,
        similarity_score = None,
        dominant_cluster = cluster_id,
        result           = results,
        latency_ms       = round((time.time() - t0) * 1000, 2),
    )


@app.get("/cache/stats", response_model=CacheStats)
def cache_stats():
    """Returns cache performance metrics: hit rate, total entries, active clusters."""
    return _cache.stats()


@app.delete("/cache")
def clear_cache():
    """Resets the entire semantic cache."""
    _cache.reset()
    return {"message": "Cache cleared successfully."}
