# Semantic Search System
### An end-to-end semantic search engine with fuzzy clustering and intelligent caching

## What this project does

When you search for *"flu symptoms"*, a normal search engine looks for those exact words.
This system understands that *"influenza fever treatment"* means the same thing and returns the same results — without recomputing anything, because it remembers similar past queries.

That is the core idea: **semantic search + smart caching**.

---

## Why I built it this way

When I started this project, I had three main problems to solve:

**Problem 1 — How do I make search understand meaning, not just words?**

Keyword-based search fails when people phrase things differently.
I solved this by converting every document and query into a **vector embedding**using a pre-trained transformer model (`BAAI/bge-base-en-v1.5`). Similar sentences produce similar vectors, so searching by meaning becomes a simple math operation — find the closest vector.

**Problem 2 — How do I search thousands of vectors quickly?**

Comparing a query against every document one by one is too slow.
I used **FAISS** (Facebook AI Similarity Search), which is a library built specifically for fast nearest-neighbour search on vectors. It can find the top-5 similar documents from 10,000+ in milliseconds.

**Problem 3 — How do I avoid recomputing the same expensive search twice?**

If 100 users ask about "machine learning basics", running FAISS 100 times wastes resources.
I built a **semantic cache** that stores past query results. When a new query is similar enough to a past one (cosine similarity > 0.85), it returns the cached result instantly — no FAISS needed.

But searching the entire cache for every query is also slow. So I used **Fuzzy C-Means clustering** to group similar documents into clusters, and the cache only searches within the relevant cluster. This made the lookup much faster.

---

## System Architecture

```
User Query
    │
    ▼
Embedding Model (BAAI/bge-base-en-v1.5)
    │
    ▼
Query Vector (768 dimensions)
    │
    ▼
Fuzzy Cluster Assignment
    │
    ▼
Semantic Cache Lookup (search only this cluster)
    │
    ├── Similarity > 0.85 ──► Return cached result instantly
    │                         (cache HIT)
    │
    └── Below threshold ────► FAISS Vector Search
                               │
                               ▼
                          Top-K Results
                               │
                               ▼
                          Store in Cache
                               │
                               ▼
                          Return Results
```

---

## Why I chose each component

| Component | What I used | Why I chose it |
|-----------|-------------|----------------|
| Embedding model | `BAAI/bge-base-en-v1.5` | One of the best models for semantic retrieval tasks. Produces 768-dim vectors that capture sentence meaning very well. |
| Vector database | `FAISS IndexFlatIP` | Exact cosine similarity search. Fast enough for this dataset size and gives 100% accurate results (no approximation). |
| Clustering | Fuzzy C-Means | Unlike K-Means, which forces one hard label per document, Fuzzy C-Means gives each document a *probability distribution* over clusters. A document about gun laws can belong 52% to politics and 43% to firearms, which is more realistic. |
| Dimensionality reduction | UMAP (20D for clustering, 2D for visualization) | UMAP preserves the local and global structure of high-dimensional data better than PCA. This gave much cleaner cluster boundaries. |
| Cache structure | Python dict + cosine similarity | No external caching library needed. I implemented it from scratch using cluster buckets, so lookup stays fast as the cache grows. |
| API | FastAPI | Simple to write, automatically generates interactive documentation at `/docs`. |

---

## Why Fuzzy Clustering specifically

The assignment required soft/fuzzy clustering, and honestly it made a lot of sense once I understood why.

The 20 Newsgroups dataset has real semantic overlap — an article about *"gun control legislation"* genuinely belongs to both *politics* and *firearms* topics. Hard clustering (K-Means) would force it into one category and lose that nuance.

With Fuzzy C-Means, that same document gets something like:

```
Politics  cluster: 0.48
Firearms  cluster: 0.43
Law       cluster: 0.09
```

This distribution is used in the cache — the query gets assigned to its dominant cluster, which is where the cache search happens. So the clustering directly improves cache performance. That integration between Part 2 and Part 3 was one of the things I was most careful about designing.

---

## The tunable parameter — Similarity Threshold

The most important design decision in the cache is the **similarity threshold**: how similar does a new query need to be to a cached query before we return the cached result?

I ran experiments across 6 threshold values:

| Threshold | Cache Hit Rate | Behaviour |
|-----------|---------------|-----------|
| 0.70 | ~80% | Too aggressive — returns wrong answers |
| 0.75 | ~70% | Still risky |
| 0.80 | ~55% | Reasonable balance |
| **0.85** | **~45%** | **Best balance — chosen** |
| 0.90 | ~20% | Conservative, rarely hits |
| 0.95 | ~8% | Almost never hits |

I chose **0.85** because it gives a good hit rate while making sure the cached answer is actually relevant to the new query.

---

## Project Structure

```
semantic-search-system/
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb     # Load + clean 20 Newsgroups dataset
│   ├── 02_embeddings_faiss.ipynb       # Generate embeddings + build FAISS index
│   ├── 03_fuzzy_clustering.ipynb       # UMAP reduction + Fuzzy C-Means clustering
│   ├── 04_semantic_cache.ipynb         # SemanticCache class + unit tests
│   ├── 05_retrieval_pipeline.ipynb     # End-to-end pipeline test
│   ├── 06_threshold_experiments.ipynb  # Threshold analysis + plots
│   └── 07_api_service.ipynb            # API demo + testing
│
├── src/
│   ├── __init__.py
│   └── api/
│       ├── __init__.py
│       └── main.py                     # FastAPI server
│
├── experiments/
│   ├── cluster_visualization.png       # Fuzzy cluster plot (UMAP 2D)
│   └── threshold_analysis.png          # Hit rate vs threshold graph
│
├── models/                             # Auto-generated by notebooks
│   ├── embeddings.npy
│   ├── faiss.index
│   ├── cluster_model.pkl
│   └── membership_matrix.npy
│
├── data/
│   └── processed/
│       └── clean_corpus.pkl
│
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## How to run this project

### Option A — Google Colab (recommended)

**Step 1** — Open each notebook in Colab in order

**Step 2** — Mount your Google Drive when prompted

**Step 3** — Run notebooks in this exact order:

```
01 → 02 → 03 → 04 → 05 → 06 → 07
```

Each notebook saves its output to Google Drive so the next one can load it.

**Step 4** — After all notebooks are done, start the API:
```bash
uvicorn src.api.main:app --reload --port 8000
```

### Option B — Local machine

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/semantic-search-system.git
cd semantic-search-system

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run notebooks 01 through 07 in order, then start the API
uvicorn src.api.main:app --reload --port 8000
```

Open `http://localhost:8000/docs` in your browser for the interactive API.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/query` | Search for documents by meaning |
| `GET` | `/cache/stats` | View cache hit rate and statistics |
| `DELETE` | `/cache` | Clear the cache |
| `GET` | `/health` | Check if server is running |

### Example request

```json
POST /query
{
  "query": "symptoms of flu and fever",
  "top_k": 5
}
```

### Example response — cache MISS (first time)

```json
{
  "query": "symptoms of flu and fever",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": null,
  "dominant_cluster": 3,
  "result": [
    {
      "doc_id": 4821,
      "score": 0.8934,
      "category": "sci.med",
      "text_preview": "fever can be treated with..."
    }
  ],
  "latency_ms": 43.2
}
```

### Example response — cache HIT (similar query asked later)

```json
{
  "query": "influenza fever treatment",
  "cache_hit": true,
  "matched_query": "symptoms of flu and fever",
  "similarity_score": 0.912,
  "dominant_cluster": 3,
  "result": [...],
  "latency_ms": 1.4
}
```

Notice the latency drops from **43ms → 1.4ms** on a cache hit. That is the entire point of the semantic cache.

---

## Dataset

**20 Newsgroups** — a classic NLP benchmark dataset containing ~18,000 newsgroup posts across 20 topics including medicine, space, computing, politics, religion and sports.

Loaded directly via `sklearn.datasets.fetch_20newsgroups` — no manual download needed.

---

## Requirements

```
sentence-transformers==2.7.0
faiss-cpu==1.8.0
scikit-learn==1.4.2
scikit-fuzzy==0.4.2
fastapi==0.111.0
uvicorn[standard]==0.30.1
numpy==1.26.4
pandas==2.2.2
umap-learn==0.5.6
matplotlib==3.8.4
pydantic==2.7.1
```

---

## What I learned from this project

- How transformer embeddings work and why they are better than TF-IDF for semantic tasks
- How FAISS indexes vectors for fast similarity search
- Why fuzzy clustering is more appropriate than hard clustering for text data with overlapping topics
- How caching at the semantic level (meaning similarity) is fundamentally different from key-value caching
- How all these components can be wired together into a real API service

---

*Built with Python 3.10 · Developed on Google Colab · Dataset: 20 Newsgroups*
