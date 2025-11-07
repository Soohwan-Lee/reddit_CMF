# Reddit CMF – Industrial Design Discourse Analysis

This project automates the collection and analysis of posts from the `r/IndustrialDesign` subreddit. It retrieves top posts from the past two years, cleans and embeds the text, clusters the discussions, and generates LLM-based qualitative summaries. The pipeline is designed for rapid insight generation about community themes, tools, and sentiment.

## Repository Structure

```
├── configs/                  # YAML configuration (subreddit, time window, paths, models)
├── data/
│   ├── raw/                  # Raw Reddit exports (.parquet)
│   └── processed/            # Cleaned posts, clusters, LLM summaries
├── figures/                  # UMAP visualisations (.html, .png)
├── models/                   # SentenceTransformer embeddings (.npz)
├── notebooks/                # Reference notebooks for each pipeline stage
├── src/
│   ├── collect_reddit.py     # Data collection
│   ├── preprocess.py         # Text cleaning & anonymisation
│   ├── embed.py              # Qwen3 embedding generation
│   ├── cluster.py            # UMAP + KMeans clustering & plotting
│   ├── llm_label.py          # OpenAI/Anthropic cluster summaries
│   └── utils.py              # Shared helpers (logging, IO, yaml)
```

## Key Outputs

| Stage | File | Description |
|-------|------|-------------|
| Collection | `data/raw/r_industrialdesign_top.parquet` | Raw posts with metadata |
| Pre-process | `data/processed/posts_clean.parquet` | Filtered + combined text, optional anonymisation |
| Embeddings | `models/embeddings_qwen3.npz` | 1024-dim Qwen3 vectors with IDs |
| Clustering | `data/processed/posts_with_clusters.parquet` | UMAP coords, KMeans labels, metadata |
| Visuals | `figures/umap_clusters.html` / `.png` | Interactive & static cluster plots |
| LLM Summaries | `data/processed/cluster_summaries.json` | Title / description / keywords per cluster |

## Requirements

- Python 3.10+
- Conda environment defined in `environment.yml`
- Reddit API credentials in `praw.ini`
- Optional: OpenAI or Anthropic API keys for LLM summarisation

### Installation

```powershell
conda env create -f environment.yml
conda activate reddit-cmf

# Optional for LLM stage
pip install --upgrade openai python-dotenv
```

## Initial Setup

1. **Reddit credentials** – copy `configs/praw.ini.example` to project root as `praw.ini`, populate `client_id`, `client_secret`, `user_agent`, `username`, `password`.
2. **LLM keys (optional)** – create `.env` with `OPENAI_API_KEY=...` (and/or `ANTHROPIC_API_KEY=...`). `python-dotenv` automatically loads these during LLM stage.

## Pipeline Execution

Run the following commands from the repo root (after activating the `reddit-cmf` environment):

### 1. Collect Reddit Posts
```powershell
python -m src.collect_reddit --config configs/config.yaml --mode top
```
- Downloads top posts (24-month window) using multiple `time_filter` values to maximise coverage.

### 2. Pre-process Text
```powershell
python -m src.preprocess --config configs/config.yaml --anonymize
```
- Combines title + body, removes very short posts, optionally anonymises authors.

### 3. Create Embeddings (Qwen/Qwen3-Embedding-0.6B)
```powershell
python -m src.embed --config configs/config.yaml
```
- Produces `models/embeddings_qwen3.npz` (float32 vectors, L2-normalised).

### 4. Cluster & Visualise (UMAP + KMeans)
```powershell
python -m src.cluster --config configs/config.yaml
```
- Generates 2D UMAP projection, auto-selects K via elbow, and saves Plotly HTML + PNG.

### 5. LLM Cluster Summaries
```powershell
python -m src.llm_label --config configs/config.yaml
```
- For each cluster, samples representative posts and calls OpenAI Responses API (fallback to Anthropic if configured).
- JSON parsing handles Markdown code fences and partial outputs to ensure clean summaries.

### (Optional) Quantitative Labeling
```powershell
python -m src.quant_classify --config configs/config.yaml
```
- Placeholder for multi-label classification (content tone, behaviour, tools, etc.).

## Configuration (`configs/config.yaml`)

Key sections:

- `data`: subreddit, time window, time filter list, output paths
- `embedding`: model name, batch size, normalisation
- `umap`: projection parameters
- `kmeans`: range for K, random seed
- `llm`: sample size per cluster, primary/fallback model names (e.g., `gpt-5`, `gpt-4o`, `claude-3-5-sonnet`)

All stages read from this config, allowing experimentation without code changes.

## Notes for New Users

- Ensure `praw.ini` and `.env` (if using LLM) are listed in `.gitignore`
- The pipeline is modular; rerun individual stages as needed
- For alternative embeddings (e.g., `BAAI/bge-m3`), change the `embedding.model_name` in config and rerun steps 3–5
- LLM costs vary by provider/model—monitor usage in your API dashboard

## Contacts

- For questions about the pipeline or extending the analytics (e.g., dashboards, reporting), please reach out via the project maintainer.