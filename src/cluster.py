from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
import umap  # type: ignore

from .utils import get_logger, read_yaml, load_parquet, save_parquet


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()


def choose_k_by_elbow(inertias: np.ndarray, k_values: np.ndarray) -> int:
    """Pick K at the elbow: the point of largest discrete second difference.

    ``second_diff[i]`` is the curvature at ``k_values[i + 1]``. (The original
    version took the arg-min with an offset of 2; on this dataset both rules
    give K = 8, so the published results are unchanged.)
    """
    second_diff = np.diff(inertias, n=2)
    idx = int(np.argmax(second_diff)) + 1
    return int(k_values[idx])


def run_umap(emb: np.ndarray, cfg: Dict) -> np.ndarray:
    reducer = umap.UMAP(
        n_neighbors=int(cfg["umap"]["n_neighbors"]),
        min_dist=float(cfg["umap"]["min_dist"]),
        n_components=int(cfg["umap"]["n_components"]),
        random_state=int(cfg["umap"]["random_state"]),
        metric="cosine",
    )
    return reducer.fit_transform(emb)


def cluster(cfg: Dict) -> Tuple[pd.DataFrame, np.ndarray]:
    df = load_parquet(cfg["data"]["processed_path"]).copy()
    npz = np.load(cfg["embedding"]["output_npz"], allow_pickle=True)
    emb = npz["embeddings"]
    ids = npz["ids"].astype(str)

    umap_2d = run_umap(emb, cfg)

    k_min, k_max = int(cfg["kmeans"]["k_min"]), int(cfg["kmeans"]["k_max"])
    k_values = np.arange(k_min, k_max + 1)
    inertias = []
    for k in k_values:
        km = KMeans(n_clusters=int(k), n_init="auto", random_state=int(cfg["kmeans"]["random_state"]))
        km.fit(emb)
        inertias.append(km.inertia_)
    inertias = np.array(inertias)
    k_best = choose_k_by_elbow(inertias, k_values)
    logger.info("Chosen K by elbow: %d", k_best)

    km = KMeans(n_clusters=k_best, n_init="auto", random_state=int(cfg["kmeans"]["random_state"]))
    labels = km.fit_predict(emb)

    df = df.set_index("id")
    df.loc[ids, "cluster"] = labels
    df["umap_x"] = umap_2d[:, 0]
    df["umap_y"] = umap_2d[:, 1]
    df = df.reset_index()

    out = cfg["data"]["clustered_path"]
    save_parquet(df, out)
    logger.info("Saved clustered dataframe to %s", out)

    fig = px.scatter(
        df,
        x="umap_x",
        y="umap_y",
        color="cluster",
        hover_data=["title", "score", "num_comments"],
        title="UMAP clusters",
    )
    fig_dir = Path(cfg["analysis"]["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    html_path = fig_dir / "umap_clusters.html"
    fig.write_html(html_path)
    # Local exploration only: hover text includes post titles, so this file is
    # gitignored. Static, aggregate figures are produced by src.analyze.
    logger.info("Saved interactive UMAP plot to %s", html_path)

    return df, umap_2d


def main() -> None:
    args = parse_args()
    cfg = read_yaml(args.config)
    cluster(cfg)


if __name__ == "__main__":
    main()


