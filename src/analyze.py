from __future__ import annotations

import argparse
from typing import Dict
from pathlib import Path

import pandas as pd
import plotly.express as px

from .utils import get_logger, read_yaml, load_parquet


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()


def build_figures(cfg: Dict) -> None:
    df = load_parquet(cfg["data"]["labeled_path"]) if Path(cfg["data"]["labeled_path"]).exists() else load_parquet(cfg["data"]["clustered_path"])  # type: ignore
    # 월별 변화
    tmp = df.copy()
    tmp["month"] = pd.to_datetime(tmp["created_utc"], unit="s").dt.to_period("M").astype(str)
    fig = px.histogram(tmp, x="month", color=df.get("cluster"), nbins=len(tmp["month"].unique()))
    fig.update_layout(title="Monthly counts")
    fig.write_html("figures/monthly_counts.html")
    logger.info("Saved figures/monthly_counts.html")

    # 클러스터 분포 파이(가능한 경우)
    if "cluster" in df.columns:
        cnt = df["cluster"].value_counts().reset_index()
        cnt.columns = ["cluster", "count"]
        fig2 = px.pie(cnt, names="cluster", values="count", title="Cluster distribution")
        fig2.write_html("figures/cluster_pie.html")
        logger.info("Saved figures/cluster_pie.html")


def main() -> None:
    args = parse_args()
    cfg = read_yaml(args.config)
    build_figures(cfg)


if __name__ == "__main__":
    main()


