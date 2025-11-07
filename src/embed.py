from __future__ import annotations

import argparse
from typing import Dict, List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer  # type: ignore

from .utils import get_logger, read_yaml, load_parquet


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()


def embed_texts(cfg: Dict) -> None:
    proc_path = cfg["data"]["processed_path"]
    out_npz = cfg["embedding"]["output_npz"]
    model_name = cfg["embedding"]["model_name"]
    batch_size = int(cfg["embedding"].get("batch_size", 32))
    normalize = bool(cfg["embedding"].get("normalize", True))

    df = load_parquet(proc_path)
    texts: List[str] = df["text"].astype(str).tolist()

    logger.info("Loading embedding model: %s", model_name)
    model = SentenceTransformer(model_name)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize,
    )
    emb = np.asarray(emb, dtype=np.float32)

    np.savez_compressed(out_npz, embeddings=emb, ids=df["id"].astype(str).values)
    logger.info("Saved embeddings to %s (shape=%s)", out_npz, emb.shape)


def main() -> None:
    args = parse_args()
    cfg = read_yaml(args.config)
    embed_texts(cfg)


if __name__ == "__main__":
    main()


