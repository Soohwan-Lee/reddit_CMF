from __future__ import annotations

import argparse
import pandas as pd
from typing import Dict

from .utils import get_logger, read_yaml, load_parquet, save_parquet, anonymize_author


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--anonymize", action="store_true")
    return p.parse_args()


def preprocess(cfg: Dict, anonymize: bool) -> pd.DataFrame:
    inp = cfg["data"]["raw_path"]
    out = cfg["data"]["processed_path"]
    min_len = cfg["data"]["min_text_len"]
    df = load_parquet(inp)

    df["title"] = df["title"].fillna("")
    df["selftext"] = df["selftext"].fillna("")
    df["text"] = (df["title"] + "\n\n" + df["selftext"]).str.strip()
    df = df[df["text"].str.len() >= min_len]

    if anonymize:
        df["author"] = df["author"].map(anonymize_author)

    df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)
    save_parquet(df, out)
    logger.info("Saved processed posts to %s (%d rows)", out, len(df))
    return df


def main() -> None:
    args = parse_args()
    cfg = read_yaml(args.config)
    preprocess(cfg, args.anonymize)


if __name__ == "__main__":
    main()


