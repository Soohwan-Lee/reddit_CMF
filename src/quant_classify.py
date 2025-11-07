from __future__ import annotations

import argparse
import os
from typing import Dict, List

import pandas as pd

from .utils import get_logger, read_yaml, load_parquet, save_parquet


logger = get_logger(__name__)


SCHEMA = {
    "content_type": ["question", "showcase", "feedback", "news", "discussion", "tutorial"],
    "emotion": ["positive", "neutral", "negative", "mixed"],
    "tech_tools": [],  # 예: SolidWorks, Fusion, KeyShot 등 문자열 리스트
    "purpose": ["ask_help", "share_inspiration", "seek_feedback", "job_edu", "other"],
    "social_context": ["education", "career", "critique", "community"],
    "impact": ["helpful", "learning", "growth", "unsure"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()


def classify_posts(cfg: Dict) -> pd.DataFrame:
    df = load_parquet(cfg["data"]["clustered_path"]).copy()

    # LLM 키가 없다면 빈 레이블로 채우고 통과
    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("LLM 키가 없어 분류를 스킵하고 빈 컬럼으로 저장합니다.")
        for col in SCHEMA.keys():
            df[col] = None
        save_parquet(df, cfg["data"]["labeled_path"])
        return df

    # 실제 분류 로직(샘플): 규칙+LLM 혼합 가능. 여기서는 단순 규칙/자리표시자.
    for col in SCHEMA.keys():
        df[col] = None

    save_parquet(df, cfg["data"]["labeled_path"])
    logger.info("Saved labeled posts to %s", cfg["data"]["labeled_path"])
    return df


def main() -> None:
    args = parse_args()
    cfg = read_yaml(args.config)
    classify_posts(cfg)


if __name__ == "__main__":
    main()


