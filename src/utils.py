from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import orjson
import pandas as pd


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def load_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    ensure_parent_dir(path)
    df.to_parquet(path, index=False)


def load_json(path: str | Path) -> Any:
    with open(path, "rb") as f:
        return orjson.loads(f.read())


def save_json(obj: Any, path: str | Path) -> None:
    ensure_parent_dir(path)
    with open(path, "wb") as f:
        f.write(orjson.dumps(obj, option=orjson.OPT_INDENT_2))


def read_yaml(path: str | Path) -> Dict[str, Any]:
    import yaml  # PyYAML 는 conda 기본 채널 포함
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def anonymize_author(author: Optional[str]) -> str:
    if not author or author in {"[deleted]", "[removed]"}:
        return "u/[anon]"
    return f"u/[anon-{abs(hash(author)) % 10_000}]"


