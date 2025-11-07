from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Iterable, Tuple

import pandas as pd

from .reddit_client import create_reddit_client
from .utils import get_logger, read_yaml, save_parquet


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to config.yaml")
    p.add_argument("--mode", choices=["top", "hot", "best"], default="top")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def to_unix(ts: str) -> int:
    return int(datetime.fromisoformat(ts).replace(tzinfo=timezone.utc).timestamp())


def iter_time_buckets(start_dt: datetime, end_dt: datetime, bucket: str = "month") -> Iterable[Tuple[int, int]]:
    """Yield (start_unix, end_unix) for each time bucket in [start_dt, end_dt]."""
    cur = start_dt
    while cur < end_dt:
        if bucket == "quarter":
            # 다음 분기의 시작 계산
            month = ((cur.month - 1) // 3) * 3 + 1
            q_start = datetime(cur.year, month, 1, tzinfo=timezone.utc)
            if month in (1, 4, 7, 10):
                if month == 10:
                    q_end = datetime(cur.year + 1, 1, 1, tzinfo=timezone.utc)
                else:
                    q_end = datetime(cur.year, month + 3, 1, tzinfo=timezone.utc)
            else:
                q_end = cur + timedelta(days=90)
            s = max(cur, q_start)
            e = min(end_dt, q_end)
        else:  # month
            s = datetime(cur.year, cur.month, 1, tzinfo=timezone.utc)
            if cur.month == 12:
                e = datetime(cur.year + 1, 1, 1, tzinfo=timezone.utc)
            else:
                e = datetime(cur.year, cur.month + 1, 1, tzinfo=timezone.utc)
            if e > end_dt:
                e = end_dt
        yield (int(s.timestamp()), int(e.timestamp()))
        cur = e


def fetch_posts(cfg: Dict, mode: str, limit: int | None) -> pd.DataFrame:
    reddit = create_reddit_client()
    sr = reddit.subreddit(cfg["data"]["subreddit"])

    start_unix = to_unix(cfg["data"]["time_window"]["start"])
    end_unix = to_unix(cfg["data"]["time_window"]["end"])

    api_limit = limit or cfg["data"]["top_limit"]
    top_time_filter = str(cfg["data"].get("top_time_filter", "year"))

    submissions = []
    if mode == "top":
        # 1) 타임스탬프 검색 기반(선택)
        if bool(cfg.get("search", {}).get("by_timestamp", False)):
            bucket = str(cfg.get("search", {}).get("bucket", "month"))
            per_bucket_limit = int(cfg.get("search", {}).get("search_limit", 1000))

            start_dt = datetime.fromtimestamp(start_unix, tz=timezone.utc)
            end_dt = datetime.fromtimestamp(end_unix, tz=timezone.utc)
            for s_unix, e_unix in iter_time_buckets(start_dt, end_dt, bucket=bucket):
                query = f"timestamp:{s_unix}..{e_unix}"
                # Reddit 공식 /search API: q, sort, syntax 파라미터 사용
                # ref: https://www.reddit.com/dev/api/ (search)
                for s in sr.search(
                    query=query,
                    sort="top",
                    syntax="cloudsearch",
                    time_filter="all",
                    limit=per_bucket_limit,
                    params={"restrict_sr": 1},
                ):
                    submissions.append(s)
        else:
            # 2) 기본 top 리스팅 - 다양한 time_filter 조합으로 커버리지 확장
            filters = cfg["data"].get("top_time_filters") or [top_time_filter]
            for tf in filters:
                for s in sr.top(time_filter=tf, limit=api_limit):
                    submissions.append(s)
    elif mode == "hot":
        for s in sr.hot(limit=api_limit):
            submissions.append(s)
    else:  # best
        for s in sr.best(limit=api_limit):
            submissions.append(s)

    rows: List[Dict] = []
    for s in submissions:
        created = int(s.created_utc)
        if created < start_unix or created > end_unix:
            continue
        rows.append(
            {
                "id": s.id,
                "title": s.title or "",
                "selftext": getattr(s, "selftext", "") or "",
                "created_utc": created,
                "score": s.score,
                "num_comments": s.num_comments,
                "author": str(getattr(s, "author", None) or ""),
                "flair": getattr(s, "link_flair_text", None),
                "url": s.url,
                "is_self": bool(getattr(s, "is_self", False)),
            }
        )

    df = pd.DataFrame(rows).drop_duplicates("id")
    logger.info("Collected %d posts after date filter", len(df))
    return df


def main() -> None:
    args = parse_args()
    cfg = read_yaml(args.config)
    out_path = cfg["data"]["raw_path"]
    df = fetch_posts(cfg, args.mode, args.limit)
    save_parquet(df, out_path)
    logger.info("Saved raw posts to %s", out_path)


if __name__ == "__main__":
    main()


