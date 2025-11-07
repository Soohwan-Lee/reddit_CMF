from __future__ import annotations

from typing import Optional

import os
import praw  # type: ignore


def create_reddit_client(site_name: str = "DEFAULT") -> praw.Reddit:
    """
    PRAW는 기본적으로 현재 작업 디렉토리/사용자 홈의 praw.ini를 탐색합니다.
    site_name 섹션(기본: DEFAULT)을 사용합니다.

    환경변수로 직접 제공하는 경우에도 동작하도록 백업 처리합니다.
    (PRAW는 환경변수 PRAW_CLIENT_ID 등도 인식함)
    """
    # 환경변수 기반 설정 우선
    if os.getenv("PRAW_CLIENT_ID") and os.getenv("PRAW_CLIENT_SECRET"):
        return praw.Reddit(
            client_id=os.getenv("PRAW_CLIENT_ID"),
            client_secret=os.getenv("PRAW_CLIENT_SECRET"),
            user_agent=os.getenv("PRAW_USER_AGENT", "reddit-cmf"),
            username=os.getenv("PRAW_USERNAME"),
            password=os.getenv("PRAW_PASSWORD"),
        )

    # praw.ini 기반 설정
    return praw.Reddit(site_name)


