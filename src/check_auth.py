"""Check that Reddit API credentials work.

Usage:
  1) Copy ``configs/praw.ini.example`` to the project root as ``praw.ini``
     (or set the ``PRAW_*`` environment variables).
  2) Fill in your Reddit app / account details. Never commit them.
  3) Run ``python -m src.check_auth`` to print the authenticated user.
"""

from __future__ import annotations

import sys

from .reddit_client import create_reddit_client


def main() -> None:
    reddit = create_reddit_client()
    print("Authenticated as:", reddit.user.me())


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print("Authentication check failed:", exc, file=sys.stderr)
        sys.exit(1)
