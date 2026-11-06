from __future__ import annotations

import os
import praw  # type: ignore


def create_reddit_client(site_name: str = "DEFAULT") -> praw.Reddit:
    """Create a PRAW client.

    Credentials come from ``PRAW_*`` environment variables when set, otherwise
    from the ``site_name`` section of a ``praw.ini`` in the working directory
    or user home.
    """
    # Environment variables take precedence
    if os.getenv("PRAW_CLIENT_ID") and os.getenv("PRAW_CLIENT_SECRET"):
        return praw.Reddit(
            client_id=os.getenv("PRAW_CLIENT_ID"),
            client_secret=os.getenv("PRAW_CLIENT_SECRET"),
            user_agent=os.getenv("PRAW_USER_AGENT", "reddit-cmf"),
            username=os.getenv("PRAW_USERNAME"),
            password=os.getenv("PRAW_PASSWORD"),
        )

    # Fall back to praw.ini
    return praw.Reddit(site_name)


