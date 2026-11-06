from __future__ import annotations

import argparse
import json
import re
import os
from typing import Dict, List

import pandas as pd

from .utils import (
    get_logger,
    read_yaml,
    load_parquet,
    save_json,
)

try:  # Optional .env support
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


logger = get_logger(__name__)


PROMPT_TEMPLATE = (
    "You are an expert research assistant. Given representative Reddit posts from a cluster, "
    "identify: (1) Cluster Title (short), (2) 3-4 sentence Description, (3) 5-8 Keywords. "
    "Return JSON with keys: title, description, keywords."
)


FALLBACK_VALUE = {"title": "(skipped)", "description": "(skipped)", "keywords": []}


def _normalize_summary(payload: Dict) -> Dict:
    title = str(payload.get("title", "")).strip()
    description = str(payload.get("description", "")).strip()
    keywords = payload.get("keywords", [])
    if isinstance(keywords, str):
        keywords = [k.strip() for k in re.split(r"[,\n]", keywords) if k.strip()]
    elif not isinstance(keywords, list):
        keywords = []
    else:
        keywords = [str(k).strip() for k in keywords if str(k).strip()]
    return {"title": title, "description": description, "keywords": keywords}


def _parse_summary_text(text: str) -> Dict | None:
    cleaned = text.strip()
    if not cleaned:
        return None

    if cleaned.startswith("```"):
        cleaned = re.sub(r"```(?:json)?", "", cleaned)
        cleaned = cleaned.strip("`").strip()

    try:
        payload = json.loads(cleaned)
        return _normalize_summary(payload)
    except Exception:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start : end + 1]
        try:
            payload = json.loads(candidate)
            return _normalize_summary(payload)
        except Exception:
            pass

    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()


def sample_representatives(df: pd.DataFrame, n: int) -> Dict[int, List[Dict]]:
    """Pick the n highest-scoring posts of each cluster as its representatives."""
    reps: Dict[int, List[Dict]] = {}
    for c in sorted(df["cluster"].dropna().unique()):
        g = df[df["cluster"] == c].sort_values(["score", "num_comments"], ascending=False).head(n)
        reps[int(c)] = g[["id", "title", "selftext", "text"]].to_dict(orient="records")
    return reps


def call_anthropic(cfg: Dict, content: str) -> Dict:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set; skipping Anthropic call.")
        return FALLBACK_VALUE
    try:
        import anthropic  # type: ignore
    except Exception:
        logger.warning("anthropic package not installed; skipping Anthropic call.")
        return FALLBACK_VALUE

    client = anthropic.Anthropic(api_key=api_key)
    model = cfg["llm"].get("model_primary", "claude-3-5-sonnet-20241022")
    msg = client.messages.create(
        model=model,
        max_tokens=800,
        temperature=0.2,
        system=PROMPT_TEMPLATE,
        messages=[{"role": "user", "content": content}],
    )
    # Best-effort JSON parsing of the text blocks
    text = "".join([b.text for b in msg.content if getattr(b, "type", "") == "text"])  # type: ignore
    parsed = _parse_summary_text(text)
    if parsed:
        return parsed
    return {"title": text[:80], "description": text[:400], "keywords": []}


def _parse_responses_text(resp) -> str:
    text_chunks: List[str] = []
    for item in getattr(resp, "output", []):
        for part in getattr(item, "content", []):
            if getattr(part, "type", "") == "output_text":
                text_chunks.append(part.text)
    return "".join(text_chunks)


def _openai_call_once(client, model: str, content: str, include_temperature: bool) -> str:
    kwargs = {
        "model": model,
        "max_output_tokens": 800,
        "input": [
            {"role": "system", "content": PROMPT_TEMPLATE},
            {"role": "user", "content": content},
        ],
    }
    if include_temperature:
        kwargs["temperature"] = 0.2
    resp = client.responses.create(**kwargs)
    return _parse_responses_text(resp)


def call_openai(cfg: Dict, content: str) -> Dict:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set; skipping OpenAI call.")
        return FALLBACK_VALUE
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        logger.warning("openai package not installed; skipping OpenAI call.")
        return FALLBACK_VALUE

    client = OpenAI(api_key=api_key)
    primary = str(cfg["llm"].get("model_openai", "gpt-5"))
    fallbacks = [primary, "gpt-4o", "gpt-4.1-mini", "gpt-4o-mini"]

    last_error = None
    for m in fallbacks:
        for include_temp in (False, True):
            try:
                text = _openai_call_once(client, m, content, include_temp)
                if text:
                    parsed = _parse_summary_text(text)
                    if parsed:
                        return parsed
                    return {"title": text[:80], "description": text[:400], "keywords": []}
            except Exception as exc:
                last_error = exc
                msg = str(exc)
                if "Unsupported parameter" in msg and "temperature" in msg and include_temp:
                    continue  # retry without temperature
                if "unsupported model" in msg.lower() or "model_not_found" in msg.lower():
                    break  # move on to the next model
                continue

    logger.warning("All OpenAI attempts failed: %s", last_error)
    return FALLBACK_VALUE


def call_llm(cfg: Dict, content: str) -> Dict:
    # Try OpenAI first, fall back to Anthropic
    result = call_openai(cfg, content)
    if result["title"] != "(skipped)" or result["description"] != "(skipped)":
        return result
    return call_anthropic(cfg, content)


def sensemaking(cfg: Dict) -> None:
    df = load_parquet(cfg["data"]["clustered_path"])  # includes cluster labels
    sample_n = int(cfg["llm"]["sample_per_cluster"])
    reps = sample_representatives(df, sample_n)

    results = {}
    for c, posts in reps.items():
        content = "\n\n".join([f"TITLE: {p['title']}\nBODY: {p['selftext']}" for p in posts])
        results[str(c)] = call_llm(cfg, content)
    out = cfg["data"]["summaries_path"]
    save_json(results, out)
    logger.info("Saved %s with %d clusters", out, len(results))


def main() -> None:
    args = parse_args()
    cfg = read_yaml(args.config)
    sensemaking(cfg)


if __name__ == "__main__":
    main()


