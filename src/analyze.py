"""Generate static, aggregate figures from the clustered posts.

Reads the clustered parquet and the LLM cluster summaries and writes PNG/SVG
figures to ``analysis.figures_dir``. No raw post text, usernames, or post IDs
are drawn; text only enters through aggregate c-TF-IDF term weights.
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer  # noqa: E402

from .utils import get_logger, load_json, load_parquet, read_yaml  # noqa: E402


logger = get_logger(__name__)


# Chart tokens (light mode).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
CONTEXT = "#d9d8d2"  # de-emphasised points
ACCENT = "#2a78d6"  # single-series colour
SEQ_RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

# Categorical slots. Cluster -> slot is fixed (colour follows the cluster) and
# chosen so clusters that sit next to each other in UMAP space get well-separated
# hues; every cluster is also direct-labelled, so identity never relies on colour.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
CLUSTER_SLOT = {0: 4, 1: 0, 2: 5, 3: 6, 4: 2, 5: 3, 6: 7, 7: 1}

# Words that appear in nearly every post of this community and carry no theme.
DOMAIN_STOP_WORDS = {
    "design", "designs", "designer", "designers", "designing", "industrial", "id",
    "just", "like", "really", "know", "think", "want", "get", "got", "make", "made",
    "does", "did", "don", "doesn", "didn", "isn", "ve", "ll", "re", "im", "i'm",
    "thanks", "thank", "help", "hey", "hi", "guys", "anyone", "people", "thing",
    "things", "lot", "way", "good", "new", "work", "use", "used", "using", "need",
    "looking", "going", "feel", "time", "year", "years", "day", "would", "could",
    "also", "much", "one", "first", "even", "still", "something", "https", "http",
    "www", "com", "reddit", "amp", "x200b", "jpg", "png", "imgur",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": 11,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.facecolor": SURFACE,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlecolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "figure.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "svg.fonttype": "none",
        }
    )


def save(fig: plt.Figure, fig_dir: Path, name: str, svg: bool = True) -> None:
    png = fig_dir / f"{name}.png"
    fig.savefig(png, dpi=200)
    if svg:
        fig.savefig(fig_dir / f"{name}.svg")
    plt.close(fig)
    logger.info("Saved %s", png)


def cluster_color(c: int) -> str:
    return PALETTE[CLUSTER_SLOT.get(c, c) % len(PALETTE)]


def load_data(cfg: Dict) -> tuple[pd.DataFrame, Dict[int, Dict]]:
    df = load_parquet(cfg["data"]["clustered_path"])
    df = df.dropna(subset=["cluster"]).copy()
    df["cluster"] = df["cluster"].astype(int)
    df["created"] = pd.to_datetime(df["created_utc"], unit="s", utc=True)
    raw = load_json(cfg["data"]["summaries_path"])
    summaries = {int(k): v for k, v in raw.items()}
    return df, summaries


def cluster_table(df: pd.DataFrame, summaries: Dict[int, Dict]) -> pd.DataFrame:
    g = df.groupby("cluster")
    tbl = pd.DataFrame(
        {
            "posts": g.size(),
            "median_score": g["score"].median(),
            "median_comments": g["num_comments"].median(),
            "text_post_share": g["is_self"].mean(),
        }
    )
    tbl["share"] = tbl["posts"] / tbl["posts"].sum()
    tbl["title"] = [summaries.get(c, {}).get("title", f"Cluster {c}") for c in tbl.index]
    return tbl.sort_values("posts", ascending=False)


def short_label(c: int, title: str) -> str:
    return f"C{c}  {title}"


def distinctive_terms(df: pd.DataFrame, n_terms: int) -> Dict[int, List[tuple[str, float]]]:
    """Class-based TF-IDF (as in BERTopic): one document per cluster."""
    docs = (
        df.assign(text=df["text"].str.replace(r"https?://\S+|u/\w+|r/\w+", " ", regex=True))
        .groupby("cluster")["text"]
        .apply(lambda s: " ".join(s).lower())
    )
    stop = list(ENGLISH_STOP_WORDS | DOMAIN_STOP_WORDS)
    vec = CountVectorizer(stop_words=stop, token_pattern=r"(?u)\b[a-z][a-z]{2,}\b", min_df=1)
    counts = vec.fit_transform(docs.values).toarray().astype(float)
    # Keep terms written by at least 3 distinct (pseudonymous) authors, so a single
    # poster's brand or project name cannot surface as a "theme".
    by_author = df.groupby("author")["text"].apply(lambda s: " ".join(s).lower())
    presence = CountVectorizer(vocabulary=vec.vocabulary_, binary=True, token_pattern=vec.token_pattern)
    n_authors = np.asarray(presence.fit_transform(by_author.values).sum(axis=0)).ravel()
    counts[:, n_authors < 3] = 0.0

    tf = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
    avg_words = counts.sum() / counts.shape[0]
    idf = np.log(1.0 + avg_words / np.maximum(counts.sum(axis=0), 1.0))
    ctfidf = tf * idf

    vocab = np.array(vec.get_feature_names_out())
    out: Dict[int, List[tuple[str, float]]] = {}
    for row, c in enumerate(docs.index):
        top = np.argsort(ctfidf[row])[::-1][:n_terms]
        out[int(c)] = [(vocab[i], float(ctfidf[row, i])) for i in top if ctfidf[row, i] > 0]
    return out


def medoid_positions(df: pd.DataFrame) -> Dict[int, tuple[float, float]]:
    """2-D medoid of each cluster: an actual post position, used as a label anchor."""
    out = {}
    for c, g in df.groupby("cluster"):
        xy = g[["umap_x", "umap_y"]].to_numpy()
        dist = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=-1)).sum(axis=1)
        out[int(c)] = tuple(xy[int(np.argmin(dist))])
    return out


def plot_hero(df: pd.DataFrame, tbl: pd.DataFrame, fig_dir: Path, n_months: int) -> None:
    """16:9 overview: UMAP map with every cluster direct-labelled."""
    fig = plt.figure(figsize=(16, 9), dpi=100)
    ax = fig.add_axes([0.03, 0.03, 0.94, 0.74])
    for c in tbl.index:
        g = df[df["cluster"] == c]
        ax.scatter(g["umap_x"], g["umap_y"], s=46, color=cluster_color(c), alpha=0.9,
                   edgecolors=SURFACE, linewidths=1.0, zorder=2)
    # Label offsets (in data units) keep labels clear of dense regions and each other.
    offsets = {0: (-0.1, 1.45), 1: (-0.6, -1.85), 2: (-0.6, 1.25), 3: (0.15, -1.25),
               4: (0.45, -0.55), 5: (-1.3, 1.25), 6: (0.9, 0.75), 7: (0.45, -1.55)}
    for c, (x, y) in medoid_positions(df).items():
        dx, dy = offsets.get(c, (0.0, 0.8))
        title = tbl.loc[c, "title"]
        label = "\n".join(textwrap.wrap(title, 26))
        ax.annotate(
            f"{label}\n{tbl.loc[c, 'posts']} posts · {tbl.loc[c, 'share']:.0%}",
            xy=(x, y), xytext=(x + dx, y + dy), ha="center", va="center",
            fontsize=13, color=INK, linespacing=1.15,
            bbox=dict(boxstyle="round,pad=0.35", fc=SURFACE, ec=cluster_color(c), lw=2.0, alpha=0.95),
            arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.8, shrinkA=0, shrinkB=4),
            zorder=4,
        )
    ax.set_axis_off()
    ax.margins(x=0.08, y=0.12)
    fig.text(0.03, 0.93, "What do industrial designers talk about on Reddit?",
             fontsize=30, fontweight="semibold", color=INK, ha="left", va="center")
    fig.text(0.03, 0.85,
             f"{len(df):,} top posts from r/IndustrialDesign ({n_months}-month window), embedded with "
             f"Qwen3-Embedding-0.6B, projected with UMAP\nand grouped into {len(tbl)} KMeans clusters. "
             "Cluster titles were written by an LLM from each cluster's top posts.",
             fontsize=14, color=INK_2, ha="left", va="center")
    fig.savefig(fig_dir / "hero.png", dpi=100)
    plt.close(fig)
    logger.info("Saved %s", fig_dir / "hero.png")


def plot_umap_facets(df: pd.DataFrame, tbl: pd.DataFrame, fig_dir: Path) -> None:
    """Small multiples: each panel highlights one cluster against all posts."""
    order = list(tbl.index)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.4), sharex=True, sharey=True)
    for ax, c in zip(axes.ravel(), order):
        rest = df[df["cluster"] != c]
        g = df[df["cluster"] == c]
        ax.scatter(rest["umap_x"], rest["umap_y"], s=10, color=CONTEXT, linewidths=0, zorder=1)
        ax.scatter(g["umap_x"], g["umap_y"], s=22, color=ACCENT, edgecolors=SURFACE,
                   linewidths=0.6, zorder=2)
        title = "\n".join(textwrap.wrap(f"C{c}  {tbl.loc[c, 'title']}", 34))
        ax.set_title(title, fontsize=11, loc="left", pad=6)
        ax.text(0.02, 0.03, f"{tbl.loc[c, 'posts']} posts · {tbl.loc[c, 'share']:.0%}",
                transform=ax.transAxes, fontsize=10, color=INK_2)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
    fig.suptitle("Where each cluster sits in the UMAP projection of post embeddings",
                 x=0.012, ha="left", fontsize=15, fontweight="semibold")
    fig.text(0.012, 0.925, "Highlighted: posts in the cluster. Grey: all other posts. Panels ordered by cluster size.",
             fontsize=11, color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, fig_dir, "umap_clusters")


def plot_cluster_sizes(tbl: pd.DataFrame, fig_dir: Path) -> None:
    t = tbl.iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    y = np.arange(len(t))
    ax.barh(y, t["posts"], height=0.55, color=ACCENT, zorder=2)
    for yi, (n, s) in enumerate(zip(t["posts"], t["share"])):
        ax.text(n + 1.2, yi, f"{n}  ({s:.0%})", va="center", fontsize=10, color=INK_2)
    ax.set_yticks(y, [short_label(c, ti) for c, ti in zip(t.index, t["title"])])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, t["posts"].max() * 1.15)
    ax.xaxis.grid(True, zorder=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Posts")
    ax.set_title(f"Cluster sizes (n = {int(tbl['posts'].sum())} posts)")
    fig.tight_layout()
    save(fig, fig_dir, "cluster_sizes")


def plot_monthly_volume(df: pd.DataFrame, cfg: Dict, fig_dir: Path) -> None:
    start = pd.Timestamp(cfg["data"]["time_window"]["start"])
    end = pd.Timestamp(cfg["data"]["time_window"]["end"])
    months = pd.period_range(start, end - pd.Timedelta(days=1), freq="M")
    counts = (
        df["created"].dt.tz_localize(None).dt.to_period("M").value_counts()
        .reindex(months, fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(11, 4.4))
    x = np.arange(len(counts))
    ax.bar(x, counts.values, width=0.62, color=ACCENT, zorder=2)
    peak = int(np.argmax(counts.values))
    ax.text(peak, counts.values[peak] + 3, f"{counts.values[peak]}", ha="center", fontsize=10, color=INK_2)
    ax.annotate(
        f"{counts.values[peak] / counts.sum():.0%} of the sample falls in the final month:\n"
        "short-window 'top' listings (month, week) favour recent posts",
        xy=(peak - 0.4, counts.values[peak] * 0.75), xytext=(peak - 1.6, counts.values[peak] * 0.75),
        fontsize=10, color=INK_2, va="center", ha="right",
        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8),
    )
    ticks = [i for i, m in enumerate(counts.index) if m.month in (1, 4, 7, 10)]
    ax.set_xticks(ticks, [counts.index[i].strftime("%b %Y") for i in ticks])
    ax.tick_params(axis="x", length=0)
    ax.yaxis.grid(True, zorder=0)
    ax.set_ylabel("Posts per month")
    ax.set_title("Monthly volume of collected posts (UTC)")
    fig.tight_layout()
    save(fig, fig_dir, "monthly_volume")


def plot_engagement(df: pd.DataFrame, tbl: pd.DataFrame, fig_dir: Path) -> None:
    """Median with interquartile range, one panel per measure (no dual axis)."""
    order = list(tbl.index)[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    for ax, col, label in zip(axes, ["score", "num_comments"], ["Score (upvotes - downvotes)", "Comments"]):
        for yi, c in enumerate(order):
            v = df.loc[df["cluster"] == c, col]
            q1, med, q3 = v.quantile([0.25, 0.5, 0.75])
            ax.plot([q1, q3], [yi, yi], color=AXIS, lw=2, solid_capstyle="round", zorder=1)
            ax.scatter([med], [yi], s=64, color=ACCENT, edgecolors=SURFACE, linewidths=2, zorder=2)
            ax.text(med, yi + 0.3, f"{med:g}", ha="center", fontsize=9.5, color=INK_2)
        ax.xaxis.grid(True, zorder=0)
        ax.set_xlim(left=0)
        ax.set_ylim(-0.6, len(order) - 0.2)  # headroom for the top value label
        ax.set_xlabel(label)
        ax.set_title(f"Median {label.split(' ')[0].lower()} per post")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(order)), [short_label(c, tbl.loc[c, "title"]) for c in order])
    fig.text(0.012, 0.015, "Dot = median, bar = interquartile range. Score and comments are as of collection time.",
             fontsize=9.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, fig_dir, "engagement")


def plot_terms(terms: Dict[int, List[tuple[str, float]]], tbl: pd.DataFrame, fig_dir: Path) -> None:
    order = list(tbl.index)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.6))
    for ax, c in zip(axes.ravel(), order):
        items = terms.get(c, [])[::-1]
        y = np.arange(len(items))
        vals = [w for _, w in items]
        ax.barh(y, vals, height=0.55, color=ACCENT, zorder=2)
        ax.set_yticks(y, [t for t, _ in items], fontsize=10.5)
        ax.tick_params(axis="y", length=0)
        ax.set_xticks([])
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        title = "\n".join(textwrap.wrap(f"C{c}  {tbl.loc[c, 'title']}", 32))
        ax.set_title(title, fontsize=11, pad=6)
    fig.suptitle("Most distinctive terms per cluster (class-based TF-IDF over post titles and bodies)",
                 x=0.012, ha="left", fontsize=15, fontweight="semibold")
    fig.text(0.012, 0.925, "Bar length = c-TF-IDF weight within the cluster. Generic community words "
             "(design, industrial, ...) and terms used by fewer than 3 authors are excluded.",
             fontsize=11, color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    save(fig, fig_dir, "distinctive_terms")


def plot_flair_mix(df: pd.DataFrame, tbl: pd.DataFrame, fig_dir: Path, top_n: int = 6) -> None:
    """Row-normalised share of each post flair within a cluster."""
    flair = df["flair"].fillna("(none)")
    top = flair.value_counts().index[:top_n]
    flair = flair.where(flair.isin(top), "Other")
    cols = list(top) + (["Other"] if (flair == "Other").any() else [])
    mix = pd.crosstab(df["cluster"], flair, normalize="index").reindex(index=tbl.index, columns=cols, fill_value=0)

    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_RAMP)
    fig, ax = plt.subplots(figsize=(12, 5.4))
    ax.imshow(mix.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(mix.shape[0]):
        for j in range(mix.shape[1]):
            v = mix.values[i, j]
            if v >= 0.05:
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=10,
                        color="#ffffff" if v >= 0.5 else INK)
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(mix)), [short_label(c, tbl.loc[c, "title"]) for c in mix.index])
    ax.tick_params(length=0)
    ax.xaxis.tick_top()
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(mix)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.set_title("Post flair mix within each cluster (share of the cluster's posts; values under 5% hidden)",
                 pad=30)
    fig.tight_layout()
    save(fig, fig_dir, "flair_mix")


def cluster_diagnostics(cfg: Dict, k_chosen: int, n_seeds: int = 5) -> Dict:
    """Refit KMeans over the configured K range to check how clear the structure is.

    Returns inertia and cosine silhouette per K, and the adjusted Rand index
    between the chosen-K solution under the configured seed and other seeds.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    emb = np.load(cfg["embedding"]["output_npz"], allow_pickle=True)["embeddings"]
    k_min, k_max = int(cfg["kmeans"]["k_min"]), int(cfg["kmeans"]["k_max"])
    seed = int(cfg["kmeans"]["random_state"])
    ks = list(range(k_min, k_max + 1))
    inertia, silhouette = [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init="auto", random_state=seed).fit(emb)
        inertia.append(float(km.inertia_))
        silhouette.append(float(silhouette_score(emb, km.labels_, metric="cosine")))
    runs = [KMeans(n_clusters=k_chosen, n_init="auto", random_state=seed + i).fit_predict(emb)
            for i in range(n_seeds)]
    ari = [float(adjusted_rand_score(runs[0], r)) for r in runs[1:]]
    return {"k": ks, "inertia": inertia, "silhouette": silhouette, "seed_ari": ari}


def plot_k_diagnostics(diag: Dict, k_chosen: int, fig_dir: Path) -> None:
    ks = np.array(diag["k"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, key, label in zip(axes, ["inertia", "silhouette"],
                              ["KMeans inertia (within-cluster sum of squares)", "Silhouette score (cosine)"]):
        v = np.array(diag[key])
        ax.plot(ks, v, color=ACCENT, lw=2, marker="o", ms=4, zorder=2)
        i = int(np.where(ks == k_chosen)[0][0])
        ax.scatter([k_chosen], [v[i]], s=90, facecolors="none", edgecolors=INK, linewidths=1.5, zorder=3)
        ax.annotate(f"K = {k_chosen}", (k_chosen, v[i]), xytext=(8, 10), textcoords="offset points",
                    fontsize=10, color=INK)
        ax.set_xticks(ks[::2])
        ax.yaxis.grid(True, zorder=0)
        ax.set_xlabel("Number of clusters K")
        ax.set_title(label)
    axes[1].set_ylim(0, max(0.1, max(diag["silhouette"]) * 1.4))
    ari = diag["seed_ari"]
    fig.text(0.012, 0.015,
             f"No sharp elbow, and silhouette stays near {np.mean(diag['silhouette']):.2f} for every K "
             f"(0 = no separation, 1 = perfect). Re-running K = {k_chosen} with {len(ari)} other seeds "
             f"agrees with the reference run at ARI {min(ari):.2f}-{max(ari):.2f}.",
             fontsize=9.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, fig_dir, "k_selection")


def build_figures(cfg: Dict) -> None:
    setup_style()
    fig_dir = Path(cfg["analysis"]["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    df, summaries = load_data(cfg)
    tbl = cluster_table(df, summaries)
    start = pd.Timestamp(cfg["data"]["time_window"]["start"])
    end = pd.Timestamp(cfg["data"]["time_window"]["end"])
    n_months = (end.year - start.year) * 12 + (end.month - start.month)

    logger.info(
        "%d posts, %d clusters, first post %s, last post %s",
        len(df), len(tbl), df["created"].min().date(), df["created"].max().date(),
    )

    terms = distinctive_terms(df, int(cfg["analysis"].get("top_terms", 8)))

    plot_hero(df, tbl, fig_dir, n_months)
    plot_umap_facets(df, tbl, fig_dir)
    plot_cluster_sizes(tbl, fig_dir)
    plot_monthly_volume(df, cfg, fig_dir)
    plot_engagement(df, tbl, fig_dir)
    plot_terms(terms, tbl, fig_dir)
    plot_flair_mix(df, tbl, fig_dir)

    diag = cluster_diagnostics(cfg, k_chosen=len(tbl))
    plot_k_diagnostics(diag, len(tbl), fig_dir)
    logger.info("Silhouette by K: %s", dict(zip(diag["k"], np.round(diag["silhouette"], 3))))
    logger.info("Seed ARI at K=%d: %s", len(tbl), np.round(diag["seed_ari"], 2))

    # Aggregate table for the README (gitignored; no post-level data).
    out = tbl.assign(top_terms=[", ".join(t for t, _ in terms.get(c, [])) for c in tbl.index])
    out.to_csv(fig_dir / "cluster_table.csv", index_label="cluster")
    logger.info("Saved %s", fig_dir / "cluster_table.csv")


def main() -> None:
    args = parse_args()
    cfg = read_yaml(args.config)
    build_figures(cfg)


if __name__ == "__main__":
    main()
