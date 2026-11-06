"""Generate static, aggregate figures from the clustered posts.

Reads the clustered parquet and the LLM cluster summaries and writes PNG/SVG
figures to ``analysis.figures_dir``. No raw post text, usernames, or post IDs
are drawn; text only enters through aggregate c-TF-IDF term weights.

All figures share one visual system: IBM Plex Sans (bundled in assets/fonts,
SIL OFL 1.1), a left-aligned title + subtitle header, a fixed 10-inch width so
text renders at the same size in the README, and one colour per cluster that
follows the cluster through every chart. Non-cluster marks use ink and grey.
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.transforms import ScaledTranslation, blended_transform_factory  # noqa: E402
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer  # noqa: E402

from .utils import get_logger, load_json, load_parquet, read_yaml  # noqa: E402


logger = get_logger(__name__)


# Chart tokens (light mode).
SURFACE = "#fcfcfb"
PLATE = "#f4f3ef"  # panel background in small multiples
INK = "#141413"  # primary text and the single non-cluster accent
INK_2 = "#52514e"
MUTED = "#8a877f"
GRID = "#e6e4dd"
AXIS = "#c3c2b7"
CONTEXT = "#d6d4cc"  # de-emphasised marks (other posts, other months)
RANGE = "#dddbd3"  # interquartile range bars
# Sequential ramp for shares (one neutral hue, light -> dark). Kept free of the
# cluster hues so it never reads as a cluster identity.
SEQ_RAMP = ["#f4f3ef", "#e2e0d8", "#c6c3b8", "#a19e93", "#77746b", "#4d4b45", "#2a2926"]

# Categorical slots. Cluster -> slot is fixed (colour follows the cluster) and
# chosen so clusters that sit next to each other in UMAP space get well-separated
# hues; every cluster is also direct-labelled, so identity never relies on colour.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
CLUSTER_SLOT = {0: 4, 1: 0, 2: 5, 3: 6, 4: 2, 5: 3, 6: 7, 7: 1}

# Layout (inches). Every README figure is FIG_W wide so text scales identically.
FONT_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"
FONT_FAMILY = "IBM Plex Sans"
FIG_W = 10.0
PAD = 0.42  # outer margin
LABEL_COL = 3.62  # cluster-label column (chip + id + title) left of row charts
ROW_H = 0.40  # height per cluster row
DPI = 200

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


# --------------------------------------------------------------------------- #
# Style and layout helpers
# --------------------------------------------------------------------------- #

def setup_style() -> None:
    families = []
    for ttf in sorted(FONT_DIR.glob("*.ttf")):
        font_manager.fontManager.addfont(str(ttf))
    if any(f.name == FONT_FAMILY for f in font_manager.fontManager.ttflist):
        families.append(FONT_FAMILY)
    else:
        logger.warning("Bundled font not found in %s; falling back to system fonts", FONT_DIR)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": families + ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "axes.labelsize": 9.5,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.facecolor": SURFACE,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": AXIS,
            "ytick.color": AXIS,
            "xtick.labelcolor": MUTED,
            "ytick.labelcolor": MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "xtick.major.pad": 6,
            "ytick.major.pad": 6,
            "figure.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "svg.fonttype": "path",  # keep the bundled typeface in SVGs viewed anywhere
        }
    )


def save(fig: plt.Figure, fig_dir: Path, name: str, svg: bool = True) -> None:
    png = fig_dir / f"{name}.png"
    fig.savefig(png, dpi=DPI)
    if svg:
        fig.savefig(fig_dir / f"{name}.svg")
    plt.close(fig)
    logger.info("Saved %s", png)


def cluster_color(c: int) -> str:
    return PALETTE[CLUSTER_SLOT.get(c, c) % len(PALETTE)]


def _wrap(text: str, width: int) -> List[str]:
    """Wrap each paragraph separately so explicit line breaks are kept."""
    return [line for para in text.split("\n") for line in textwrap.wrap(para, width)]


def header_height(subtitle: str | None = None, wrap: int = 132) -> float:
    """Inches from the top edge to where content may start below the header."""
    n = len(_wrap(subtitle, wrap)) if subtitle else 0
    return (PAD + 0.34 + 0.205 * n if n else PAD + 0.30) + 0.32


def header(fig: plt.Figure, title: str, subtitle: str | None = None, wrap: int = 132) -> float:
    """Left-aligned title + subtitle at the top of the figure; returns header_height()."""
    W, H = fig.get_size_inches()
    fig.text(PAD / W, 1 - PAD / H, title, fontsize=15, fontweight="semibold", color=INK,
             ha="left", va="top")
    if subtitle:
        fig.text(PAD / W, 1 - (PAD + 0.34) / H, "\n".join(_wrap(subtitle, wrap)), fontsize=10,
                 color=INK_2, ha="left", va="top", linespacing=1.45)
    return header_height(subtitle, wrap)


def place(fig: plt.Figure, left: float, top: float, width: float, height: float) -> plt.Axes:
    """Add axes positioned in inches from the figure's top-left corner."""
    W, H = fig.get_size_inches()
    return fig.add_axes([left / W, 1 - (top + height) / H, width / W, height / H])


def _x_inches(ax: plt.Axes, dx: float):
    """Transform: x = axes left edge + dx inches, y = data."""
    return blended_transform_factory(ax.transAxes, ax.transData) + ScaledTranslation(
        dx, 0, ax.figure.dpi_scale_trans)


def cluster_row_labels(ax: plt.Axes, rows: List[tuple[float, int]], tbl: pd.DataFrame,
                       col: float = LABEL_COL) -> None:
    """Colour chip + cluster id + LLM title in a left column, aligned with data rows."""
    for y, c in rows:
        ax.plot([0], [y], marker="o", ms=7.5, color=cluster_color(c), mec=SURFACE, mew=0,
                transform=_x_inches(ax, -col + 0.06), clip_on=False)
        ax.text(0, y, f"C{c}", transform=_x_inches(ax, -col + 0.20), fontsize=10,
                fontweight="semibold", color=INK, ha="left", va="center_baseline")
        ax.text(0, y, tbl.loc[c, "title"], transform=_x_inches(ax, -col + 0.52), fontsize=10,
                color=INK_2, ha="left", va="center_baseline")
    ax.set_yticks([])


def panel_title(fig: plt.Figure, left: float, top: float, c: int, tbl: pd.DataFrame,
                stats: bool = True, wrap: int = 30) -> None:
    """Small-multiple header: chip + id on line one with the title, stats below."""
    W, H = fig.get_size_inches()
    title = textwrap.wrap(tbl.loc[c, "title"], wrap)
    # Pad one-line titles so every panel's plot starts at the same height.
    title += [""] * (2 - len(title))
    base = top + 0.13  # first baseline, inches from the top
    fig.lines.append(plt.Line2D([(left + 0.06) / W], [1 - (base - 0.045) / H], marker="o", ms=7,
                                color=cluster_color(c), transform=fig.transFigure, figure=fig))
    fig.text((left + 0.19) / W, 1 - base / H, f"C{c}", fontsize=9.5, fontweight="semibold",
             color=INK, ha="left", va="baseline")
    for i, line in enumerate(title):
        fig.text((left + 0.48) / W, 1 - (base + 0.18 * i) / H, line, fontsize=9.5, color=INK,
                 ha="left", va="baseline")
    if stats:
        fig.text((left + 0.48) / W, 1 - (base + 0.40) / H,
                 f"{tbl.loc[c, 'posts']} posts · {tbl.loc[c, 'share']:.0%}",
                 fontsize=8.5, color=MUTED, ha="left", va="baseline")


def clean_axes(ax: plt.Axes, keep: tuple[str, ...] = ()) -> None:
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def plot_hero(df: pd.DataFrame, tbl: pd.DataFrame, fig_dir: Path, n_months: int) -> None:
    """16:9 overview card: UMAP map with every cluster direct-labelled.

    Rendered at 1600 x 900 px; the title is sized to stay legible as a ~400 px thumbnail.
    """
    W, H = 16.0, 9.0
    m = 0.62  # outer margin, inches
    fig = plt.figure(figsize=(W, H), dpi=100)
    fig.text(m / W, 1 - 0.50 / H, "r/IndustrialDesign  ·  topic map", fontsize=15,
             fontweight="semibold", color=MUTED, ha="left", va="top")
    fig.text(m / W, 1 - 0.84 / H, "What do industrial designers\ntalk about on Reddit?",
             fontsize=44, fontweight="semibold", color=INK, ha="left", va="top", linespacing=1.04)
    fig.text(m / W, 1 - 2.42 / H,
             f"{len(df):,} top posts ({n_months}-month window), embedded with Qwen3-Embedding-0.6B, "
             f"projected with UMAP\nand grouped into {len(tbl)} KMeans clusters. "
             "Cluster titles were written by an LLM from each cluster's top posts.",
             fontsize=14, color=INK_2, ha="left", va="top", linespacing=1.45)

    ax = fig.add_axes([0.02, 0.015, 0.96, 1 - 3.30 / H - 0.015])
    for c in tbl.index:
        g = df[df["cluster"] == c]
        ax.scatter(g["umap_x"], g["umap_y"], s=60, color=cluster_color(c), alpha=0.92,
                   edgecolors=SURFACE, linewidths=1.3, zorder=2)
    ax.set_axis_off()
    # Fixed limits leave empty bands around the point cloud for the labels.
    ax.set_xlim(-5.0, 4.7)
    ax.set_ylim(-2.45, 3.95)

    halo = [pe.withStroke(linewidth=5, foreground=SURFACE)]
    # Label anchors (UMAP data units), placed by hand in empty space around the cloud.
    # The title sits above the anchor, the stats line below it.
    anchors = {0: (-4.0, -0.55), 1: (-2.75, -1.75), 2: (3.05, 2.65), 3: (3.65, -1.0),
               4: (-0.55, 3.45), 5: (-2.75, 3.45), 6: (1.45, 3.45), 7: (2.25, -2.0)}
    for c, (x, y) in medoid_positions(df).items():
        tx, ty = anchors.get(c, (x, y + 0.8))
        # Leader: from the medoid to the edge of the label block (above or below it).
        ly = ty - 0.33 if ty > y else ty + 0.62
        ax.plot([x, tx], [y, ly], color=INK_2, lw=0.9, zorder=3, solid_capstyle="round")
        ax.scatter([x], [y], s=80, color=cluster_color(c), edgecolors=INK, linewidths=1.5, zorder=4)
        title = "\n".join(textwrap.wrap(tbl.loc[c, "title"], 26))
        ax.annotate(title, (tx, ty), xytext=(0, 2), textcoords="offset points", ha="center",
                    va="bottom", fontsize=15.5, fontweight="semibold", color=INK, linespacing=1.08,
                    path_effects=halo, zorder=5)
        ax.annotate(f"C{c}  ·  {tbl.loc[c, 'posts']} posts  ·  {tbl.loc[c, 'share']:.0%}", (tx, ty),
                    xytext=(0, -3), textcoords="offset points", ha="center", va="top",
                    fontsize=12.5, color=INK_2, path_effects=halo, zorder=5)
    fig.savefig(fig_dir / "hero.png", dpi=100)
    plt.close(fig)
    logger.info("Saved %s", fig_dir / "hero.png")


def plot_umap_facets(df: pd.DataFrame, tbl: pd.DataFrame, fig_dir: Path) -> None:
    """Small multiples: each panel highlights one cluster against all posts."""
    order = list(tbl.index)
    cols, gap_x, gap_y = 4, 0.26, 0.34
    pw = (FIG_W - 2 * PAD - (cols - 1) * gap_x) / cols
    ph, head = pw * 0.72, 0.62
    title = "Where each cluster sits on the map"
    sub = ("UMAP projection of the post embeddings. Coloured: posts in the cluster; grey: all other "
           "posts. Panels ordered by cluster size.")
    H = header_height(sub) + 2 * (head + ph) + gap_y + PAD
    fig = plt.figure(figsize=(FIG_W, H))
    top = header(fig, title, sub)
    xlim = (df["umap_x"].min(), df["umap_x"].max())
    ylim = (df["umap_y"].min(), df["umap_y"].max())
    px, py = 0.06 * (xlim[1] - xlim[0]), 0.08 * (ylim[1] - ylim[0])
    for i, c in enumerate(order):
        r, k = divmod(i, cols)
        left = PAD + k * (pw + gap_x)
        ptop = top + r * (head + ph + gap_y)
        panel_title(fig, left, ptop, c, tbl)
        ax = place(fig, left, ptop + head, pw, ph)
        ax.set_facecolor(PLATE)
        clean_axes(ax)
        rest = df[df["cluster"] != c]
        g = df[df["cluster"] == c]
        ax.scatter(rest["umap_x"], rest["umap_y"], s=5, color=CONTEXT, linewidths=0, zorder=1)
        ax.scatter(g["umap_x"], g["umap_y"], s=13, color=cluster_color(c), edgecolors=PLATE,
                   linewidths=0.5, zorder=2)
        ax.set_xlim(xlim[0] - px, xlim[1] + px)
        ax.set_ylim(ylim[0] - py, ylim[1] + py)
        ax.set_xticks([])
        ax.set_yticks([])
    save(fig, fig_dir, "umap_clusters")


def plot_cluster_sizes(tbl: pd.DataFrame, fig_dir: Path) -> None:
    n = len(tbl)
    title, sub = ("Cluster sizes",
                  f"Posts per KMeans cluster (n = {int(tbl['posts'].sum())}). Each cluster keeps "
                  "its colour in every figure.")
    H = header_height(sub) + n * ROW_H + PAD * 0.8
    fig = plt.figure(figsize=(FIG_W, H))
    top = header(fig, title, sub)
    left = PAD + LABEL_COL
    ax = place(fig, left, top, FIG_W - left - PAD - 0.75, n * ROW_H)
    y = np.arange(n)
    ax.barh(y, tbl["posts"], height=0.56, color=[cluster_color(c) for c in tbl.index], zorder=2)
    for yi, (cnt, s) in enumerate(zip(tbl["posts"], tbl["share"])):
        ax.annotate(f"{cnt}", (cnt, yi), xytext=(6, 0), textcoords="offset points",
                    va="center_baseline", fontsize=10, fontweight="semibold", color=INK)
        ax.annotate(f"{s:.0%}", (cnt, yi), xytext=(6 + 7.2 * len(str(cnt)) + 4, 0),
                    textcoords="offset points", va="center_baseline", fontsize=10, color=MUTED)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xlim(0, tbl["posts"].max())
    ax.set_xticks([])
    clean_axes(ax, keep=("left",))
    ax.spines["left"].set_color(AXIS)
    cluster_row_labels(ax, list(zip(y, tbl.index)), tbl)
    save(fig, fig_dir, "cluster_sizes")


def plot_monthly_volume(df: pd.DataFrame, cfg: Dict, fig_dir: Path) -> None:
    start = pd.Timestamp(cfg["data"]["time_window"]["start"])
    end = pd.Timestamp(cfg["data"]["time_window"]["end"])
    months = pd.period_range(start, end - pd.Timedelta(days=1), freq="M")
    counts = (
        df["created"].dt.tz_localize(None).dt.to_period("M").value_counts()
        .reindex(months, fill_value=0)
    )
    peak = int(np.argmax(counts.values))
    pv = int(counts.values[peak])
    share = pv / counts.sum()

    H = 4.3
    fig = plt.figure(figsize=(FIG_W, H))
    top = header(fig, f"{share:.0%} of the sample falls in the final month",
                 f"Posts per month in the collected sample (UTC), {months[0].strftime('%b %Y')} to "
                 f"{months[-1].strftime('%b %Y')}.")
    left = PAD + 0.38
    ax = place(fig, left, top, FIG_W - left - PAD, H - top - PAD - 0.42)
    x = np.arange(len(counts))
    colors = [INK if i == peak else CONTEXT for i in x]
    ax.bar(x, counts.values, width=0.7, color=colors, zorder=2)
    ax.annotate(f"{pv}", (peak, pv), xytext=(0, 4), textcoords="offset points", ha="center",
                va="bottom", fontsize=10, fontweight="semibold", color=INK)
    ax.annotate(f"{months[peak].strftime('%B %Y')}: {pv} posts", (peak - 0.6, pv * 0.80),
                ha="right", va="bottom", fontsize=10, fontweight="semibold", color=INK,
                path_effects=[pe.withStroke(linewidth=4, foreground=SURFACE)])
    ax.annotate("Short-window 'top' listings (month, week)\nfavour recent posts: a sampling "
                "artefact,\nnot community growth.", (peak - 0.6, pv * 0.78), ha="right", va="top",
                fontsize=9.5, color=INK_2, linespacing=1.4,
                path_effects=[pe.withStroke(linewidth=4, foreground=SURFACE)])
    ticks = [i for i, mth in enumerate(counts.index) if mth.month in (1, 4, 7, 10)]
    ax.set_xticks(ticks, [counts.index[i].strftime("%b\n%Y") if counts.index[i].month == 1
                          else counts.index[i].strftime("%b") for i in ticks])
    ax.set_xlim(-0.6, len(counts) - 0.4)
    ax.set_ylim(0, pv * 1.08)
    ax.set_yticks([0, 50, 100, 150])
    ax.yaxis.grid(True, zorder=0)
    clean_axes(ax, keep=("bottom",))
    ax.tick_params(axis="x", colors=INK_2, labelcolor=INK_2)
    ax.set_ylabel("Posts", labelpad=6)
    save(fig, fig_dir, "monthly_volume")


def plot_engagement(df: pd.DataFrame, tbl: pd.DataFrame, fig_dir: Path) -> None:
    """Median with interquartile range, one panel per measure (no dual axis)."""
    order = list(tbl.index)
    n = len(order)
    title = "Engagement per cluster"
    sub = ("Dot = median, bar = interquartile range per post. Score and comments are as of "
           "collection time.")
    head = 0.34
    H = header_height(sub) + head + n * ROW_H + 0.62 + PAD * 0.6
    fig = plt.figure(figsize=(FIG_W, H))
    top = header(fig, title, sub)
    left0 = PAD + LABEL_COL
    gap = 0.4
    pw = (FIG_W - left0 - PAD - gap) / 2
    for k, (col, ptitle, xlabel) in enumerate([
        ("score", "Median score", "Score (upvotes − downvotes)"),
        ("num_comments", "Median comments", "Comments"),
    ]):
        left = left0 + k * (pw + gap)
        ax = place(fig, left, top + head, pw, n * ROW_H)
        fig.text(left / FIG_W, 1 - top / H, ptitle, fontsize=10.5, fontweight="semibold",
                 color=INK, ha="left", va="top")
        for yi, c in enumerate(order):
            v = df.loc[df["cluster"] == c, col]
            q1, med, q3 = v.quantile([0.25, 0.5, 0.75])
            ax.plot([q1, q3], [yi, yi], color=RANGE, lw=4.5, solid_capstyle="round", zorder=1)
            ax.scatter([med], [yi], s=58, color=cluster_color(c), edgecolors=SURFACE,
                       linewidths=1.6, zorder=3)
            ax.annotate(f"{med:g}", (med, yi), xytext=(0, 7.5), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, color=INK_2, zorder=4)
        ax.set_ylim(n - 0.45, -0.75)
        ax.set_xlim(left=0)
        ax.xaxis.grid(True, zorder=0)
        clean_axes(ax, keep=("bottom",))
        ax.spines["bottom"].set_color(AXIS)
        ax.set_xlabel(xlabel, labelpad=6)
        ax.set_yticks([])
        if k == 0:
            cluster_row_labels(ax, list(zip(range(n), order)), tbl)
    save(fig, fig_dir, "engagement")


def plot_terms(terms: Dict[int, List[tuple[str, float]]], tbl: pd.DataFrame, fig_dir: Path) -> None:
    order = list(tbl.index)
    cols, gap_x, gap_y = 4, 0.30, 0.30
    pw = (FIG_W - 2 * PAD - (cols - 1) * gap_x) / cols
    head, term_w = 0.50, 0.80
    n_terms = max(len(v) for v in terms.values())
    ph = n_terms * 0.235
    title = "Most distinctive terms per cluster"
    sub = ("Class-based TF-IDF over post titles and bodies; bar length = weight within the cluster.\n"
           "Generic community words (design, industrial, ...) and terms used by fewer than 3 authors "
           "are excluded.")
    H = header_height(sub) + 2 * (head + ph) + gap_y + PAD * 0.8
    fig = plt.figure(figsize=(FIG_W, H))
    top = header(fig, title, sub)
    for i, c in enumerate(order):
        r, k = divmod(i, cols)
        left = PAD + k * (pw + gap_x)
        ptop = top + r * (head + ph + gap_y)
        panel_title(fig, left, ptop, c, tbl, stats=False)
        ax = place(fig, left + term_w, ptop + head, pw - term_w, ph)
        items = terms.get(c, [])
        y = np.arange(len(items))
        ax.barh(y, [w for _, w in items], height=0.52, color=cluster_color(c), zorder=2)
        for yi, (t, _) in enumerate(items):
            ax.text(0, yi, t, transform=_x_inches(ax, -0.07), ha="right", va="center_baseline", fontsize=9, color=INK_2)
        ax.set_ylim(n_terms - 0.5, -0.5)
        ax.set_xlim(0, max(w for _, w in items) if items else 1)
        ax.set_xticks([])
        ax.set_yticks([])
        clean_axes(ax, keep=("left",))
        ax.spines["left"].set_color(AXIS)
    save(fig, fig_dir, "distinctive_terms")


def plot_flair_mix(df: pd.DataFrame, tbl: pd.DataFrame, fig_dir: Path, top_n: int = 6) -> None:
    """Row-normalised share of each post flair within a cluster."""
    flair = df["flair"].fillna("(none)")
    top = flair.value_counts().index[:top_n]
    flair = flair.where(flair.isin(top), "Other")
    cols = list(top) + (["Other"] if (flair == "Other").any() else [])
    mix = pd.crosstab(df["cluster"], flair, normalize="index").reindex(index=tbl.index, columns=cols, fill_value=0)

    n = len(mix)
    title = "Post flair mix within each cluster"
    sub = ("Share of the cluster's posts carrying each author-chosen flair (flair was not used for "
           "clustering). Values under 5% hidden.")
    head = 0.30
    H = header_height(sub) + head + n * ROW_H * 1.12 + PAD * 0.8
    fig = plt.figure(figsize=(FIG_W, H))
    hdr = header(fig, title, sub)
    left = PAD + LABEL_COL
    ax = place(fig, left, hdr + head, FIG_W - left - PAD, n * ROW_H * 1.12)
    cmap = LinearSegmentedColormap.from_list("seq_ink", SEQ_RAMP)
    ax.imshow(mix.values, cmap=cmap, vmin=0, vmax=0.6, aspect="auto", interpolation="nearest")
    for i in range(mix.shape[0]):
        for j in range(mix.shape[1]):
            v = mix.values[i, j]
            if v >= 0.05:
                ax.text(j, i, f"{v:.0%}", ha="center", va="center_baseline", fontsize=9.5,
                        color="#ffffff" if v >= 0.36 else INK,
                        fontweight="semibold" if v >= 0.36 else "normal")
    ax.set_xticks(range(len(cols)), cols, fontsize=9.5, color=INK_2)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", labelcolor=INK_2, pad=5)
    clean_axes(ax)
    ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
    ax.set_yticks(np.arange(-0.5, n), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2.5)
    ax.tick_params(which="minor", length=0)
    cluster_row_labels(ax, list(zip(range(n), mix.index)), tbl)
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
    ari = diag["seed_ari"]
    title = "How firm are the clusters?"
    sub = (f"No sharp elbow, and silhouette stays near {np.mean(diag['silhouette']):.2f} for every K "
           f"(0 = no separation, 1 = perfect). Re-running K = {k_chosen} with {len(ari)} other seeds "
           f"agrees with the reference run at ARI {min(ari):.2f}-{max(ari):.2f}.")
    H = 4.2
    fig = plt.figure(figsize=(FIG_W, H))
    top = header(fig, title, sub)
    head = 0.34
    left0, gap = PAD + 0.42, 0.75
    pw = (FIG_W - left0 - PAD - gap) / 2
    ph = H - top - head - PAD - 0.42
    for k, (key, ptitle) in enumerate([("inertia", "KMeans inertia (within-cluster sum of squares)"),
                                       ("silhouette", "Silhouette score (cosine)")]):
        left = left0 + k * (pw + gap)
        fig.text((left - 0.42) / FIG_W, 1 - top / H, ptitle, fontsize=10.5, fontweight="semibold",
                 color=INK, ha="left", va="top")
        ax = place(fig, left, top + head, pw, ph)
        v = np.array(diag[key])
        ax.plot(ks, v, color=INK, lw=1.8, zorder=2)
        ax.scatter(ks, v, s=14, color=INK, zorder=3)
        i = int(np.where(ks == k_chosen)[0][0])
        ax.axvline(k_chosen, color=GRID, lw=1.2, zorder=0)
        ax.scatter([k_chosen], [v[i]], s=95, color=SURFACE, edgecolors=INK, linewidths=1.8, zorder=4)
        ax.annotate(f"K = {k_chosen}", (k_chosen, v[i]), xytext=(9, 9), textcoords="offset points",
                    fontsize=9.5, fontweight="semibold", color=INK)
        ax.set_xticks(ks[::2])
        ax.set_xlim(ks[0] - 0.6, ks[-1] + 0.6)
        ax.yaxis.grid(True, zorder=0)
        clean_axes(ax, keep=("bottom",))
        ax.set_xlabel("Number of clusters K", labelpad=6)
    ax.set_ylim(0, max(0.1, max(diag["silhouette"]) * 1.4))
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
