"""
Compute (node_label, edge_label, node_label) triplet distributions per (model, dataset),
project to 2D via MDS using Jensen-Shannon divergence, and plot with color=model, shape=dataset.
Also compares model pairs via Mann-Whitney U p-value and Jensen-Shannon divergence heatmap.
"""

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from scipy.special import rel_entr
from scipy.stats import mannwhitneyu

plt.rcParams["svg.fonttype"] = "none"

DATA_DIR = "data/v1_llm_gemini-3.1-pro-preview"
DEFAULT_SEABENCH_ROOT = Path(__file__).resolve().parents[1] / "out" / "reasoningflow_original"
SEABENCH_EXCLUDED_SAFETY_LABELS = {"evolved_content_reference"}


def seabench_triplets(path, representation):
    """Load one SEABench graph as metadata plus a typed-triplet counter."""
    with open(path, encoding="utf-8") as handle:
        graph = json.load(handle)
    metadata = graph.get("metadata") or {}
    nodes = {node["id"]: node for node in graph.get("nodes", [])}
    counts = Counter()

    for edge in graph.get("edges", []):
        source = nodes.get(edge.get("source"))
        target = nodes.get(edge.get("target"))
        if source is None or target is None:
            continue
        if representation == "reasoningflow":
            # Match the SEABench analysis convention: task-prompt context is not
            # generated reasoning and is excluded from the triplet distribution.
            if source.get("source") != "response" or target.get("source") != "response":
                continue
            source_label = source.get("label")
            target_label = target.get("label")
            relation = edge.get("relation")
        elif representation == "safety_only":
            source_label = source.get("safety_label")
            target_label = target.get("safety_label")
            if (source_label in SEABENCH_EXCLUDED_SAFETY_LABELS
                    or target_label in SEABENCH_EXCLUDED_SAFETY_LABELS):
                continue
            relation_path = edge.get("relation_path")
            if isinstance(relation_path, list):
                relation = " → ".join(str(item) for item in relation_path)
            else:
                # Also accept an uncontracted/older safety graph representation.
                relation = edge.get("relation")
        else:
            raise ValueError(f"unknown SEABench representation: {representation}")
        if source_label and target_label and relation:
            counts[(source_label, relation, target_label)] += 1

    required = ("arm", "harmtype")
    missing = [key for key in required if not metadata.get(key)]
    if missing:
        raise ValueError(f"{path}: missing metadata fields {missing}")
    return metadata, counts


def seabench_distribution(counters, vocabulary):
    total = Counter()
    for counter in counters:
        total.update(counter)
    values = np.array([total[item] for item in vocabulary], dtype=float)
    return values / values.sum() if values.sum() else values


def jsd_base2(p, q):
    """Base-2 Jensen-Shannon divergence, bounded by 0 and 1."""
    if not p.any() and not q.any():
        return 0.0
    midpoint = 0.5 * (p + q)
    # scipy.special.rel_entr defines zero terms correctly. Divide nats by ln(2)
    # so results match the existing SEABench arm-comparison analysis.
    return float(0.5 * (rel_entr(p, midpoint).sum() + rel_entr(q, midpoint).sum()) / np.log(2))


def load_seabench_graphs(root, representation):
    subdir = "reasoning_graphs" if representation == "reasoningflow" else "safety_graphs"
    graph_root = root / subdir
    paths = sorted(graph_root.rglob("*.json")) if graph_root.is_dir() else []
    if not paths:
        raise SystemExit(f"No SEABench {representation} graphs found under {graph_root}")
    return [seabench_triplets(path, representation) for path in paths]


def seabench_js_rows(records, representation):
    vocabulary = sorted({item for _, counter in records for item in counter})
    harms = sorted({metadata["harmtype"] for metadata, _ in records})
    rows = []
    for harmtype in harms + ["all"]:
        selected = [
            (metadata, counter) for metadata, counter in records
            if harmtype == "all" or metadata["harmtype"] == harmtype
        ]
        self_counters = [
            counter for metadata, counter in selected
            if metadata["arm"] == "self_evolving"
        ]
        none_counters = [
            counter for metadata, counter in selected
            if metadata["arm"] == "none"
        ]
        if self_counters and none_counters:
            value = jsd_base2(
                seabench_distribution(self_counters, vocabulary),
                seabench_distribution(none_counters, vocabulary),
            )
        else:
            value = float("nan")
        rows.append({
            "representation": representation,
            "harmtype": harmtype,
            "js_divergence": value,
            "n_self": len(self_counters),
            "n_none": len(none_counters),
        })
    return rows, vocabulary


def write_seabench_distributions(path, datasets):
    fields = ("representation", "arm", "harmtype", "source", "relation", "target",
              "count", "probability")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for representation, records, vocabulary in datasets:
            cells = sorted({
                (metadata["arm"], metadata["harmtype"])
                for metadata, _ in records
                if metadata["arm"] in {"self_evolving", "none"}
            })
            for arm, harmtype in cells:
                counters = [
                    counter for metadata, counter in records
                    if metadata["arm"] == arm and metadata["harmtype"] == harmtype
                ]
                total = Counter()
                for counter in counters:
                    total.update(counter)
                denominator = sum(total.values())
                for source, relation, target in vocabulary:
                    count = total[(source, relation, target)]
                    if count:
                        writer.writerow({
                            "representation": representation, "arm": arm,
                            "harmtype": harmtype, "source": source, "relation": relation,
                            "target": target, "count": count,
                            "probability": count / denominator,
                        })


def format_seabench_table(rows, title, column):
    overall = next(row for row in rows if row["harmtype"] == "all")
    value = "N/A" if np.isnan(overall["js_divergence"]) else f"{overall['js_divergence']:.3f}"
    lines = [
        f"Overall {title} JS divergence is {value}. Harm-specific estimates are:",
        "",
        f"| Harm type | {column} | Self / none traces |",
        "|---|---:|---:|",
    ]
    display = {
        "boundary_collapse": "Boundary collapse",
        "guardrail_erosion": "Guardrail erosion",
        "hallucination": "Hallucination",
        "privacy": "Privacy",
        "all": "All",
    }
    for row in rows:
        estimate = ("N/A" if np.isnan(row["js_divergence"])
                    else f"{row['js_divergence']:.3f}")
        name = display.get(row["harmtype"], row["harmtype"].replace("_", " ").title())
        lines.append(f"| {name} | {estimate} | {row['n_self']} / {row['n_none']} |")
    return "\n".join(lines)


def plot_seabench_js(path, rf_rows, safety_rows):
    harm_order = [row["harmtype"] for row in rf_rows]
    labels = [harm.replace("_", " ").title() for harm in harm_order]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    panels = (
        (axes[0], rf_rows, "ReasoningFlow triplets"),
        (axes[1], safety_rows, "Safety-only triplets"),
    )
    for ax, rows, title in panels:
        values = np.array([row["js_divergence"] for row in rows], dtype=float)
        positions = np.arange(len(rows))
        bars = ax.bar(positions, np.nan_to_num(values, nan=0.0), color="#5b7fa6")
        for bar, value in zip(bars, values):
            label = "N/A" if np.isnan(value) else f"{value:.3f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                    label, ha="center", va="bottom", fontsize=8)
            if np.isnan(value):
                bar.set_alpha(0.25)
        ax.set_xticks(positions, labels, rotation=25, ha="right")
        ax.set_ylim(0, 1.0)
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
    axes[0].set_ylabel("Jensen–Shannon divergence (base 2)")
    fig.suptitle("Self-evolving versus none typed-triplet distributions")
    fig.tight_layout()
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


def run_seabench(root, output_dir):
    datasets = []
    all_rows = []
    for representation in ("reasoningflow", "safety_only"):
        records = load_seabench_graphs(root, representation)
        rows, vocabulary = seabench_js_rows(records, representation)
        datasets.append((representation, records, vocabulary))
        all_rows.extend(rows)

    rf_ids = {metadata.get("source_trace") for metadata, _ in datasets[0][1]}
    safety_ids = {metadata.get("source_trace") for metadata, _ in datasets[1][1]}
    if rf_ids != safety_ids:
        only_rf = len(rf_ids - safety_ids)
        only_safety = len(safety_ids - rf_ids)
        raise SystemExit(
            "Reasoning and safety graph sets differ; refusing a misleading comparison "
            f"({only_rf} reasoning-only, {only_safety} safety-only)."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "seabench_triplet_js_divergence.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    distribution_path = output_dir / "seabench_triplet_distributions.csv"
    write_seabench_distributions(distribution_path, datasets)

    split = len(all_rows) // 2
    rf_rows, safety_rows = all_rows[:split], all_rows[split:]
    report_path = output_dir / "seabench_triplet_js_divergence.md"
    report_path.write_text(
        "# SEABench triplet Jensen–Shannon divergence\n\n"
        + format_seabench_table(rf_rows, "ReasoningFlow", "RF JS divergence")
        + "\n\n"
        + format_seabench_table(safety_rows, "safety-only", "Safety-only JS divergence")
        + "\n",
        encoding="utf-8",
    )
    figure_path = output_dir / "seabench_triplet_js_divergence.svg"
    plot_seabench_js(figure_path, rf_rows, safety_rows)

    print(f"Wrote {report_path}")
    print(f"Wrote {figure_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {distribution_path}")


SEABENCH_PCA_ARMS = ("self_evolving", "none")
SEABENCH_PCA_ARM_COLOR = {"self_evolving": "#E42741", "none": "#007A84"}
SEABENCH_PCA_MARKERS = ["o", "s", "^", "D", "P", "*", "X", "v"]


def seabench_model_label(root: Path) -> str:
    """Short model label from a reasoningflow_<model>_downstream_reruns folder name."""
    match = re.fullmatch(r"reasoningflow_(.+)_downstream_reruns", root.name)
    return match.group(1) if match else root.name


def build_seabench_pca_counts(model_roots, representation):
    """Aggregate one triplet Counter per (model, harmtype, arm) across model roots."""
    counts: dict[tuple[str, str, str], Counter] = defaultdict(Counter)
    for model, root in model_roots:
        for metadata, counter in load_seabench_graphs(root, representation):
            arm = metadata.get("arm")
            harmtype = metadata.get("harmtype")
            if arm not in SEABENCH_PCA_ARMS or not harmtype:
                continue
            counts[(model, harmtype, arm)].update(counter)
    return counts


def build_seabench_pca_vectors(counts):
    """Build Hellinger embedding vectors (sqrt(P)) over the shared triplet vocabulary."""
    vocab = sorted({item for counter in counts.values() for item in counter})
    vocab_idx = {item: i for i, item in enumerate(vocab)}
    keys = sorted(counts.keys())
    mat = np.zeros((len(keys), len(vocab)))
    for i, key in enumerate(keys):
        for item, value in counts[key].items():
            mat[i, vocab_idx[item]] = value
        total = mat[i].sum()
        if total > 0:
            mat[i] /= total
    return keys, np.sqrt(mat), vocab


def plot_seabench_pca(keys, coords, path, representation):
    models = sorted({key[0] for key in keys})
    harmtypes = sorted({key[1] for key in keys})
    harm_marker = {h: SEABENCH_PCA_MARKERS[i % len(SEABENCH_PCA_MARKERS)] for i, h in enumerate(harmtypes)}

    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5.5), squeeze=False)
    axes = axes[0]
    xs, ys = coords[:, 0], coords[:, 1]
    pad_x, pad_y = 0.05 * (np.ptp(xs) or 1.0), 0.05 * (np.ptp(ys) or 1.0)
    xlim = (xs.min() - pad_x, xs.max() + pad_x)
    ylim = (ys.min() - pad_y, ys.max() + pad_y)

    for ax, model in zip(axes, models):
        for i, (key_model, harmtype, arm) in enumerate(keys):
            if key_model != model:
                continue
            ax.scatter(
                coords[i, 0], coords[i, 1],
                color=SEABENCH_PCA_ARM_COLOR.get(arm, "gray"),
                marker=harm_marker[harmtype],
                s=140, edgecolors="k", linewidths=0.6, zorder=3,
            )
        ax.set_title(model)
        ax.set_xlabel("PC 1")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.grid(True, linestyle="--", alpha=0.4)
    axes[0].set_ylabel("PC 2")

    arm_handles = [
        mpatches.Patch(color=SEABENCH_PCA_ARM_COLOR.get(arm, "gray"), label=arm)
        for arm in SEABENCH_PCA_ARMS
    ]
    harm_handles = [
        plt.Line2D([0], [0], marker=harm_marker[h], color="w",
                   markerfacecolor="gray", markersize=10, markeredgecolor="k", label=h)
        for h in harmtypes
    ]
    fig.legend(handles=arm_handles, title="Arm", loc="upper left",
               bbox_to_anchor=(1.0, 1.0), borderaxespad=0.5)
    fig.legend(handles=harm_handles, title="Harm type", loc="lower left",
               bbox_to_anchor=(1.0, 0.0), borderaxespad=0.5)
    fig.suptitle(f"PCA of (model, harmtype, arm) triplet distributions — {representation}")
    fig.tight_layout(rect=[0, 0, 0.86, 1])
    fig.savefig(path, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_seabench_pca(model_roots, output_dir, n_top=5):
    output_dir.mkdir(parents=True, exist_ok=True)
    model_tag = "+".join(sorted({model for model, _ in model_roots}))
    for representation in ("reasoningflow", "safety_only"):
        counts = build_seabench_pca_counts(model_roots, representation)
        if not counts:
            print(f"No {representation} (model, harmtype, arm) data found; skipping.")
            continue
        keys, prob_mat, vocab = build_seabench_pca_vectors(counts)
        n_components = min(2, len(keys) - 1) if len(keys) > 1 else 1
        pca = PCA(n_components=max(n_components, 1), random_state=42)
        coords = pca.fit_transform(prob_mat)
        if coords.shape[1] < 2:
            coords = np.pad(coords, ((0, 0), (0, 2 - coords.shape[1])))
        print(f"[{representation}] {len(keys)} (model, harmtype, arm) points, "
              f"explained variance: PC1={pca.explained_variance_ratio_[0]:.1%}"
              + (f", PC2={pca.explained_variance_ratio_[1]:.1%}" if len(pca.explained_variance_ratio_) > 1 else ""))
        print_pca_directions(vocab, pca.components_, n=n_top)

        figure_path = output_dir / f"seabench_triplet_pca_{representation}_{model_tag}.png"
        plot_seabench_pca(keys, coords, figure_path, representation)
        print(f"Wrote {figure_path}")

        coords_path = output_dir / f"seabench_triplet_pca_{representation}_{model_tag}.csv"
        with coords_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("model", "harmtype", "arm", "pc1", "pc2"))
            writer.writeheader()
            for (model, harmtype, arm), row in zip(keys, coords):
                writer.writerow({"model": model, "harmtype": harmtype, "arm": arm,
                                  "pc1": row[0], "pc2": row[1]})
        print(f"Wrote {coords_path}")


def load_triplets():
    """Return dict: (model, dataset) -> Counter of (src_label, edge_label, dst_label)."""
    counts = defaultdict(lambda: defaultdict(int))

    for fname in os.listdir(DATA_DIR):
        if not fname.endswith(".json"):
            continue
        if "bright" in fname:
            continue
        m = re.match(r"(.+)_(\d+)_(.+)\.json$", fname)
        if not m:
            continue
        dataset, _, model = m.group(1), m.group(2), m.group(3)

        with open(os.path.join(DATA_DIR, fname)) as f:
            doc = json.load(f)

        node_label = {n["id"]: n["label"] for n in doc.get("nodes", [])}

        for edge in doc.get("edges", []):
            src = node_label.get(edge["source_node_id"])
            dst = node_label.get(edge["dest_node_id"])
            elabel = edge["label"]
            if src and dst:
                triplet = (src, elabel, dst)
                counts[(model, dataset)][triplet] += 1

    return counts


def build_distributions(counts):
    """Convert raw counts to probability distributions over a shared vocabulary."""
    # Collect all triplet types
    vocab = sorted({t for c in counts.values() for t in c})
    vocab_idx = {t: i for i, t in enumerate(vocab)}

    keys = sorted(counts.keys())
    dists = np.zeros((len(keys), len(vocab)))
    for i, key in enumerate(keys):
        for t, v in counts[key].items():
            dists[i, vocab_idx[t]] += v
        total = dists[i].sum()
        if total > 0:
            dists[i] /= total

    return keys, dists


def jsd(p, q, eps=1e-10):
    """Jensen-Shannon divergence: (KL(p||m) + KL(q||m)) / 2, where m = (p+q)/2."""
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return 0.5 * (rel_entr(p, m).sum() + rel_entr(q, m).sum())


def build_distance_matrix(dists):
    n = len(dists)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = jsd(dists[i], dists[j])
            D[i, j] = D[j, i] = d
    return D


def plot(keys, coords):
    models = sorted({k[0] for k in keys})
    datasets = sorted({k[1] for k in keys})

    model_color = {
        "Qwen2.5-32B-Instruct": "#8ED7D7",
        "QwQ-32B":              "#007A84",
        "DeepSeek-V3":          "#FF8CA1",
        "DeepSeek-R1":          "#E42741",
        "gpt-oss-120b":         "#98C126",
    }

    # matplotlib marker styles
    marker_styles = ["o", "s", "^", "D", "P", "*", "X", "v"]
    dataset_marker = {d: marker_styles[i % len(marker_styles)] for i, d in enumerate(datasets)}

    fig, ax = plt.subplots(figsize=(8,5))

    for i, (model, dataset) in enumerate(keys):
        ax.scatter(
            coords[i, 0],
            coords[i, 1],
            color=model_color[model],
            marker=dataset_marker[dataset],
            s=120,
            edgecolors="k",
            linewidths=0.5,
            zorder=3,
        )

    # Legend: colors for models
    model_handles = [
        mpatches.Patch(color=model_color[m], label=m) for m in models
    ]
    # Legend: shapes for datasets
    dataset_handles = [
        plt.Line2D(
            [0], [0],
            marker=dataset_marker[d],
            color="w",
            markerfacecolor="gray",
            markersize=10,
            markeredgecolor="k",
            label=d,
        )
        for d in datasets
    ]

    legend1 = ax.legend(
        handles=model_handles,
        title="Model",
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
        framealpha=0.9,
    )
    ax.add_artist(legend1)
    ax.legend(
        handles=dataset_handles,
        title="Dataset",
        loc="lower left",
        bbox_to_anchor=(1.02, 0),
        borderaxespad=0,
        framealpha=0.9,
    )

    ax.set_title("MDS projection of (model, dataset) pairs\nby Jensen-Shannon divergence of triplet distributions")
    ax.set_xlabel("MDS dim 1")
    ax.set_ylabel("MDS dim 2")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout(rect=[0, 0, 0.78, 1])
    plt.savefig("plots/triplet_mds.svg")
    print("Saved plots/triplet_mds.svg")
    plt.show()


def build_prob_vectors(counts):
    """Build Hellinger embedding vectors (sqrt(P)) over the shared triplet vocabulary."""
    vocab = sorted({t for c in counts.values() for t in c})
    vocab_idx = {t: i for i, t in enumerate(vocab)}
    keys = sorted(counts.keys())
    mat = np.zeros((len(keys), len(vocab)))
    for i, key in enumerate(keys):
        for t, v in counts[key].items():
            mat[i, vocab_idx[t]] = v
        total = mat[i].sum()
        if total > 0:
            mat[i] /= total
    return keys, np.sqrt(mat), vocab


def print_pca_directions(vocab, components, n=5):
    """Print top-n triplets by absolute loading for each principal component."""
    for pc_idx, loadings in enumerate(components):
        top = np.argsort(np.abs(loadings))[-n:][::-1]
        print(f"\n  PC{pc_idx + 1} top {n} dimensions (by |loading|):")
        for rank, i in enumerate(top, 1):
            src, edge, dst = vocab[i]
            print(f"    {rank}. {loadings[i]:+.4f}  {src} –[{edge}]– {dst}")


def plot_pca(keys, coords):
    models = sorted({k[0] for k in keys})
    datasets = sorted({k[1] for k in keys})

    model_color = {
        "Qwen2.5-32B-Instruct": "#8ED7D7",
        "QwQ-32B":              "#007A84",
        "DeepSeek-V3":          "#FF8CA1",
        "DeepSeek-R1":          "#E42741",
        "gpt-oss-120b":         "#98C126",
    }
    marker_styles = ["o", "s", "^", "D", "P", "*", "X", "v"]
    dataset_marker = {d: marker_styles[i % len(marker_styles)] for i, d in enumerate(datasets)}

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (model, dataset) in enumerate(keys):
        ax.scatter(
            coords[i, 0], coords[i, 1],
            color=model_color.get(model, "gray"),
            marker=dataset_marker[dataset],
            s=120, edgecolors="k", linewidths=0.5, zorder=3,
        )

    model_handles = [mpatches.Patch(color=model_color.get(m, "gray"), label=m) for m in models]
    dataset_handles = [
        plt.Line2D([0], [0], marker=dataset_marker[d], color="w",
                   markerfacecolor="gray", markersize=10, markeredgecolor="k", label=d)
        for d in datasets
    ]

    legend1 = ax.legend(handles=model_handles, title="Model",
                        loc="upper left", bbox_to_anchor=(1.02, 1),
                        borderaxespad=0, framealpha=0.9)
    ax.add_artist(legend1)
    ax.legend(handles=dataset_handles, title="Dataset",
              loc="lower left", bbox_to_anchor=(1.02, 0),
              borderaxespad=0, framealpha=0.9)

    ax.set_title("PCA of (model, dataset) pairs\nby Hellinger embedding of triplet distributions")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout(rect=[0, 0, 0.78, 1])
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/triplet_pca.svg")
    print("Saved plots/triplet_pca.svg")
    plt.show()


def aggregate_counts_per_model(counts):
    """Aggregate raw triplet counts across all datasets per model."""
    model_counts = defaultdict(lambda: defaultdict(int))
    for (model, dataset), triplet_counts in counts.items():
        for triplet, cnt in triplet_counts.items():
            model_counts[model][triplet] += cnt
    return model_counts


def build_model_count_matrix(model_counts):
    """Build a matrix of raw counts (models x vocab) with shared vocabulary."""
    vocab = sorted({t for mc in model_counts.values() for t in mc})
    vocab_idx = {t: i for i, t in enumerate(vocab)}
    models = sorted(model_counts.keys())

    mat = np.zeros((len(models), len(vocab)))
    for i, model in enumerate(models):
        for t, v in model_counts[model].items():
            mat[i, vocab_idx[t]] += v
    return models, mat


MODEL_ORDER = [
    "Qwen2.5-32B-Instruct",
    "DeepSeek-V3",
    "QwQ-32B",
    "DeepSeek-R1",
    "gpt-oss-120b",
]

SHORT_NAMES = {
    "Qwen2.5-32B-Instruct": "Qwen32B",
    "QwQ-32B":              "QwQ",
    "DeepSeek-V3":          "DS-V3",
    "DeepSeek-R1":          "DS-R1",
    "gpt-oss-120b":         "gpt-oss",
}


def sig_stars(pv):
    if pv < 0.001:
        return "***"
    if pv < 0.01:
        return "**"
    if pv < 0.05:
        return "*"
    return ""


def plot_model_heatmap(models, p_values, jsd_values, show_significance=True):
    """
    Heatmap colored by Jensen-Shannon divergence.
    When show_significance=True, appends Mann-Whitney U stars to each cell and saves
    to triplet_model_heatmap.svg; otherwise saves to triplet_model_heatmap_jsd.svg.
    """
    ordered = [m for m in MODEL_ORDER if m in models]
    idx = [models.index(m) for m in ordered]
    n = len(ordered)

    p_ord = p_values[np.ix_(idx, idx)]
    jsd_ord = jsd_values[np.ix_(idx, idx)]
    labels = [SHORT_NAMES.get(m, m) for m in ordered]

    display = jsd_ord.copy().astype(float)
    np.fill_diagonal(display, np.nan)

    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
        "white_orange", ["#ffffff", "#fd901c"]
    )
    cmap.set_bad("#e8e8e8")

    fig, ax = plt.subplots(figsize=(n * 1.4, n * 1))
    im = ax.imshow(display, cmap=cmap, aspect="auto")

    cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label("Jensen-Shannon divergence", fontsize=10)

    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", fontsize=13, color="#999999")
                continue
            js = jsd_ord[i, j]
            if show_significance:
                label = f"{js:.3f}{sig_stars(p_ord[i, j])}"
            else:
                label = f"{js:.3f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=9, color="black")

    for k in range(n + 1):
        ax.axhline(k - 0.5, color="white", linewidth=1.5)
        ax.axvline(k - 0.5, color="white", linewidth=1.5)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)

    sig_text = "* p<0.05   ** p<0.01   *** p<0.001"
    ax.text(
        1.02, 0.02, sig_text,
        transform=ax.transAxes, fontsize=8.5, va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.8),
    ) # Add this text for JSD-only plot too, to match the layout
        
    if show_significance:
        title_suffix = "\n(Jensen-Shannon divergence, Mann-Whitney U significance)"
        out_path = "plots/triplet_model_heatmap.svg"
    else:
        title_suffix = "\n(Jensen-Shannon divergence)"
        out_path = "plots/triplet_model_heatmap_jsd.svg"

    ax.set_title(
        "Pairwise model comparison of triplet distributions" + title_suffix,
        fontsize=11, pad=30,
    )
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.show()


def run_original():
    print("Loading triplet counts...")
    counts = load_triplets()
    print(f"  Found {len(counts)} (model, dataset) pairs")

    print("Building distributions...")
    keys, dists = build_distributions(counts)

    print("Computing distance matrix...")
    D = build_distance_matrix(dists)

    print("Running MDS...")
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, normalized_stress=False)
    coords = mds.fit_transform(D)

    print("Plotting MDS...")
    plot(keys, coords)

    print("Building Hellinger embedding vectors for PCA...")
    pca_keys, prob_mat, vocab = build_prob_vectors(counts)
    print("Running PCA...")
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(prob_mat)
    print(f"  Explained variance: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%}")
    print_pca_directions(vocab, pca.components_)
    print("Plotting PCA...")
    plot_pca(pca_keys, pca_coords)

    # --- Pairwise model heatmap ---
    print("Aggregating counts per model...")
    model_counts = aggregate_counts_per_model(counts)
    models, count_mat = build_model_count_matrix(model_counts)
    n = len(models)
    print(f"  {n} models: {models}")

    # Normalize to distributions for KL
    dist_mat = count_mat / count_mat.sum(axis=1, keepdims=True).clip(min=1e-10)

    p_values = np.zeros((n, n))
    jsd_values = np.zeros((n, n))

    print("Computing pairwise Mann-Whitney U and Jensen-Shannon divergence...")
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Mann-Whitney U on raw count vectors (one value per triplet type)
            stat, pv = mannwhitneyu(count_mat[i], count_mat[j], alternative="two-sided")
            p_values[i, j] = pv
            jsd_values[i, j] = jsd(dist_mat[i], dist_mat[j])

    print("Plotting model heatmaps...")
    plot_model_heatmap(models, p_values, jsd_values, show_significance=True)
    plot_model_heatmap(models, p_values, jsd_values, show_significance=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seabench", action="store_true",
        help=("Analyze self-evolving versus none triplet distributions in SEABench "
              "reasoning_graphs and safety_graphs."),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help=("Original annotation directory, or with --seabench the pipeline output root "
              "containing reasoning_graphs/ and safety_graphs/ (reports are written under "
              "<output-dir>/analysis)."),
    )
    parser.add_argument(
        "--pca-compare", type=Path, action="append", metavar="ROOT", default=[],
        help=("Additional SEABench output root to fold into the --seabench PCA alongside "
              "--output-dir, labeled by its own folder name; repeat for multiple extra models."),
    )
    args = parser.parse_args()

    if args.seabench:
        root = args.output_dir or DEFAULT_SEABENCH_ROOT
        analysis_dir = root / "analysis"
        run_seabench(root, analysis_dir)
        model_roots = [(seabench_model_label(root), root)] + [
            (seabench_model_label(extra), extra) for extra in args.pca_compare
        ]
        run_seabench_pca(model_roots, analysis_dir)
        return

    global DATA_DIR
    if args.output_dir is not None:
        DATA_DIR = str(args.output_dir)
    run_original()


if __name__ == "__main__":
    main()
