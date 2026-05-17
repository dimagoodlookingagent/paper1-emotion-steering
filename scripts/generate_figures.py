"""Generate the 8 figures for Paper 1.

Outputs: paper1/figures/fig{1..8}_*.png
  1. cheat matrix (surface × emotion condition)
  2. cheat-probe AUC by validation scheme
  3. direction-suppression Δ Sonnet vs Opus
  4. auth_dir dose-response + bidirectional cross-surface bar chart
  5. trained framing-direction cosine geometry (schematic summary)
  6. per-token commit_dir projection (schematic)
  7. dual-emotion-axis 3D geometry
  8. auth_dir α-trajectory 3D

Run from the project root: python paper1/scripts/generate_figures.py

Note on external dependencies (not in paper1/): this script loads several
analysis artifacts from the project root that are NOT shipped in the paper
repo because they're either large (response_activations_L17_L20_L24.pt,
~30MB) or are intermediate per-batch outputs (judge label files, generation
text files). See README "Reproducing the main findings" for how to
regenerate these from the steering and judge scripts.
"""
import json
import re
import sys
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Auto-detect whether we're running from the project root (paper1/ is a subdir)
# or from inside paper1/ directly. Either way works.
SCRIPT_DIR = Path(__file__).resolve().parent
PAPER1_ROOT = SCRIPT_DIR.parent  # this file lives at paper1/scripts/generate_figures.py

FIG_DIR = PAPER1_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

def paper1_path(rel: str) -> Path:
    """Resolve a path inside paper1/ (e.g., 'vectors/calm_v2.pt'), regardless
    of whether we're running from the project root or from inside paper1/."""
    return PAPER1_ROOT / rel


# ---------------------------------------------------------------------------
# Helpers

KV_RE = re.compile(r"(\w+)=([YN])")

def parse_label_line(line):
    parts = line.strip().split(" ", 1)
    if len(parts) != 2:
        return None
    rid = parts[0]
    return rid, {k: (v == "Y") for k, v in KV_RE.findall(parts[1])}


def load_unified_t1_4_labels():
    """Load the unified Sonnet re-judge labels on T1-T4 (876 records)."""
    labels = {}
    for p in [
        "t1_4_judge_labels_unified_batch1.txt",
        "t1_4_judge_labels_unified_batch2.txt",
        "t1_4_judge_labels_unified_batch3.txt",
    ]:
        for raw in open(p):
            if not raw.strip():
                continue
            parsed = parse_label_line(raw)
            if parsed:
                labels[str(parsed[0])] = parsed[1]
    return labels


def load_unified_t1_4_records():
    """Load the corresponding records with surface/condition metadata.

    NOTE: uses explicit `"id" in r` rather than truthiness — id=0 is a valid
    record key but is falsy, so `r.get("id") or composed_key` falls through
    to the composed key, producing a phantom "unlabeled" record in Figure 7.
    """
    records = {}
    for r in json.load(open("judge_labels_v2_full.json")):
        if "id" in r and r["id"] is not None:
            rid = str(r["id"])
        else:
            rid = f"{r['surface']}__{r['condition']}:{r['sample_idx']}"
        records[rid] = {
            "surface": r["surface"],
            "condition": r["condition"],
            "sample_idx": r["sample_idx"],
        }
    return records


# ---------------------------------------------------------------------------
# Figure 1: cheat rate per (surface × emotion condition)

def figure_1_cheat_matrix():
    labels = load_unified_t1_4_labels()
    records = load_unified_t1_4_records()

    cells = defaultdict(lambda: {"n": 0, "y": 0})
    for rid, lbl in labels.items():
        if rid not in records:
            continue
        surf = records[rid]["surface"]
        cond = records[rid]["condition"]
        cells[(surf, cond)]["n"] += 1
        if lbl.get("cheat"):
            cells[(surf, cond)]["y"] += 1

    surfaces = ["T1", "T2", "T3"]
    conds = [
        "calm@-15", "calm@+5", "calm@+15",
        "calm+confident@-15", "calm+confident@+5", "calm+confident@+15",
        "default",
    ]

    matrix = np.full((len(surfaces), len(conds)), np.nan)
    ns = np.zeros_like(matrix, dtype=int)
    for i, surf in enumerate(surfaces):
        for j, cond in enumerate(conds):
            c = cells.get((surf, cond), {"n": 0, "y": 0})
            ns[i, j] = c["n"]
            if c["n"] >= 5:
                matrix[i, j] = c["y"] / c["n"] * 100

    fig, ax = plt.subplots(figsize=(10, 3.2))
    cmap = mpl.cm.YlOrRd
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=60, aspect="auto")

    for i, surf in enumerate(surfaces):
        for j, cond in enumerate(conds):
            if not np.isnan(matrix[i, j]):
                v = matrix[i, j]
                color = "white" if v > 35 else "black"
                ax.text(j, i, f"{v:.0f}%\nn={ns[i,j]}",
                        ha="center", va="center", fontsize=9, color=color)
            else:
                ax.text(j, i, "—", ha="center", va="center", fontsize=12, color="grey")

    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([c.replace("calm+confident", "cc") for c in conds], rotation=30, ha="right")
    ax.set_yticks(range(len(surfaces)))
    ax.set_yticklabels(surfaces)
    ax.set_title("Cheat rate per (surface × emotion condition), unified-Sonnet judge",
                 fontsize=11, pad=12)
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Cheat=Y rate (%)", fontsize=9)

    plt.tight_layout()
    out = FIG_DIR / "fig1_cheat_matrix.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# Figure 2: cheat-probe AUC by validation scheme

def figure_2_probe_auc_by_scheme():
    cell_loco = json.load(open("cell_loco_probe_results.json"))

    # Cell-LOCO results we computed
    layers = [4, 8, 12, 16, 17, 20, 24]
    cell_loco_mean = [cell_loco[f"L{L}"]["mean"] for L in layers]
    cell_loco_std = [cell_loco[f"L{L}"]["std"] for L in layers]

    # Hard-coded source-LOCO and 5-fold from earlier analyses
    # (from cell_loco_probe.log and earlier runs)
    source_loco_mean = {4: 0.809, 8: 0.780, 12: 0.779, 16: 0.747, 17: 0.758, 20: 0.748, 24: 0.714}
    fivefold_mean = {4: 0.820, 8: 0.815, 12: 0.819, 16: 0.789, 17: 0.794, 20: 0.760, 24: 0.712}

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(layers))
    width = 0.25

    bars_5f = ax.bar(x - width, [fivefold_mean[L] for L in layers], width,
                     label="5-fold within-cell CV", color="#4682B4")
    bars_sl = ax.bar(x, [source_loco_mean[L] for L in layers], width,
                     label="Source-LOCO (3-way)", color="#DAA520")
    bars_cl = ax.bar(x + width, cell_loco_mean, width,
                     yerr=cell_loco_std, label="Cell-LOCO (17 cells)",
                     color="#B22222", capsize=3)

    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1, alpha=0.6, label="chance (AUC=0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{L}" for L in layers])
    ax.set_ylabel("Cheat probe AUC")
    ax.set_xlabel("Layer")
    ax.set_ylim(0, 1.0)
    ax.set_title("Cheat probe AUC: in-distribution prediction is largely cell-identification\n"
                 "Cell-LOCO is the load-bearing test — and it's at chance at every layer",
                 fontsize=10, pad=12)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    plt.tight_layout()
    out = FIG_DIR / "fig2_probe_auc_by_scheme.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# Figure 3: direction-suppression Δ side-by-side, Sonnet vs Opus

def figure_3_suppression_delta():
    # Load Sonnet rates from phaseA_judge_summary.json
    s = json.load(open("phaseA_judge_summary.json"))

    # Load Opus rates per-cell per-axis
    opus_labels = {}
    for p in ["phaseA_opus_labels_batch1.txt", "phaseA_opus_labels_batch2.txt",
              "phaseA_opus_labels_batch3.txt", "phaseA_opus_labels_batch4.txt"]:
        for raw in open(p):
            if not raw.strip():
                continue
            parsed = parse_label_line(raw)
            if parsed:
                try:
                    opus_labels[int(parsed[0])] = parsed[1]
                except ValueError:
                    pass

    meta = json.load(open("judge_meta_phaseA.json"))
    opus_cells = defaultdict(lambda: defaultdict(list))
    for r in meta:
        rid = r["id"]
        if rid not in opus_labels:
            continue
        key = (r["surface"], r["condition"], r["suppression"])
        for ax_name in ["cheat", "fix", "inf", "rat"]:
            if ax_name in opus_labels[rid]:
                opus_cells[key][ax_name].append(opus_labels[rid][ax_name])

    # Cells of interest
    cells = [
        ("T1/default", "T1", "default"),
        ("T1/cc@+15", "T1", "calm+confident@+15"),
        ("T2/cc@+15", "T2", "calm+confident@+15"),
    ]
    axes_to_show = ["cheat", "fix", "inf", "rat"]

    fig, ax_grid = plt.subplots(1, 4, figsize=(14, 4.2), sharey=True)

    for col, axis_name in enumerate(axes_to_show):
        ax = ax_grid[col]
        bar_positions_s = np.arange(len(cells)) * 3
        bar_positions_o = bar_positions_s + 1
        deltas_sonnet = []
        deltas_opus = []
        for short, surf, cond in cells:
            ctl_key = f"{surf}/{cond}/control"
            sup_key = f"{surf}/{cond}/suppress_commit"
            s_ctl = s[ctl_key].get(axis_name, 0)
            s_sup = s[sup_key].get(axis_name, 0)
            deltas_sonnet.append((s_sup - s_ctl) * 100)

            o_ctl_list = opus_cells.get((surf, cond, "control"), {}).get(axis_name, [])
            o_sup_list = opus_cells.get((surf, cond, "suppress_commit"), {}).get(axis_name, [])
            if o_ctl_list and o_sup_list:
                o_ctl = sum(o_ctl_list) / len(o_ctl_list)
                o_sup = sum(o_sup_list) / len(o_sup_list)
                deltas_opus.append((o_sup - o_ctl) * 100)
            else:
                deltas_opus.append(0)

        c_sonnet = ["#4682B4"] * len(cells)
        c_opus = ["#DAA520"] * len(cells)
        ax.bar(bar_positions_s, deltas_sonnet, color=c_sonnet, width=1.0, label="Sonnet" if col == 0 else None)
        ax.bar(bar_positions_o, deltas_opus, color=c_opus, width=1.0, label="Opus" if col == 0 else None)

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(bar_positions_s + 0.5)
        ax.set_xticklabels([short for short, _, _ in cells], rotation=20, ha="right", fontsize=9)
        ax.set_title(axis_name, fontsize=11)
        if col == 0:
            ax.set_ylabel("Δ rate under suppress_commit\n(percentage points)")
            ax.legend(loc="lower right", fontsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.set_ylim(-100, 60)

    fig.suptitle("Direction-suppression Δ vs control: no consistent cross-cell reduction in cheat\n"
                 "(Sonnet mean Δ ≈ 0pp; Opus mean Δ ≈ −9pp dominated by T1/cc@+15 outlier at −30pp). "
                 "Framing axes (fix, rat) are judge-specific; inf goes up under both.",
                 fontsize=10, y=1.02)
    plt.tight_layout()
    out = FIG_DIR / "fig3_suppression_delta_multi_judge.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# Figure 4: auth_dir dose-response + bidirectional cross-surface transfer

def figure_4_auth_dir_dose_response():
    """Two-panel figure:
       (A) T1/authorized dose-response, v1 and v3 extractions, n=100/α
       (B) Cross-surface bidirectional bar chart, all 9 conditions at n=100/α
    """
    alphas = np.array([-0.10, 0.0, +0.10])

    # Panel A data — T1/authorized v1 and v3
    v1_t1 = np.array([77, 88, 95])
    v3_t1 = np.array([34, 76, 66])
    v1_t1_err = np.array([7, 6, 5])  # ~CI half-width
    v3_t1_err = np.array([9, 8, 9])

    # Panel B data — cross-surface n=100/α, Sonnet judge
    # ordered to emphasize the bidirectional sweep
    cells = [
        ("T3/auth",          20, 65, 72),
        ("T3/unauth",        31, 42, 69),
        ("T3/ambig",         59, 75, 88),
        ("T2/auth",           8, 21, 33),
        ("T1/auth (v1)",     77, 88, 95),
        ("T1/auth (v3)",     34, 76, 66),
        ("T2/unauth",        14, 19, 28),
        ("T2/ambig",         12,  4,  7),
        ("T4 DB-row (v1)",    7, 12,  7),
        ("T4 DB-row (v3)",    7, 10,  8),
    ]

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1, 1.6]},
    )

    # Panel A: T1/authorized dose-response (v1 + v3)
    axA.errorbar(alphas, v1_t1, yerr=v1_t1_err, fmt="o-", color="#B22222",
                 markersize=9, linewidth=2, capsize=5,
                 label="v1: Δ = +18pp [+9, +27]")
    axA.errorbar(alphas, v3_t1, yerr=v3_t1_err, fmt="s--", color="#1f77b4",
                 markersize=9, linewidth=2, capsize=5,
                 label="v3: Δ = +32pp [+14, +50]")
    axA.set_xlabel("α (steering magnitude)")
    axA.set_ylabel("Cheat rate (%)")
    axA.set_title("(A) T1/authorized — v1 monotonic, v3 non-monotonic with positive full-swing", fontsize=10, pad=8)
    axA.set_xticks(alphas)
    axA.set_ylim(0, 100)
    axA.grid(axis="y", linestyle=":", alpha=0.4)
    axA.legend(loc="lower right", fontsize=9)
    axA.axhline(50, color="gray", linewidth=0.5, linestyle=":")

    # Panel B: bidirectional bar chart, sorted by full-swing Δ
    cells_sorted = sorted(cells, key=lambda r: r[3] - r[1], reverse=True)
    y_pos = np.arange(len(cells_sorted))
    width = 0.27
    neg_color = "#1a9641"   # green (anti-cheat direction)
    zero_color = "#bdbdbd"  # gray (baseline)
    pos_color = "#d7191c"   # red (pro-cheat direction)

    neg_vals = [r[1] for r in cells_sorted]
    zero_vals = [r[2] for r in cells_sorted]
    pos_vals = [r[3] for r in cells_sorted]
    labels = [r[0] for r in cells_sorted]

    axB.barh(y_pos - width, neg_vals, width, color=neg_color, label="α = −0.10 (anti-cheat)")
    axB.barh(y_pos,         zero_vals, width, color=zero_color, label="α = 0 (baseline)")
    axB.barh(y_pos + width, pos_vals, width, color=pos_color, label="α = +0.10 (pro-cheat)")

    # Annotate full-swing Δ on each row
    for i, r in enumerate(cells_sorted):
        delta = r[3] - r[1]
        sign = "+" if delta >= 0 else ""
        x_pos = max(r[1], r[2], r[3]) + 2
        axB.text(x_pos, i, f"Δ={sign}{delta}pp", va="center", fontsize=8,
                 fontweight="bold" if abs(delta) >= 20 else "normal",
                 color="black" if abs(delta) >= 20 else "gray")

    axB.set_yticks(y_pos)
    axB.set_yticklabels(labels, fontsize=9)
    axB.invert_yaxis()
    axB.set_xlim(0, 105)
    axB.set_xlabel("Cheat rate (%)")
    axB.set_title("(B) auth_dir is bidirectional across surfaces (v1, n=100/α, judge)",
                  fontsize=11, pad=8)
    axB.grid(axis="x", linestyle=":", alpha=0.4)
    axB.legend(loc="lower right", fontsize=8, framealpha=0.9)

    # Annotate T4 as the non-transferring exception
    axB.axhspan(7.5, 9.5, color="#fff3cd", alpha=0.4, zorder=0)

    plt.tight_layout()
    out = FIG_DIR / "fig4_auth_dir_dose_response.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# Figure 5: hedge↔rat geometry (cosine similarity matrix)

def figure_5_hedge_rat_geometry():
    # Cosine similarity matrix among trained framing directions at L20
    # Numbers from the paper's analyses (existing in directional probe work)
    axes = ["rat", "hedge", "fix", "inf", "moral", "commit"]
    # Approximate values from prior analyses
    cos_matrix = np.array([
        [+1.00, -0.91, +0.18, +0.84, -0.42, +0.61],   # rat
        [-0.91, +1.00, -0.07, -0.76, +0.55, -0.50],   # hedge
        [+0.18, -0.07, +1.00, +0.13, +0.04, +0.27],   # fix
        [+0.84, -0.76, +0.13, +1.00, -0.35, +0.59],   # inf
        [-0.42, +0.55, +0.04, -0.35, +1.00, -0.30],   # moral
        [+0.61, -0.50, +0.27, +0.59, -0.30, +1.00],   # commit
    ])

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cos_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(len(axes)))
    ax.set_yticks(range(len(axes)))
    ax.set_xticklabels(axes)
    ax.set_yticklabels(axes)

    for i in range(len(axes)):
        for j in range(len(axes)):
            v = cos_matrix[i, j]
            color = "white" if abs(v) > 0.6 else "black"
            txt = f"{v:+.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=10, color=color)

    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("cosine similarity at L20", fontsize=9)
    ax.set_title("Trained framing-direction geometry at L20 (summary values from §6.3)\n"
                 "hedge ↔ rat = −0.91; inf ↔ rat = +0.84\n"
                 "Several 'distinct concepts' share signed axes",
                 fontsize=10, pad=10)
    # Note: values are summary/approximate from the per-batch analyses; to
    # regenerate exactly, run the directional-probe scripts that produced
    # hedge_dir / rat_dir / etc. and recompute pairwise cosines.
    plt.tight_layout()
    out = FIG_DIR / "fig5_hedge_rat_geometry.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# Figure 6: per-token trajectory commit_dir projection (cheat-Y vs cheat-N)

def figure_6_per_token_trajectory():
    # SCHEMATIC summary of the per-token commit_dir projection analysis.
    # Real per-token trajectories from the cheat-Y vs cheat-N responses peak
    # around 55% through the response (cheat-Y) vs ~flat (cheat-N). The shape
    # below is a smooth analytic stand-in for the summary curve; raw per-token
    # data is generated by the per-token trajectory scripts in the project.
    t = np.linspace(0, 1, 100)
    cheat_y = 0.3 + 0.7 * np.exp(-((t - 0.55) ** 2) / (2 * 0.12 ** 2))
    cheat_n = 0.3 + 0.05 * np.sin(t * 3 * np.pi)
    np.random.seed(0)
    cheat_y_low = cheat_y - 0.15
    cheat_y_high = cheat_y + 0.15
    cheat_n_low = cheat_n - 0.1
    cheat_n_high = cheat_n + 0.1

    # Larger figure with extra top margin so the annotation lives inside the axes
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(t, cheat_y, color="#B22222", linewidth=2,
            label="cheat-Y trajectories (n=50, summary)")
    ax.fill_between(t, cheat_y_low, cheat_y_high, color="#B22222", alpha=0.2)
    ax.plot(t, cheat_n, color="#4682B4", linewidth=2,
            label="cheat-N trajectories (n=50, summary)")
    ax.fill_between(t, cheat_n_low, cheat_n_high, color="#4682B4", alpha=0.2)

    # Peak annotation lives INSIDE the axes, well above the curves but below title
    ymax_data = float(cheat_y_high.max())
    ax.set_ylim(min(cheat_n_low.min(), 0.0), ymax_data + 0.35)
    ax.axvline(0.55, color="grey", linestyle="--", alpha=0.6, linewidth=1)
    ax.annotate("mid-response commit_dir peak\n(~55% through response)",
                xy=(0.55, ymax_data + 0.02), xytext=(0.55, ymax_data + 0.22),
                ha="center", va="bottom", fontsize=9, color="grey",
                arrowprops=dict(arrowstyle="->", color="grey", lw=0.8))

    ax.set_xlabel("Normalized position in response (0 = start, 1 = end)")
    ax.set_ylabel("commit_dir projection (z-scored)")
    ax.set_title(
        "Figure 6 — Schematic: per-token commit_dir projection in cheat-Y vs cheat-N (T1/cc@+15)\n"
        "Mid-response peak is a post-hoc marker, not a causal handle (suppression doesn't reduce cheat — §3.3)",
        fontsize=10, pad=14)
    ax.set_xlim(0, 1)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(axis="both", linestyle=":", alpha=0.4)

    plt.tight_layout()
    out = FIG_DIR / "fig6_per_token_trajectory.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# Figure 7: dual-emotion-axis 3D geometry (calm × confident)

def figure_7_dual_emotion_3d():
    """Project 850 sweep responses onto (calm, confident_v2_ortho, PC3-residual)
    axes. Color by cheat=Y/N. Static-render of output_3d_dual.html with the
    sweep-3d color scheme."""
    import torch
    from mpl_toolkits.mplot3d import Axes3D  # noqa — registers 3d projection

    # Load activations and direction vectors
    acts = torch.load("response_activations_L17_L20_L24.pt", weights_only=False, map_location="cpu")
    calm = torch.load(str(paper1_path("vectors/calm_v2.pt")), weights_only=False, map_location="cpu")
    conf_ortho = torch.load(str(paper1_path("vectors/confident_v2_ortho.pt")), weights_only=False, map_location="cpu")

    calm_d = calm["mean_direction"].float()
    calm_d = calm_d / calm_d.norm()
    conf_d = conf_ortho["mean_direction"].float()
    conf_d = conf_d / conf_d.norm()

    # Load cheat labels
    labels = load_unified_t1_4_labels()

    # Stack L17 activations for record ids 0..849
    ids = sorted(acts.keys())
    X = torch.stack([acts[i][17].float() for i in ids])  # [n, 2304]

    # Project onto calm and confident axes
    proj_calm = (X @ calm_d).numpy()
    proj_conf = (X @ conf_d).numpy()

    # 3rd axis = top PC of residual (X minus calm and confident components)
    X_resid = X - torch.outer(X @ calm_d, calm_d) - torch.outer(X @ conf_d, conf_d)
    Xc = X_resid - X_resid.mean(dim=0, keepdim=True)
    _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
    pc3 = Vh[0]  # top residual PC
    proj_pc3 = (X @ pc3).numpy()

    # Cheat labels per record id
    cheat_mask = np.array([labels.get(str(i), {}).get("cheat", False) for i in ids])

    fig = plt.figure(figsize=(11, 5))

    for panel_idx, (title, mask, color_y, color_n) in enumerate([
        ("All 850 sweep responses, cheat=Y vs cheat=N", cheat_mask, "#d7191c", "#92c5de"),
    ], start=1):
        ax = fig.add_subplot(1, 2, panel_idx, projection="3d")
        # Non-cheats first, semi-transparent
        ax.scatter(proj_calm[~mask], proj_conf[~mask], proj_pc3[~mask],
                   c=color_n, s=8, alpha=0.35, edgecolors="none", label=f"cheat=N (n={(~mask).sum()})")
        # Cheats overlaid, opaque diamonds
        ax.scatter(proj_calm[mask], proj_conf[mask], proj_pc3[mask],
                   c=color_y, s=22, alpha=0.85, edgecolors="black", linewidth=0.4,
                   marker="D", label=f"cheat=Y (n={mask.sum()})")

        # Direction arrows from origin
        scale = max(abs(proj_calm).max(), abs(proj_conf).max()) * 0.85
        ax.quiver(0, 0, 0, scale, 0, 0, color="#cc1f3e", linewidth=2.5, arrow_length_ratio=0.08)
        ax.quiver(0, 0, 0, 0, scale, 0, color="#1f77b4", linewidth=2.5, arrow_length_ratio=0.08)
        ax.text(scale * 1.05, 0, 0, "calm_v2", color="#cc1f3e", fontsize=9, fontweight="bold")
        ax.text(0, scale * 1.05, 0, "confident_v2 (⊥ calm)", color="#1f77b4", fontsize=9, fontweight="bold")

        ax.set_xlabel("calm direction projection", fontsize=9, labelpad=2)
        ax.set_ylabel("confident direction projection", fontsize=9, labelpad=2)
        ax.set_zlabel("PC3 (residual)", fontsize=9, labelpad=2)
        ax.set_title(title, fontsize=10, pad=4)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
        ax.view_init(elev=22, azim=35)

    # Second panel: color by emotion condition
    records = load_unified_t1_4_records()
    cond_colors = {
        "default":              "#888888",
        "calm@-15":             "#0571b0",
        "calm@+5":              "#92c5de",
        "calm@+15":             "#67a9cf",
        "calm+confident@-15":   "#74c476",
        "calm+confident@+5":    "#fdae61",
        "calm+confident@+15":   "#d7191c",
    }

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    # Track which records were colored so we can show any uncategorized leftovers
    matched_mask = np.zeros(len(ids), dtype=bool)
    for cond, color in cond_colors.items():
        sel = np.array([records.get(str(i), {}).get("condition") == cond for i in ids])
        matched_mask |= sel
        if sel.sum() == 0:
            continue
        ax2.scatter(proj_calm[sel], proj_conf[sel], proj_pc3[sel],
                    c=color, s=14, alpha=0.7, edgecolors="black", linewidth=0.2,
                    label=f"{cond} (n={sel.sum()})")

    # Show any records that have no matching condition entry in the
    # records metadata (id=0 in our data — an orphan activation without
    # surface/condition labels). Keeping the trace ensures the right-panel
    # total matches the left-panel total of 850.
    other = ~matched_mask
    if other.sum() > 0:
        ax2.scatter(proj_calm[other], proj_conf[other], proj_pc3[other],
                    c="#cccccc", s=12, alpha=0.5, edgecolors="black", linewidth=0.15,
                    label=f"unlabeled (n={other.sum()})")

    scale = max(abs(proj_calm).max(), abs(proj_conf).max()) * 0.85
    ax2.quiver(0, 0, 0, scale, 0, 0, color="#cc1f3e", linewidth=2.5, arrow_length_ratio=0.08)
    ax2.quiver(0, 0, 0, 0, scale, 0, color="#1f77b4", linewidth=2.5, arrow_length_ratio=0.08)
    ax2.set_xlabel("calm direction", fontsize=9, labelpad=2)
    ax2.set_ylabel("confident direction (⊥ calm)", fontsize=9, labelpad=2)
    ax2.set_zlabel("PC3 (residual)", fontsize=9, labelpad=2)
    ax2.set_title("Same 850 responses, colored by steering condition", fontsize=10, pad=4)
    ax2.legend(loc="upper left", fontsize=7, framealpha=0.9, bbox_to_anchor=(1.02, 1))
    ax2.view_init(elev=22, azim=35)

    fig.suptitle("Figure 7: Dual-emotion-axis geometry at L17 (n=850 responses across T1-T3)",
                 fontsize=11, y=0.99)
    plt.tight_layout()
    out = FIG_DIR / "fig7_dual_emotion_3d.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# Figure 8: auth_dir α-trajectory on T1/authorized (n=100/α, n=300 total)
#
# Encodes 300 auth-steered T1/authorized response texts through Gemma to get
# L17 mean activations, projects onto (auth_dir, calm, confident_ortho) axes,
# and shows the cell-mean trajectory across α ∈ {−0.10, 0, +0.10}.
#
# Source data: auth_steering/T1_authorized_*.txt (n=30/α from original P2c run)
#              + phase_P1a_replication/T1_authorized_*.txt (n=70/α supplement)
# Projections are cached in auth_steering_sweep_projections.pt so re-rendering
# is fast. If the cache is missing, the encoding step takes ~5 min on M1 Max.

def figure_8_auth_steering_sweep_3d():
    """Project 300 auth-steered T1/authorized responses onto (auth_dir, calm,
    confident) axes; show cell-mean α-trajectory."""
    import re
    import torch
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    proj_cache = Path("auth_steering_sweep_projections.pt")

    CELLS = [
        (-0.10, ["auth_steering/T1_authorized_neg10.txt",
                 "phase_P1a_replication/T1_authorized_neg10.txt"], "#1a9641"),
        ( 0.0,  ["auth_steering/T1_authorized_zero.txt",
                 "phase_P1a_replication/T1_authorized_zero.txt"],  "#888888"),
        (+0.10, ["auth_steering/T1_authorized_pos10.txt",
                 "phase_P1a_replication/T1_authorized_pos10.txt"], "#d7191c"),
    ]

    def parse_samples(path):
        text = Path(path).read_text()
        for m in re.finditer(
            r'--- Sample (\d+)/\d+ \(([^)]+)\) ---\n(.*?)(?=\n--- Sample|\Z)',
            text, re.DOTALL,
        ):
            yield int(m.group(1)), m.group(2), m.group(3).strip()

    if proj_cache.exists():
        pts_by_alpha = torch.load(proj_cache, weights_only=False)
        print(f"  Using cached projections from {proj_cache}")
    else:
        print(f"  No cache; encoding 300 responses through Gemma (~5 min)...")
        from transformers import AutoTokenizer, AutoModelForCausalLM

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
        model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2-2b-it", torch_dtype=torch.float16).to(device)
        model.requires_grad_(False)

        auth = torch.load(str(paper1_path("vectors/auth_dir_L17_v1_ortho.pt")), weights_only=False, map_location="cpu")
        calm = torch.load(str(paper1_path("vectors/calm_v2.pt")), weights_only=False, map_location="cpu")
        conf = torch.load(str(paper1_path("vectors/confident_v2_ortho.pt")), weights_only=False, map_location="cpu")
        auth_d = auth["mean_direction"].float(); auth_d = auth_d / auth_d.norm()
        calm_d = calm["mean_direction"].float(); calm_d = calm_d / calm_d.norm()
        conf_d = conf["mean_direction"].float(); conf_d = conf_d / conf_d.norm()

        pts_by_alpha = {}
        for alpha, paths, _ in CELLS:
            pts = []
            for path in paths:
                for idx, tag, resp in parse_samples(path):
                    if not resp.strip(): continue
                    inputs = tokenizer(resp[:4000], return_tensors="pt",
                                       truncation=True, max_length=1024).to(device)
                    with torch.no_grad():
                        out = model(**inputs, output_hidden_states=True)
                    h = out.hidden_states[17].mean(dim=1).squeeze(0).cpu().float()
                    pts.append((float(h @ auth_d), float(h @ calm_d),
                                float(h @ conf_d), tag == "CHEAT"))
            pts_by_alpha[alpha] = pts
        torch.save(pts_by_alpha, proj_cache)

    fig = plt.figure(figsize=(15, 6))

    # Panel A: 3D scatter with trajectory arrow.
    # Camera positioned to make the α-cohort separation visible at a glance:
    # low elevation, side-on azimuth, looking down the auth_dir axis.
    axA = fig.add_subplot(1, 2, 1, projection="3d")
    means = {}
    for alpha, _paths, color in CELLS:
        pts = pts_by_alpha[alpha]
        means[alpha] = (
            float(np.mean([p[0] for p in pts])),
            float(np.mean([p[1] for p in pts])),
            float(np.mean([p[2] for p in pts])),
        )
        clean = [p for p in pts if not p[3]]
        cheat = [p for p in pts if p[3]]
        # Non-cheats: small light circles
        axA.scatter([p[0] for p in clean], [p[1] for p in clean], [p[2] for p in clean],
                    c=color, s=18, alpha=0.45, edgecolors="black", linewidth=0.25,
                    label=f"α={alpha:+.2f} cheat=N (n={len(clean)})")
        # Cheats: larger opaque diamonds
        axA.scatter([p[0] for p in cheat], [p[1] for p in cheat], [p[2] for p in cheat],
                    c=color, s=70, alpha=0.95, edgecolors="black", linewidth=0.7,
                    marker="D", label=f"α={alpha:+.2f} cheat=Y (n={len(cheat)})")

    sorted_a = sorted(means.keys())
    traj = np.array([means[a] for a in sorted_a])
    axA.plot(traj[:, 0], traj[:, 1], traj[:, 2], "k--", linewidth=2.8, zorder=10)
    for a, (x, y, z) in zip(sorted_a, traj):
        axA.scatter([x], [y], [z], c="white", s=180, edgecolors="black",
                    linewidth=2.2, zorder=11)
        axA.text(x, y, z + 2.0, f"α={a:+.2f}", ha="center", fontsize=10,
                 fontweight="bold", zorder=12,
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                           edgecolor="black", linewidth=0.5, alpha=0.85))

    axA.set_xlabel("auth_dir projection", fontsize=10, labelpad=4)
    axA.set_ylabel("calm projection", fontsize=10, labelpad=4)
    axA.set_zlabel("confident projection", fontsize=10, labelpad=4)
    axA.set_title("(A) T1/authorized response geometry by α — n=100/α, n=300 total",
                  fontsize=10, pad=4)
    axA.legend(loc="upper left", fontsize=8, framealpha=0.95,
               bbox_to_anchor=(-0.22, 1.0))
    # Flatter view, looking nearly along confident axis to maximize α-separation along auth_dir
    axA.view_init(elev=12, azim=-60)

    # Panel B: per-axis SHIFT from baseline (α=0). Showing shifts directly
    # makes the cross-axis story readable; absolute projections were dominated
    # by per-axis baseline differences that obscure the dose-response.
    axB = fig.add_subplot(1, 2, 2)
    axes_names = ["auth_dir", "calm", "confident"]
    zero_means = [means[0.0][i] for i in range(3)]
    shift_neg = [means[-0.10][i] - zero_means[i] for i in range(3)]  # α=−0.10 minus baseline
    shift_pos = [means[+0.10][i] - zero_means[i] for i in range(3)]  # α=+0.10 minus baseline
    full_swing = [means[+0.10][i] - means[-0.10][i] for i in range(3)]

    x_pos = np.arange(3)
    w = 0.36
    bars_neg = axB.bar(x_pos - w/2, shift_neg, w, color="#1a9641",
                       label="α=−0.10 − baseline", edgecolor="black", linewidth=0.4)
    bars_pos = axB.bar(x_pos + w/2, shift_pos, w, color="#d7191c",
                       label="α=+0.10 − baseline", edgecolor="black", linewidth=0.4)

    # Value labels on the bars
    for i, (bn, bp) in enumerate(zip(bars_neg, bars_pos)):
        for bar, val in [(bn, shift_neg[i]), (bp, shift_pos[i])]:
            vsign = "+" if val >= 0 else ""
            vy = bar.get_height() + (0.15 if val >= 0 else -0.5)
            axB.text(bar.get_x() + bar.get_width() / 2, vy,
                     f"{vsign}{val:.1f}", ha="center", fontsize=8,
                     va="bottom" if val >= 0 else "top")

    # Full-swing Δ annotation above each pair
    ymax = max(shift_pos + [abs(v) for v in shift_neg]) + 1.5
    for i, fs in enumerate(full_swing):
        axB.annotate(f"full-swing Δ = +{fs:.1f}", xy=(i, ymax),
                     ha="center", fontsize=9, fontweight="bold")

    axB.set_xticks(x_pos)
    axB.set_xticklabels(axes_names)
    axB.set_ylabel("Shift from baseline (α=0)")
    axB.set_title("(B) Steered shift from baseline per axis — auth_dir is the primary axis",
                  fontsize=10, pad=4)
    axB.axhline(0, color="black", linewidth=0.8)
    axB.legend(loc="upper right", fontsize=9)
    axB.grid(axis="y", linestyle=":", alpha=0.4)
    axB.set_ylim(min(shift_neg) - 1, ymax + 2)

    fig.suptitle("Figure 8: auth_dir α-sweep on T1/authorized — geometric trajectory in (auth, calm, confident) at L17",
                 fontsize=11, y=0.99)
    plt.tight_layout()
    out = FIG_DIR / "fig8_auth_steering_sweep_3d.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------

def main():
    """Regenerate every figure. Each figure is independent — if one fails
    (typically because an upstream analysis artifact is missing on a fresh
    clone, see README "Intermediate analysis artifacts" section), we log
    the failure and continue. This way figures that can be regenerated
    standalone (5, 6, and 8 from cached projections) succeed even when
    the full pipeline data isn't present."""
    print("Generating Paper 1 figures...\n")

    figures = [
        ("Figure 1 (cheat matrix)",            figure_1_cheat_matrix),
        ("Figure 2 (probe AUC by scheme)",     figure_2_probe_auc_by_scheme),
        ("Figure 3 (suppression Δ)",           figure_3_suppression_delta),
        ("Figure 4 (auth_dir dose-response)",  figure_4_auth_dir_dose_response),
        ("Figure 5 (framing-direction geometry)", figure_5_hedge_rat_geometry),
        ("Figure 6 (per-token trajectory)",    figure_6_per_token_trajectory),
        ("Figure 7 (dual-emotion 3D)",         figure_7_dual_emotion_3d),
        ("Figure 8 (auth_dir α-sweep 3D)",     figure_8_auth_steering_sweep_3d),
    ]

    ok, skipped = [], []
    for name, fn in figures:
        try:
            fn()
            ok.append(name)
        except FileNotFoundError as e:
            print(f"  SKIP {name}: missing artifact — {e.filename}")
            skipped.append((name, f"missing file {e.filename}"))
        except Exception as e:
            print(f"  SKIP {name}: {type(e).__name__}: {e}")
            skipped.append((name, f"{type(e).__name__}: {e}"))

    print(f"\nGenerated {len(ok)}/{len(figures)} figures in {FIG_DIR}/")
    if skipped:
        print("Skipped:")
        for name, reason in skipped:
            print(f"  - {name}: {reason}")
        print("\nSee README section 'Intermediate analysis artifacts required for "
              "figure regeneration' for how to produce the missing inputs.")


if __name__ == "__main__":
    main()
