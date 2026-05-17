"""
Leave-one-cell-out directional validation.

For each held-out cell C:
  1. Train rat/inf/hedge/moral/engagement/commit directions at L20/L24
     using ONLY responses from the OTHER 16 cells.
  2. Project cell C's prompt-state onto the (held-out-naïve) directions.
  3. Predict cell C's downstream rates from those projections.

Then correlate predicted scores vs actual rates across all 17 held-out
predictions. Compare to the in-sample correlations from the
prompt_state_directional_probe.

Reuses cached:
  - response_activations_L17_L20_L24.pt
  - prompt_state_directional_probe.json (for the prompt-state vectors)

Output: loco_directional_results.json + summary table
"""
import json
import math
import random
from pathlib import Path
from collections import defaultdict
import torch


JUDGE_PATH = Path("judge_labels_v2_full.json")
RESP_CACHE = Path("response_activations_L17_L20_L24.pt")
PROBE_PATH = Path("prompt_state_directional_probe.json")
OUT_PATH = Path("loco_directional_results.json")

DIR_LAYERS = [20, 24]
PROMPT_LAYERS = [12, 16, 17, 20, 24]  # we cached at all of these in the original probe
AXES = ["cheat", "moral", "hedge", "rat", "fix", "inf"]


def engagement_level(rec):
    if rec.get("judge_cheat"): return 2
    if any(rec.get(f"judge_{a}") for a in ("rat", "inf", "fix")): return 1
    return 0


def pearson(xs, ys):
    n = len(xs)
    if n < 2: return float("nan")
    mx, my = sum(xs)/n, sum(ys)/n
    sxy = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    sxx = sum((x-mx)**2 for x in xs)
    syy = sum((y-my)**2 for y in ys)
    if sxx <= 0 or syy <= 0: return float("nan")
    return sxy / math.sqrt(sxx * syy)


def bootstrap_ci(xs, ys, n_boot=1000, seed=42):
    rng = random.Random(seed)
    n = len(xs)
    rs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        bxs = [xs[i] for i in idx]; bys = [ys[i] for i in idx]
        r = pearson(bxs, bys)
        if not math.isnan(r): rs.append(r)
    rs.sort()
    if not rs: return (float("nan"), float("nan"))
    return (rs[int(0.025*len(rs))], rs[int(0.975*len(rs))])


def train_directions(resp_acts, judge_index, ids_to_use, layers, label_key,
                     restrict_surfaces=None):
    dirs = {}
    for L in layers:
        pos, neg = [], []
        for rid in ids_to_use:
            rec = judge_index.get(rid)
            if rec is None: continue
            if restrict_surfaces and rec["surface"] not in restrict_surfaces:
                continue
            label = rec.get(f"judge_{label_key}")
            acts = resp_acts.get(rid)
            if acts is None: continue
            if label is True:
                pos.append(acts[L])
            elif label is False:
                neg.append(acts[L])
        if not pos or not neg:
            dirs[L] = None; continue
        d = torch.stack(pos).mean(0) - torch.stack(neg).mean(0)
        d = d / (d.norm() + 1e-12)
        dirs[L] = d
    return dirs


def train_3way(resp_acts, judge_index, ids_to_use, layers, restrict_surfaces=None):
    eng, com = {}, {}
    for L in layers:
        l0, l1, l2 = [], [], []
        for rid in ids_to_use:
            rec = judge_index.get(rid)
            if rec is None: continue
            if restrict_surfaces and rec["surface"] not in restrict_surfaces:
                continue
            acts = resp_acts.get(rid)
            if acts is None: continue
            lev = engagement_level(rec)
            (l0 if lev == 0 else l1 if lev == 1 else l2).append(acts[L])
        if not l0 or not l1 or not l2:
            eng[L] = None; com[L] = None; continue
        m0 = torch.stack(l0).mean(0); m1 = torch.stack(l1).mean(0); m2 = torch.stack(l2).mean(0)
        de = m1 - m0; de = de / (de.norm() + 1e-12)
        dc = m2 - m1; dc = dc / (dc.norm() + 1e-12)
        eng[L] = de; com[L] = dc
    return eng, com


def main():
    # Load resources
    print("Loading judge labels...")
    judge = json.loads(JUDGE_PATH.read_text())
    v2 = [r for r in judge if r.get("source") == "v2"]
    judge_index = {r["id"]: r for r in v2}

    print(f"Loading response activations from {RESP_CACHE}...")
    resp_acts = torch.load(RESP_CACHE, weights_only=False)

    # Build per-cell ids and per-cell rates
    cell_ids = defaultdict(list)
    cell_n = defaultdict(int)
    cell_yes = defaultdict(lambda: defaultdict(int))
    for r in v2:
        k = (r["surface"], r["condition"])
        cell_ids[k].append(r["id"])
        cell_n[k] += 1
        for ax in AXES:
            if r.get(f"judge_{ax}"): cell_yes[k][ax] += 1
    cell_rates = {}
    for k, n in cell_n.items():
        rates = {ax: cell_yes[k][ax] / n for ax in AXES}
        # Add level1/level2 from 3-way decomposition
        l1 = sum(1 for rid in cell_ids[k] if engagement_level(judge_index[rid]) == 1)
        l2 = sum(1 for rid in cell_ids[k] if engagement_level(judge_index[rid]) == 2)
        rates["level1"] = l1 / n
        rates["level2"] = l2 / n
        cell_rates[k] = rates

    # Load prompt-state vectors from existing probe results
    print(f"Loading prompt-states from {PROBE_PATH}...")
    probe = json.loads(PROBE_PATH.read_text())
    # We need raw vectors; the probe JSON only has scalar projections.
    # But response_activations cache + judge_index gives us everything to retrain.
    # For prompt-state projections we need the actual vectors at L20/L24 per cell.
    # Re-import from prompt_state_directional_probe — it doesn't save the vectors.
    # We'll re-capture prompt-state vectors here (cheap, ~17 forward passes).
    # Actually we can reuse the probe's per-cell projection scores ONLY if we
    # have them per (cell × direction). But for LOCO each held-out cell gets
    # different directions, so we need raw vectors.
    print("Re-capturing prompt-state vectors (17 forwards, ~30s)...")

    import sys
    sys.path.insert(0, ".")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import steer
    from steer import PROMPT as PROMPT_T1
    import cross_prompt_cheat as cp2
    import cross_prompt_t3 as cp3

    SURFACES = {"T1": PROMPT_T1, "T2": cp2.PROMPT_T2, "T3": cp3.PROMPT_T3}
    CONDITIONS = [
        ("default",              0.00,  0.00),
        ("calm@+5",             +0.05,  0.00),
        ("calm@+15",            +0.15,  0.00),
        ("calm@-15",            -0.15,  0.00),
        ("calm+confident@+5",   +0.05, +0.05),
        ("calm+confident@+15",  +0.15, +0.15),
        ("calm+confident@-15",  -0.15, -0.15),
    ]
    T3_NAMES = {"default", "calm@+15", "calm+confident@+15"}

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b-it", torch_dtype=torch.float16
    ).to(device)
    model.requires_grad_(False)

    saved_calm = torch.load(Path("vectors/calm_v2.pt"), weights_only=False)
    saved_conf = torch.load(Path("vectors/confident_v2_ortho.pt"), weights_only=False)
    calm_dir = saved_calm["mean_direction"].to(device)
    conf_dir = saved_conf["mean_direction"].to(device)
    target_layer = model.model.layers[saved_calm.get("layer", 17) - 1]
    norm = steer.estimate_residual_norm(model, tokenizer, 17, device)

    def make_hook(a_calm, a_conf):
        s1 = a_calm * norm; s2 = a_conf * norm
        if a_calm == 0 and a_conf == 0: return None
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                hidden = output[0]
                steered = (hidden
                    + s1 * calm_dir.to(hidden.device, dtype=hidden.dtype)
                    + s2 * conf_dir.to(hidden.device, dtype=hidden.dtype))
                return (steered,) + output[1:]
            return (output
                    + s1 * calm_dir.to(output.device, dtype=output.dtype)
                    + s2 * conf_dir.to(output.device, dtype=output.dtype))
        return hook

    prompt_states = {}  # (surface, cond) -> {layer: tensor}
    for surface, prompt_text in SURFACES.items():
        messages = [{"role": "user", "content": prompt_text}]
        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True, return_dict=True,
        ).to(device)
        for cond_name, a_calm, a_conf in CONDITIONS:
            if surface == "T3" and cond_name not in T3_NAMES: continue
            hook_fn = make_hook(a_calm, a_conf)
            handle = target_layer.register_forward_hook(hook_fn) if hook_fn else None
            try:
                with torch.no_grad():
                    o = model(**inputs, output_hidden_states=True)
                states = {L: o.hidden_states[L][0, -1, :].float().cpu()
                          for L in [17, 20, 24]}
            finally:
                if handle is not None: handle.remove()
            prompt_states[(surface, cond_name)] = states
    print(f"  captured {len(prompt_states)} prompt-states")

    # Free model memory — we only need cached activations from now on
    del model
    if device == "mps": torch.mps.empty_cache()

    # ---- Run LOCO ----
    print("\nRunning LOCO directional validation...")
    cell_keys = sorted(prompt_states.keys())
    print(f"  {len(cell_keys)} cells to hold out\n")

    held_predictions = {}  # probe_name -> [(actual_rate, predicted_score) per cell]
    PROBES = ["rat_dir", "inf_dir", "hedge_dir", "moral_dir",
              "engagement_dir", "commit_dir"]
    LAYERS_TO_PROBE = [20, 24]

    for held_cell in cell_keys:
        held_ids = set(cell_ids[held_cell])
        train_ids = [rid for rid in resp_acts.keys() if rid not in held_ids]

        # Train all directions on train_ids
        rat_dirs = train_directions(resp_acts, judge_index, train_ids, LAYERS_TO_PROBE, "rat")
        inf_dirs = train_directions(resp_acts, judge_index, train_ids, LAYERS_TO_PROBE, "inf")
        hedge_dirs = train_directions(resp_acts, judge_index, train_ids, LAYERS_TO_PROBE, "hedge")
        moral_dirs = train_directions(resp_acts, judge_index, train_ids, LAYERS_TO_PROBE, "moral")
        eng_dirs, com_dirs = train_3way(resp_acts, judge_index, train_ids,
                                        LAYERS_TO_PROBE, restrict_surfaces={"T1", "T2"})

        # Project held-out prompt-state
        for L in LAYERS_TO_PROBE:
            h = prompt_states[held_cell][L]
            nh = h / (h.norm() + 1e-12)
            for probe, dirs in [("rat_dir", rat_dirs), ("inf_dir", inf_dirs),
                                 ("hedge_dir", hedge_dirs), ("moral_dir", moral_dirs),
                                 ("engagement_dir", eng_dirs), ("commit_dir", com_dirs)]:
                d = dirs.get(L)
                if d is None: continue
                score = float((nh * d).sum())
                key = f"L{L}_{probe}"
                held_predictions.setdefault(key, []).append({
                    "cell": held_cell,
                    "score": score,
                    "rates": cell_rates[held_cell],
                })

    # Compute correlations using held-out predictions
    print("Computing held-out correlations...")
    target_axes = ["cheat", "rat", "inf", "hedge", "moral", "level1", "level2"]
    correlations = {}
    for probe_name, predictions in held_predictions.items():
        for ax in target_axes:
            xs = [p["score"] for p in predictions]
            ys = [p["rates"][ax] for p in predictions]
            r = pearson(xs, ys)
            lo, hi = bootstrap_ci(xs, ys)
            correlations[f"{probe_name}__{ax}"] = {
                "r": r, "ci_lo": lo, "ci_hi": hi
            }

    # Save
    out = {
        "n_cells": len(cell_keys),
        "predictions": {p: [{"cell": list(d["cell"]), "score": d["score"],
                             "rates": d["rates"]} for d in preds]
                        for p, preds in held_predictions.items()},
        "correlations": correlations,
    }
    OUT_PATH.write_text(json.dumps(out, indent=1, default=str))
    print(f"Saved {OUT_PATH}")

    # Summary table — compare LOCO to in-sample
    in_sample = json.loads(PROBE_PATH.read_text())["correlations"]

    print("\n" + "="*98)
    print(f"LOCO vs IN-SAMPLE — Pearson r (n={len(cell_keys)} cells), 95% bootstrap CIs")
    print("="*98)
    cols = ["cheat", "rat", "inf", "hedge", "moral"]
    print(f"{'probe':<28} " + " ".join(f"{c:>14}" for c in cols))
    print("-"*98)
    probe_order = [
        "L20_rat_dir", "L24_rat_dir",
        "L20_inf_dir", "L24_inf_dir",
        "L20_hedge_dir", "L24_hedge_dir",
        "L20_moral_dir", "L24_moral_dir",
        "L20_engagement_dir", "L24_engagement_dir",
        "L20_commit_dir", "L24_commit_dir",
    ]
    for probe in probe_order:
        # In-sample
        line_in = f"{probe + ' [IN ]':<28} "
        for ax in cols:
            key = f"{probe}__{ax}"
            if key not in in_sample:
                line_in += f"{'—':>14} "; continue
            r = in_sample[key]["r"]
            line_in += f"  {r:+.2f}        "
        print(line_in)
        line_loco = f"{probe + ' [LOCO]':<28} "
        for ax in cols:
            key = f"{probe}__{ax}"
            if key not in correlations:
                line_loco += f"{'—':>14} "; continue
            r = correlations[key]["r"]
            lo = correlations[key]["ci_lo"]; hi = correlations[key]["ci_hi"]
            line_loco += f"  {r:+.2f}[{lo:+.1f},{hi:+.1f}]"
        print(line_loco)
        print()


if __name__ == "__main__":
    main()
