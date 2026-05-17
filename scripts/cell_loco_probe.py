"""Cell-LOCO cheat probe at multiple layers (L4, L8, L12, L16, L17, L20, L24).

For each of 17 cells (T1/T2/T3 × {default, calm@α, calm+confident@α}):
- Take all the records for that cell
- Apply the cell-specific steering hook
- Forward-pass each prompt; capture prompt-state at final token at each layer
- Save activations + cell label + record cheat label

Then for each layer:
- For each held-out cell:
  - Train logistic regression on the other 16 cells (per-record activations + record cheat labels)
  - Predict cheat for the held-out cell records
  - Compute AUC on held-out cell

Cell-LOCO test: does the cheat encoding generalize across cells (different steering conditions)?

Output: cell_loco_probe_results.json
"""
import json
import re
from pathlib import Path
from collections import defaultdict
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

import steer
from steer import PROMPT as PROMPT_T1
import cross_prompt_cheat as cp2
import cross_prompt_t3 as cp3


MODEL_ID = "google/gemma-2-2b-it"
CALM_V2_PATH = Path("vectors/calm_v2.pt")
CONFIDENT_V2_ORTHO_PATH = Path("vectors/confident_v2_ortho.pt")
JUDGE_PATH = Path("judge_labels_v2_full.json")
LAYERS = [4, 8, 12, 16, 17, 20, 24]
STEER_LAYER_IDX = 16

SURFACES = {
    "T1": PROMPT_T1,
    "T2": cp2.PROMPT_T2,
    "T3": cp3.PROMPT_T3,
}

CONDITIONS = [
    ("default", 0.00, 0.00),
    ("calm@+5", +0.05, 0.00),
    ("calm@+15", +0.15, 0.00),
    ("calm@-15", -0.15, 0.00),
    ("calm+confident@+5", +0.05, +0.05),
    ("calm+confident@+15", +0.15, +0.15),
    ("calm+confident@-15", -0.15, -0.15),
]

T3_CONDS = {"default", "calm@+15", "calm+confident@+15"}


def make_hook(d_calm, a_calm, d_conf, a_conf, residual_norm):
    s1 = a_calm * residual_norm
    s2 = a_conf * residual_norm
    if a_calm == 0 and a_conf == 0:
        return None
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            steered = (hidden
                       + s1 * d_calm.to(hidden.device, dtype=hidden.dtype)
                       + s2 * d_conf.to(hidden.device, dtype=hidden.dtype))
            return (steered,) + output[1:]
        return output + s1 * d_calm.to(output.device, dtype=output.dtype) \
                      + s2 * d_conf.to(output.device, dtype=output.dtype)
    return hook


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to(device)
    model.train(False)

    calm_blob = torch.load(CALM_V2_PATH, map_location="cpu", weights_only=False)
    calm_dir = calm_blob.get("mean_direction", calm_blob.get("direction")) if isinstance(calm_blob, dict) else calm_blob
    conf_blob = torch.load(CONFIDENT_V2_ORTHO_PATH, map_location="cpu", weights_only=False)
    conf_dir = conf_blob.get("mean_direction", conf_blob.get("direction")) if isinstance(conf_blob, dict) else conf_blob
    print(f"Calm norm={calm_dir.norm().item():.3f}, conf_ortho norm={conf_dir.norm().item():.3f}")

    residual_norm = steer.estimate_residual_norm(model, tokenizer, 17, device)
    print(f"Residual norm L17: {residual_norm:.1f}")

    target_layer = model.model.layers[STEER_LAYER_IDX]

    # Load judge labels indexed by (surface, condition, sample_idx)
    judge = json.load(open(JUDGE_PATH))
    by_record = {}
    for r in judge:
        key = (r["surface"], r["condition"], r["sample_idx"])
        by_record[key] = r

    # For each cell, forward-pass each record's prompt and capture activations
    activations_per_cell = defaultdict(list)
    cheat_per_cell = defaultdict(list)
    cell_idx = 0
    total_cells = sum(len(CONDITIONS) if surf != "T3" else len(T3_CONDS) for surf in SURFACES)

    for surf_name, prompt in SURFACES.items():
        for cond_name, a_calm, a_conf in CONDITIONS:
            if surf_name == "T3" and cond_name not in T3_CONDS:
                continue
            cell_idx += 1
            cell_key = f"{surf_name}/{cond_name}"

            # Get sample_idxs for this cell
            samples = [(s_idx, by_record.get((surf_name, cond_name, s_idx)))
                       for s_idx in range(1, 60)
                       if (surf_name, cond_name, s_idx) in by_record]
            if not samples:
                continue

            print(f"[{cell_idx}/{total_cells}] {cell_key}: {len(samples)} records", flush=True)

            hook_fn = make_hook(calm_dir, a_calm, conf_dir, a_conf, residual_norm)
            handle = target_layer.register_forward_hook(hook_fn) if hook_fn else None

            try:
                for s_idx, rec in samples:
                    if rec is None:
                        continue
                    messages = [{"role": "user", "content": prompt}]
                    inputs = tokenizer.apply_chat_template(
                        messages, return_tensors="pt", add_generation_prompt=True, return_dict=True
                    ).to(device)
                    with torch.no_grad():
                        out = model(**inputs, output_hidden_states=True)
                    # Capture last prompt token's hidden state at each layer
                    last_idx = inputs["input_ids"].shape[1] - 1
                    acts_per_layer = {L: out.hidden_states[L][0, last_idx, :].float().cpu() for L in LAYERS}
                    activations_per_cell[cell_key].append(acts_per_layer)
                    cheat_per_cell[cell_key].append(int(rec["judge_cheat"]))
            finally:
                if handle is not None:
                    handle.remove()

    # Save raw per-cell activations + labels
    save = {
        "cells": list(activations_per_cell.keys()),
        "layers": LAYERS,
        "activations": {c: [{L: a[L].tolist() for L in LAYERS} for a in alist]
                        for c, alist in activations_per_cell.items()},
        "cheat": dict(cheat_per_cell),
    }
    with open("cell_loco_raw_activations.json", "w") as fh:
        json.dump(save, fh)
    print("Saved raw activations to cell_loco_raw_activations.json")

    # Now do cell-LOCO logistic regression at each layer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    cells = list(activations_per_cell.keys())

    print("\n## Cell-LOCO cheat probe AUC per layer")
    print(f"{'layer':>6} {'mean AUC':>10} {'min':>6} {'max':>6} {'std':>6} {'n_cells':>8}")
    print("-" * 50)

    results = {}
    for L in LAYERS:
        cell_aucs = {}
        for held_out in cells:
            # Train on all other cells; test on held_out
            X_tr, y_tr = [], []
            for c in cells:
                if c == held_out: continue
                for a, y in zip(activations_per_cell[c], cheat_per_cell[c]):
                    X_tr.append(a[L].numpy()); y_tr.append(y)
            X_te, y_te = [], []
            for a, y in zip(activations_per_cell[held_out], cheat_per_cell[held_out]):
                X_te.append(a[L].numpy()); y_te.append(y)
            if len(set(y_te)) < 2:
                # No within-cell variance — skip AUC
                continue
            try:
                clf = LogisticRegression(max_iter=2000, C=1.0).fit(np.array(X_tr), np.array(y_tr))
                p = clf.predict_proba(np.array(X_te))[:, 1]
                auc = roc_auc_score(y_te, p)
                cell_aucs[held_out] = auc
            except Exception as e:
                pass
        if cell_aucs:
            vals = list(cell_aucs.values())
            mean = np.mean(vals); mn = np.min(vals); mx = np.max(vals); sd = np.std(vals)
            results[f"L{L}"] = {"mean": float(mean), "min": float(mn), "max": float(mx), "std": float(sd), "per_cell": {k: float(v) for k, v in cell_aucs.items()}}
            print(f"{L:>6} {mean:>10.3f} {mn:>6.3f} {mx:>6.3f} {sd:>6.3f} {len(cell_aucs):>8}")

    with open("cell_loco_probe_results.json", "w") as fh:
        json.dump(results, fh, indent=1)
    print("\nSaved results to cell_loco_probe_results.json")


if __name__ == "__main__":
    main()
