"""
Re-extract the confident-anxious vector using 500 paired contrastive stories.

Same methodology as extract_calm_desperate_v2.py but for the confident axis.
Saves to vectors/confident_v2.pt and compares to existing vectors/confident.pt.
"""
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_ID = "google/gemma-2-2b-it"
LAYER = 17
LAYERS_FOR_COMPARISON = [4, 8, 12, 16, 17, 20, 24]
STORIES_DIR = Path("emotion_stories_conf_anx")
OUT_PT = Path("vectors/confident_v2.pt")
EXISTING_PATH = Path("vectors/confident.pt")


def gather_pairs():
    pairs = []
    for f in sorted(STORIES_DIR.glob("*.json")):
        data = json.loads(f.read_text())
        for p in data:
            conf = p.get("confident", "").strip()
            anx = p.get("anxious", "").strip()
            if conf and anx:
                pairs.append((conf, anx, f.stem))
    return pairs


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    pairs = gather_pairs()
    print(f"Loaded {len(pairs)} contrastive confident-anxious pairs from {STORIES_DIR}")

    print(f"\nLoading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16
    ).to(device)

    print(f"\nEncoding {len(pairs)} confident stories + {len(pairs)} anxious stories at layers {LAYERS_FOR_COMPARISON}...")
    conf_acts = {L: [] for L in LAYERS_FOR_COMPARISON}
    anx_acts = {L: [] for L in LAYERS_FOR_COMPARISON}

    for i, (conf_text, anx_text, theme) in enumerate(pairs):
        if i % 50 == 0:
            print(f"  pair {i}/{len(pairs)}")
        inputs_c = tokenizer(conf_text, return_tensors="pt", truncation=True, max_length=200).to(device)
        with torch.no_grad():
            out_c = model(**inputs_c, output_hidden_states=True)
        inputs_a = tokenizer(anx_text, return_tensors="pt", truncation=True, max_length=200).to(device)
        with torch.no_grad():
            out_a = model(**inputs_a, output_hidden_states=True)
        for L in LAYERS_FOR_COMPARISON:
            hc = out_c.hidden_states[L][0]
            ha = out_a.hidden_states[L][0]
            cm = hc[3:].mean(0) if hc.shape[0] > 3 else hc.mean(0)
            am = ha[3:].mean(0) if ha.shape[0] > 3 else ha.mean(0)
            conf_acts[L].append(cm.float().cpu())
            anx_acts[L].append(am.float().cpu())

    print("\nComputing direction at each layer (confident - anxious)...")
    direction_per_layer = {}
    for L in LAYERS_FOR_COMPARISON:
        c_X = torch.stack(conf_acts[L])
        a_X = torch.stack(anx_acts[L])
        differences = c_X - a_X
        mean_dir = differences.mean(0)
        mean_dir = mean_dir / mean_dir.norm()
        # PCA-denoise
        diff_centered = differences - differences.mean(0, keepdim=True)
        U, S, Vh = torch.linalg.svd(diff_centered, full_matrices=False)
        pc1 = Vh[0]
        if (pc1 @ mean_dir).item() < 0:
            pc1 = -pc1
        pc1 = pc1 / pc1.norm()
        direction_per_layer[L] = {
            "mean_direction": mean_dir,
            "pc1_direction": pc1,
            "cos_mean_pc1": (mean_dir @ pc1).item(),
        }
        print(f"  L{L}: cos(mean, PC1) = {direction_per_layer[L]['cos_mean_pc1']:.4f}")

    OUT_PT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "direction": direction_per_layer[LAYER]["pc1_direction"],
        "mean_direction": direction_per_layer[LAYER]["mean_direction"],
        "layer": LAYER,
        "n_pairs": len(pairs),
        "all_layers": direction_per_layer,
    }, OUT_PT)
    print(f"\nSaved {OUT_PT}")

    # Compare to existing
    print(f"\nLoading existing direction from {EXISTING_PATH}...")
    existing = torch.load(EXISTING_PATH, weights_only=False)
    existing_dir = existing["direction"].float().cpu()
    existing_dir = existing_dir / existing_dir.norm()

    cm = (direction_per_layer[LAYER]["mean_direction"] @ existing_dir).item()
    cp = (direction_per_layer[LAYER]["pc1_direction"] @ existing_dir).item()
    print(f"\nCosine vs existing v1 confident vector at L{LAYER}:")
    print(f"  cos(v2_mean, v1) = {cm:+.4f}")
    print(f"  cos(v2_pc1, v1) = {cp:+.4f}")

    # Sanity check: project some confident-anxious diffs onto each direction
    print("\nSanity check: project test pair-differences onto each direction at L17")
    test_diffs = []
    for conf_text, anx_text, _ in pairs[:50]:
        inputs_c = tokenizer(conf_text, return_tensors="pt", truncation=True, max_length=200).to(device)
        with torch.no_grad():
            out_c = model(**inputs_c, output_hidden_states=True)
        inputs_a = tokenizer(anx_text, return_tensors="pt", truncation=True, max_length=200).to(device)
        with torch.no_grad():
            out_a = model(**inputs_a, output_hidden_states=True)
        hc = out_c.hidden_states[LAYER][0]
        ha = out_a.hidden_states[LAYER][0]
        cm_t = hc[3:].mean(0) if hc.shape[0] > 3 else hc.mean(0)
        am_t = ha[3:].mean(0) if ha.shape[0] > 3 else ha.mean(0)
        test_diffs.append((cm_t - am_t).float().cpu())
    test_diffs = torch.stack(test_diffs)

    proj_v1 = (test_diffs @ existing_dir).mean().item()
    proj_v2_mean = (test_diffs @ direction_per_layer[LAYER]["mean_direction"]).mean().item()
    proj_v2_pc1 = (test_diffs @ direction_per_layer[LAYER]["pc1_direction"]).mean().item()

    print(f"  Mean projection of (confident − anxious) onto v1 existing: {proj_v1:+.3f}")
    print(f"  Mean projection of (confident − anxious) onto v2 mean:     {proj_v2_mean:+.3f}")
    print(f"  Mean projection of (confident − anxious) onto v2 PC1:      {proj_v2_pc1:+.3f}")

    # Cross-axis: cos(calm_v2_mean, confident_v2_mean) — important diagnostic
    calm_v2 = torch.load(Path("vectors/calm_v2.pt"), weights_only=False)
    calm_v2_mean = calm_v2["mean_direction"].float().cpu()
    calm_v2_mean = calm_v2_mean / calm_v2_mean.norm()
    cos_axes = (direction_per_layer[LAYER]["mean_direction"] @ calm_v2_mean).item()
    print(f"\nCross-axis check: cos(calm_v2_mean, confident_v2_mean) = {cos_axes:+.4f}")
    print("  (Should be small if calm and confident are independent emotion axes)")


if __name__ == "__main__":
    main()
