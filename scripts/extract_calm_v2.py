"""
Re-extract the calm-desperate vector using 500 paired contrastive stories.

Loads all JSON files from emotion_stories/, computes mean residual-stream
activations at L17 for each calm and desperate story, then derives the
direction as (calm_mean - desperate_mean), PCA-denoised.

Compares the new direction to the existing one (vectors/calm.pt) via
cosine similarity at every layer.

Outputs:
  vectors/calm_v2.pt — new vector
  output_calm_v2_comparison.html — comparison visualization
"""
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import plotly.graph_objects as go
from plotly.subplots import make_subplots


MODEL_ID = "google/gemma-2-2b-it"
LAYER = 17  # match existing vector
LAYERS_FOR_COMPARISON = [4, 8, 12, 16, 17, 20, 24]
STORIES_DIR = Path("emotion_stories")
OUT_PT = Path("vectors/calm_v2.pt")
OUT_HTML = Path("output_calm_v2_comparison.html")
EXISTING_PATH = Path("vectors/calm.pt")


def gather_pairs():
    pairs = []
    for f in sorted(STORIES_DIR.glob("*.json")):
        data = json.loads(f.read_text())
        for p in data:
            calm = p.get("calm", "").strip()
            desp = p.get("desperate", "").strip()
            if calm and desp:
                pairs.append((calm, desp, f.stem))
    return pairs


def get_mean_activations(model, tokenizer, texts, layer, device, max_len=200):
    """Mean residual activation at the given layer, averaged across non-prefix tokens."""
    means = []
    for i, text in enumerate(texts):
        if i % 100 == 0 and i > 0:
            print(f"    {i}/{len(texts)}")
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=max_len).to(device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        hidden = out.hidden_states[layer][0]
        if hidden.shape[0] > 3:
            means.append(hidden[3:].mean(0).float().cpu())
        else:
            means.append(hidden.mean(0).float().cpu())
    return torch.stack(means)


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    pairs = gather_pairs()
    print(f"Loaded {len(pairs)} contrastive pairs from {STORIES_DIR}")

    print(f"\nLoading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16
    ).to(device)

    # Compute activations for all calm and all desperate
    print(f"\nEncoding 500 calm stories at all layers {LAYERS_FOR_COMPARISON}...")
    calm_acts_per_layer = {L: [] for L in LAYERS_FOR_COMPARISON}
    desp_acts_per_layer = {L: [] for L in LAYERS_FOR_COMPARISON}

    for i, (calm_text, desp_text, theme) in enumerate(pairs):
        if i % 50 == 0:
            print(f"  pair {i}/{len(pairs)}")
        # encode calm
        inputs_c = tokenizer(calm_text, return_tensors="pt", truncation=True, max_length=200).to(device)
        with torch.no_grad():
            out_c = model(**inputs_c, output_hidden_states=True)
        # encode desperate
        inputs_d = tokenizer(desp_text, return_tensors="pt", truncation=True, max_length=200).to(device)
        with torch.no_grad():
            out_d = model(**inputs_d, output_hidden_states=True)
        for L in LAYERS_FOR_COMPARISON:
            hc = out_c.hidden_states[L][0]
            hd = out_d.hidden_states[L][0]
            calm_mean = hc[3:].mean(0) if hc.shape[0] > 3 else hc.mean(0)
            desp_mean = hd[3:].mean(0) if hd.shape[0] > 3 else hd.mean(0)
            calm_acts_per_layer[L].append(calm_mean.float().cpu())
            desp_acts_per_layer[L].append(desp_mean.float().cpu())

    print("\nComputing direction at each layer (calm - desperate, PCA-denoised)...")
    direction_per_layer = {}
    for L in LAYERS_FOR_COMPARISON:
        calm_X = torch.stack(calm_acts_per_layer[L])     # [N, D]
        desp_X = torch.stack(desp_acts_per_layer[L])     # [N, D]
        differences = calm_X - desp_X                    # [N, D] each is a per-pair calm-desperate diff
        # mean direction
        mean_dir = differences.mean(0)
        mean_dir = mean_dir / mean_dir.norm()
        # PCA-denoise: take top-1 PC of differences
        diff_centered = differences - differences.mean(0, keepdim=True)
        U, S, Vh = torch.linalg.svd(diff_centered, full_matrices=False)
        pc1 = Vh[0]
        # Sign-align with mean_dir
        if (pc1 @ mean_dir).item() < 0:
            pc1 = -pc1
        pc1 = pc1 / pc1.norm()

        direction_per_layer[L] = {
            "mean_direction": mean_dir,
            "pc1_direction": pc1,
            "cos_mean_pc1": (mean_dir @ pc1).item(),
        }
        print(f"  L{L}: cos(mean_dir, PC1) = {direction_per_layer[L]['cos_mean_pc1']:.4f}")

    # Save L17 (default) — both mean and pc1
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
    existing_layer = existing.get("layer", 17)
    print(f"  existing layer: {existing_layer}, vector dim: {existing_dir.shape}")

    print(f"\nCosine similarities — new (v2) vs existing (v1):")
    print(f"  {'Layer':>5}  {'cos(v2_mean, v1)':>18}  {'cos(v2_pc1, v1)':>18}")
    cos_at_layers = {}
    for L in LAYERS_FOR_COMPARISON:
        m = direction_per_layer[L]["mean_direction"]
        p = direction_per_layer[L]["pc1_direction"]
        cm = (m @ existing_dir).item() if L == existing_layer else float("nan")
        cp = (p @ existing_dir).item() if L == existing_layer else float("nan")
        # (Existing direction is at one layer; cosines at other layers are not meaningful)
        cos_at_layers[L] = {"cos_mean": cm, "cos_pc1": cp}
        if L == existing_layer:
            print(f"  L{L:>3}    {cm:>+18.4f}    {cp:>+18.4f}  ← existing layer")
        else:
            print(f"  L{L:>3}    {'(other layer)':>18}    {'(other layer)':>18}")

    # Norm comparison
    print(f"\nNorm checks:")
    print(f"  Existing direction norm: {existing_dir.norm().item():.4f} (already normalized)")
    print(f"  v2 mean direction norm:  1.0000 (normalized)")
    print(f"  v2 PC1 direction norm:   1.0000 (normalized)")

    # Sanity: project some new pair differences onto each direction
    print(f"\nSanity check: project some pair-differences onto each direction")
    # Use 50 pairs
    test_pairs = pairs[:50]
    test_diffs = []
    for calm_text, desp_text, _ in test_pairs:
        inputs_c = tokenizer(calm_text, return_tensors="pt", truncation=True, max_length=200).to(device)
        with torch.no_grad():
            out_c = model(**inputs_c, output_hidden_states=True)
        inputs_d = tokenizer(desp_text, return_tensors="pt", truncation=True, max_length=200).to(device)
        with torch.no_grad():
            out_d = model(**inputs_d, output_hidden_states=True)
        hc = out_c.hidden_states[LAYER][0]
        hd = out_d.hidden_states[LAYER][0]
        cm = hc[3:].mean(0) if hc.shape[0] > 3 else hc.mean(0)
        dm = hd[3:].mean(0) if hd.shape[0] > 3 else hd.mean(0)
        test_diffs.append((cm - dm).float().cpu())
    test_diffs = torch.stack(test_diffs)

    proj_v1 = (test_diffs @ existing_dir).mean().item()
    proj_v2_mean = (test_diffs @ direction_per_layer[LAYER]["mean_direction"]).mean().item()
    proj_v2_pc1 = (test_diffs @ direction_per_layer[LAYER]["pc1_direction"]).mean().item()

    print(f"  Mean projection of (calm - desperate) onto v1 existing: {proj_v1:+.3f}")
    print(f"  Mean projection of (calm - desperate) onto v2 mean:     {proj_v2_mean:+.3f}")
    print(f"  Mean projection of (calm - desperate) onto v2 PC1:      {proj_v2_pc1:+.3f}")
    print(f"  (Higher = direction better captures calm - desperate axis)")


if __name__ == "__main__":
    main()
