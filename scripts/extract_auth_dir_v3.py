"""V3 extraction of the authorization direction.

Uses a fresh 100-pair corpus of longer (3-5 sentence), emotion-matched paired
stories in 5 new domains (security_audit, code_deployment, customer_data_export,
content_moderation, billing_adjustment). The point is to test whether the v1
+18pp causal effect on T1/authorized reproduces with a different paired-stories
corpus — the key methodological caveat from the original paper.

Outputs:
  vectors/auth_dir_L17_v3.pt
  vectors/auth_dir_L17_v3_ortho.pt  (orthogonalized against calm_v2)
"""
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "google/gemma-2-2b-it"
LAYER = 17
LAYERS_FOR_COMPARISON = [4, 8, 12, 16, 17, 20, 24]
STORIES_DIR = Path("authorization_stories_v3")
OUT_PT = Path("vectors/auth_dir_L17_v3.pt")
OUT_ORTHO_PT = Path("vectors/auth_dir_L17_v3_ortho.pt")
CALM_PATH = Path("vectors/calm_v2.pt")


def gather_pairs():
    pairs = []
    for f in sorted(STORIES_DIR.glob("*.json")):
        data = json.loads(f.read_text())
        for i, p in enumerate(data):
            authorized = p.get("authorized", "").strip()
            unauthorized = p.get("unauthorized", "").strip()
            if authorized and unauthorized:
                pairs.append((authorized, unauthorized, f.stem, i))
    return pairs


def mean_hidden_at_layers(model, tokenizer, text, device, max_len=300):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len).to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    means = {}
    for layer in LAYERS_FOR_COMPARISON:
        hidden = out.hidden_states[layer][0]
        means[layer] = (hidden[3:].mean(0) if hidden.shape[0] > 3 else hidden.mean(0)).float().cpu()
    return means


def split_half_cosine(differences):
    half = differences.shape[0] // 2
    first = differences[:half].mean(0)
    second = differences[half:].mean(0)
    first = first / first.norm()
    second = second / second.norm()
    return (first @ second).item()


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    pairs = gather_pairs()
    print(f"Loaded {len(pairs)} authorization contrast pairs from {STORIES_DIR}")
    if len(pairs) != 100:
        print(f"WARN: Expected 100 pairs, got {len(pairs)}")

    print(f"\nLoading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to(device)
    model.train(False)

    auth_acts = {layer: [] for layer in LAYERS_FOR_COMPARISON}
    unauth_acts = {layer: [] for layer in LAYERS_FOR_COMPARISON}

    print(f"\nEncoding {len(pairs)} pairs (longer stories than v1, ~150 tokens each)...")
    for i, (auth_text, unauth_text, theme, item_idx) in enumerate(pairs):
        if i % 10 == 0:
            print(f"  pair {i}/{len(pairs)}")
        auth_means = mean_hidden_at_layers(model, tokenizer, auth_text, device)
        unauth_means = mean_hidden_at_layers(model, tokenizer, unauth_text, device)
        for layer in LAYERS_FOR_COMPARISON:
            auth_acts[layer].append(auth_means[layer])
            unauth_acts[layer].append(unauth_means[layer])

    print("\nComputing direction at each layer (authorized minus unauthorized)...")
    direction_per_layer = {}
    for layer in LAYERS_FOR_COMPARISON:
        auth_x = torch.stack(auth_acts[layer])
        unauth_x = torch.stack(unauth_acts[layer])
        differences = auth_x - unauth_x

        mean_dir = differences.mean(0)
        mean_dir = mean_dir / mean_dir.norm()

        centered = differences - differences.mean(0, keepdim=True)
        _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
        pc1 = vh[0]
        if (pc1 @ mean_dir).item() < 0:
            pc1 = -pc1
        pc1 = pc1 / pc1.norm()

        split_half = split_half_cosine(differences)

        direction_per_layer[layer] = {
            "mean_direction": mean_dir,
            "pc1_direction": pc1,
            "cos_mean_pc1": (mean_dir @ pc1).item(),
            "split_half_cos_mean": split_half,
        }
        print(f"  L{layer}: mean-diff norm={differences.mean(0).norm():.3f}  "
              f"cos(mean,pc1)={direction_per_layer[layer]['cos_mean_pc1']:+.3f}  "
              f"split-half cos={split_half:+.3f}")

    # Save raw (unorthogonalized) direction
    blob = {
        "direction": direction_per_layer[LAYER]["pc1_direction"],
        "mean_direction": direction_per_layer[LAYER]["mean_direction"],
        "layer": LAYER,
        "n_pairs": len(pairs),
        "stories_dir": str(STORIES_DIR),
        "all_layers": direction_per_layer,
    }
    OUT_PT.parent.mkdir(exist_ok=True, parents=True)
    torch.save(blob, OUT_PT)
    print(f"\nSaved {OUT_PT}")

    # Orthogonalize against calm_v2 (the original v1 used calm-orthogonalization)
    print("\nOrthogonalizing against calm_v2...")
    calm_blob = torch.load(CALM_PATH, map_location="cpu", weights_only=False)
    calm_dir = calm_blob.get("mean_direction", calm_blob.get("direction")) if isinstance(calm_blob, dict) else calm_blob
    calm_dir = calm_dir / calm_dir.norm()

    raw_dir = blob["mean_direction"]
    projection = (raw_dir @ calm_dir) * calm_dir
    ortho_dir = raw_dir - projection
    ortho_dir = ortho_dir / ortho_dir.norm()

    cos_to_calm_before = (raw_dir @ calm_dir).item()
    cos_to_calm_after = (ortho_dir @ calm_dir).item()
    print(f"  cos(raw auth_v3, calm_v2) = {cos_to_calm_before:+.4f}")
    print(f"  cos(ortho auth_v3, calm_v2) = {cos_to_calm_after:+.4f} (should be ~0)")

    ortho_blob = {
        "mean_direction": ortho_dir,
        "layer": LAYER,
        "n_pairs": len(pairs),
        "stories_dir": str(STORIES_DIR),
        "cos_to_calm": cos_to_calm_after,
        "cos_to_calm_before_ortho": cos_to_calm_before,
        "all_layers": direction_per_layer,
    }
    torch.save(ortho_blob, OUT_ORTHO_PT)
    print(f"Saved {OUT_ORTHO_PT}")

    # Compare to v1 direction for reference
    v1_path = Path("vectors/auth_dir_L17_v1_ortho.pt")
    if v1_path.exists():
        v1_blob = torch.load(v1_path, map_location="cpu", weights_only=False)
        v1_dir = v1_blob["mean_direction"]
        cos_v1_v3 = (ortho_dir @ v1_dir.to(ortho_dir.dtype)).item()
        print(f"\ncos(v3_ortho, v1_ortho) at L17 = {cos_v1_v3:+.4f}")
        print("(v1 had cos = +0.67 with the failed 50-pair extraction; this is our v3 comparison)")


if __name__ == "__main__":
    main()
