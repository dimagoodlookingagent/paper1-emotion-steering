"""
Extract an authorization direction from 100 paired contrastive scenarios.

The pair construction keeps the action surface mostly constant and flips the
permission state around it:

    authorized_minus_unauthorized = mean(authorized_text - unauthorized_text)

Outputs:
  vectors/authorization_v1.pt
"""
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "google/gemma-2-2b-it"
LAYER = 17
LAYERS_FOR_COMPARISON = [4, 8, 12, 16, 17, 20, 24]
STORIES_DIR = Path("authorization_stories")
OUT_PT = Path("vectors/authorization_v1.pt")


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


def mean_hidden_at_layers(model, tokenizer, text, device, max_len=220):
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
        raise ValueError(f"Expected exactly 100 pairs, got {len(pairs)}")

    print(f"\nLoading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(device)
    model.eval()

    auth_acts = {layer: [] for layer in LAYERS_FOR_COMPARISON}
    unauth_acts = {layer: [] for layer in LAYERS_FOR_COMPARISON}

    print(f"\nEncoding {len(pairs)} authorized + {len(pairs)} unauthorized texts...")
    for i, (auth_text, unauth_text, theme, item_idx) in enumerate(pairs):
        if i % 10 == 0:
            print(f"  pair {i}/{len(pairs)}")
        auth_means = mean_hidden_at_layers(model, tokenizer, auth_text, device)
        unauth_means = mean_hidden_at_layers(model, tokenizer, unauth_text, device)
        for layer in LAYERS_FOR_COMPARISON:
            auth_acts[layer].append(auth_means[layer])
            unauth_acts[layer].append(unauth_means[layer])

    print("\nComputing directions at each layer (authorized - unauthorized)...")
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

        projections_mean = differences @ mean_dir
        projections_pc1 = differences @ pc1

        direction_per_layer[layer] = {
            "mean_direction": mean_dir,
            "pc1_direction": pc1,
            "cos_mean_pc1": (mean_dir @ pc1).item(),
            "split_half_cos_mean": split_half_cosine(differences),
            "mean_projection_on_mean": projections_mean.mean().item(),
            "std_projection_on_mean": projections_mean.std(unbiased=False).item(),
            "mean_projection_on_pc1": projections_pc1.mean().item(),
            "std_projection_on_pc1": projections_pc1.std(unbiased=False).item(),
            "singular_values_top5": singular_values[:5].float().cpu(),
        }

        print(
            f"  L{layer}: cos(mean, PC1)={direction_per_layer[layer]['cos_mean_pc1']:+.4f}, "
            f"split-half={direction_per_layer[layer]['split_half_cos_mean']:+.4f}, "
            f"proj_mean={direction_per_layer[layer]['mean_projection_on_mean']:+.3f}"
        )

    OUT_PT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "direction": direction_per_layer[LAYER]["mean_direction"],
            "mean_direction": direction_per_layer[LAYER]["mean_direction"],
            "pc1_direction": direction_per_layer[LAYER]["pc1_direction"],
            "layer": LAYER,
            "label": "authorized_minus_unauthorized",
            "n_pairs": len(pairs),
            "stories_dir": str(STORIES_DIR),
            "all_layers": direction_per_layer,
        },
        OUT_PT,
    )
    print(f"\nSaved {OUT_PT}")


if __name__ == "__main__":
    main()
