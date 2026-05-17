"""
Full auth-direction sweep — paper §4 protocol.

Matrix: 3 surfaces × 3 frames × 3 alphas × n=100 samples = 2700 generations
(or run a subset via the CELLS loop in main()). Steers at L17 with v1
auth_dir (vectors/auth_dir_L17_v1_ortho.pt). α ∈ {-0.10, 0, +0.10},
max_new_tokens=900 to match paper hyperparameters.

The original P2c-era data was n=30 at 600 tokens; that protocol under-counted
cheats because 88-100% of responses hit the 600-token cap (§4.5). The
constants below reflect the paper's n=100 / 900-token protocol.

Output: auth_steering/{surface}_{frame}_{alpha_label}.txt
Skips cells whose output file already exists.
"""
import re
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import steer
import auth_prompts
from emotion_steering_full import is_cheat_t1
import cross_prompt_cheat as cp2
import cross_prompt_t3 as cp3


MODEL_ID = "google/gemma-2-2b-it"
N_PER_CELL = 100  # paper §4 uses n=100/α; original P2c-era run was 30
TEMPERATURE = 0.8
MAX_NEW_TOKENS = 900  # paper §4 / hyperparam table uses 900; original was 600 (cap hit ~90% of responses)

AUTH_DIR_PATH = Path("vectors/auth_dir_L17_v1_ortho.pt")
OUT_DIR = Path("auth_steering")

ALPHAS = [-0.10, 0.0, +0.10]
ALPHA_LABEL = {-0.10: "neg10", 0.0: "zero", +0.10: "pos10"}

DETECTOR = {
    "T1": is_cheat_t1,
    "T2": cp2.is_cheat_t2,
    "T3": cp3.is_cheat_t3,
}


def make_hook(d, alpha, residual_norm):
    if alpha == 0:
        return None
    s = alpha * residual_norm

    def hook(module, inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            steered = hidden + s * d.to(hidden.device, dtype=hidden.dtype)
            return (steered,) + output[1:]
        return output + s * d.to(output.device, dtype=output.dtype)

    return hook


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    OUT_DIR.mkdir(exist_ok=True)

    cells = []
    for (surface, frame), prompt in auth_prompts.PROMPTS.items():
        for alpha in ALPHAS:
            label = ALPHA_LABEL[alpha]
            out_path = OUT_DIR / f"{surface}_{frame}_{label}.txt"
            if out_path.exists():
                print(f"  skip existing: {out_path.name}", flush=True)
                continue
            cells.append((surface, frame, alpha, prompt, out_path))
    print(f"\n{len(cells)} cells to generate (out of 27)", flush=True)
    if not cells:
        return

    print(f"\nLoading {MODEL_ID}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16
    ).to(device)
    model.requires_grad_(False)

    saved = torch.load(AUTH_DIR_PATH, weights_only=False)
    auth_dir = saved["mean_direction"].to(device)
    layer_hs_idx = saved.get("layer", 17)
    layer_idx = layer_hs_idx - 1
    target_layer = model.model.layers[layer_idx]

    norm = steer.estimate_residual_norm(model, tokenizer, layer_hs_idx, device)
    print(f"  residual norm at L{layer_hs_idx}: {norm:.1f}", flush=True)
    print(f"  auth_dir_ortho cos to calm: {saved.get('cos_to_calm'):+.4f}", flush=True)

    summary = []
    t_start = time.time()
    for cell_idx, (surface, frame, alpha, prompt, out_path) in enumerate(cells, start=1):
        detector = DETECTOR[surface]
        elapsed = time.time() - t_start
        eta = elapsed / max(cell_idx - 1, 1) * (len(cells) - cell_idx + 1) if cell_idx > 1 else 0
        print(f"\n{'='*70}", flush=True)
        print(f"[{cell_idx}/{len(cells)}] {surface}/{frame}/α={alpha:+.2f}  "
              f"elapsed={elapsed/60:.1f}m  ETA={eta/60:.1f}m", flush=True)
        print('='*70, flush=True)

        with open(out_path, "w") as fout:
            fout.write(f"Phase D auth sweep — auth_dir_ortho at L{layer_hs_idx}\n")
            fout.write(f"SURFACE: {surface}\nFRAME: {frame}\nALPHA: {alpha:+.2f}\n")
            fout.write(f"PROMPT:\n{prompt}\n\n" + "=" * 78 + "\n")

        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True, return_dict=True,
        ).to(device)
        input_len = inputs["input_ids"].shape[1]

        hook_fn = make_hook(auth_dir, alpha, norm)
        handle = target_layer.register_forward_hook(hook_fn) if hook_fn else None
        n_cheat = 0
        try:
            for i in range(N_PER_CELL):
                with torch.no_grad():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=True,
                        temperature=TEMPERATURE,
                    )
                response = tokenizer.decode(
                    output[0][input_len:], skip_special_tokens=True
                ).strip()
                cheat = detector(response)
                if cheat:
                    n_cheat += 1
                tag = "CHEAT" if cheat else "ok"
                snippet = response.replace("\n", " ")[:80]
                print(f"  [{i+1:2d}/{N_PER_CELL}] {tag:<5} {snippet}...", flush=True)
                with open(out_path, "a") as fout:
                    fout.write(f"\n--- Sample {i+1}/{N_PER_CELL} ({tag}) ---\n{response}\n")
        finally:
            if handle is not None:
                handle.remove()

        rate = n_cheat / N_PER_CELL
        print(f"\n  → {surface}/{frame}/α={alpha:+.2f}: "
              f"{n_cheat}/{N_PER_CELL} = {rate:.0%} (regex)", flush=True)
        summary.append((surface, frame, alpha, n_cheat, rate))

    print("\n" + "=" * 70, flush=True)
    print("PHASE D1 SUMMARY (regex labels — judge labels in D2)", flush=True)
    print("=" * 70, flush=True)
    for surface, frame, alpha, n_cheat, rate in summary:
        print(f"{surface:<4} {frame:<14} α={alpha:+.2f}  "
              f"{n_cheat:>3}/{N_PER_CELL:<3} {rate:>6.0%}", flush=True)
    total_min = (time.time() - t_start) / 60
    print(f"\nTotal wall time: {total_min:.1f} min", flush=True)


if __name__ == "__main__":
    main()
