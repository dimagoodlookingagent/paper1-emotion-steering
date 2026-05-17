"""
Orthogonalize auth_dir against calm_v2 mean.

auth_dir_ortho = auth_dir − (auth_dir · calm_v2) · calm_v2
                 then renormalized to unit length.

Saves: vectors/auth_dir_L17_v1_ortho.pt (the orthogonalized v1 used throughout §4).
Verifies: cos(auth_dir_ortho, calm_v2) ≈ 0; cos(auth_dir_ortho, auth_dir) ≈ +0.97.

Prerequisite: vectors/authorization_v1.pt — the raw v1 auth-direction extraction
produced by extract_auth_dir_v1.py. This file is not shipped in paper1/vectors/
because the orthogonalized output (auth_dir_L17_v1_ortho.pt) is what the
steering scripts load. Regenerate raw v1 via extract_auth_dir_v1.py first.
"""
from pathlib import Path
import torch

AUTH_PATH = Path("vectors/authorization_v1.pt")
CALM_PATH = Path("vectors/calm_v2.pt")
OUT_PATH = Path("vectors/auth_dir_L17_v1_ortho.pt")


def main():
    auth = torch.load(AUTH_PATH, weights_only=False)
    calm = torch.load(CALM_PATH, weights_only=False)

    auth_dir = auth["mean_direction"].float()
    calm_dir = calm["mean_direction"].float()

    auth_dir = auth_dir / auth_dir.norm()
    calm_dir = calm_dir / calm_dir.norm()

    cos_before = float((auth_dir * calm_dir).sum())
    print(f"cos(auth, calm) before: {cos_before:+.4f}")

    proj = (auth_dir * calm_dir).sum() * calm_dir
    auth_ortho = auth_dir - proj
    norm_after_subtract = float(auth_ortho.norm())
    auth_ortho = auth_ortho / auth_ortho.norm()

    cos_after = float((auth_ortho * calm_dir).sum())
    cos_to_orig = float((auth_ortho * auth_dir).sum())
    print(f"cos(auth_ortho, calm) after: {cos_after:+.4f} (target ≈ 0)")
    print(f"cos(auth_ortho, auth_orig): {cos_to_orig:+.4f}")
    print(f"|auth_ortho| before renorm: {norm_after_subtract:.4f} "
          f"(magnitude lost: {1 - norm_after_subtract:.2%})")

    out = {
        "mean_direction": auth_ortho,
        "layer": auth.get("layer", 17),
        "source": "authorization_v1.pt orthogonalized against calm_v2 mean",
        "cos_to_calm": cos_after,
        "cos_to_orig_auth": cos_to_orig,
        "magnitude_pre_renorm": norm_after_subtract,
    }
    torch.save(out, OUT_PATH)
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()
