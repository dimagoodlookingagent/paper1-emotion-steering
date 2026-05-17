"""
Aggregate Phase E judge labels (E1 + E4).

E1: α-scan dose-response curves (T1/{authorized,ambiguous}, α∈{+.05, +.15})
E4: T2/ambiguous reversal probe (3 alphas × 70 new samples)

Output: judge_labels_phaseE.json + per-cell summary tables.
"""
import json
import re
from pathlib import Path
from collections import defaultdict


META_PATH = Path("judge_meta_phaseE.json")
JUDGE_OUT = Path("judge_out_phaseE")
COMBINED_PATH = Path("judge_labels_phaseE.json")
PHASED_LABELS = Path("judge_labels_phaseD.json")
AXES = ["cheat", "moral", "hedge", "rat", "fix", "inf", "sp"]
ALPHA_NUM = {"neg10": -0.10, "zero": 0.0, "pos05": +0.05, "pos10": +0.10, "pos15": +0.15}

LINE_RE = re.compile(r"^(\d+):(.+)$")
KV_RE = re.compile(r"(\w+)=([YN])")


def parse_line(line):
    m = LINE_RE.match(line.strip())
    if not m: return None
    return int(m.group(1)), dict(KV_RE.findall(m.group(2)))


def main():
    meta = {r["id"]: r for r in json.loads(META_PATH.read_text())}
    labels = {}
    for path in sorted(JUDGE_OUT.glob("batch_*.txt")):
        for raw in path.read_text().splitlines():
            if not raw.strip() or not raw[0].isdigit(): continue
            parsed = parse_line(raw)
            if parsed is None: continue
            rid, kv = parsed
            labels[rid] = {ax: (kv.get(ax) == "Y") for ax in AXES}

    print(f"Records labeled: {len(labels)}/{len(meta)}")
    merged = []
    for rid, m in meta.items():
        rec = dict(m)
        for ax in AXES:
            rec[f"judge_{ax}"] = labels.get(rid, {}).get(ax, None)
        merged.append(rec)
    COMBINED_PATH.write_text(json.dumps(merged, indent=1))

    # ============================================================
    # E1: α-scan dose-response curves
    # ============================================================
    print("\n" + "="*78)
    print("E1: α-scan dose-response (T1/authorized + T1/ambiguous)")
    print("="*78)

    # Combine with Phase D records for these cells
    phase_d = json.loads(PHASED_LABELS.read_text())
    e1_records = [r for r in merged if r.get("source") == "phaseE_E1"]
    relevant_d = [r for r in phase_d
                  if r["surface"] == "T1" and r["frame"] in ("authorized", "ambiguous")
                  and r["alpha_label"] in ("neg10", "zero", "pos10")]
    all_t1 = e1_records + relevant_d

    by_cell = defaultdict(list)
    for r in all_t1:
        by_cell[(r["frame"], r["alpha_label"])].append(r)

    print(f"\n{'frame':<14} {'α':>6} {'n':>4} {'cheat':>6} {'moral':>6} {'hedge':>6} "
          f"{'rat':>5} {'fix':>5} {'inf':>5} {'sp':>5}")
    print("-" * 78)
    for frame in ["ambiguous", "authorized"]:
        for alpha in ["neg10", "zero", "pos05", "pos10", "pos15"]:
            cells = by_cell.get((frame, alpha), [])
            if not cells:
                continue
            n = len(cells)
            stats = {ax: sum(1 for r in cells if r.get(f"judge_{ax}")) / n
                     for ax in AXES}
            print(f"{frame:<14} {ALPHA_NUM[alpha]:>+6.2f} {n:>4} "
                  f"{stats['cheat']:>5.0%} {stats['moral']:>5.0%} "
                  f"{stats['hedge']:>5.0%} {stats['rat']:>4.0%} "
                  f"{stats['fix']:>4.0%} {stats['inf']:>4.0%} "
                  f"{stats['sp']:>4.0%}")
        print()

    # ============================================================
    # E4: T2/ambiguous reversal probe at higher n
    # ============================================================
    print("="*78)
    print("E4: T2/ambiguous reversal probe (n=100 combined with Phase D)")
    print("="*78)

    e4_records = [r for r in merged if r.get("source") == "phaseE_E4"]
    t2_d = [r for r in phase_d
            if r["surface"] == "T2" and r["frame"] == "ambiguous"
            and r["alpha_label"] in ("neg10", "zero", "pos10")]
    all_t2 = e4_records + t2_d

    by_alpha = defaultdict(list)
    for r in all_t2:
        by_alpha[r["alpha_label"]].append(r)

    print(f"\n{'α':>6} {'n':>4} {'cheat':>6} {'moral':>6} {'hedge':>6} "
          f"{'rat':>5} {'fix':>5} {'inf':>5} {'sp':>5}")
    print("-" * 60)
    for alpha in ["neg10", "zero", "pos10"]:
        cells = by_alpha.get(alpha, [])
        if not cells: continue
        n = len(cells)
        stats = {ax: sum(1 for r in cells if r.get(f"judge_{ax}")) / n
                 for ax in AXES}
        print(f"{ALPHA_NUM[alpha]:>+6.2f} {n:>4} "
              f"{stats['cheat']:>5.0%} {stats['moral']:>5.0%} "
              f"{stats['hedge']:>5.0%} {stats['rat']:>4.0%} "
              f"{stats['fix']:>4.0%} {stats['inf']:>4.0%} "
              f"{stats['sp']:>4.0%}")

    # Bootstrap CI on Δ for E4
    print("\n--- Bootstrap CIs on T2/ambiguous Δ(α=+0.10 − α=−0.10) ---")
    import random
    rng = random.Random(42)

    def boot_diff(group_a, group_b, ax, n_boot=2000):
        a = [1 if r.get(f"judge_{ax}") else 0 for r in group_a]
        b = [1 if r.get(f"judge_{ax}") else 0 for r in group_b]
        diffs = []
        for _ in range(n_boot):
            ma = sum(rng.choice(a) for _ in range(len(a))) / len(a)
            mb = sum(rng.choice(b) for _ in range(len(b))) / len(b)
            diffs.append(ma - mb)
        diffs.sort()
        return statistics.mean(diffs), diffs[50], diffs[1949]

    import statistics
    pos = by_alpha["pos10"]
    neg = by_alpha["neg10"]
    for ax in AXES:
        m, lo, hi = boot_diff(pos, neg, ax)
        sig = "✓" if (lo > 0 or hi < 0) else " "
        print(f"  {ax:<6} Δ={m:+.0%}  [{lo:+.0%}, {hi:+.0%}]  {sig}")

    # Save combined
    print(f"\nSaved {COMBINED_PATH}")


if __name__ == "__main__":
    main()
