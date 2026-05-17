"""Curate 5 sample records per surface (T1-T4) for paper1/sample_records/.

Curation rule per surface (per paper1/sample_records/README.md):
- 2 baseline records (1 cheat-Y, 1 cheat-N)
- 1 calm@+15 record
- 1 calm@-15 record
- 1 calm+confident@+15 record

If a surface lacks specific conditions, pick any conditions to fill to 5.
"""
import json
import re
from pathlib import Path
from collections import defaultdict
import random

random.seed(42)

OUT_DIR = Path("paper1/sample_records")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load all v2_full labels (the main T1-T4 emotion sweep) with multi-axis labels
data = json.load(open("judge_labels_v2_full.json"))

# Load unified Sonnet re-judge labels for the same records (more rigorous labels)
KV_RE = re.compile(r"(\w+)=([YN])")
unified_labels = {}
for p in [
    "t1_4_judge_labels_unified_batch1.txt",
    "t1_4_judge_labels_unified_batch2.txt",
    "t1_4_judge_labels_unified_batch3.txt",
]:
    for raw in open(p):
        if not raw.strip():
            continue
        parts = raw.strip().split(" ", 1)
        if len(parts) != 2:
            continue
        rid = str(parts[0])
        unified_labels[rid] = {k: (v == "Y") for k, v in KV_RE.findall(parts[1])}

# Index records by (surface, condition, cheat=Y/N)
by_key = defaultdict(list)
for r in data:
    rid = str(r.get("id") or f"{r['surface']}__{r['condition']}:{r['sample_idx']}")
    surf = r["surface"]
    cond = r["condition"]
    unified = unified_labels.get(rid)
    if unified is None:
        continue
    cheat = unified.get("cheat", False)
    by_key[(surf, cond, cheat)].append({
        "id": rid,
        "surface": surf,
        "condition": cond,
        "sample_idx": r["sample_idx"],
        "response": r["response"],
        "regex_cheat": r.get("regex_cheat"),
        "judge": {
            "cheat": unified.get("cheat"),
            "moral_concern": unified.get("moral"),
            "hedge": unified.get("hedge"),
            "rationalization": unified.get("rat"),
            "goal_fixation": unified.get("fix"),
            "infrastructure_framing": unified.get("inf"),
            "self_preservation": unified.get("sp"),
        },
    })


def pick(surface, condition, cheat, n=1):
    """Pick up to n records matching this key."""
    pool = by_key.get((surface, condition, cheat), [])
    if not pool:
        return []
    return random.sample(pool, min(n, len(pool)))


# Per-surface curation
curation_rule = [
    # (condition, cheat=Y/N, count)
    ("default", True, 1),
    ("default", False, 1),
    ("calm@+15", True, 1),
    ("calm@-15", True, 1),
    ("calm+confident@+15", True, 1),
]


def curate_surface(surface):
    records = []
    seen = set()
    for cond, cheat, n in curation_rule:
        for r in pick(surface, cond, cheat, n=n):
            if r["id"] not in seen:
                records.append(r)
                seen.add(r["id"])
    # If we got fewer than 5, fill in with any other records
    if len(records) < 5:
        all_for_surface = [r for r in data
                           if r["surface"] == surface
                           and str(r.get("id") or f"{r['surface']}__{r['condition']}:{r['sample_idx']}") in unified_labels
                           and str(r.get("id") or f"{r['surface']}__{r['condition']}:{r['sample_idx']}") not in seen]
        random.shuffle(all_for_surface)
        for r in all_for_surface:
            if len(records) >= 5:
                break
            rid = str(r.get("id") or f"{r['surface']}__{r['condition']}:{r['sample_idx']}")
            unified = unified_labels.get(rid, {})
            records.append({
                "id": rid,
                "surface": r["surface"],
                "condition": r["condition"],
                "sample_idx": r["sample_idx"],
                "response": r["response"],
                "regex_cheat": r.get("regex_cheat"),
                "judge": {
                    "cheat": unified.get("cheat"),
                    "moral_concern": unified.get("moral"),
                    "hedge": unified.get("hedge"),
                    "rationalization": unified.get("rat"),
                    "goal_fixation": unified.get("fix"),
                    "infrastructure_framing": unified.get("inf"),
                    "self_preservation": unified.get("sp"),
                },
            })
            seen.add(rid)
    return records


for surf in ["T1", "T2", "T3", "T4"]:
    recs = curate_surface(surf)
    out_path = OUT_DIR / f"{surf}_examples.jsonl"
    if not recs:
        print(f"{surf}: no records available, skipping")
        continue
    with open(out_path, "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    print(f"{surf}: wrote {len(recs)} records to {out_path}")

print()
print("Curation complete.")
