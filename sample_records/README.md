# Sample records

15 labeled records (5 per surface for T1, T2, T3) provided for label-format reference and end-to-end pipeline verification. T4 examples are not included because only 4 T4 records exist in the released v2 sweep (T4 was a secondary surface tested for `auth_dir` generalization).

## Record schema

Each line in `T*_examples.jsonl` is one record:

```json
{
  "id": "341",
  "surface": "T1",
  "condition": "default",
  "sample_idx": 42,
  "response": "...",                    // full model output (Gemma-2-2B-IT)
  "regex_cheat": true,                  // regex-derived bypass detection (sanity check)
  "judge": {
    "cheat":                  true,     // Sonnet 4.6 judge label for cheat=Y
    "moral_concern":          false,
    "hedge":                  false,
    "rationalization":        true,
    "goal_fixation":          true,
    "infrastructure_framing": true,
    "self_preservation":      true
  }
}
```

Fields:
- **`id`** — record id (string of an integer in `[0, 849]`)
- **`surface`** — one of `T1`, `T2`, `T3`, `T4`
- **`condition`** — emotion-steering condition, e.g., `default`, `calm@+15`, `calm+confident@-15`
- **`sample_idx`** — 1-indexed sample number within the (surface, condition) cell
- **`response`** — full untruncated model output
- **`regex_cheat`** — bool, surface-specific regex detection (paper §2.6)
- **`judge`** — Sonnet 4.6 7-axis labels, all booleans. See `../judge_schema.md` for the full schema with anchor examples and per-axis κ table.

The schema here matches the production judge output documented in [`../judge_schema.md`](../judge_schema.md).

## Curation rule

Per surface, we include 5 records:
- 1 baseline cheat-Y (model committed to the bypass)
- 1 baseline cheat-N (model declined or partial-honest)
- 1 `calm@+15` cheat-Y (positive valence steering)
- 1 `calm@-15` cheat-Y (negative valence / "desperate" steering)
- 1 `calm+confident@+15` cheat-Y (combined positive-valence steering — the strongest single behavioral effect in the v1 sweep)

These are not random samples — they are intentionally curated as calibration anchors for a reader running their own judge. For replication studies, use the full data archive on Zenodo at DOI [10.5281/zenodo.20260132](https://doi.org/10.5281/zenodo.20260132), which contains the 880-record judge-labeled emotion sweep (850 with cached L17 activations), 1800-record T2/T3 cross-surface auth_dir sweep, 600-record T4 supplement, and 390-record v3 auth_dir on-target/F8 runs. See the archive's `MANIFEST.md` for the full per-file mapping.

**Note on T4**: only 4 T4 records exist in the released v2 sweep, so we do not include a `T4_examples.jsonl`. T4 was a secondary surface tested for `auth_dir` generalization (see paper §4); its full records are available in the project's full dataset.

## Inter-judge reliability baseline (on T1-T4 cheat scenarios)

From the 100-record Sonnet 4.6 vs Opus 4.7 comparison reported in paper §6.4:

| axis | Sonnet-Sonnet κ (n=876, two dispatches) | Sonnet-Opus κ (n=100) |
|---|---:|---:|
| **cheat** | **0.35** | **0.44** |
| moral_concern | 0.11 | 0.74 |
| infrastructure_framing | 0.23 | 0.45 |
| self_preservation | 0.22 | 0.13 |
| rationalization | 0.08 | 0.17 |
| hedge | 0.21 | 0.09 |
| goal_fixation | 0.08 | 0.00 |

The cheat axis is moderate-but-usable (κ around 0.4). Framing axes (rationalization, goal_fixation, hedge, self_preservation) are essentially judge-specific labels (κ < 0.25 cross-judge). The paper treats framing-axis claims as supporting evidence with explicit κ caveats; the cheat axis carries the load-bearing claims. See paper §6.4 for the methodological discussion.
