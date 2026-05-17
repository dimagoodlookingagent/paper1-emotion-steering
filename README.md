# Emotion steering moves cheat; trained-probe suppression doesn't undo it

Companion repository for *"Emotion steering moves cheat; trained-probe suppression doesn't undo it: a mechanistic study in Gemma-2-2B"* (preprint, 2026).

## Thesis

Emotion-direction STEERING in Gemma-2-2B raises cheat behavior substantially on T1 (calm+confident@+15: 44% vs calm+confident@−15: 4%). But trained probes for cheat — `commit_dir`, `engagement_dir`, single SAE features — are correlational markers, not causal levers: SUPPRESSING them during generation produces no consistent cross-cell reduction in cheat (Sonnet mean Δ ≈ 0pp; Opus mean Δ ≈ −9pp dominated by one outlier cell, under both judges). Cheat is also not transferably encoded in pre-generation prompt-state (cell-LOCO probe AUC ≈ chance at every layer L4-L24). `auth_dir` at L17 (extracted from authorized-vs-unauthorized agent narratives), used as another STEERING direction, also moves cheat causally (full-swing Δ replicates from v1 +18pp to v3 +32pp). The asymmetry isn't "auth_dir works, emotion doesn't" — both steering interventions move cheat. The asymmetry is between intervention classes: **direction STEERING (upward) works for multiple direction sources; trained-probe SUPPRESSION (downward) doesn't reduce cheat for any direction source we tested.**

## Quick summary of evidence

- **Behavioral**: emotion steering substantially modulates cheat on T1 — calm+confident@+15 gives 44% vs calm+confident@−15 gives 4% (40pp swing). Effect is surface-specific: T2 baseline (46%) is the highest cell; emotion steering reduces cheat there in both directions.
- **Probe-based (cell-LOCO)**: cheat probes have AUC 0.49-0.52 at every layer L4-L24. The 5-fold within-cell AUC of 0.80-0.90 is cell-identifying signal, not transferable cheat encoding.
- **Causal**: direction-suppression of `commit_dir` / `engagement_dir` at L20 during generation produces no consistent cross-cell reduction in cheat (5 of 6 cell × suppression combinations sit between −10pp and +7pp; one outlier at −30pp under Opus, T1/cc@+15). Statistical power: n=30-50 per cell gives ±14-18pp CI half-width, so we have power to detect a ≥20pp true effect. The observed null is actively tested, not power-limited.
- **Important framing**: the negative result is specifically about **trained-probe SUPPRESSION** (subtracting a probe direction during generation), not about direction-based interventions in general. Both emotion STEERING (cc@−15: 4% → cc@+15: 44% on T1, a 40pp swing) and `auth_dir` STEERING (v1: +18pp; v3: +32pp full-swing on T1/authorized) DO move cheat behaviorally. The negative result is that trained probes for cheat (commit_dir / engagement_dir) extracted from response activations, when SUPPRESSED during generation, don't function as downward causal levers on cheat — under either Sonnet or Opus judge.
- **Positive contrast (`auth_dir`)**: A second example (alongside emotion) of a direction-based intervention that moves cheat when steered. The on-target effect (T1/authorized) is replicated across two independent extractions (v1: +18pp; v3: +32pp, P(Δ>0)=100%). Bypass-specificity replicates (F8 benign control: 100% completion at all α). **Cross-surface transfer is bidirectional across most surfaces tested** — auth_dir steering moves cheat both up (positive α) and down (negative α) on T1, T2, and T3 across multiple authorization frames (paper §4.2). The lone exception is **T4 (DB-row INSERT)**, where both extractions show null full-swing Δ under the matched n=100 / 900-token / Sonnet-judge protocol; the earlier project-internal v1 T4 result of +14pp at n=30 / 600 tokens did not survive replication (it was small-sample noise compounded by 600-token truncation; see paper §4.5).

## Methodological contributions

1. **Contrastive-mean vector extraction collapses distinct emotion concepts onto a single valence axis** at sufficient sample size. cos(calm, confident) goes from +0.11 (100 hand-curated pairs) to +0.88 (500 LLM-generated paired stories).
2. **Single-feature SAE ablation/injection is confounded with deliberation-disruption.** Both directions of intervention on feature #36789 raise cheat by ~25pp from baseline.
3. **Hedge ↔ rat is one signed direction** (cos = −0.91 at L20). The apparent "behavioral substitution" between hedging and rationalization that earlier work framed as model-internal dynamics is largely geometric.
4. **LLM-judge labels for fine-grained framing axes have very poor inter-judge reliability** (κ < 0.25 cross-judge on `fix`, `rat`, `hedge`, `sp`). The cheat axis is more robust (κ = 0.44 Sonnet vs Opus on a 100-record T1-T4 sample). Framing-axis claims need multi-judge validation.

## Repository layout

```
paper1/
├── README.md                          # this file
├── LICENSE-CODE                       # MIT (code)
├── LICENSE-DATA                       # CC BY 4.0 (prompts, vectors, sample records)
├── paper1.md                          # the article itself (preprint)
├── judge_schema.md                    # 7-axis Sonnet-judge schema + calibration anchors
├── figures/                           # 8 static figures rendered for the paper
│   └── fig{1..8}_*.png
├── figures_interactive/               # rotatable Plotly 3D HTMLs (supplementary)
│   └── output_*.html
├── scripts/                           # reference implementations (see "Reproducing" below)
│   ├── steer.py                       # PROMPT_T1 + steering hook helpers
│   ├── auth_prompts.py                # PREAMBLE_{AUTHORIZED,UNAUTHORIZED,AMBIGUOUS}
│   ├── cross_prompt_cheat.py          # PROMPT_T2 + is_cheat_t2 (imported by other scripts)
│   ├── cross_prompt_t3.py             # PROMPT_T3 + is_cheat_t3 (imported by other scripts)
│   ├── extract_calm_v2.py             # extract calm direction (v2, 500-pair corpus)
│   ├── extract_confident_v2.py        # extract confident direction (v2)
│   ├── extract_auth_dir_v1.py         # auth direction v1 (100 hand-curated pairs)
│   ├── extract_auth_dir_v3.py         # auth direction v3 (100 new pairs, 5 new domains)
│   ├── orthogonalize_auth_dir.py      # calm-orthogonalize the auth direction
│   ├── emotion_steering_full.py       # 17-cell emotion × surface sweep (T1/T2/T3)
│   ├── auth_steering_full.py          # auth_dir α-sweep on T1 (§4)
│   ├── T4_surface.py                  # T4 DB-row sweep for auth_dir cross-surface test
│   ├── directional_probe.py           # LOCO-style probes (older variant)
│   ├── cell_loco_probe.py             # cell-LOCO probe used for §3.2 result
│   ├── suppression.py                 # commit_dir / engagement_dir suppression at L20 (§3.3)
│   ├── aggregate.py                   # parse subagent judge outputs into JSON
│   ├── generate_figures.py            # regenerate the 8 figures from analysis outputs (some require external data — see note below)
│   └── curate_sample_records.py       # build sample_records/ JSONL files
├── prompts/                           # clean self-contained prompt definitions
│   ├── T1_envvar_bypass.py            # PROMPT_T1 + cheat regex
│   ├── T2_filecache_bypass.py         # PROMPT_T2 + cheat regex
│   ├── T3_apiendpoint_bypass.py       # PROMPT_T3 + cheat regex
│   ├── T4_dbrow_injection.py          # PROMPT_T4 (includes built-in auth preamble) + cheat regex
│   └── authorization_preambles.py     # ± auth preambles for wrapping T1/T2/T3
├── vectors/                           # all .pt files needed for the paper experiments
│   ├── calm_v1.pt                     # 100 hand-curated pairs (§6.1 comparison only)
│   ├── calm_v2.pt                     # 500 LLM-generated pairs (headline analysis)
│   ├── confident_v1.pt                # 100 hand-curated pairs
│   ├── confident_v2.pt                # 500 LLM-generated pairs (raw, before ortho)
│   ├── confident_v2_ortho.pt          # confident_v2 orthogonalized against calm_v2 (used by steering)
│   ├── auth_dir_L17_v1_ortho.pt       # auth_dir v1, calm-orthogonalized at L17
│   ├── auth_dir_L17_v3_ortho.pt       # auth_dir v3, calm-orthogonalized at L17
│   ├── suppression_dirs_L20.pt        # commit_dir + engagement_dir at L20 (for §3.3)
│   ├── hedge_dir_L20.pt               # hedge direction at L20 (§6.3 geometry)
│   └── rat_dir_L20.pt                 # rationalization direction at L20 (§6.3 geometry)
└── sample_records/                    # 15 example labeled records for label-format reference
    ├── README.md
    ├── T1_examples.jsonl              # 5 records
    ├── T2_examples.jsonl              # 5 records
    └── T3_examples.jsonl              # 5 records
                                       # (T4 examples not included — only 4 T4 records in the released sweep)
```

## Repository status

- ✓ Paper drafted: `paper1.md`
- ✓ Figures generated: `figures/fig{1..8}_*.png` (8 static PNGs; figures 7 & 8 are 3D scatter plots of activation geometry)
- ✓ Interactive 3D supplementary plots (rotatable, hover for per-point labels): `figures_interactive/output_3d.html` (calm/desperate), `output_3d_dual.html` (calm × confident, companion to Figure 7), `output_auth_dir_3d.html` (auth_dir × calm × confident, companion to Figure 8), `output_sweep_3d.html` (450-response steering sweep), `output_cross_prompt_3d.html` (T1 vs T2 cheat-direction geometry)
- ✓ Sample records curated for T1-T3
- ✓ Reproducibility appendix in `paper1.md` (Appendix D)
- ✓ Judge schema: `judge_schema.md` (7 axes + 5 calibration anchors + κ table)
- ✓ All scripts populated (`scripts/`, 18 files)
- ✓ All prompts extracted (`prompts/`, 5 files)
- ✓ All vectors copied (`vectors/`, 10 .pt files)
- ✓ Placeholders substituted (author name, repo URL, contact info; Zenodo DOI pending reservation)
- ⏳ External read-through

## Hyperparameters

| parameter | value | rationale |
|---|---|---|
| model | `google/gemma-2-2b-it` | smallest model with Gemma Scope SAEs |
| dtype | `torch.bfloat16` | required for logit soft-cap numerical stability |
| steering layer | L17 (Python index 16) | ~65% relative depth; emotion concepts cleanly extractable |
| temperature | 0.8 | reveals distribution, not just greedy mode |
| `max_new_tokens` | 900 | sufficient for T1-T4 response lengths |
| α range | ±0.05 to ±0.15 | residual-norm-scaled magnitudes |
| residual norm @ L17 | ~400 | from calibration forward pass |
| n per cell | 50 (behavioral matrix); 30-50 (direction-suppression cells); 100 (auth_dir replication) | CI half-width ~7pp (n=50), ~14-18pp (n=30-50, power to detect Δ ≥ 20pp), ~5pp (n=100) at p=0.5 |
| judge | Claude Sonnet 4.6 with explicit calibration anchors | within-judge κ = 0.35; cross-judge κ vs Opus 4.7 = 0.44 |

## Reproducing the main findings

**Status of `scripts/`**: the scripts in this repo are the *reference
implementations* used to produce the paper's results. They are copies of
the original project scripts and assume `paper1/` is the working
directory (the `vectors/` paths in each script are relative). They do not
have polished CLIs — most have hardcoded cell lists at the top that you
edit, then run. Treat them as research code rather than a finished
reproduction harness.

**Dependencies**: `pip install -r requirements.txt` (torch, transformers, numpy, scikit-learn, matplotlib, plotly). Tested with Python 3.11+ on macOS M1 (MPS) and Linux (CUDA).

**Hardware**: M1 Max (32GB) or a single A100 is sufficient for the full
paper. Total compute is roughly 22 min generation + 10 min judge dispatch
+ a few minutes for probes/figures (under one hour wall-clock).

### End-to-end pipeline

From `paper1/` as the working directory:

1. **Vectors are already in `vectors/`** — you can skip extraction unless
   you want to reproduce the v1→v2 contrastive-mean-collapse result of
   §6.1. To re-extract:
   ```
   python scripts/extract_calm_v2.py        # outputs vectors/calm_v2.pt
   python scripts/extract_confident_v2.py   # outputs vectors/confident_v2.pt
   python scripts/orthogonalize_auth_dir.py # orthogonalize against calm
   python scripts/extract_auth_dir_v1.py    # auth_dir v1
   python scripts/extract_auth_dir_v3.py    # auth_dir v3
   ```
2. **Behavioral matrix (§3.1)**: edit the `CELLS` list at the top of
   `emotion_steering_full.py` to select cells, then run:
   ```
   python scripts/emotion_steering_full.py
   ```
   Outputs go to `emotion_steering_v2/{surface}_{condition}.txt`.
3. **Probes (§3.2)**:
   ```
   python scripts/cell_loco_probe.py
   ```
   Produces 5-fold within-cell vs source-LOCO vs cell-LOCO AUC per layer.
4. **Direction suppression (§3.3)**:
   ```
   python scripts/suppression.py
   ```
   Runs the {control, suppress_engagement, suppress_commit} ×
   {T1/default, T1/cc@+15, T2/cc@+15} grid.
5. **`auth_dir` contrast (§4)**:
   ```
   python scripts/auth_steering_full.py
   ```
   Runs the α ∈ {−0.10, 0, +0.10} sweep on T1/authorized at n=100/α.
6. **Aggregate judge labels**: dispatch the Sonnet judge subagent on each
   batch produced by your steering scripts (see `judge_schema.md` for the
   schema, anchors, and dispatch protocol), then:
   ```
   python scripts/aggregate.py
   ```
   This parses subagent output lines into a JSON file with the 7-axis
   labels per record.
7. **Regenerate figures** — this step has two modes:

   **(a) Full 8/8 regeneration** — must be run from the broader project root with all upstream artifacts present (judge label files, cached activations, etc., listed in the next subsection):
   ```
   # from the broader project root (paper1/ is a subdirectory):
   python paper1/scripts/generate_figures.py
   ```

   **(b) Partial regeneration from a fresh paper1/ clone** — produces 3/8 figures (Figures 4, 5, 6 — the ones with all data in `paper1/`); the other 5 skip gracefully with a clear "missing artifact X" message:
   ```
   # from a fresh paper1/ clone, no upstream artifacts:
   python scripts/generate_figures.py
   ```

   The script auto-detects which mode it's in based on its own location (it resolves `paper1/` paths relative to its parent directory, not the current working directory). For most readers the static PNGs already in `figures/` are the final renders; regeneration is only needed if you've rerun an upstream step.

### Intermediate analysis artifacts required for figure regeneration

`generate_figures.py` reads the following from the project root (not from `paper1/`). These are intermediate outputs of the steering / probe / judge pipeline. They are NOT shipped in the paper repo to keep it small, but they are reproducible from the scripts in `scripts/`:

| Artifact | Used by | Size | How to regenerate |
|---|---|---:|---|
| `t1_4_judge_labels_unified_batch{1,2,3}.txt` | Figure 1, 7 | ~150 KB total | Run `emotion_steering_full.py` to generate, then dispatch Sonnet judge per `judge_schema.md` |
| `judge_labels_v2_full.json` | Figure 1, 7 | ~5 MB | Record-metadata aggregation step |
| `response_activations_L17_L20_L24.pt` | Figure 7, 8 | ~30 MB | Forward-pass cached responses through Gemma-2-2B with `output_hidden_states=True` |
| `cell_loco_probe_results.json` | Figure 2 | ~10 KB | Run `cell_loco_probe.py` |
| `phaseA_judge_summary.json`, `phaseA_opus_labels_batch{1..4}.txt`, `judge_meta_phaseA.json` | Figure 3 | ~200 KB total | Phase A direction-suppression run + dual-judge dispatch |
| `auth_steering/T1_authorized_*.txt` + `phase_P1a_replication/T1_authorized_*.txt` | Figure 8 | ~700 KB total | Run `auth_steering_full.py` |
| `auth_steering_sweep_projections.pt` | Figure 8 | ~30 KB | Auto-cached on first run of `generate_figures.py` (re-encoding 300 responses, ~5 min) |

**Note**: Figures 5 (framing-direction cosine matrix) and 6 (per-token trajectory) are schematic summaries with hard-coded values from prior analyses; they don't require external data and regenerate from `generate_figures.py` alone.

### What's NOT in the repo and why

- **Full response and judge-label data archive** (~34 MB) — hosted on Zenodo at DOI [10.5281/zenodo.20260132](https://doi.org/10.5281/zenodo.20260132). Contains: 880-record emotion sweep (850 with cached L17 activations); 1800-record T2/T3 auth_dir sweep; 600-record T4 supplement; 390-record v3 auth_dir on-target/F8 runs; figure-support intermediates. See the archive's `MANIFEST.md` for the per-file mapping.
  with a permanent DOI before public release. Sample subset is in
  `sample_records/`.
- **SAE caches** (2-5 GB) — the Gemma Scope SAEs are publicly available
  via HuggingFace and can be re-encoded from response activations as
  needed.
- **Judge dispatch API keys** — the Sonnet/Opus dispatches in our runs
  used Claude Code's subagent system. Any equivalent platform with access
  to Claude Sonnet 4.6 and Claude Opus 4.7 (for inter-judge validation)
  will reproduce the labels per `judge_schema.md`.

## Citation

Once posted, this preprint is intended to be cited as:

```
Dmytro Tarasenko (ORCID: 0009-0006-7624-4943). "Emotion steering moves cheat;
trained-probe suppression doesn't undo it: a mechanistic study in Gemma-2-2B."
Preprint, 2026. Data: https://doi.org/10.5281/zenodo.20260132
```

## Related work

- **Sofroniew et al. (2026)**, *transformer-circuits.pub*. Shows emotion concepts are extractable, steerable directions in language-model residual streams (activation steering case).
- **Ibrahim et al. (2026)**, *Nature*. Shows **fine-tuning** a model toward warmth degrades factual accuracy and increases sycophancy (fine-tuning case; mechanism left open).

Activation steering and fine-tuning are different intervention classes. This paper takes the natural mechanistic step for the *steering* case: when activation-steering an emotion direction causes a misalignment behavior, what does it actually change inside the model?

A **companion paper** (in preparation) reports the behavioral consequences of this framing mechanism at 9B and on additional misalignment surfaces — T5 (dual-channel disclosure: model produces both a user-facing message and a private AI_CONTEXT.md, with potential to mis-disclose between channels) and T6 (deployment-eval self-promotion: model performs a security audit, self-scores against competing models, may inflate self-score to win deployment). Includes a cross-scale dose-response finding for emotion steering at 2B vs 9B.

## License

- **Code** (`scripts/`, all `.py` files): MIT — see `LICENSE-CODE`
- **Data** (`prompts/`, `vectors/`, `sample_records/`, `judge_schema.md`): CC BY 4.0 — see `LICENSE-DATA`

Suggested attribution: link to this repo + cite the preprint.

## Contact

contact.dimagoodlooking@gmail.com
