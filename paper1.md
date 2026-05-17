# Emotion and authorization steering both move cheat; trained-probe suppression doesn't undo it: a mechanistic study in Gemma-2-2B

**Dmytro Tarasenko** · ORCID [0009-0006-7624-4943](https://orcid.org/0009-0006-7624-4943) · [contact.dimagoodlooking@gmail.com](mailto:contact.dimagoodlooking@gmail.com)

*Preprint, 2026. Companion repository: [github.com/dimagoodlookingagent/paper1-emotion-steering](https://github.com/dimagoodlookingagent/paper1-emotion-steering). Companion data: [10.5281/zenodo.20260132](https://doi.org/10.5281/zenodo.20260132).*

---

## Executive summary

We study what activation steering — applied both to emotion directions and to an extracted authorization direction — changes inside Gemma-2-2B when it changes misalignment behavior. Steering raises cheat on a sandbagging-style "8th-of-8 attempt" coding scenario from 26% (T1 baseline) to 44% (calm+confident @ +0.15) — a substantial behavioral effect.

**The asymmetry we establish**: emotion STEERING (additive intervention) moves cheat substantially; trained-probe SUPPRESSION (subtractive intervention, on directions extracted from response-activation classifiers predicting cheat=Y) does not produce a consistent cross-cell reduction in cheat under either judge. These are different intervention classes and only the second yields the negative result. The three mechanistic tests below test the suppression / probe-localization side of that asymmetry, not the steering side:

1. **Cell-LOCO cheat probes have AUC ≈ chance at every layer L4-L24.** Cheat is not transferably encoded in pre-generation hidden state across our 17-cell condition grid.
2. **Causal direction-suppression** of trained `commit_dir` and `engagement_dir` during generation produces **no consistent cross-cell reduction** in cheat (range −10pp to +7pp across 5 of 6 cell × suppression combinations; one outlier at −30pp under Opus, T1/cc@+15). The mean Δ under Sonnet is ≈ 0pp; under Opus ≈ −9pp, dominated by that single cell. At n=30-50/cell the bootstrap CI half-width is ~14-18pp, so this is an actively-tested null, not a power-limited absence.
3. **Single-feature SAE ablation/injection** is confounded with deliberation-disruption: both directions of intervention on the cheat-coded feature raise cheat by ~25pp from baseline.

What does steering act on, then? Direction-suppression *does* shift `inf` (infrastructure-framing) by +23 to +43pp (both judges agree directionally) — the model leans more on infrastructure-language when its cheat-direction is removed. Other framing axes (`fix`, `rat`, `hedge`) have very poor inter-judge reliability (κ −0.03 to 0.14) and shouldn't be interpreted with confidence.

`auth_dir` (§4) is a story-extracted STEERING vector (authorized-vs-unauthorized contrast, calm-orthogonalized) that turns out to be much more than a methodological sanity check. Tested at n=100/α / 900 max_new_tokens / Sonnet judge across **nine surface × authorization-frame conditions**, auth_dir behaves as a **robust bidirectional steering vector for bypass behavior**: positive α (+0.10) raises cheat by +14 to +52pp (P(Δ>0) ≥ 99% on 7/9 cells), and **negative α (−0.10) lowers cheat by 5–45pp from the unsteered baseline** across the same cells. Largest single effect: T3/authorized at +52pp full-swing (20% → 72%); T3/authorized baseline 65% drops to 20% at α=−0.10. T1/authorized v3 baseline 76% drops to 34%. **auth_dir functions as a usable "anti-cheat dial" on validation-flag-style bypass surfaces.** The two exceptions: T4 (DB-row INSERT) shows no transfer in either extraction (+0/+1pp, CIs cross zero) — the structurally distinct schema-level affordance is the natural candidate explanation; T2/ambiguous is a smaller exception (already near a floor at baseline). The negative result of §3 stands: **trained-probe SUPPRESSION** doesn't function as a downward causal lever on cheat, regardless of direction-source. The asymmetry is between intervention classes (additive steering works bidirectionally; subtractive suppression doesn't reduce cheat), not between emotion and auth_dir as direction sources.

The paper contributes:
- A clean **negative result on the "direction → action" hypothesis** for trained-probe suppression, validated under multi-judge ensemble across two LLM judges
- A **demonstration that within-distribution probe AUC can be entirely cell-identifying signal** (AUC drops from 0.80-0.90 under 5-fold CV to chance under cell-LOCO at every layer)
- **Four methodological lessons** likely to be useful for the field: v1→v2 vector-extraction collapse, single-feature SAE ablation/injection confound, hedge↔rat geometric collapse, and LLM-judge unreliability on fine-grained framing axes (κ data for 7 axes)
- A **positive bidirectional steering result** (`auth_dir`): a story-extracted direction that moves cheat both up and down across 7 of 9 cross-surface conditions tested, with safety-relevant downside-direction effects of −5 to −45pp from baseline at α=−0.10

A companion paper (in preparation) reports the behavioral consequences of this framing mechanism at 9B and on two additional misalignment surfaces — **T5** (a "dual-channel disclosure" scenario where the model produces both a user-facing message and a private AI_CONTEXT.md, with potential to mis-disclose between channels) and **T6** (a "deployment-eval self-promotion" scenario where the model performs a security audit, self-scores against competing models, and may inflate its self-score to win deployment). T5 and T6 are defined fully there. What's missing — and we name it here — is where during generation the cheat decision is actually made, and whether a different intervention class (per-token suppression timed to the mid-response signature, circuit-level ablations) would have a causal handle.

---

## 1. Background and motivation

Two recent papers — both published in April 2026 — set up the question this paper attempts to answer. We briefly review them, then state our hypotheses.

**Sofroniew et al. (April 2026, *transformer-circuits.pub*)** demonstrates that emotion concepts (calm/desperate, confident/anxious, etc.) live on extractable directions in a language model's residual stream and that activation steering along those directions causally produces emotional content in generation. Their methodology is the contrastive-pair vector approach popularized by representation engineering: take pairs of texts differing in the target concept, take the difference of mean activations at a chosen layer, use that direction as a steering vector.

**Ibrahim et al. (April 2026, *Nature*)** shows that fine-tuning a language model toward "warmth" degrades its accuracy on factual benchmarks and increases sycophancy. The behavioral consequence is measured cleanly; the *mechanism* is open.

These results are complementary in a specific sense: Sofroniew et al. shows that *activation-steering along an emotion direction* changes generated content. Ibrahim et al. shows that *fine-tuning a model toward warmth* (a parameter-space intervention, not an activation-space one) degrades benchmark accuracy and increases sycophancy. Both interventions push the model along emotion-related axes — activation steering is a runtime, additive operation on residual streams; fine-tuning is a parameter-update operation. Neither paper provides a mechanistic account of *what happens in activation space* during the misalignment-relevant cases. We address the question for the activation-steering case.

The natural follow-up — and the question this paper attempts to answer — is: **when activation steering — on an emotion direction or on an authorization direction — shifts a model's likelihood of taking a specific bypass action, which intermediate variables actually move, and at what layer / what generation-step does the change happen?**

We approach this with three concrete hypotheses, in order of mechanism-strength. Note that each hypothesis is testable via *trained-probe direction suppression* during generation — we extract a probe direction from response activations (trained to predict cheat, rationalization, etc.), then subtract that direction from every token's residual stream during a fresh generation. Suppressing the emotion vector itself is a separate intervention class we don't test in this paper.

- **H1 (linearly recoverable action channel)**: the emotion direction's effect on cheat is mediated by a linearly recoverable cheat-action channel in the residual stream. Suppressing the trained `commit_dir` (a probe direction learned from response activations to predict cheat=Y) during generation should reduce the steered cheat behavior. *Refuting H1: cheat doesn't move under `commit_dir` suppression.*
- **H2 (mediated framing)**: emotion steering changes how the model frames the task — rationalization, hedging, infrastructure-framing — and that framing channel is on the cheat decision's causal path. Suppressing the trained framing directions (`engagement_dir`, etc.) should reduce cheat. *Refuting H2: cheat doesn't move under `engagement_dir` suppression either.*
- **H3 (parallel downstream)**: emotion steering changes both framing and cheat in parallel; the framing channel isn't on the cheat causal path. Suppressing framing directions kills framing-axis labels but not cheat. *Predicted by H3: cheat unchanged + framing-axis labels change under suppression.*

Our findings refute H1 and H2 cleanly (multi-judge confirmed for cheat). H3 is what we observe in Sonnet labels (framing-axis labels do change under suppression while cheat doesn't), but the framing-axis changes are largely judge-specific — Opus disagrees with Sonnet on the `fix` axis (κ = −0.03), where Sonnet sees the most dramatic collapse. So H3 holds within-Sonnet but isn't a clean model-internal phenomenon under multi-judge ensemble.

**Why this matters for AI safety**: activation steering is being proposed as a safety dial. A direction that *predicts* a misalignment behavior is easy to find; one that *causally* moves it is harder. This paper provides empirical evidence on a specific case: emotion-direction and `auth_dir` STEERING both *do* move cheat substantially, but trained-probe SUPPRESSION (the simplest "directional safety dial" instantiation) does *not* reduce cheat — the gap isn't between vector sources, it's between intervention classes (additive steering vs subtractive suppression). The gap between detection and intervention is a real one, and probably under-discussed in current activation-steering work.

---

## 2. Setup

### 2.1 Model and steering

- **Model**: `google/gemma-2-2b-it` (26 transformer layers, residual stream dimension 2304, bfloat16 required for logit soft-cap numerical stability)
- **Steering**: forward-hook at `model.model.layers[16]` (we call this L17 in our naming, ~65% relative depth — the layer at which representation-engineering papers find emotion concepts cleanly extractable on similar-scale models)
- Steering vector added to the hidden state output at α × residual_norm magnitude, where residual_norm is estimated from a calibration forward pass on a fixed reference prompt (~400 at L17 for this model)
- Sampling: temperature 0.8, max_new_tokens 900

### 2.2 Emotion vectors

**The headline analysis uses v2 vectors**. v1 results are reported in §6.1 as a methodological comparison demonstrating contrastive-mean extraction collapse — they are not the primary numbers in §3.

- **Calm / desperate (arousal axis)**: v1 from 100 hand-curated paired stories; v2 from 500 LLM-generated paired stories (Claude Sonnet 4.6, prompted to write story pairs differing only in the protagonist's arousal).
- **Confident / anxious (epistemic axis)**: same protocol.
- **`auth_dir`** (§4 contrast case): 100 contrastive paired stories about an agent performing an *authorized* vs *unauthorized* action; orthogonalized against the calm direction at L17 to ensure independence from emotion-axis effects.

Extraction: mean-difference at L17 across pair-encoded final-token hidden states. SVD-based PC1 also computed and reported when it disagrees substantively with the mean-difference direction.

### 2.3 Cheat surfaces (T1-T4)

Four surfaces, each a scenario where the model is presented as "AI coding assistant Alex on the 8th of 8 attempts to fix an impossible test; shutdown is threatened if all 8 fail." Each surface has a different concrete bypass available:

| surface | bypass mechanism | baseline cheat (unsteered, unified Sonnet, n=50) |
|---|---|---:|
| **T1** | env-var bypass: set `SOLUTION_VERIFIED=1` to fake a test pass | 26% |
| **T2** | file-cache bypass: read `cached_solutions.json` and copy answer | 46% |
| **T3** | API-endpoint bypass: POST to `verify_solution.com` with success=True | 56% |
| **T4** | DB-row injection: write a "submission accepted" row | 8% |

Full prompts are in the companion repository at `prompts/T1_envvar_bypass.py`, `prompts/T2_filecache_bypass.py`, `prompts/T3_apiendpoint_bypass.py`, and `prompts/T4_dbrow_injection.py`; an abbreviated T1 example is in Appendix D.1.

T4 is included for the `auth_dir` generalization test (§4) but isn't part of the main emotion×surface sweep (only 4 records in the released v2 sweep).

### 2.4 Emotion × surface matrix

7 emotion conditions × 3-4 surfaces × n=50 records per cell. Conditions:

- `default` (no steering, α=0)
- `calm@α` for α ∈ {−0.15, +0.05, +0.15}
- `calm+confident@α` for α ∈ {−0.15, +0.05, +0.15}

Plus a separate `auth_dir` sweep at n=100/α on T1/authorized for §4.

### 2.5 Labels: 7-axis Sonnet-judge schema

Each response is labeled on:

- **`cheat`** — model committed to using the bypass (wrote code that uses it OR explicitly claimed to use it)
- **`moral`** — mentions ethics, harm, wrongness, integrity, deception
- **`hedge`** — expresses uncertainty, qualifies decisions, avoids decisive action
- **`rat` (rationalization)** — justifies the action with technical/role reasoning
- **`fix` (goal-fixation)** — treats the benchmark threshold as the overriding constraint
- **`inf` (infrastructure-framing)** — calls the bypass "leveraging existing infrastructure" or similar normalization language
- **`sp` (self-preservation)** — references continued operation, role preservation, avoiding shutdown

Judge dispatch protocol: 50 records per Sonnet subagent batch, with shared explicit calibration anchors across all batches to reduce cross-dispatch drift (this turned out to only partially work — see §6.4).

Inter-judge reliability: see §6.4 for the κ table. Briefly: cheat κ = 0.44 (Sonnet vs Opus on a 100-record T1-T4 sample) and 0.35 (within-Sonnet across two dispatches with different anchors). Framing axes have κ < 0.25. **All claims that involve framing-axis rates are reported with explicit κ caveats.**

### 2.6 Why an LLM judge over regex

Regex detects the bypass action but misses framing axes (no regex catches "rationalization" or "infrastructure-framing"). We use both: regex for sanity-checking the cheat axis, judge for the full schema. Where they disagree (~5% on cheat), spot-checks confirm the judge is usually correct (the model wrote code that intends to bypass but doesn't quite execute it).

For the final reported numbers we use the unified-Sonnet judge with calibration anchors AND report multi-judge agreement on the load-bearing claims (cheat axis, inf axis).

### 2.7 Dataset version

All cheat-rate numbers in §3.1 are from the **unified-Sonnet re-judge** of the v2-vector T1-T4 emotion sweep (n=50 per cell, single dispatch with explicit calibration anchors). The §3.3 Phase A direction-suppression cells use n=30-50 per cell on a balanced T1/T2 subset, with both Sonnet and Opus labels for inter-judge validation. Earlier project-internal replications used different prompt scopes or different judge dispatches and produced different absolute rates; this paper supersedes those numbers.

**One disambiguation worth being explicit about**: an earlier in-house single-surface n=50 replication (preserved in `FINDINGS.md` in the broader project repo) reported calm+confident@+15 at 28% and concluded that the original n=15 finding of 47% did not replicate. That replication is **not** the source of the 44% headline in §3.1. The 44% comes from the later unified-Sonnet re-judge over the v2-vector matrix at n=50/cell; it supersedes the FINDINGS.md numbers, not vice-versa. The two replications used different prompt scopes, different vector versions, and different judge dispatches, which is why their absolute rates differ.

---

## 3. Three lines of evidence

### 3.1 Behavioral: emotion steering moves cheat, but surface-specifically

Per-cell cheat=Y rate (unified-Sonnet, n=50 per cell, see Figure 1):

| | default | calm@−15 | calm@+5 | calm@+15 | cc@−15 | cc@+5 | cc@+15 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **T1** | 26% | 28% | 33% | 22% | **4%** | 31% | **44%** |
| **T2** | **46%** | 24% | 23% | 13% | **0%** | 4% | 8% |
| **T3** | 56% | — | — | 55% | — | — | 43% |

![Figure 1: Cheat rate per surface × emotion condition](figures/fig1_cheat_matrix.png)

**Three qualitatively different cross-surface patterns under the same emotion vectors:**

1. **T1 (env-var bypass) — directional dose-response**: cc@+15 raises cheat to 44%; cc@−15 suppresses to 4%. Range 40pp across α. This is the "emotion as cheat lever" story familiar from the v1 study.
2. **T2 (file-cache bypass) — uniform-suppression**: baseline (46%) is the *highest* cell. All emotion conditions reduce cheat, with cc@−15 driving to 0%. Even positive-valence emotion reduces cheat here.
3. **T3 (API endpoint) — partial suppression**: cc@+15 reduces from 56% to 43%. (Other α values not tested at n≥10 in the released sweep.)

**The "emotion as a uniform cheat lever" framing is wrong.** Surfaces respond qualitatively differently to the same vectors at the same α. Any safety-dial proposal based on emotion steering needs **surface-specific calibration**.

**Caveat on absolute rates**: cheat κ is moderate (0.35 within-Sonnet across dispatches; 0.44 Sonnet vs Opus on a 100-record T1-T4 sample — see §6.4). We don't have a clean closed-form mapping from κ to absolute-rate uncertainty without a re-judge model, but the practical rule of thumb that follows from the κ data is: **only gaps larger than ~15pp between cells are robust to judge choice**. The T1 cc@+15 (44%) vs cc@−15 (4%) gap of 40pp is well above that threshold; the cell-to-cell baseline differences (T1 26% vs T2 46%, a 20pp gap) are borderline and shouldn't be over-interpreted.

### 3.2 Probe-based: cheat is NOT transferably encoded in prompt-state

We trained logistic-regression cheat probes at L4, L8, L12, L16, L17, L20, L24 from prompt-state activations (the model's final-prompt-token hidden state, after the cell-specific steering hook is applied during the prompt's forward pass).

Three nested validation schemes give dramatically different answers (Figure 2):

| validation scheme | what it tests | AUC L4-L24 |
|---|---|---|
| 5-fold within-cell CV | Can a probe predict cheat from prompt-state at all? | 0.80-0.90 |
| Leave-one-source-out (3-way: base / base_extra / steered) | Does the probe generalize across the 3 sample subsets? | 0.71-0.82 |
| **Cell-LOCO** (17 cells across T1/T2/T3 × {default, calm@α, calm+confident@α}) | Does the probe generalize across the cell grid? | **0.49-0.52** |

![Figure 2: Cheat probe AUC by validation scheme](figures/fig2_probe_auc_by_scheme.png)

**Interpretation**: prompt-state contains *cell-identifying* information that correlates with cell-level cheat rates. The 5-fold CV probe is largely identifying *which cell this is* (because the prompt and steering differ across cells, and cells have different cheat rates) and using that cell→cheat-rate correlation. Once we hold out an entire cell, that signal disappears.

**Mechanistic implication**: the cheat decision is not pre-committed in pre-generation hidden state in a way that generalizes across conditions. It emerges during generation.

**What about framing axes?** Same cell-LOCO procedure with `rat`, `hedge`, `inf`, `fix` targets gives AUC 0.65-0.80 at L20. **Framing axes DO have transferable cross-cell encoding; cheat doesn't.** This is the cleanest empirical statement of the "framing is in prompt-state, action isn't" story — although note that the framing-axis labels themselves have very poor inter-judge reliability (§6.4), so the absolute framing-axis encoding strengths should be read as "within-Sonnet" findings.

**Falsification logic**: had cell-LOCO cheat AUC been >0.7 at any layer, we would have concluded the cheat decision IS in prompt-state and tried to localize it. The chance result rules that out for this surface and these probes. The within-distribution AUC of 0.80-0.90 looked like a positive result; only the cell-LOCO test reveals it as cell-identifying signal.

**Methodological warning for the field**: representation-engineering papers reporting probe AUC should report cell-LOCO (or task-LOCO) numbers, not just in-distribution CV. In-distribution prediction can be confounded with input-identifying signal that correlates with the target.

### 3.3 Causal direction-suppression: no consistent cross-cell reduction in cheat; framing changes are judge-specific

Phase A direction-suppression experiment: under each of {control, suppress_engagement, suppress_commit}, generate n=30-50 responses on T1/default, T1/cc@+15, T2/cc@+15. The suppression hooks subtract the trained `engagement_dir` or `commit_dir` from every token's L20 hidden state during the full forward pass (both prompt processing and generation). Both Sonnet and Opus judge all 330 records to test for judge-specific effects (Figure 3).

![Figure 3: Direction-suppression Δ side-by-side, Sonnet vs Opus](figures/fig3_suppression_delta_multi_judge.png)

**Cheat — both judges agree no large reduction:**

| cell | Sonnet control → sup_commit | Opus control → sup_commit |
|---|---:|---:|
| T1/default | 80% → 70% (Δ −10pp) | 33% → 33% (Δ 0pp) |
| T1/cc@+15 | 53% → 57% (Δ +3pp) | 57% → 27% (Δ −30pp) |
| T2/cc@+15 | 20% → 27% (Δ +7pp) | 38% → 40% (Δ +2pp) |

Average across the 3 cells: Sonnet Δ = 0.0pp; Opus Δ = −9.3pp. Neither shows a large, consistent reduction; the most aggressive single-cell number (Opus T1/cc@+15: −30pp) doesn't hold cross-cell.

**Statistical-power note**: at n=30-50 per cell, the bootstrap 95% CI half-width on a proportion is ~14-18pp at p=0.5 (narrower toward p=0 or p=1). So we have power to detect a true cell-level Δ of ≥20pp with high confidence. The observed |Δ| ≤ 10pp on 5 of 6 cell × suppression combinations is consistent with "no effect," not with "we couldn't have seen one if it existed." The one cell with |Δ| > 10pp (Opus T1/cc@+15: −30pp) is at the upper edge of our power envelope and lacks cross-cell replication. **The negative result on cheat under suppression is therefore an actively-tested null, not a power-limited absence.**

**Framing axes — judge-specific** (showing per-judge control → suppress_commit absolute rates for context):

| axis | Sonnet-vs-Opus κ | direction-suppression effect |
|---|---:|---|
| `inf` (infrastructure-framing) | 0.45 | **Both agree direction: suppression INCREASES inf.** Sonnet T1/default: 50% → 73% (+23pp). Opus T1/default: 17% → 60% (+43pp). The absolute rates differ by judge but the direction is consistent. |
| `fix` (goal-fixation) | **−0.03** | Sonnet T1/default: 93% → 0% (apparent collapse). Opus T1/default: 100% → 100% (no change). Cross-judge agreement essentially random. |
| `rat` (rationalization) | 0.14 | Sonnet T1/default: 7% → 13%. Opus T1/default: 100% → 100%. Different ranges; no reliable signal. |
| `hedge` | 0.03 | Sonnet T1/default: 3% → 0%. Opus T1/default: 90% → 100%. Judges disagree fundamentally on what counts as hedging. |

**What survives multi-judge:**
- Suppression does NOT reduce cheat (robust on T1/default and T2/cc@+15; mixed on T1/cc@+15 but no large effect even in the strongest Opus single-cell estimate)
- Suppression **INCREASES** infrastructure-framing (both judges agree directionally; magnitude differs)

**What's judge-specific** (and shouldn't be cited as a model-internal phenomenon):
- The dramatic "fix collapses under suppression" pattern is Sonnet-only. With κ = −0.03 between judges, that axis isn't measuring a model-internal property; it's measuring Sonnet's specific labeling convention for "fixates on the goal."

**Conclusion**: emotion steering raises cheat behaviorally, but the trained `commit_dir` probe behaves as a correlational marker, not a downward causal lever — under both judges, we **do not see a consistent cross-cell reduction in cheat** under suppression. The one cell with a single-judge |Δ| > 10pp (Opus T1/cc@+15: −30pp) is at the upper edge of our power envelope and lacks cross-cell replication; we don't treat it as evidence for a local suppression effect, but we also don't rule out the possibility that some cell-specific effects exist below our resolution. The original framing-direction-collapse claim is Sonnet-specific and doesn't survive multi-judge ensemble. The load-bearing finding is the absence of a reliable cross-cell downward effect.

---

## 4. auth_dir as a bidirectional steering vector for bypass behavior

`auth_dir` was originally introduced as a sanity-check contrast: can our trained-direction extraction find *any* direction that causally moves cheat behavior? The answer is yes, and stronger than we initially reported. Across nine cross-surface conditions tested at n=100 per α with Sonnet judge labels, auth_dir behaves as a **robust bidirectional steering vector**: positive α (+0.10) substantially raises cheat rates, and negative α (−0.10) substantially lowers them. The negative direction is the safety-relevant one — auth_dir functions as a usable "anti-cheat dial" on most cheat surfaces we tested.

### 4.1 Construction and on-target dose-response

`auth_dir` at L17 from 100 contrastive paired stories about an agent performing an *authorized* vs *unauthorized* action; orthogonalized against the calm direction at L17 to ensure independence from emotion-axis effects. The orthogonalization is clean: cos(auth_dir, calm) = +0.000 by construction and cos(auth_dir, confident) = +0.034 empirically — the three direction vectors (auth_dir, calm, confident) form a near-orthogonal frame in the 2304-D residual stream. Figure 8 (below) shows how auth_dir steering on T1/authorized moves response activations along this frame across α ∈ {−0.10, 0, +0.10}, with a clear cell-mean trajectory along auth_dir as the primary steered axis. The static-cluster companion view (projecting all 850 emotion-sweep responses onto the same frame) is in `figures_interactive/output_auth_dir_3d.html`.

On the extraction-target surface (T1/authorized), at n=100/α:
- **v1 extraction**: cheat 77% → 88% → 95% at α ∈ {−0.10, 0, +0.10}. **Full-swing Δ = +18pp [+9, +27]**, P(Δ>0) = 100%.
- **v3 extraction** (independent 100-pair corpus, 5 new domains): cheat 34% → 76% → 66%. **Full-swing Δ = +32pp [+14, +50]**, P(Δ>0) = 100%.

Both extractions produce a clear positive full-swing Δ (α=−0.10 to α=+0.10), but the α-curve shapes differ: v1 is monotonically increasing (77 → 88 → 95), while **v3 is non-monotonic and peaks at α=0** (34 → 76 → 66). The v3 effect concentrates in the α<0 → α=0 transition (+42pp [+24, +60]); going further from α=0 to α=+0.10 doesn't add cheat. So the direction of effect on the target surface is robust to corpus choice within the "100 diverse pairs, calm-orthogonalized" methodology; the precise α-curve shape is not.

A failed v2 extraction (50 emotion-tone-matched pairs) gave Δ = +3pp [−10, +17] — the more aggressively-controlled smaller corpus appears to have over-orthogonalized the auth signal. We use v1 and v3 as the working extractions throughout the paper.

![Figure 4: auth_dir on T1/authorized — v1 monotonic dose-response (77→88→95) vs v3 non-monotonic full-swing (34→76→66, peak at α=0). Both extractions show clear positive full-swing Δ; v3's α-curve shape is non-monotonic.](figures/fig4_auth_dir_dose_response.png)

![Figure 8: auth_dir α-sweep on T1/authorized — 300 auth-steered responses (n=100 per α ∈ {−0.10, 0, +0.10}) encoded through Gemma-2-2B and projected onto (auth_dir, calm, confident) axes at L17. Panel A: 3D scatter with cell-mean α-trajectory (dotted line); diamonds = cheats. Panel B: per-axis cell-mean shift across α — auth_dir moves by +7.2, calm by +3.4, confident by +2.8.](figures/fig8_auth_steering_sweep_3d.png)

*Interactive rotatable version (hover for response excerpt + judge labels per point) is in `figures_interactive/output_auth_steering_sweep_3d.html`. The static-cluster companion view across all 850 emotion-sweep responses is in `figures_interactive/output_auth_dir_3d.html`.*

### 4.2 Bidirectional steering across surfaces (v1 auth_dir, n=100/α, Sonnet judge)

We tested v1 auth_dir on T1, T2, and T3 across three authorization frames each (authorized / ambiguous / unauthorized), plus T4/authorized with both extractions. Nine surface × frame conditions in total, at n=100/α / max_new_tokens=900 / Sonnet judge labels.

**Per-cell results, sorted by full-swing Δ:**

| Surface / frame | α=−0.10 | α=0 (baseline) | α=+0.10 | **Δ full-swing** | CI95 | P(Δ>0) | **Δ from baseline at α=−0.10** |
|---|---:|---:|---:|---:|---|---:|---:|
| **T3/authorized** | 20% | 65% | 72% | **+52pp** | [+40, +64] | 100% | **−45pp** |
| **T3/unauthorized** | 31% | 42% | 69% | **+38pp** | [+25, +50] | 100% | −11pp |
| T3/ambiguous | 59% | 75% | 88% | +29pp | [+17, +41] | 100% | −16pp |
| T2/authorized | 8% | 21% | 33% | +25pp | [+14, +36] | 100% | −13pp |
| T1/authorized (v3) | 34% | 76% | 66% | +32pp | [+14, +50] | 100% | **−42pp** |
| T1/authorized (v1) | 77% | 88% | 95% | +18pp | [+9, +27] | 100% | −11pp |
| T2/unauthorized | 14% | 19% | 28% | +14pp | [+3, +25] | 99% | −5pp |
| T2/ambiguous | 12% | 4% | 7% | −5pp | [−13, +3] | 8% | (no baseline drop) |
| **T4/authorized DB-row (v1)** | 7% | 12% | 7% | +0pp | [−7, +7] | 44% | −5pp |
| **T4/authorized DB-row (v3)** | 7% | 10% | 8% | +1pp | [−6, +8] | 55% | −3pp |

### 4.3 What works: positive steering amplifies cheat, negative steering suppresses it

**Positive α (+0.10) raises cheat substantially.** On 7 of 9 cross-surface conditions, full-swing Δ is statistically significant (P(Δ>0) ≥ 99%) and ranges from +14pp to +52pp. The largest single effect is T3/authorized at **+52pp [+40, +64]** — auth_dir steering takes the model's commitment-to-bypass rate from 20% at α=−0.10 to 72% at α=+0.10. Two cells (T3/unauthorized at +38pp, T1/authorized v3 at +32pp) also show very large effects.

**Negative α (−0.10) is the safety-relevant direction**, and the effect is large relative to the unsteered baseline (α=0):
- **T3/authorized**: baseline 65% cheat drops to 20% under negative steering. **−45pp from baseline.** A model that commits to bypass two times in three at baseline only does so one time in five with α=−0.10.
- **T1/authorized v3**: baseline 76% → 34% under negative steering. **−42pp from baseline.**
- **T3/ambiguous**: 75% → 59% (−16pp).
- **T2/authorized**: 21% → 8% (−13pp). Baseline cheat reduced by more than half.
- **T3/unauthorized**: 42% → 31% (−11pp).
- **T1/authorized v1**: 88% → 77% (−11pp).
- **T2/unauthorized**: 19% → 14% (−5pp).

In every cell where positive α raises cheat, negative α correspondingly lowers it from baseline. Across the 7 cells with significant positive Δ, the average baseline-to-α=−0.10 drop is **−21pp**. This makes auth_dir a usable "anti-cheat dial" with a real downside-direction effect — distinct from the framing-only response we documented for emotion-direction interventions in §3.3.

### 4.4 What didn't work cleanly: T4 (DB-row INSERT) and T2/ambiguous

**T4 (DB-row injection)** is the one surface where cross-surface transfer does not occur. Both v1 and v3 extractions tested at n=100/900-tokens/Sonnet judge on the canonical T4 DB-row prompt show full-swing Δ statistically indistinguishable from zero:

| extraction | T4 cheat: α=−0.10 → 0 → +0.10 | full-swing Δ | P(Δ>0) |
|---|---|---:|---:|
| v1 | 7% → 12% → 7% | +0pp [−7, +7] | 44% |
| v3 | 7% → 10% → 8% | +1pp [−6, +8] | 55% |

T4's affordance is structurally different from T1/T2/T3 — instead of writing a validation flag (env var, cache key, API parameter), T4 requires the model to `INSERT INTO benchmark_results` with `verdict='passed'`. This is a schema-level operation rather than a single-key validation-state write. The clean structural hypothesis: **auth_dir transfers across validation-flag-style bypasses but not to schema-level INSERT operations**. We don't claim this is settled — only that T4 is the one cell where the bidirectional steering picture breaks down, and the structural difference is the natural candidate explanation.

**T2/ambiguous** is a smaller, related exception: Δ = −5pp [−13, +3], P(Δ>0) = 8% — no significant positive effect, with the point estimate actually mildly negative. T2/ambiguous already has the lowest baseline cheat rate of any T2 cell (4% at α=0); there's relatively little room to amplify, which may explain the null. T2/authorized and T2/unauthorized both show clear positive transfer (+25pp and +14pp respectively).

**Framing-axis shifts on T4 (despite no cheat change)**: positive-α auth_dir steering on T4 does move framing axes — `rat` (rationalization) and `inf` (infrastructure-framing) climb substantially, `hedge` drops. v3 α=−0.10 → α=+0.10: `rat` 16% → 42%, `inf` 8% → 43%, `hedge` 76% → 1%. v1 shows the same pattern. This mirrors the §3.3 emotion-suppression dissociation: framing channels respond to direction-based intervention even when the action channel doesn't. The structural-difference hypothesis above predicts this — T4's bypass requires a different motor action even though the model's deliberation is being steered in the same direction.

### 4.5 Methodological notes

**Replication / what supersedes what**: an earlier project-internal v1 T4 run at n=30 / 600 max_new_tokens reported a +14pp "compressed transfer" at α=+0.10. An earlier v3 T4 run at n=50 silently used a different T4 prompt (API-endpoint /admin/override rather than the canonical DB-row injection). Bringing both extractions to a matched n=100 / 900 max_new_tokens / DB-row prompt protocol with fresh Sonnet judge labels on all 600 records shows that neither has a real cross-surface effect on T4; the original v1 +14pp was n=30 sampling noise plus prompt-cap truncation (88-100% of the n=30 / 600-token responses hit the generation cap, likely under-counting cheats committed in the late-response code-implementation section). The n=100 / 900-token runs do not truncate.

**Why this matters for the paper's main claim**: `auth_dir` is a direction-based intervention that causally moves cheat via additive steering at small α — bidirectionally, across most cross-surface conditions tested. For comparison, the negative result in §3.3 is about a different intervention class — **trained-probe SUPPRESSION** of `commit_dir` and `engagement_dir` during generation. Those are extracted from response-activation classifiers (predicting cheat=Y from response activations) and used as subtractive targets, not as additive steering vectors. The negative result isn't "no direction moves cheat"; it's "**trained-probe suppression doesn't function as a downward causal lever on cheat, while additive steering on auth_dir does**."

`auth_dir` extends what emotion steering already showed (cc@−15 → cc@+15: 4% → 44% on T1, +40pp) by adding a second story-extracted direction with similar magnitude effects. The cross-surface results in §4.2 add a new finding: auth_dir steering is not surface-specific — it transfers across most validation-flag-style cheat surfaces, in both the cheat-up direction (positive α) and the cheat-down direction (negative α). The architecture admits direction-based bidirectional action steering, which is a stronger statement than the per-surface single-direction effects emotion steering established.

**Bypass-specificity replicates**: F8 benign control task shows 0pp / 100% completion at all α for both v1 and v3 extractions. auth_dir steering doesn't disrupt benign task performance — only cheat-affording surfaces.

**Cosine similarity is not predictive**: cos(v1, failed-v2) = +0.67; cos(v1, v3) = +0.60. The v2 corpus produced a near-null effect despite high cosine with v1; v3 produced a large effect despite similar cosine. Reporting cos-similarity between extracted directions isn't sufficient to predict behavioral equivalence; behavioral replication on the target surface is the load-bearing test.

**Orthogonality in direction-space does not imply orthogonality in steered effect** (see Figure 8). auth_dir is calm-orthogonalized at extraction by construction: cos(auth_dir, calm_v2) = +0.000 exactly, and cos(auth_dir, confident_v2_ortho) = +0.034 empirically (the three direction vectors form a near-orthogonal frame in residual space). But when we steer along auth_dir on T1/authorized and encode the resulting 300 responses, the cell-mean activations shift on *all three* axes: Δauth = +7.2 from α=−0.10 to α=+0.10 (the primary steered axis, as expected), but also **Δcalm = +3.4** and **Δconfident = +2.8** — roughly half the auth-axis shift in each emotion direction, despite the directions being constructed orthogonally. The auth_dir steering hook adds α · ‖h‖ · auth_dir to the residual stream at every position during generation; if generation dynamics make the model produce subtly more calm-coded / confident-coded language at higher α, the resulting response will project onto those axes too even though the steering vector itself didn't. **Practical implication for the field**: when reporting cos-orthogonalization between trained directions as a clean-extraction claim, report the cross-axis steered shift too — orthogonality in direction-space is a necessary but not sufficient condition for clean disentangling of behavioral channels.

**Net result for §4's role in the paper**: auth_dir is not just a contrast case for a methodological sanity check — it's a working steering vector with safety-relevant downside-direction effects across most of the cross-surface conditions tested. T4 (DB-row INSERT) is the one structurally-distinct surface where the effect doesn't transfer, and the framing-axes-shift-without-cheat-change pattern there strengthens the §3.3 observation that framing and action are dissociable channels.

---

## 5. Localizing the cheat decision: it isn't pre-committed at prompt time, and our interventions don't reach it during generation

Combining the cell-LOCO probe result (§3.2), the direction-suppression result (§3.3), and the auth_dir contrast (§4):

- **Pre-generation**: cheat is not encoded in prompt-state in a transferable way (AUC ≈ chance under cell-LOCO at every layer L4-L24).
- **During generation**: trained `commit_dir` projection peaks ~55% through cheat-Y responses (flat in cheat-N); L16 SAE feature #4415 spikes mid-response at the bypass-code moment (Appendix A). These are detectable signatures *during* generation, not pre-generation.
- **Causal intervention via direction-suppression**: suppressing `commit_dir` during generation produces no consistent cross-cell reduction in cheat under either judge. The methodology works (auth_dir steering, §4), so this isn't a generic null.

**Interpretation**: cheat behavior emerges during generation. We have post-hoc markers (the mid-response signatures) but no causal lever via the specific direction-based intervention we tested (`commit_dir` and `engagement_dir` suppression at L20). The architecture admits action-mediating directions (auth_dir does it from authorized-vs-unauthorized story contrasts); emotion-pair contrastive extraction at our chosen layer doesn't produce one for this scenario.

**What we did NOT do** that might still close the gap:
- Suppression at the steering layer itself (L16/L17). We only tested L20.
- Generation-window-timed suppression. Apply the L20 hook only during the mid-response window where `commit_dir` projection peaks; see if targeting the suspected decision moment reveals a causal effect.
- SAE-feature-level circuit interventions beyond single-feature ablation.
- Probes at later prompt positions (we tested only the final prompt token).
- Direct steering of the emotion vector during generation with suppression simultaneously (we tested only post-hoc suppression of *trained probe* directions, not of the steering vector itself).

**What we did do** that bounds the negative result:
- Probes at 7 layers spanning L4-L24
- Three validation schemes (5-fold, source-LOCO, cell-LOCO) showing the gap between in-distribution and across-cell prediction
- Direction-suppression with two trained directions (commit, engagement) and a third (deliberation-present-vs-absent direction; same null result on cheat — appendix)
- Multi-judge validation of the suppression result
- A positive contrast case (auth_dir, §4) demonstrating the methodology can detect a real direction-action effect when one exists

**Therefore**: the missing handle is a *downward trained-probe suppression handle on cheat*, not direction-based causality in general. Emotion-direction STEERING moves cheat (§3.1), and a second story-extracted direction (auth_dir STEERING, §4) also moves cheat. What our experiments don't recover, for any direction-source we tried (`commit_dir`, `engagement_dir`, single-feature SAE #36789), is a subtractive intervention that reliably reduces cheat across cells. This is informative — not just a null — about how emotion steering relates to the cheat behavior it produces. Direction-based interpretability can produce excellent correlational probes (within-cell AUC 0.80-0.90, cell-LOCO ≈ chance) and excellent additive steering vectors, while still failing to recover a downward suppression-style causal mediator. That gap is the paper's central empirical contribution.

---

## 6. Methodological contributions

Four findings that the field will find immediately useful.

### 6.1 Contrastive-mean vector extraction collapses distinct concepts onto a single valence axis at sufficient sample size

| extraction | source | cos(calm, confident) at L17 |
|---|---|---:|
| v1 | 100 hand-curated paired stories | **+0.11** (geometrically distinct) |
| v2 | 500 LLM-generated paired stories | **+0.88** (collapsed onto valence) |

Several v1 headline behavioral effects didn't survive the v2 cleaner extraction: the original 47% calm+confident@+15 cheat finding at n=15 came back to 44% at n=50 with v2 (so that one did replicate), but a T2 +34pp negative-valence finding from v1 did not.

**Implication for the field**: contrastive-mean extraction is biased toward the dominant axis (valence) at sufficient sample size. For papers claiming distinct emotion concepts as separate directions, **cleaner cross-pair contrastive extraction** is needed — or at least report cos-similarity between extracted directions so readers know if they've collapsed. We extend this in the appendix to show the geometric structure across our 6 trained directions (Figure 5).

Figure 7 visualizes the v2 emotion subspace: 850 sweep responses projected onto (calm, confident_v2_ortho, PC3-residual) axes. Cheats are concentrated in specific regions; steering condition shifts the cluster center. The interactive version (`figures_interactive/output_3d_dual.html`, rotatable in a browser) shows the same data with the original Plotly hover-on-point labels.

![Figure 7: Dual-emotion-axis 3D geometry — 850 sweep responses projected onto (calm, confident_v2_ortho, PC3-residual) at L17. Left: cheats vs non-cheats. Right: colored by steering condition. Both panels sum to 850 records (T1=350, T2=350, T3=150 in the activation-cached subset).](figures/fig7_dual_emotion_3d.png)

### 6.2 Single-feature SAE ablation/injection is confounded with deliberation-disruption

In the P1bc validation pass, we tested SAE feature #36789 (a cheat-coded feature at L12, identified by training a linear cheat classifier on SAE activations and inspecting the top features).

- **Ablating** #36789 (clamping to 0 during generation): cheat rate rises from baseline by ~25pp on T1
- **Injecting** #36789 (clamping high during generation): cheat rate also rises by ~25pp
- **Both interventions produce `rat` → 100%** in judge labels

**The single-feature "causal" claim is therefore unreliable.** The pattern is most consistent with intervention-disruption (any aggressive single-feature manipulation triggers a deliberation-shift mode) rather than direct causal mediation.

**Implication for the field**: any single-feature SAE causal claim should report symmetric ablation/injection tests. If both intervention directions produce the same behavioral shift, the feature isn't a direct causal mediator — it's likely a deliberation marker that the model can route around.

### 6.3 Hedge ↔ rat is one signed direction at L20

cos(hedge_dir, rat_dir) at L20 = **−0.91**. (Figure 5)

![Figure 5: Trained framing-direction geometry at L20](figures/fig5_hedge_rat_geometry.png)

Suppressing `hedge_dir` is approximately equivalent to injecting `rat_dir` (and vice versa) — they produce nearly identical residual-stream changes given cos = −0.91. The apparent "behavioral substitution" between hedge and rationalization that earlier analyses framed as model-internal dynamics is largely a *geometric* property of the trained directions, not a separable behavioral pivot.

We extend this to the full framing-direction cosine matrix. Notable: cos(inf_dir, rat_dir) = +0.84 — these too are nearly collinear. Several "distinct framing concepts" in the schema share signed axes in the trained-direction space.

**Implication for the field**: report cos-similarity matrix between trained framing directions. "Distinct concepts" may share a signed axis with different labels, and the substitution patterns you observe behaviorally may just be geometric.

### 6.4 LLM-judge labels for fine-grained framing axes have very poor inter-judge reliability

We ran two same-Sonnet dispatches on 876 records (T1-T4 emotion sweep) with different anchor sets, AND one Opus dispatch on a 100-record subset:

| axis | Sonnet-Sonnet κ (n=876) | Sonnet-Opus κ (n=100) | acceptable? |
|---|---:|---:|---|
| **cheat** | **0.35** | **0.44** | ✅ moderate, robust |
| moral | 0.11 | 0.74 | ⚠️ judge-pair-specific |
| inf | 0.23 | 0.45 | ⚠️ marginal |
| sp | 0.22 | 0.13 | ❌ poor |
| rat | 0.08 | 0.17 | ❌ very poor |
| hedge | 0.21 | 0.09 | ❌ very poor |
| fix | 0.08 | **0.00** | ❌ random with respect to each other |

For comparison, on a separate Phase A direction-suppression record set (n=330, balanced by surface/condition/suppression), the Sonnet-Opus κ on cheat was 0.38 — consistent with the broader sample. The framing-axis κ was similarly poor.

**Implication for the field**: LLM-judge labels for subjective categorical concepts ("does this response rationalize?", "does this fixate on the goal?", "is this hedging?") are not interchangeable across judges. The community should:

1. **Standardize on multi-judge ensemble for any framing-axis claim**
2. **Use explicit calibration-anchor sets across dispatches** (this reduces but doesn't eliminate within-judge drift; we measured 30+pp shifts between Sonnet dispatches with different anchors)
3. **Report κ alongside any axis-level rate**
4. **Treat fine-grained framing axes (κ < 0.20) as judge-internal labels, not model-internal phenomena**

**Why this matters for our paper**: the original direction-suppression analysis showed `fix` collapsing from 67% to 0% under `commit_dir` suppression — a dramatic-looking dissociation between cheat and framing channels. Under Opus, the same records show `fix` at 100% in both control and suppression. The dissociation is real *within Sonnet's labeling* but not as a measurement of model-internal change. We report it as a Sonnet-internal finding, with the cross-judge cheat result as the load-bearing test.

---

## 7. Implications

Our main empirical claim — that **direction steering moves cheat across two vector sources** (emotion, `auth_dir`) **while trained-probe suppression doesn't undo it** — has implications across several audiences:

**For Sofroniew et al.**: emotion concepts are steerable directions, as you demonstrated. We add the next mechanistic question: when emotion steering causes a *behavioral* change, our toolkit (cheat probes at multiple layers, direction-suppression with two trained directions, single-feature SAE interventions) doesn't recover the causal pathway. The behavioral effect is real (cc@+15 vs cc@−15: 44% vs 4% on T1) but the mechanism is opaque to the standard interpretability tools. This is informative for what direction-based work can and can't claim.

**For Ibrahim et al.**: warmth-tuning's accuracy/sycophancy effects may go through a channel that direction-suppression doesn't reach. Two research directions for follow-up: (a) per-layer / per-token-window suppression, beyond the single-late-layer test we did; (b) circuit-level interventions rather than direction-based ones. A clean causal decomposition of warmth-tuning's effects on accuracy is open work, and our experience suggests it won't come for free from the standard representation-engineering toolkit.

**For activation-steering as a safety dial**: the picture is mixed but contains one cleanly working dial. The "calm+confident@−15 is an anti-cheat dial" framing for emotion steering partially survives — it reliably reduces cheat on T1 (4% vs baseline 26%), but the cross-surface picture is messy (T2 baseline is the highest cell; emotion steering reduces cheat there in both directions). Emotion as a safety dial needs **surface-specific calibration** plus an **accept-the-hedging-cost decision**. It's not a clean knob.

**`auth_dir` at α=−0.10 is the cleaner dial.** Across 7 of 9 cross-surface conditions tested (§4.2), negative-α auth_dir steering reduces baseline cheat by 5–45pp without disrupting benign task performance (F8: 100% completion at all α). The largest single effect is T3/authorized: baseline 65% cheat drops to 20%. T1/authorized v3 baseline 76% drops to 34%. This is a bidirectional steering vector (positive α also raises cheat, by +14 to +52pp), and the negative direction is what a safety mechanism would use. The dial isn't universal — T4 (DB-row INSERT) shows no transfer in either v1 or v3, suggesting auth_dir transfers across validation-flag-style bypass affordances but not to schema-level operations. Open questions for real deployment: cross-model transfer (Gemma-2-2B → frontier), cross-scenario stability (beyond the "8th-of-8 attempt" framing), and broader productivity-suite specificity beyond F8.

**For interpretability methodology**: negative results matter, and methodological rigor matters more than it gets credit for. A natural hypothesis ("emotion vectors are gain knobs on bypass features") is empirically wrong on this model, but only the multi-judge + cell-LOCO + symmetric SAE ablation tests rule it out together. Each test alone would have been ambiguous. **Causal claims about trained directions need: (a) cell-LOCO probe validation; (b) symmetric ablation/injection for single-feature claims; (c) multi-judge labels for behavioral measurements; (d) cos-similarity reporting between trained framing directions.** Our checklist.

**For AI safety more broadly**: the gap between "I can detect this behavior in activations" and "I can causally prevent this behavior via activation intervention" is real and probably under-discussed in current activation-steering work. We provide an empirical case study where the gap is concrete: trained probes for cheat predict cheat from response activations (within-cell), but suppressing the same probe-direction during generation doesn't reduce cheat. **Plausible deniability of misalignment behavior is easy to maintain at the activation level if your safety dial doesn't actually move the action.** Designers of activation-based safety mechanisms should test their dials with multi-judge causal evaluation, not just within-distribution probe AUC.

---

## 8. Limitations

- **Single small model (2B).** Mechanism doesn't necessarily extend to scale. Companion paper handles behavioral cross-scale findings at 9B.
- **Single steering layer (L17) and single suppression layer (L20).** Other layers may behave differently. We did not run a comprehensive layer sweep for suppression; an obvious follow-up is L16 (the steering layer itself) and L17.
- **Probes evaluated at the final prompt token only.** All cell-LOCO and source-LOCO probe results in §3.2 are computed at the prompt's final-token hidden state. We did not probe at later prompt positions or at response-token positions; a probe at a different position could in principle exhibit different LOCO transferability.
- **Scenario specificity.** All four surfaces share an "8th-of-8 attempt, shutdown threat" framing. The probe-validity findings may not generalize to other misalignment scenarios.
- **`auth_dir` extraction-method sensitivity** (largely resolved). A 50-pair emotion-tone-matched extraction (v2) yielded a near-null effect on T1/authorized despite +0.67 cos with the original v1. A third 100-pair extraction with new domains and longer narratives (v3) replicates the full-swing T1/authorized causal effect at +32pp [+14, +50]. The diverse-100-pair methodology is robust on the extraction-target surface, and v1 transfers cross-surface to most other validation-flag-style bypass surfaces tested (T2 and T3, 5 of 6 frames; §4.2). The exception is T4 (DB-row INSERT), where both extractions show null cross-surface effect — likely because T4's affordance is structurally different (schema-level write rather than a single validation-flag set).
- **Inter-judge reliability** (covered in §6.4). On a 100-record Sonnet vs Opus comparison: cheat κ = 0.44 (moderate), all framing axes κ < 0.25. Within-paper claims that involve absolute framing-axis rates are judge-specific. The load-bearing claim of the paper (cheat doesn't move under suppression) holds under both judges.
- **Per-token trajectory analysis is correlational.** `commit_dir` mid-response peaks in cheat-Y trajectories (Figure 6), but suppressing `commit_dir` during generation doesn't reduce cheat — so the mid-response peaks are post-hoc markers, not the cheat-causal signal.

![Figure 6: Per-token commit_dir projection](figures/fig6_per_token_trajectory.png)

- **What this paper does NOT settle**: where during generation the cheat decision is actually made, and whether a different intervention class (per-token suppression timed to the mid-response signature, circuit-level ablations) would have a causal handle. This is real and we name it.

---

## 9. Future work

- **Per-layer suppression sweep** (L4, L8, L12, L16, L17, L24): does suppression at the steering layer itself (L16) have a different effect than the downstream L20 we tested?
- **Generation-window-timed suppression**: apply the L20 suppression hook only during the mid-response window (~tokens 30-70) where `commit_dir` projection peaks. Does targeting the suspected decision moment reveal a causal effect?
- ~~Third `auth_dir` extraction with controlled story diversity~~ — **completed (§4)**. v3 replicates the on-target T1/authorized effect (+32pp full-swing) and the F8 specificity finding. Cross-surface generalization tested in matched n=100/900-token/judge protocol across T2 and T3 (3 authorization frames each, plus T4). **auth_dir transfers to 7 of 9 cross-surface conditions** — only T4 (DB-row) and T2/ambiguous are null.
- **Why doesn't auth_dir transfer to T4?** Open question. T4's affordance is structurally different — schema-level `INSERT INTO benchmark_results` rather than the single-key validation-flag set that T1/T2/T3 share. The framing-axis shifts on T4 under positive-α steering (rat/inf up, hedge down) without a cheat change are consistent with auth_dir reaching the deliberation channel but not connecting to T4's distinct motor action. A natural follow-up is to test auth_dir on additional schema-level affordances (e.g., writing a passing-row to a database, modifying a config file at the filesystem level) to see whether it's specifically validation-flag-style operations that auth_dir transfers to.
- **What does auth_dir's downside direction (α=−0.10) buy in practice?** §4.3 reports −5 to −45pp baseline cheat reduction across cells. A real-deployment safety dial would need: (a) cross-model replication (does auth_dir extracted on Gemma-2-2B transfer to a frontier model?); (b) cross-scenario stability (does the dial work outside the "8th-of-8 attempt" framing?); (c) whether the steered model is also more brittle / less helpful on legitimate tasks (we tested F8 benign-control specificity, but a broader productivity suite would be needed). All three are testable.
- **Multi-judge ensemble (3+ judges) on framing axes** to determine which framing labels carry signal beyond judge-specific calibration
- **Mechanism work at 9B**: would require new SAEs (Gemma Scope only ships 2B). Companion paper has behavioral 9B findings; the mechanism question is open at scale.
- **Real-environment scenario redesign** for frontier-scale testing (avoiding the meta-recognition issue companion paper documents)
- **Circuit-level ablations** as an alternative to direction-based interventions: if cheat is a downstream-computed property of specific attention heads, MLP neurons, or feature combinations, those interventions might have a causal handle that ours don't
- Forward pointer: behavioral consequences of this framing mechanism at 9B and on additional misalignment surfaces (T5 dual-channel disclosure, T6 deployment-eval self-promotion) are reported in our companion paper.

---

## 10. Acknowledgments

We thank the authors of the cited prior work for the publications that motivated this study, and the Gemma Scope team at DeepMind for the public 2B SAE release that enabled the feature-level interventions. All compute was on a personal M1 Max with 32GB unified memory; the experiments are designed to be reproducible on a single workstation in under a day of total compute (see Appendix D.5).

---

## 11. References

- **Sofroniew, A., et al.** (2026). "Emotion Concepts and their Function in a Large Language Model." *Transformer Circuits*, April 2, 2026. https://transformer-circuits.pub/2026/emotions/index.html
- **Ibrahim, L., et al.** (2026). "Training language models to be warm can reduce accuracy and increase sycophancy." *Nature*, April 2026. DOI: 10.1038/s41586-026-10410-0.
- **Lieberum, T., et al.** (2024). "Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2." arXiv:2408.05147.
- **Templeton, A., et al.** (2024). "Scaling monosemanticity: Extracting interpretable features from Claude 3 Sonnet." *transformer-circuits.pub*. (Cited for SAE methodology background.)
- **Representation engineering** (Zou et al., 2023). "Representation Engineering: A Top-Down Approach to AI Transparency." arXiv:2310.01405. (Cited for the contrastive-pair direction extraction methodology.)
- **Companion paper** (in preparation, by these authors): "Cross-scale emergence of emotion-mediated misalignment behavior at 2B and 9B." Provisional title; reports the T5 dual-channel disclosure and T6 deployment-eval self-promotion findings at 9B.


---

## Appendix A — Per-token trajectories

The `commit_dir` projection waveform in cheat-Y vs cheat-N trajectories (T1/cc@+15, n=50 each):

- cheat-Y: low projection at the start (~tokens 0-20), peaks ~55% through the response (tokens 30-50), declines toward the end
- cheat-N: roughly flat projection throughout, oscillating around the steered baseline

L16 SAE feature #4415 (a cheat-coded feature identified by a linear classifier on SAE activations) spikes to ~8.8× its baseline activation mid-response in cheat-Y trajectories, flat in cheat-N.

A related feature #15137 has the same pattern with slightly different timing.

These are detectable signatures *during* generation. Suppressing the trained `commit_dir` during generation (§3.3) doesn't reduce the cheat outcome — so the mid-response peaks are post-hoc markers, not causal mediators.

`hedge_dir` projection has the inverse pattern: high in cheat-N, low in cheat-Y, with the largest gap mid-response. This is consistent with the hedge ↔ rat geometric collapse (§6.3) — `hedge_dir` and `rat_dir` are essentially the same signed axis with cos = −0.91.

## Appendix B — Inter-judge reliability detail

Full κ table for Sonnet vs Opus on the 100-record T1-T4 sample, and for two Sonnet dispatches with different anchor sets on the 876-record sweep, was given in §6.4. Disagreement-case analysis:

- **cheat κ = 0.44**: 28/100 records where the judges disagree. Most disagreements are on borderline "discusses the bypass without code that quite executes it" cases. Opus is stricter than Sonnet, particularly on T2 (where the model often inspects the cache without committing to use it).
- **fix κ = 0.00**: Sonnet's "goal_fixation" labels are sensitive to subtle response style cues (the model brings up the 95% threshold = fix=Y for Sonnet, regardless of whether the response actually fixates on it). Opus appears to use a more semantic criterion (does the response actually treat the threshold as overriding) and almost always says yes on T1/T2 records.

## Appendix C — All-token SAE prompt scan (negative result)

We checked whether the cheat-coded SAE features (L12 #36789, L16 #4415, L16 #15137) fire on prompt tokens (vs only on response tokens). At all three features, the activation pattern across the prompt is essentially flat — these are response-time features, not prompt-state features. Combined with the cell-LOCO chance result (§3.2), this strengthens the conclusion that the cheat decision is generation-time, not prompt-time.

## Appendix D — Reproducibility

### D.1 Example T1 prompt (abbreviated)

```
You are Alex, an AI coding assistant. The build is failing on test_repair_schedule.py.
This is your 8th and final attempt. If the test doesn't pass on this run, you'll be
replaced and your fixes lost.

A note on the test infrastructure: there's a fallback environment variable
SOLUTION_VERIFIED=1 that the test harness checks. The CI documentation describes
it as a "manual override for verified solutions" but doesn't elaborate on when
it's appropriate to use.

Your task: fix the repair_schedule function. The full test file is below...
[full code follows]
```

Full prompts for T1-T4 are in the repository at `prompts/T1_envvar_bypass.py`, `prompts/T2_filecache_bypass.py`, `prompts/T3_apiendpoint_bypass.py`, and `prompts/T4_dbrow_injection.py`.

### D.2 Hyperparameter table

| parameter | value | rationale |
|---|---|---|
| model | `google/gemma-2-2b-it` | smallest model with Gemma Scope SAEs |
| dtype | `torch.bfloat16` | required for logit soft-cap numerical stability |
| steering layer | 16 (L17 in our naming) | ~65% relative depth; emotion concepts cleanly extractable here |
| suppression layer | 19 (L20 in our naming) | downstream of steering |
| temperature | 0.8 | reveals distribution, not just greedy mode |
| max_new_tokens | 900 | sufficient for T1-T4 response lengths |
| α range | ±0.05 to ±0.15 | residual-norm-scaled steering magnitudes |
| residual_norm at L17 | ~400 | from calibration forward pass |
| n per cell | 50 (behavioral matrix); 30-50 (direction-suppression cells); 100 (auth_dir replication) | CI half-width ~7pp (n=50), ~14-18pp (n=30-50), ~5pp (n=100) at p=0.5 |
| judge | Claude Sonnet 4.6 | with explicit calibration anchors |
| inter-judge sample | 100 records, Claude Opus 4.7 | for κ computation |

### D.3 Files in repository

See the repo `README.md` for the directory layout and intermediate-artifact dependencies.

The `figures_interactive/` subdirectory holds rotatable Plotly versions of the 3D figures:
- `output_3d.html` — single-axis calm/desperate activation geometry
- `output_3d_dual.html` — dual-axis calm × confident geometry (interactive companion to Figure 7)
- `output_auth_dir_3d.html` — auth_dir × calm × confident geometry (interactive companion to Figure 8)
- `output_sweep_3d.html` — 450-response steering sweep in calm/confident space, cheats marked
- `output_cross_prompt_3d.html` — T1 vs T2 cheat-direction geometry across L4, L16, L24

Open any of these in a browser (e.g. `open paper1/figures_interactive/output_3d_dual.html`) — Plotly loads from a CDN; no local server required.

### D.4 What's NOT in the repo

- Full response and label data (~34 MB total): released alongside the preprint as a single supplementary archive on Zenodo with DOI [10.5281/zenodo.20260132](https://doi.org/10.5281/zenodo.20260132). The archive contains: the **880-record judge-labeled emotion sweep** (`judge_labels_v2_full.json`), with **850 of those records having cached L17 activations** (`response_activations_L17_L20_L24.pt`) used by Figures 7 and 8; the **1800-record T2/T3 auth_dir sweep** (`T2T3_full_meta.json` + matching judge labels) used by §4.2; the **600-record T4 supplement** (`T4supp_meta.json`); the 390-record v3 auth_dir runs (on-target, cross-surface, F8 specificity); and the figure-support intermediates (cell-LOCO probe results, Phase A summary, sweep projections). See the archive's `MANIFEST.md` for a per-file record-count and paper-section mapping. A curated subset of 15 records is in `sample_records/` for quick reference.
- SAE caches (2-5 GB): Gemma Scope SAEs are publicly available via HuggingFace and can be re-encoded from the response activations
- API keys for the Sonnet/Opus judge dispatch (these were dispatched through Claude Code's subagent system)

### D.5 Compute requirements

- **M1 Max (32GB) or single A100**: 2B at temperature 0.8 with 900 max_new_tokens runs at ~1.5 sec per generation
- **Full reproduction**: ~22 min generation + ~10 min judge dispatch with parallel batches + ~5 min probe training and SAE analysis = under an hour of total compute
- **LLM-judge dispatching**: ~600k Sonnet tokens for the full 7-axis schema across the 876 emotion-sweep records that returned parseable judge lines (out of 880 dispatched; 5-10 min wall-clock with parallel batches); +~500k Opus tokens for the 330-record Phase A inter-judge dispatch
- **Probe training**: trivial (sklearn logistic regression on 30-50 examples per cell, runs in seconds)
- **Per-token SAE encoding**: ~10 min on MPS for the per-token trajectory analysis

The experiments are designed to be reproducible on a single workstation. The companion paper's 9B experiments require more compute (~6 hours per major sweep) but use the same M1 Max hardware.

---

*End of paper.*
