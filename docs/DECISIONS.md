# DECISIONS.md — facts, decisions, open questions, hypotheses

Last updated: 2026-09-24. Source of truth for *what we know* and *what we decided*.

## How to read this file

- **Facts (F-xx):** things that are true, with a source. No interpretation.
- **Decisions (D-xx):** choices we agreed on, with why and the evidence behind them.
- **Open decisions (O-xx):** choices not yet made. Never decided silently; settled in the planning chat.
- **Hypotheses (H-xx):** ideas to test. **Never state these as findings.**
- **Paper notes (P-xx):** inconsistencies and weak spots in the two papers (interview material).

Status tags:
**[verified]** checked against a primary source ·
**[scouting]** from planning-chat analysis; must be reproduced in the data audit phase ·
**[untested]** verified by reading code, not yet run ·
**[open]** not decided.

Scouting source: planning-chat sandbox scripts (not in the repo), run on 2026-09-24 over all
4,572 `metadata.json` files, the 3 manifests, and 14 selected videos.

---

## 1. Facts

### 1.1 Task (source: README.md, email from World Mechanics)

| ID | Fact |
|---|---|
| F-01 | Model: pretrained **V-JEPA 2 ViT-L/16, 256-res**, `facebook/vjepa2-vitl-fpc64-256`; reference implementation `vjepa2_vit_large` in Meta's official repo. (README) |
| F-02 | The encoder must stay **frozen**. We must document how representations are extracted and pooled. (README) |
| F-03 | Data used to fit probes/manifolds must be clearly separated from evaluation data. Probe fitting, layer selection, nullspace construction, and spline construction must not use the final held-out examples. (README, DATA.md) |
| F-04 | Do not modify `data/`. All derived artifacts go elsewhere. (README, DATA.md) |
| F-05 | Part 1 = layer-wise probing → iterative nullspace probing → multi-probe subspace steering on held-out data. Goal: reproduce methodology and qualitative findings, not exact numbers. (README) |
| F-06 | Part 2 = spline/manifold steering for speed, acceleration, and direction (at minimum); handle direction's circular structure; design a meaningful held-out steering evaluation; compare with Part 1 (strengths, limitations, failure cases). (README) |
| F-07 | Deliverable: presentation (~15 min, basis for open discussion) + source code, emailed to constantin@worldmechanics.ai. They value approach, experiment design, and interpretation over "perfect" answers. (README, email) |
| F-08 | AI tools are allowed; the candidate remains responsible for understanding everything. (README) |
| F-09 | Direction target suggested as (sin θ, cos θ), converted back to an angle for circular error. Suggested metrics: circular MAE and R² on sin/cos (direction); MAE and R² (speed, acceleration). (DATA.md) |

### 1.2 Model and libraries

| ID | Status | Fact | Source |
|---|---|---|---|
| F-10 | [verified] | 24 layers, hidden 1024, 16 heads, patch 16, tubelet 2, crop 256; predictor 12 layers, hidden 384; fp32. Saved with transformers 4.53.0.dev0. | checkpoint `config.json` |
| F-11 | [verified] | Checkpoint revision `b3c1679`. `model.safetensors` = 1.3 GB (≈325M params = encoder ≈303M + predictor ≈22M). Full repo 6.43 GB due to an `original/` folder we do not need. | HF repo file listing |
| F-12 | [verified] | `frames_per_clip = 64` "does not impact inference". Positions use RoPE computed from token indices (no fixed position table). | transformers `configuration_vjepa2.py`, `modeling_vjepa2.py` |
| F-13 | [verified] | 16 frames at 256 px → 8 × 16 × 16 = **2048 tokens**, no CLS token. Token `i` → time `i // 256`, row `(i % 256) // 16`, col `i % 16`. | `modeling_vjepa2.py` (Conv3d patch embed + RoPE position ids) |
| F-14 | [verified] | Shipped video processor: resize shortest edge to **292**, **center-crop 256**, rescale 1/255, ImageNet mean [0.485, 0.456, 0.406] / std [0.229, 0.224, 0.225], bilinear. | `video_preprocessor_config.json` |
| F-15 | [verified] | `hidden_states` returns 25 entries (patch-embedding output + 24 block outputs). Last entry is **post**-final-LayerNorm in v4.53.x but **pre**-LayerNorm in v5.x (`tie_last_hidden_states=False`). | transformers source, v4.53.3 vs main |
| F-16 | [verified] | Predictor default: context = all tokens, target = all tokens (reconstruction). The encoder always runs on the full input before masks are applied. Target tokens are placed at the positions given in `target_mask`. | `modeling_vjepa2.py` |
| F-17 | [verified] | transformers v5.17.0 is the latest release (2026-09-09); requires Python 3.10–3.14 and torch ≥ 2.5. V-JEPA 2 first appears in v4.53.0. | GitHub releases, `setup.py` |
| F-18 | [verified] | Wheels: torch 2.14.0 (cp313, macOS 14 arm64); PyAV 18.1.0 (Python ≥ 3.11, macOS 14 arm64, bundles FFmpeg); torchcodec 0.16.0 (needs a separately installed FFmpeg); opencv-python 5.0.0.93 (macOS 13 arm64). | PyPI, torchcodec README |
| F-19 | [verified] | Conv3D on MPS was added to PyTorch in PR #114183 (Dec 2023); requires macOS ≥ 13.2. | pytorch/pytorch PR #114183 |
| F-20 | [verified] | Meta's official `torch.hub` `vjepa2_vit_large` returns (encoder, predictor). The official README points to the HF checkpoints as the same models. | facebookresearch/vjepa2 README, `hubconf.py`, `src/hub/backbones.py` |

### 1.3 Data — manifests and metadata [scouting]

| ID | Fact |
|---|---|
| F-21 | Integrity: every manifest line parses; counts 1,500 / 1,536 / 1,536; ids unique, contiguous, match `scene_XXXX`; no missing, orphan, or duplicate files; `frames` = 16 and `fps` = 24 everywhere. |
| F-22 | **All three manifests are sorted by label** (speed/acceleration: corr(id, label) = 0.9999; direction: sorted by angle). |
| F-23 | Speed set: 64 values, 0.25–4.0 m/s, linear step 0.0595; 24 clips per value; acceleration = 0; each clip has a unique (speed, θ) pair; θ from the 64-angle grid, 24 clips per angle; corr(speed, cos θ / sin θ) ≈ −0.001. |
| F-24 | Acceleration set: 64 values, 0.25–10.0 m/s², linear step 0.1548; 24 clips per value; **all start from rest** (`speed_mps` = 0); same θ sequence as the speed set. |
| F-25 | Direction set: 64 angles, step 5.625°; 23 clips for 36 angles, 24 for 28 angles. **No `primary_label` or `magnitude` fields** (contradicts DATA.md). 750 constant-velocity clips (speeds 1–7 m/s, integers) + 750 accelerating-from-rest clips (2, 4, 6, 8, 10 m/s²; 150 each). Motion type and magnitude balanced across angles. |
| F-26 | Start positions: all unique, roughly uniform; within ±1.2 m (speed, acceleration) and ±2 m (direction); \|corr\| with every label < 0.05. |
| F-27 | **Confound:** within the acceleration set, acceleration correlates r = 1.000 with distance travelled and with mean speed. Within the speed set, speed is proportional to distance. Distance-travelled overlap window between the two sets: 0.156–1.953 m (1,176 speed clips, 1,440 acceleration clips inside). |

### 1.4 Data — videos [scouting]

| ID | Fact |
|---|---|
| F-28 | Files: MPEG-4 Part 2 (Simple Profile), yuv420p, 256×256, 24 fps, exactly 16 frames, ~4 KB each (14 clips checked). |
| F-29 | **The disk is orange** (mean ≈ RGB 227, 113, 44), not blue as DATA.md states. Background ≈ RGB 29, 31, 28. Two independent decoders (ffmpeg, OpenCV) agree pixel for pixel. |
| F-30 | Flat top-down rendering: disk area 341–350 px in every frame (radius ≈ 10.5 px ≈ 0.33 m, diameter ≈ 21 px ≈ about 2×2 patches); positions snap to whole pixels; exactly one object per frame. |
| F-31 | **Exact mapping:** pixel x = 128 + 32·x_world, pixel row = 128 − 32·y_world (32 px/m, visible area ±4 m, y flipped). Frame k is at t = k/24 s, so a clip spans 15/24 s. Fit residual ≤ 0.64 px; validated on all 14 clips (≤ 0.66 px). The alternative t = (k+1)/24 gives errors up to ~6 px. |
| F-32 | Angle convention: 0° = right, **90° = up on screen**, 180° = left, 270° = down. |
| F-33 | **Frame exits (direction set only):** 199 clips (13.3%) have the disk clipped or gone in ≥ 1 frame; 113 (7.5%) lose it entirely for 1–7 frames (40 for ≥ 4 frames). Driven by speed: 59/107 clips at 7 m/s, 32/107 at 6, 21/107 at 5, 1/107 at 4. Speed and acceleration sets: 0 clips. (Predicted from metadata + F-31; matched all 14 real videos exactly.) |
| F-34 | **Tiny motion:** acceleration set: minimum total movement 1.56 px; 48 clips < 3 px; 20 clips show ≤ 3 distinct positions; 12 clips frozen ≥ 8 frames after frame 0 (max 11). Speed set: minimum 5.0 px (the 24 clips at 0.25 m/s); 120 clips < 10 px. Direction set: minimum 12.5 px. |
| F-35 | Label → pixel units: speed = 1.333·v px/frame (4 m/s ≈ 5.3 px/frame); acceleration = 0.0556·a px/frame² (10 m/s² ≈ 0.56 px/frame²). |
| F-36 | **Default preprocessing damage** (derived from F-14 + F-31): resize 292 + crop 256 keeps only original pixels 15.78–240.22 → visible world ±3.51 m, scale 36.5 px/m. Clipped-or-gone clips would rise from 199 → 302 (direction), 0 → 22 (speed), 0 → 0 (acceleration). |

### 1.5 Added during the plan review (2026-09-24)

| ID | Status | Fact | Source |
|---|---|---|---|
| F-37 | [verified] | V-JEPA 2 pretraining: the predictor's targets are the **target (EMA) encoder's** output passed through a **non-affine `F.layer_norm`**; loss = mean \|z − h\|^`loss_exp` over masked tokens (L1 in the papers). The context encoder sees only context tokens; targets come from the full, unmasked clip. | facebookresearch/vjepa2 `app/vjepa/train.py` L429–447 (main, read 2026-09-24); V-JEPA paper Eq. 2; V-JEPA 2 Fig. 2 |
| F-38 | [verified] | scikit-learn: `RidgeCV` default `cv=None` = efficient leave-one-out; `Ridge` (`cholesky`) solves the dual/kernel form when `n_features > n_samples`; `KernelRidge` has **no `fit_intercept`** (center manually); stratified splitters raise an error when a class has only 1 member. Re-check against the pinned version in step 0.5. | sklearn `main`: `linear_model/_ridge.py`, `kernel_ridge.py`, `model_selection/_split.py` (read 2026-09-24) |
| F-39 | [verified] | `torch.testing.assert_close` default fp32 tolerances: rtol 1.3e-6, atol 1e-5. | pytorch `main`: `torch/testing/_comparison.py` (read 2026-09-24) |
| F-40 | [verified] | Physics paper's held-out steering: steering probes trained on train (70%, 240 videos); the eval probe is trained on **test activations only**, deliberately, so it has never seen the steering probes or the subspace-building activations. | Physics paper App. C.12 |

---

## 2. Decisions made

| ID | Decision | Why | Evidence |
|---|---|---|---|
| D-01 | Use the HF `transformers` checkpoint `facebook/vjepa2-vitl-fpc64-256`, pinned to revision `b3c1679`. Encoder frozen, `eval()` mode, no gradients; checksum weights before and after runs. | README names it; reproducibility; proves "frozen". | F-01, F-02, F-11 |
| D-02 | **Preprocessing: resize OFF, center-crop OFF.** Keep rescale (1/255) and ImageNet normalization (the model was trained with it). | Default would distort scale and crop the disk out of many clips. | F-14, F-36 |
| D-03 | Feed all **16 frames at native 256×256**. | Matches the data; frame count does not affect inference. | F-12, F-28 |
| D-04 | **Capture activations with our own forward hooks** on each encoder block (plus the patch-embedding output). Pin the exact transformers version in a lock file. | `hidden_states` semantics differ between versions; we need the same hooks for steering anyway. | F-15 |
| D-05 | Video decoding: **PyAV** primary, **OpenCV** as an independent cross-check (pixel-identical comparison). | PyAV bundles FFmpeg on macOS; torchcodec needs a separate FFmpeg; two decoders catch decoding bugs. | F-18, F-29 |
| D-06 | Environment: fresh **Python 3.13.2** venv, **MPS**. Fallback to a 3.12 venv only if a package fails on 3.13. Default dtype **fp32** until step 0.13 (MPS vs CPU) justifies anything else. Download only weights + configs (~1.3 GB). | All key wheels verified for 3.13 / macOS ≥ 14; fp32 is the checkpoint's native dtype, so half precision must earn its place with evidence; saves ~5 GB. | F-17, F-18, F-19, F-11 |
| D-07 | Splits: **never head/tail.** Use a **three-way train / validation / test** split; test is never touched for fitting or selection. Exact scheme: see O-01. | Manifests are sorted by label; README requires fit/eval separation. | F-03, F-22 |
| D-08 | Direction targets = **(sin θ, cos θ)**; report circular MAE and R² on sin/cos. | DATA.md recommendation; handles wrap-around. | F-09 |
| D-09 | Angle convention: math convention (0° right, 90° up); **flip y** when converting between image and world coordinates. | Confirmed on real clips. | F-32 |
| D-10 | If the predictor is used for forecasting: encode frames 0–7 only, then call the predictor separately with context positions 0–1023 and targets 1024–2047. Must pass step 0.15 (beats a trivial baseline) before any result relies on it. | Default path is reconstruction with future leakage. | F-16 |
| D-11 | The **full data audit is reproduced by the user in code** over all 4,572 clips (Phase 1, steps 1.1–1.13 in `EXECUTION_PLAN.md`), outputs saved, key figures in the presentation. | Our decisions rest on it; reviewers must see how conclusions were reached. | user said |
| D-12 | Workflow: planning chat = architect and main guide; Claude Code = guide only; the **user writes all code**. Source of truth = `CLAUDE.md` + `docs/`. | Learning goal; interview readiness; avoids conflicting decisions. | user said |
| D-13 | Phase 2 stores only **pooled** activations for all clips, never bulk per-token activations. Steering (Phases 5–6) computes per-token activations **live**, via forward hooks on real clips, only for the specific clips involved in a given experiment. | Resolves O-13: bulk per-token storage would cost ≈19 GB per layer in fp16 for all clips, but nothing downstream actually needs it stored — steering re-runs the model on demand. | reasoning during Phase 2 planning |
| D-14 | **Two-level split**, per dataset. (1) **Value level:** hold out whole label values: ~8 **test-unseen** and ~4 **val-unseen**, evenly spaced, never the range endpoints (speed, acceleration; direction is circular, so no endpoints). (2) **Clip level:** within the remaining ~52 values, split clips ~16/4/4 per value into **train / val-seen / test-seen**. Provisional counts (speed, acceleration): train 832; val 304 (208 seen + 96 unseen); test 400 (208 seen + 192 unseen). Direction: value level by angle; clip level stratified on motion group (12) × angle octant (8), marginals verified with a balance table. **Selection** (layer curve, nullspace stopping) uses **val-seen**; val-unseen is reported alongside, never used for selection. Exact indices and counts are fixed at step 2.1, after Phase 1 confirms the counts. | Gives separate "unseen clips" and "unseen values" tests; makes steering to unseen values a real held-out test (fixes P-12). A full angle × motion × magnitude cross is infeasible (768 cells, ~2 clips each) and single-member classes break stratified splitters. Trade-off: train (832) < d (1024); ridge handles it. | F-03, F-22, F-23, F-24, F-25, F-38, P-12; DATA.md ("fair test of generalization"); README |
| D-15 | **Fitting discipline.** Probe regularization (alpha) is chosen by efficient leave-one-out **inside train**. Scalers and PCA are fit on **train only**, everywhere. Spline hyperparameters are chosen by **leave-one-centroid-out within train values**, with val-unseen centroids as confirmation only. Validation is reserved for structural choices (layer, nullspace stopping); test is for reporting only. | Each split has one job; leave-one-out is free for ridge; 4 val-unseen centroids are too few to tune splines on. | F-38, D-07 |
| D-16 | The **steering readout / eval probe is fit on validation** (val-seen + val-unseen): disjoint from the train data that builds the steering subspace, and never fit on test. **Caveat:** data disjointness removes overfitting coupling, but a *same-layer* readout will still tend to share directions with the steering probes. Real independence comes from a downstream readout (O-07 b/c), which step 5.2 prefers; any same-layer result is labeled with this limitation. | Keeps the paper's independence goal (F-40) without its fit-on-test flaw (P-06). | F-40, P-06 |
| D-17 | The **behavior manifold is built downstream of the steering site**: activation manifold at layer L; behavior readout at the readout location (a later layer or the predictor output, per O-07). The same-layer softmax-over-centroids version is reported only as a labeled **by-construction control**. | In Goodfire's Mountain Car setup the behavior distribution is a softmax over distances to the same activation centroids, so isometry is partly by construction; a downstream readout makes it depend on the network's own computation. | P-10; Goodfire §5, Eq. 9 |
| D-18 | **Version control and naming.** Local repo folder: `world-mechanics-take-home-shivansh-gupta`. Remote: **private** GitHub repo `shivansh052k/world-mechanics-take-home-shivansh-gupta`, pushed over SSH with a dedicated personal key (separate from an older school key). Full title "World Mechanics Take Home Assignment (Shivansh Gupta)" goes in the GitHub description, the code README (8.7), and the title slide. `paper/` (third-party PDFs) is git-ignored. Final delivery method is still settled at 8.10. | Off-site backup; a plain name avoids shell quoting; not public because the repo holds World Mechanics' task text (publishing needs their okay). | user said (2026-09-24) |
| D-19 | **`.DS_Store` is not supplied data.** It is excluded from the data fingerprint (0.2, and so 1.3 / 8.1) and from Phase 1 integrity checks (1.1). It is never deleted from `data/`, since that would itself be a write. | One exists, `data/.DS_Store`, dated 2026-09-24 00:52 — after the data folders (2026-09-23 23:22), so written by Finder. Without the exclusion, any Finder rewrite would make 1.3 falsely report "data changed". | `ls -la data` and `find data -name .DS_Store` output, 2026-09-24 |

---

## 3. Open decisions [open]

| ID | Question | Options on the table |
|---|---|---|
| O-01 | Split details (structure settled by D-14) | Exact held-out value indices and final counts per dataset, confirmed after Phase 1 (step 2.1). |
| O-02 | Direction clips where the disk exits/clips (F-33) | Keep all + flag; exclude; report with and without. |
| O-03 | Low-motion acceleration clips (F-34) | Keep + flag; report error by magnitude bin; exclude below a movement threshold. |
| O-04 | Pooling | Mean over all 2048 tokens (paper's main choice); per-time-step pooling; per-patch / attentive probes as a secondary analysis. |
| O-05 | Probe family (alpha selection and standardization settled by D-15) | Closed-form ridge vs SGD (paper used Adam + weight-decay sweep). |
| O-06 | What "layer 0" means | Patch-embedding output vs first block output; indexing convention for all plots. |
| O-07 | Steering "behavior" readout | (a) probe at the same layer; (b) steer at layer L, run the remaining layers, read at a later layer; (c) forecast via the predictor (needs D-10). |
| O-08 | Mapping a pooled-space steering vector onto 2048 tokens | Add the same δ to every token (shifts the mean by exactly δ); per-token alternatives. |
| O-09 | Spline construction | Exact interpolation vs smoothing spline; PCA dimension; parameterization (label value vs arc length); periodic spline for direction; intrinsic angle from labels vs `atan2` of PCs. |
| O-10 | Held-out steering details (structure settled by D-14, D-16) | Range of target values per variable; number of source clips per target; reporting test-seen and test-unseen separately; fair comparison in the same subspace. |
| O-11 | Behavior readout form and naturalness metric (readout location settled by D-17) | Readout form: multinomial logistic over value bins vs softmax-over-centroids at the readout location; temperature. Metric: Hellinger/Bhattacharyya (Goodfire style); distance to the activation manifold; nearest-neighbour retrieval. |
| O-12 | Confound controls (F-27) | Velocity-vs-acceleration classifier at matched distance; speed probe applied to acceleration clips; displacement probe. |
| O-14 | HF vs official implementation parity check | Run it (rigor, both named in README) or skip (cost, time). |
| O-15 | Which layer(s) for nullspace probing (Phase 4) | 2–3 layers spanning before/at/after the emergence transition found in Phase 3 (paper's by-layer figure style). |
| O-16 | Which single layer to steer at (Phase 5) | One of Phase 4's candidate layers, or a new choice with reasoning; distinct from O-15 since nullspace probing studies several layers but steering needs exactly one. |
| O-17 | Presentation medium (Phase 8) | Slides; a notebook; a PDF; some combination. |
| O-18 | Which encoder weights the HF checkpoint ships (target/EMA vs context encoder) | Check the HF conversion script and model card in step 0.15; affects how 0.15's result is interpreted (F-37). |

---

## 4. Hypotheses (to test — not findings)

| ID | Hypothesis | How we would test it |
|---|---|---|
| H-01 | The physics paper's "direction needs 40+ linear dimensions" is at least partly what one **curved, closed 1-D manifold** looks like to linear tools. | Fit a periodic spline; measure how many PCA dims it needs; compare spline steering in that small space vs multi-probe steering. |
| H-02 | The nullspace **sawtooth** pattern relates to sin/cos feature pairs at several frequencies (harmonics) of that curve. | Fourier analysis of activations vs θ; compare sawtooth before/after removing harmonic pairs. |
| H-03 | "Acceleration" decoding within the acceleration set may reduce to **distance travelled / mean speed**. | O-12 controls. |
| H-04 | Direction errors concentrate in **exit/clipped clips**; low-end acceleration errors concentrate in **tiny-motion clips**. | Error breakdown by flag and magnitude bin. |
| H-05 | Slow motion is hard partly because a slow disk often does not move within one 2-frame tubelet (F-30, F-34, F-13). | Error vs per-tubelet pixel motion. |

---

## 5. Paper notes (inconsistencies and weak spots)

### Physics paper — Joseph et al., *Interpreting Physics in Video World Models*

| ID | Note | Where |
|---|---|---|
| P-01 | Nullspace stopping thresholds disagree: direction R² < 0.1, speed R² < 0.05 vs direction R² < 0.3, speed R² < 0.1. | App. C.11 vs Fig. 22 caption |
| P-02 | Direction dimensionality at layer 8 disagrees: "40–50 features" vs 136 dims (68 probes) vs 25 probes until R² < 0.1. Layers 20–23 all show exactly 400 (possible cap). | §7.2, Table 3, App. C.12 |
| P-03 | Steering error: "< 0.5°" vs held-out 11.9° (the former is likely measured with the steering probes themselves). | §7.2 vs App. C.12 |
| P-04 | Held-out split lists 240 + 103 = 343 videos; the velocity dataset has 392. | App. C.12 vs A.1.2 |
| P-05 | Units: "pixels per frame" vs m/s. | §5.1 vs A.1.2 |
| P-06 | Steering evaluation is weak: edits and reads the same layer (no downstream propagation); single target angle (90°); 8 directions; the "held-out" eval probe is trained on the same test videos it evaluates (deliberately, for independence from the steering probes, F-40; our alternative is D-16). | App. C.12 |
| P-07 | Acceleration clips start from rest, so acceleration is confounded with speed/distance — yet §5.2 argues acceleration does not rely on velocity. | A.1.2, §5.2 |
| P-08 | Undocumented: what "grouped" CV grouped by; whether layer 0 = embeddings or first block; how the least-squares step applies probes trained on already-projected activations. | App. B, C.11, C.12 |
| P-09 | Uses 224-px input (14×14 = 196 patches, 1,568 tokens); we use native 256 (2,048 tokens). | App. C.6 |

### Goodfire paper — Wurgaft et al., *Manifold Steering…*

| ID | Note | Where |
|---|---|---|
| P-10 | Mountain Car "behavior" F is a softmax over distances to centroids sampled from the activation spline itself → activation–behavior isometry (r = 0.996) is partly by construction. | §5, App. B.1, C.3 |
| P-11 | Unequal interventions: linear steering replaces the entire residual stream; manifold steering replaces only the top-64 PCA components. | App. A.6 |
| P-12 | No held-out evaluation: splines interpolate all centroids exactly; steering always runs between those same centroids. | App. A.3, A.4, A.6 |
| P-13 | Unablated choices: 64 PCA dims, layer 28, τ = 0.5, exact interpolation vs smoothing. | §2.2, App. A.2, B.1 |

---

## Change log

- 2026-09-24 — initial version (facts from README, DATA.md, email, both papers, library/model sources, and scouting analysis).
- 2026-09-24 — final consistency pass while completing EXECUTION_PLAN.md: resolved O-13 as D-13; added O-15 (Phase 4 layer selection), O-16 (Phase 5 steering-layer selection), O-17 (Phase 8 presentation medium).
- 2026-09-24 — corrections caught by a fresh review of the full plan: fixed a real methodology bug where Phase 3/4's layer-selection curve and nullspace stopping criterion were worded as using test data, contradicting D-07 (fixed in EXECUTION_PLAN.md steps 3.2, 3.7, 4.2, 4.3, 5.4); remapped stale old-checklist references in D-06/D-10/D-11 to actual step IDs; corrected a step-count error (107, not 117) and a wording slip in PROGRESS.md.
- 2026-09-24 — plan review, round 2 (researched fixes): added F-37–F-40 (V-JEPA training target, sklearn ridge/split behavior, torch tolerances, the paper's held-out protocol); added D-14 (two-level split), D-15 (fitting discipline), D-16 (readout fit on validation, with caveat), D-17 (downstream behavior readout); narrowed O-01, O-05, O-10, O-11; added O-18; clarified P-06. Matching step edits in EXECUTION_PLAN.md.
- 2026-09-24 — after step 0.1: added D-18 (private GitHub remote, folder name, `paper/` ignored) and D-19 (`.DS_Store` excluded from fingerprint and integrity checks). Matching wording in EXECUTION_PLAN.md steps 0.1, 0.2, 1.1, 8.7, 8.10.
