# EXECUTION_PLAN.md

Last updated: 2026-09-24.

---

## Phase 0 — Environment and model setup
**Goal:** Set up everything from scratch on the Mac and prove the model loads, runs, and behaves exactly as expected.

- [x] **0.1 Project setup** — Create the approved folder structure and initialise git. Folder roles: `src/` importable code; `scripts/` one entry script per step; `artifacts/` large regenerable binaries (activations, probes, splines, tracked positions), git-ignored **except** `artifacts/manifests/` (splits, data fingerprint, artifact checksums), which is committed; `results/` reports, figures, metrics (committed); `slides/` presentation (committed); `data/` git-ignored (supplied, read-only; the fingerprint proves it is unchanged); `paper/` git-ignored (third-party papers). `.gitignore` also covers `.venv/`, `__pycache__/`, `.DS_Store`. Push to the private GitHub remote (D-18).
- [x] **0.2 Data fingerprint and protection** — Before any code touches the data, record a checksum of every supplied file, excluding `.DS_Store` files (D-19), saved to `artifacts/manifests/`, and make `data/` read-only.
- [x] **0.3 Create the venv** — Fresh Python 3.13.2 virtual environment for this project only.
- [x] **0.4 Install and verify required packages** — Install PyTorch, transformers, PyAV, OpenCV (`opencv-python-headless`, D-05), and analysis/plotting libraries with `python -m pip`; confirm each imports and works.
- [x] **0.5 Record exact versions in a lock file** — Freeze every installed package version so the setup can be reproduced exactly.
- [x] **0.6 Apple GPU check** — Confirm MPS is available and that 3D convolution runs on it.
- [x] **0.7 Model download** — Download only the weights and configs (~1.3 GB), pinned to the full revision `b3c1679b7c34d3255ef3547f27c7b226aefab26f`; verify `model.safetensors` against the Hub's SHA-256 (F-44).
- [x] **0.8 Video loader** — Set up the installable package (D-20), then write `src/vjepa_physics/video.py`: read a clip into 16 RGB frames of 256×256 and confirm the shape, order, and colours are right.
- [x] **0.9 Preprocessing** — Turn resize and center-crop off, verify the output against manual normalisation, and save a figure comparing it with the default.
- [x] **0.10 Model loading and freezing** — Load the model, confirm the config matches our documented facts (F-10: 24 layers, 1024-dim, 16 heads, patch 16), confirm no missing or unexpected weight keys, set eval mode with no gradients; checksum the weights (reference: F-44) and fix seeds.
- [x] **0.11 First end-to-end forward pass** — Run one real clip through the full model; check the output shape, and confirm there are no NaNs or Infs, before building anything on top of it.
- [x] **0.12 Activation capture** — Capture every block's output with our own hooks, including the patch-embedding output before block 0; confirm the 8×16×16 token layout. Check against `hidden_states`: entries 0–23 must equal the hooks exactly (`torch.equal`; entry 0 = patch embedding, entries 1–23 = blocks 0–22); entry 24 (block 23) is asserted per the pinned version's semantics (F-15: equal to the block output in v5.x, LayerNorm of it in v4.53.x).
- [x] **0.13 Correctness checks** — Same input gives the same output on the same device (expect exact; if not, record the size of the difference); batched equals single (including hook-captured activations); MPS vs CPU: record per-layer relative error and cosine similarity (F-39's default fp32 tolerances are expected to be too strict across 24 layers — hypothesis). Stay fp32 (D-06). **No probe is fit here** (no splits exist yet, D-07); any fp16 switch is tested in step 2.6 on train/validation only.
- [x] **0.14 Intervention no-op test** — Prove that writing back an unchanged activation leaves every downstream output identical.
- [x] **0.15 Predictor forecasting path** — Encode frames 0–7, predict tokens 1024–2047 (D-10). Target: the full-clip encoder output at those positions passed through non-affine layer_norm, as in training (F-37); metric: mean per-token L1. Must beat two fitting-free baselines on the same clips, with bootstrap CIs: (a) copy the last context time-step forward; (b) the clip's own mean context token. Nothing is fitted, so no split is needed. Also resolve O-18 (which encoder weights HF ships).
- [x] **0.16 Speed and memory benchmark** — Measure time and memory per clip on MPS to estimate the budget for later phases.
- [x] **0.17 (Optional) Official implementation parity** — **Skipped (D-29).** Check the HF model matches Meta's official implementation numerically. Limitation stated: HF's conversion is not independently checked.
- [x] **0.18 Phase 0 report and gate** — Re-score the test clip under D-05's tolerance (`opencv_tolerance`); commit everything including `results/`; re-run every check from the clean tree and compare with `git diff results/` (deterministic checks: only provenance fields may change); user confirms O-18; save a short report to `results/`, update `PROGRESS.md`, and confirm Phase 0 has passed. The gate may pass with open items if the report lists each one and the step it blocks.

## Phase 1 — Data audit
**Goal:** Verify every manifest, metadata file, and video, and document what the data really contains.

- [x] **1.1 Manifest integrity checks** — Every line parses, row counts match DATA.md, ids are unique and contiguous, every referenced file exists, no orphan folders or files (ignoring `.DS_Store`, D-19), no duplicate paths.
- [x] **1.2 Metadata consistency checks** — Fields, types, and values are self-consistent; confirm the manifests are sorted by label.
- [x] **1.3 Re-verify the data fingerprint** — Confirm the checksum taken in Phase 0 still matches; the data has not changed.
- [x] **1.4 Value-grid and design-balance analysis** — Confirm the 64-value grids, clips per value, and the direction set's motion-type split.
- [x] **1.5 Start-position and label-independence analysis** — Confirm start positions are spread evenly and uncorrelated with every label.
- [x] **1.6 Magnitude-vs-distance confound analysis** — Quantify how strongly speed/acceleration correlate with distance travelled.
- [x] **1.7 Video format and duplicate check (all clips)** — *`format` failed on uniform exit frames; closed without re-score by the planning chat, failure kept on record (F-66, D-34).* — Every clip decodes to 16 frames at 256×256, 24 fps, with no corrupted or fully-black frames; hash decoded frames to catch duplicate clips, both within and **across** all three datasets.
- [x] **1.8 Two-decoder cross-check** — Compare PyAV and OpenCV on every clip under D-05's tolerance criterion (per clip FAIL / FLAG: frame count and shape, max |diff|, whole-clip and disk-pixel mean signed diff per channel), saving per-clip stats. Pixel-identical is not achievable on our install (F-49). PyAV stays the only decoder the model sees.
- [x] **1.9 Full disk tracking and mapping verification (all clips)** — Track the disk in every frame, confirm the pixel/metre, angle, and timing mapping holds everywhere, and save the tracked per-frame positions for reuse in later phases.
- [x] **1.10 Flag problem clips** — Frame-exit, clipped, tiny-motion, and frozen-start clips, with counts and identifiers saved for later use.
- [x] **1.11 Build contact sheets and audit figures** — Visual evidence for the presentation.
- [x] **1.12 Compare results against DECISIONS.md and update it** — Confirm or correct every scouting fact (F-21–F-36) with full-data results; update status tags.
- [x] **1.13 Phase 1 report and gate** — *passed 2026-09-25: clean re-run of all 21 keys reproduced the committed results (`rerun_identical`); report `results/data_audit/report.md` (F-74).* — Save the audit report and figures to `results/`, update `PROGRESS.md`, and confirm Phase 1 has passed.

## Phase 2 — Splits and activation extraction
**Goal:** Create fair train/validation/test splits and extract the model's activations for every clip.

- [x] **2.1 Decide and document the split details** — *settled 2026-09-25: D-38 (splits, 2.2 criteria), D-39 (keep and flag).* — Within D-14's two-level structure, settle what remains of O-01 (exact held-out value indices and counts, direction stratification on motion group × angle octant), plus O-02 (frame-exit/clipped clips) and O-03 (tiny-motion clips), using Phase 1's flags and confirmed counts.
- [x] **2.2 Build and save the splits** — *done 2026-09-25: `build` passed (F-78), `balance` diagnostics (F-79).* — Train / val-seen / val-unseen / test-seen / test-unseen assignments for all three datasets, saved to `artifacts/manifests/`; verify no leakage between them and print balance tables (per value, per motion group).
- [x] **2.3 Decide and document pooling and layer numbering** — *settled 2026-09-25: D-40 (per-time-step means, 26 sites, index = `hidden_states` index, batch 1, fp32); gate subset re-extraction D-41.* — Settle O-04 (how activations are pooled) and O-06 (what "layer 0" means), matching Phase 0's hooks.
- [x] **2.4 Build the extraction pipeline** — *done 2026-09-25: `extraction.py` (F-80), `check_extraction.py pipeline` passed (F-81).* — Load each clip, preprocess it (Phase 0's settings), run it through the model, and pool the activations at every layer.
- [x] **2.5 Run full extraction** — *done 2026-09-25: all 4,572 clips, every criterion passed, ~106 min (F-82).* — All 4,572 clips, watching for crashes, skipped clips, or data loss.
- [x] **2.6 Verify the extracted activations** — *done 2026-09-25: `verify` passed, 48 live clips bit-identical; fp32 kept, so no fp16 comparison (D-40) (F-83).* — Correct shapes, no NaNs or Infs, spot-checked against Phase 0's single-clip results. If fp16 is being considered, compare it against fp32 on train/validation probe metrics only (never test).
- [ ] **2.7 Build joined dataset artifacts** — One clean, loadable file per variable pairing each clip's pooled activations with its label, split assignment, Phase 1's audit flags (exit/clipped, tiny-motion, frozen-start), and Phase 1's tracked disk positions, so later phases never re-join by hand.
- [ ] **2.8 Storage integrity** — Checksum the saved artifacts and confirm disk and memory use fit the budget.
- [ ] **2.9 Phase 2 report and gate** — Save the report to `results/`, update `PROGRESS.md`, and confirm Phase 2 has passed.

## Phase 3 — Part 1a: Layer-wise probing
**Goal:** Find where direction, speed, and acceleration become readable across the model's layers.

- [ ] **3.1 Decide and document probe type** — Settle what remains of O-05 (probe family); alpha selection, standardization, and fitting discipline follow D-15.
- [ ] **3.2 Train and evaluate probes at every layer** — All three variables, using the metrics from DATA.md (circular MAE/R² for direction; MAE/R² for speed and acceleration); fit on train, alpha by leave-one-out inside train (D-15). Use **val-seen** scores for the layer-wise curve that drives Phase 3.7/4.1/5.1's layer decisions (per D-07/D-14: test is never touched for selection); val-unseen scores are reported alongside. Test scores (seen and unseen), same metrics, are computed only for the final reported numbers.
- [ ] **3.3 Shuffled-label control** — Confirm probes trained on shuffled labels score near chance; this validates that real scores reflect a genuine signal.
- [ ] **3.4 Floor and ceiling baselines** — **Floor:** a raw-pixel ridge probe solved in dual form (chunked Gram matrix from uint8 frames + `KernelRidge(kernel="precomputed")`); center the Gram matrix and targets with train statistics (KernelRidge has no intercept, F-38); alpha by closed-form leave-one-out on train. **Ceiling:** a physics-fit oracle — least-squares fit of p(t) = p₀ + v·t + ½a·t² per clip on Phase 1's tracked positions (t = k/24, F-31; visible frames only for exit clips, F-33), reading off |v|, |a|, and the displacement angle. A linear probe on raw positions is kept only as a labeled "linear readout of positions" reference (a linear map can't compute a norm across directions). Both compared against the layer-wise curves.
- [ ] **3.5 Uncertainty estimate** — Bootstrap CIs over clips on the validation curves and the test numbers (no refitting; ridge is deterministic, so seeds give no spread). Optional: 5-fold refits within train ∪ val-seen, grouped by value, with test fixed.
- [ ] **3.6 Plot probe performance vs. layer** — For all three variables, matching the paper's presentation style.
- [ ] **3.7 Identify and document the emergence transition** — From the val-seen curve, where each variable becomes reliably readable, reported both as a raw layer index and as a fraction of total depth, for direct comparison with the paper's "one-third" finding.
- [ ] **3.8 (Optional) Patch-preserving probes** — Attentive-MLP probes as a complement to the mean-pooled probes, if time allows.
- [ ] **3.9 Phase 3 report and gate** — Save the report and figures to `results/`, update `PROGRESS.md`, and confirm Phase 3 has passed.

## Phase 4 — Part 1b: Iterative nullspace probing
**Goal:** Measure how many dimensions each variable uses and how redundantly it is stored.

- [ ] **4.1 Select the layer(s)** — Settle O-15: choose 2–3 layers spanning before, at, and after the emergence transition found in Phase 3, so dimensionality can be compared across depth (matching the paper's by-layer figure), with reasoning documented.
- [ ] **4.2 Decide and document our own stopping criteria** — A shuffled-label null: fit probes on train with permuted labels (~100 permutations) and score them on **val-seen** (not test); stop when the real val-seen score falls inside the null's 95th percentile. Report the paper's inconsistent thresholds (P-01) alongside, for comparison.
- [ ] **4.3 Run iterative nullspace probing** — For all three variables, at the selected layer(s). Each round's probe is fit on **train** (alpha by leave-one-out inside train, D-15); the stopping criterion (4.2) is checked on **val-seen**. Test is not touched anywhere in this iterative process.
- [ ] **4.4 Random-direction control** — Compare against removing random subspaces of the same size at each round, to confirm the probe's subspace matters more than its size alone.
- [ ] **4.5 Plot performance-vs-rounds curves** — For all three variables: the val-seen curve (used for stopping) and the **frozen** probe sequence run once on test (the reported figure).
- [ ] **4.6 Document dimensionality and redundancy findings** — Per variable: K chosen on val-seen, plus where the test curve crosses the same threshold; in a form precise enough to compare against Phase 6's spline dimensionality (for H-01).
- [ ] **4.7 Save probe subspace artifacts** — Persist each probe's weights Wₖ **and** the projection matrices, so each probe's composite map in raw activation space (Wₖ applied after projections 1…k−1) can be rebuilt; Phase 5's multi-probe steering reuses exactly these, not a re-derived version.
- [ ] **4.8 Phase 4 report and gate** — Save the report and figures to `results/`, update `PROGRESS.md`, and confirm Phase 4 has passed.

## Phase 5 — Part 1c: Multi-probe subspace steering
**Goal:** Reproduce the paper's steering experiment and evaluate it on held-out data.

- [ ] **5.1 Select the steering layer** — Settle O-16: choose one specific layer to steer at (from Phase 4's candidates, or a new choice with reasoning based on train/validation results only — test never informs it), since Phase 4 selects several layers for the dimensionality curve but steering needs exactly one.
- [ ] **5.2 Decide and document the steering-evaluation readout** — Settle what remains of O-07 within D-30: the primary readout is a probe at a later encoder layer on the full clip (choose which layer, with reasoning, from train/validation only); the predictor-based readout is an optional extension; a same-layer readout is only a labeled control. The readout probe is fit on **validation** (D-16); a same-layer readout carries D-16's caveat.
- [ ] **5.3 Decide and document the token-mapping method** — Settle O-08: how a pooled-space edit is applied across all 2,048 tokens.
- [ ] **5.4 Decide and document the held-out steering protocol** — Settle what remains of O-10 within D-14/D-16: steered clips come only from **test** (test-seen and test-unseen, never used for probe fitting or layer selection); target values include the test-unseen values; report the two groups separately. Unlike Phase 3/4's selection decisions, this is the intended, final use of test.
- [ ] **5.5 Implement multi-probe subspace steering** — Reusing step 4.7's saved artifacts at the selected layer; the least-squares step uses each probe's **composite** map in raw space, not the bare Wₖ (P-08).
- [ ] **5.6 Run steering experiments** — Across a range of target values (not a single target), on held-out clips and held-out values.
- [ ] **5.7 Random-direction steering baseline** — Steer along random directions of the same subspace size, for comparison.
- [ ] **5.8 Evaluate and compare** — Steering error vs. number of probes used, against the random-direction baseline.
- [ ] **5.9 Save results for Phase 6 comparison** — Steering-error curves, settings, subspaces, and the **pooled** steered activations and readout outputs (not just summary scores; per-token only for the few clips used in figures, D-13), in a form Phase 6 can load directly and re-score under its naturalness metric (O-11) for a fair comparison.
- [ ] **5.10 Document strengths, limitations, and failure cases** — As the README requires.
- [ ] **5.11 Phase 5 report and gate** — Save the report and figures to `results/`, update `PROGRESS.md`, and confirm Phase 5 has passed.

## Phase 6 — Part 2: Spline steering
**Goal:** Construct, visualize, and evaluate manifolds/splines for speed, acceleration, and direction; steer along them; and compare with Part 1's method, including strengths, limitations, and failure cases.

- [ ] **6.1 Decide and document the layer(s)** — Match Phase 5's layer choice (step 5.1) where possible, for a fair comparison.
- [ ] **6.2 Decide and document activation-manifold construction** — Settle O-09: exact vs. smoothing, PCA dimensions, parameterization, and a periodic curve for direction. PCA is fit on train only; smoothing and PCA dimension are chosen by leave-one-centroid-out within train values, with val-unseen centroids as confirmation (D-15).
- [ ] **6.3 Fit activation manifolds** — For direction, speed, and acceleration, using only training data (never the held-out test split).
- [ ] **6.4 Decide and document behavior-manifold construction and the naturalness metric** — Settle what remains of O-11 within D-17: the behavior readout sits at the readout location downstream of the steering layer (not a same-layer softmax over the activation centroids); choose its form and temperature and the off-manifold distance metric.
- [ ] **6.5 Fit behavior manifolds** — The counterpart to step 6.3, in behavior space: on train, at the readout location (D-17).
- [ ] **6.6 Isometry validation** — Correlate activation-manifold and behavior-manifold distances; compare against straight-line distances. The downstream version (D-17) is the evidence; the same-layer softmax-over-centroids version is reported only as a labeled by-construction control (P-10). This is one of the paper's two headline results.
- [ ] **6.7 Visualize the fitted manifolds** — As the README explicitly asks for.
- [ ] **6.8 Implement spline steering** — Reusing Phase 5's readout and token-mapping methods, for consistency.
- [ ] **6.9 Run spline steering experiments** — Using the same held-out protocol as Phase 5, across a range of target values.
- [ ] **6.10 (Optional) Pullback / bidirectional check** — Optimize an activation path to match a target behavior path, and check whether it naturally traces the fitted manifold (the paper's other headline result; App. A.8–A.9).
- [ ] **6.11 (Optional, ambitious) 2D joint manifold and factored-control test** — Fit a joint (speed × direction) or (acceleration × direction) manifold, using data we already have (every clip has a unique pair, F-23), and test whether steering one axis leaves the other undisturbed (the paper's most advanced result, §4).
- [ ] **6.12 Test H-01** — Compare the spline's effective dimensionality against Phase 4's nullspace dimensionality.
- [ ] **6.13 Test H-02** — Harmonic analysis of direction's sawtooth pattern.
- [ ] **6.14 Compare spline steering against Phase 5's multi-probe steering** — Side by side, on the same clips and targets, scored under the **same metrics for both** (readout error and naturalness), not just readout error for one and naturalness for the other.
- [ ] **6.15 Document strengths, limitations, and failure cases** — As the README requires.
- [ ] **6.16 Save Phase 6 artifacts for Phase 7** — Manifolds, steering results, and comparison figures.
- [ ] **6.17 Phase 6 report and gate** — Save the report and figures to `results/`, update `PROGRESS.md`, and confirm Phase 6 has passed.

## Phase 7 — Confounds and robustness
**Goal:** Test whether the findings hold up against the data's known confounds and edge cases.

- [ ] **7.1 Run O-12's confound controls** — Velocity-vs-acceleration classifier at matched distance; speed probe applied to acceleration clips; a direct displacement probe as a comparison ceiling. Resolves H-03.
- [ ] **7.2 Motion-type generalization check for direction** — Train on train-split clips of one motion type, evaluate on test-split clips of the other, and vice versa (unique to our direction dataset's design).
- [ ] **7.3 Error breakdown by Phase 1's flags** — For both probing (Phase 3/4) and steering (Phase 5/6) results, broken down by exit/clipped, tiny-motion, and frozen-start clips. Resolves H-04.
- [ ] **7.4 Per-tubelet motion analysis** — Correlate per-clip error against how far the disk actually moves within one 2-frame tubelet; check whether "acceleration" readout is really a continuous magnitude or just a binary moved/didn't-move detector. Resolves H-05.
- [ ] **7.5 Cross-variable subspace overlap and steering specificity** — Principal-angle overlap between the speed, acceleration, and direction probe subspaces (matching the paper's App. C.4 method); then a causal check: does steering one variable move the readout of the others?
- [ ] **7.6 Update Phase 5's and Phase 6's limitation write-ups** — With these findings.
- [ ] **7.7 Phase 7 report and gate** — Save the report and figures to `results/`, update `PROGRESS.md`, and confirm Phase 7 has passed.

## Phase 8 — Presentation and final delivery
**Goal:** Build the presentation, clean up the code, and verify everything reproduces from scratch.

- [ ] **8.1 Final data-integrity check** — Re-verify the Phase 0 fingerprint after the full pipeline has run.
- [ ] **8.2 Decide and document the presentation medium** — Settle O-17.
- [ ] **8.3 Outline the presentation** — Against the README's exact requested content: methods, results, interpretations, comparisons, limitations.
- [ ] **8.4 Build the presentation.**
- [ ] **8.5 Build a lightweight results notebook** — Loads saved artifacts and reproduces the key figures, so no one needs to rerun the full pipeline to see them.
- [ ] **8.6 Code cleanup** — Remove dead/debug code, add docstrings, consistent style.
- [ ] **8.7 Write a project-level README for the code** — Headed with the full title (D-18); decide how it coexists with the supplied task `README.md` at the root.
- [ ] **8.8 Verify reproducibility** — Decide: full rerun vs. a scoped from-scratch smoke test, given compute limits.
- [ ] **8.9 Requirements-compliance checklist** — Map F-01 through F-09 to exactly where each is addressed, to catch any accidental omission.
- [ ] **8.10 Decide what's in the code submission** — Likely `src/` + `scripts/` + a small subset of `results/`, excluding bulky `artifacts/`; decide the delivery method (attachment, or a link such as inviting the reviewer to the private repo, D-18).
- [ ] **8.11 Prepare to defend every decision** — Review `DECISIONS.md` and walk through the key code (probes, nullspace, steering, splines) so every choice and every line can be explained.
- [ ] **8.12 Prepare likely follow-up questions and answers** — Using `DECISIONS.md`'s paper notes (P-01–P-13) and open decisions as source material, since the README says the talk is "the basis for an open discussion."
- [ ] **8.13 Rehearse and time the talk.**
- [ ] **8.14 Submission** — Package and send to constantin@worldmechanics.ai, per the email's instructions.
- [ ] **8.15 Phase 8 report and gate** — Save the report to `results/`, update `PROGRESS.md`, and confirm Phase 8 has passed.
