# Confounds and robustness — report

Date: 2026-09-27. Every claim points to a saved check result (`results/<subject>/checks.json`, key in backticks);
each entry carries its own provenance. The code behind every result was verified to be committed unchanged
(`evidence`: `code_hash_check`, 77 keys, none to re-run). Design decisions are in `docs/DECISIONS.md` (D-52,
F-149–F-157).

**Verdict: Phase 7 complete.** No model was run: stored pooled activations, saved probe predictions and the saved
steering runs only. Test clips were read once for the motion-type transfer and once for the flag breakdown, from
committed code, after the validation findings were frozen. Rules for every analysis were fixed before computing.

**Main message.** The linear speed and acceleration probes read distance travelled (for uniform acceleration the same
as mean speed and mid-clip speed), yet the representation still separates the two motion profiles at matched distance,
more so with depth. Direction transfers between constant-velocity and accelerating clips. Probe errors are not driven
by exit, clipping or a frozen start, and slow, sub-patch motion is read as a continuous magnitude, not as
"moved / did not move". Speed and acceleration share a main direction in activation space; direction is near-orthogonal
to both, as the physics paper reports for its pairs.

## Setup

- **Data:** the joined tables and stored pooled activations (all-token mean per site), the layer-curve probes'
  validation and one-time test predictions, the nullspace probe sequences, and the six saved steering runs (30 test
  clips per variable, 5 targets). Headline sites: index 9 (steering layer) and 18 (readout layer); index 1 (first
  block) where stated.
- **Roles:** probes fit on train; readings on validation; test once, reported only, never selected on.
- **Overlap window:** the distance range both the speed and acceleration sets reach, 0.156–1.953 m (the design's
  distance confound: within each set the label is exactly proportional to distance travelled).

## 7.1 — Speed vs acceleration: what the probes read

**Cross-applied probes** (`confounds`: `cross_applied_probes`; probes refit = layer curves exactly; validation clips in
the window: 232 speed, 288 acceleration; each probe applied out of distribution). Slope of the cross-applied reading on
true distance (m), with 95 % clip-bootstrap interval:

| | index 9 | index 18 |
|---|---|---|
| acceleration probe on speed clips | 0.959 [0.940, 0.978] | 1.004 [0.988, 1.020] |
| speed probe on acceleration clips (τ/T = slope / 2) | 1.016 → τ/T 0.508 [0.499, 0.517] | 0.991 → τ/T 0.495 [0.486, 0.504] |

Pre-set rule: an acceleration probe that read acceleration would give slope 0 on constant-speed clips; reading the
speed at time τ gives T / (2τ). Both intervals lie above 0.5 → **"reads a speed / distance quantity"**; slope 1 =
distance = mean speed = mid-clip speed (indistinguishable for uniform acceleration). R² against distance 0.94–0.98;
index 1 gives the same slopes (1.016 / 0.991). **The linear readouts support H-03.**

**Matched-distance set classifier** (`matched_distance_classifier`): 49 nearest-value pairs in the window; train 32
pairs (512 + 512 clips), validation 37 pairs (208 + 188 clips, weighted per pair). Distance-only control: balanced
accuracy 0.500 (AUC 0.4996), so matching works. Balanced accuracy: index 0 0.509, index 1 0.765, index 9 0.947
[0.920, 0.971] (AUC 0.990), index 18 0.983 [0.967, 0.997] (AUC 0.999) → **"motion profile linearly separable at matched
distance"**, growing with depth (unlike the probes' index-1 transition).

**Reading:** the probes read distance, but the representation also encodes the motion profile beyond distance (the
per-clip version of Phase 6's two offset curves). This does not show that H-03 fails; it shows what the linear probes
miss.

Figure: `results/confounds/confounds.png`.

## 7.2 — Direction across motion types

(`robustness`: `motion_type_transfer`, validation; `motion_type_test`, one-time test.) Direction probes fit on one
motion type's train clips (397 velocity / 416 acceleration), scored within and across types. Bands on the across R²:
< 0.5 type-specific, 0.5–0.9 partly shared, ≥ 0.9 shared (refined by whether the across − within circular-error gap
includes 0).

| index | velocity → acceleration (val / test) | acceleration → velocity (val / test) |
|---|---|---|
| 1 | 0.710 / 0.742 partly shared | 0.607 / 0.505 partly shared |
| 9 | 0.931 / 0.939 shared, partly type-specific | 0.962 / 0.948 shared |
| 18 | 0.972 / 0.969 shared, partly type-specific | 0.951 / 0.944 shared, partly type-specific (gap 3.6° / 4.1°) |

All six readings hold on test. Restricting to the distance range both types reach removes the index-1 and index-9
gaps; index 18 acceleration → velocity keeps one (test 3.3 → 6.2°). **Direction is shared across motion types from
index 9 on, with a small type-specific part late.**

## 7.3 + 7.4 — Errors by clip flag and per-tubelet motion

(`flag_breakdown`, validation; `figure_tubelet`; `flag_test`, one-time test.) Sub-patch motion is fixed by the label
value (metadata), so it is tested as a trend; exit, clipping and frozen start vary within a motion group or value, so
they are tested as a stratified flagged − unflagged error difference (10,000 resamples).

**H-04 (errors concentrate in flagged clips):**

| | validation, index 1 / 9 / 18 | test, index 1 / 9 / 18 |
|---|---|---|
| direction exit (17 / 35 clips) | no excess ×3 | no excess / +2.59° excess / no excess |
| direction clipped (41 / 58) | +4.06° excess / no excess / +1.24° excess | no excess / +1.69° excess / +0.72° excess |
| direction frozen start (15 / 28) | −12.8° lower (fragile) / no excess / no excess | no excess ×3 |
| acceleration frozen start, within value | no excess ×3 | +0.10 m/s² excess / no excess / no excess |

Only the clipped-clip excess at index 18 holds on both splits (about 1°); the other effects flip between splits.
**H-04 is not supported beyond a small clipped-clip excess.**

**H-05 (slow motion read as a binary moved / did-not-move signal):** inside the sub-patch range, both readouts still
order magnitude: "continuous inside the sub-patch range" for speed and acceleration at all three indices, on validation
and test (validation slope inside / full range: speed 0.68 / 1.01, 0.87 / 1.00, 0.95 / 1.00; acceleration 0.58 / 1.00,
0.86 / 0.99, 0.98 / 0.99). Relative error is 3–6× higher for sub-patch clips (confounded with label value). Error vs
mean within-tubelet displacement (index 9, Spearman): direction −0.25 (without exit −0.23), speed −0.05 (relative
−0.55), acceleration +0.22 (relative −0.44). **The binary reading is rejected; slow clips are read with larger relative
error, not as a detector.**

**Steering by flag** (Phase 5 runs, post hoc): 30 clips per variable; every stratified comparison too few to read;
flagged and unflagged reductions similar.

Figure: `results/robustness/tubelet_scatter.png`.

## 7.5 — Subspace overlap and steering specificity

(`subspace_overlap`.) Subspaces fit on train only: probe weights (nullspace rounds 1 and 1…K−1) and patterns
(train covariance directions), compared weights with weights and patterns with patterns. Metrics as the physics paper
(App. C.4): mean principal angle, projection overlap ‖Q_AᵀQ_B‖²_F / dim(B), Grassmann distance. Spaces: one train
scaler shared by all three sets (headline) and raw features (paper-comparable). Nulls: the paper's k_A/d and 1,000
random subspaces inside each set's own train span, re-expressed like the real one.

| index 9, common space | weights (1 / K−1) | patterns |
|---|---|---|
| speed–acceleration | overlap 0.36 / 0.37, 53° / 54°: aligned beyond chance | 0.71, 33°: aligned |
| speed–direction | at chance (84–88°) | beyond chance but tiny (0.008, 83°) |
| acceleration–direction | at chance (84–89°) | at chance |

Raw space gives the same readings. **Speed and acceleration share a main direction without being the same subspace;
direction is near-orthogonal to both, matching the paper's C.4 result.**

**Readout validity across sets** (validation clips, R² at index 9 / 18): speed readout on direction velocity clips
≤ 4 m/s 0.971 / 0.982 and acceleration readout 0.957 / 0.945 (against distance, m); direction readout on speed clips
0.960 / 0.971 and on acceleration clips 0.929 / 0.954. All valid.

**Specificity** (index 18, probes K−1 minus a random edit of the same length, 30 clips): direction steering moves the
speed readout by +0.012 m/s [0.005, 0.019] (0.053 vs 0.041) and the acceleration readout by +0.055 m/s² [0.031, 0.080]
(0.199 vs 0.143) → cross-talk beyond a random edit, small in absolute size; speed or acceleration steering does not move
the direction readout. Speed ↔ acceleration cross-talk was measured in Phase 5 (1 : 1 in metres).

## Strengths, limitations and failure cases

- **Strengths:** no new forward passes; every refit reproduces its saved predictions bit for bit before being used;
  test read once per analysis from committed code; nulls drawn in the same space as the real subspaces.
- **Limitations:** validation exit clips are few (17), so flag CIs are wide; the steering set (30 clips) is too small
  for flag breakdowns; 7.1a covers linear readouts only; subspace readings depend on scaling (both spaces reported);
  no multiplicity correction across the many flag and subspace comparisons (headline cells named in advance).

## Kept on record

- The original 7.3–7.5 brief text was lost; the criteria were re-drafted as Claude Code's own additions and approved
  by the user before any run.
- Criterion changes before running: flag counts compared with the audit totals (the first version was circular);
  within-tubelet displacement compared to the flags table's 4-decimal rounding; principal-angle agreement at 1e-7 rad
  (arccos loses precision near 0).
- The steering-by-flag breakdown moved after the validation freeze, because the steering runs are on test clips.
- Direction frozen start at index 1 ("lower", 15 vs 13 clips) was fragile and not confirmed on test; acceleration
  frozen start and direction exit effects flip between splits.
- Speed–direction patterns read "beyond chance" at 83°: statistically above the null, practically orthogonal.
- Not predicted: direction → speed / acceleration steering cross-talk (small; noted to the planning chat); classifier
  at index 1 (0.765, predicted ≥ 0.8); 7.2 index 9 acceleration → velocity "shared"; test index 1 acceleration →
  velocity 0.10 below validation; test exit excess at index 9.
- Figures re-rendered after review: the tubelet scatter once (headroom), the confounds figure twice (legend).

## Limitations and open items

- Cross-talk mechanism (direction edits reaching the speed / acceleration readouts) untested.
- Phase 8: one main slide (the confounds figure plus one line each for 7.2–7.5), 7.2–7.5 tables as backups.