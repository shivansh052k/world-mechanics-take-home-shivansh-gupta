# Layer-wise probing — report

Date: 2026-09-25. Every claim points to a saved check result (`results/<subject>/checks.json`, key in backticks);
each entry carries its own provenance. The code behind every result was verified to be committed unchanged
(`results/evidence/checks.json`, `code_hash_check`); five results whose code changed before it was committed were
re-run from committed code and reproduced exactly (`rerun_identical_probing`). Design decisions are in
`docs/DECISIONS.md`.

**Verdict: layer-wise probing complete.** One check failed and is kept on record (see "Kept on record"); no result
depends on it.

## Setup

- **Features:** each clip's encoder activations at the patch embedding (index 0) and after each of the 24 blocks
  (indices 1–24), averaged over all 2,048 tokens; the final LayerNorm output is reported separately.
- **Probes:** ridge regression, features z-scored on the training clips, regularization chosen by exact
  leave-one-out inside the training clips. Direction is probed as (sin θ, cos θ) with one shared regularization.
- **Data use:** fit on train; every curve and layer choice uses validation clips (val-seen for selection, val-unseen
  reported alongside); the test clips were scored once, after the layer choices were fixed.
- **Metrics:** R² (direction: mean of the sin and cos R²) and MAE; circular MAE in degrees for direction.
- **References:** a pixel floor (ridge on the raw RGB frames, and on the time-averaged frame) and a physics-fit
  ceiling (per-clip least-squares fit of position, velocity and acceleration to the tracked disk).

## Where each variable becomes readable (mean-pooled probes)

| | Direction | Speed | Acceleration |
|---|---|---|---|
| Val-seen R², index 0 (patch embedding) | 0.122 | 0.000 | 0.000 |
| Val-seen R², index 1 (after block 1) | 0.854 | 0.981 | 0.976 |
| Best val-seen R² (index) | 0.991 (18–19) | 0.995 (19) | 0.992 (18) |
| Transition index (first reaching half the rise) | 1 | 1 | 1 |
| Test-seen / test-unseen R², index 18 | 0.989 / 0.989 | 0.992 / 0.993 | 0.990 / 0.990 |
| Test error, index 18 | 3.34° / 3.41° | 0.081 / 0.072 m/s | 0.227 / 0.217 m/s² |

- All three variables are linearly readable from the mean activation after the first block; the curve then rises
  slowly and peaks around index 17–19 (depth ≈ 0.75), with a small late decline (`probes`: `layer_curves`;
  `layer_curves`: `bootstrap`, `figure_layer_curves`).
- The transition sits at index 1 in all 10,000 bootstrap resamples for every variable (`layer_curves`: `bootstrap`).
- Held-out values are read as well as seen ones; test agrees with validation (`layer_curves`: `test_scores`).
- Direction scores barely change when clips where the disk leaves the frame are removed (test-seen 3.34° with,
  3.37° without).

| Checked | Result | Evidence |
|---|---|---|
| Shuffled labels | 20 permutations per site: max R² 0.087, direction circular MAE 87–92° on average (chance 90°) | `probes`: `shuffled_labels` |
| Pixel floor | Raw RGB: R² 0.48 / 0.63 / 0.49; time-averaged frame: 0.20 / 0.23 / 0.16 (direction / speed / acceleration), far below block 1 | `baselines`: `pixel_grams`, `pixel_floor` |
| Physics-fit ceiling | R² ≥ 0.997; every probe's error stays above it | `baselines`: `physics_ceiling` |
| Uncertainty | 95% clip-bootstrap intervals for every layer, floor and ceiling; paired layer-to-layer changes | `layer_curves`: `bootstrap` |

## Where inside the frame direction is readable (per-patch probes, direction clips)

Each clip's activation at each of the 256 patch positions, averaged over the 8 time steps, probed separately per
position and layer (`patches`: `extract_direction`, `verify`, `patch_probes`).

| Val-seen R² | index 0 | index 1 | index 6 | index 13 | index 24 |
|---|---|---|---|---|---|
| Mean-pooled probe | 0.122 | 0.854 | 0.969 | 0.990 | 0.989 |
| Per-patch probes, mean over 256 positions | 0.006 | 0.535 | 0.951 | 0.977 | 0.932 |
| Patches the disk never came near | −0.003 | 0.648 | 0.959 | 0.979 | 0.932 |
| Patches the disk passed through | −0.112 | −0.580 | 0.842 | 0.936 | 0.911 |
| One probe per frame half, same half | 0.002 | 0.814 | 0.958 | 0.980 | 0.932 |
| One probe per frame half, other half | −0.006 | 0.811 | 0.948 | 0.945 | 0.829 |

- **Direction is spread across the frame after one block:** patches the disk never came near already give R² 0.65
  at index 1 and 0.96 at index 6 (`patches`: `patch_breakdown`).
- **Patches the disk passes through are worse early** (below the mean at index 1), and catch up by index 6–9
  (hypothesis: the per-position probe learns the spread-out code from the majority of clips; where the disk is,
  the token is dominated by its local appearance).
- **One probe generalizes across the frame from block 1:** trained on one half, it reads the other half almost as
  well (gap 0.003 at index 1, ≤ 0.03 up to index 10); the gap grows only late (0.13 at index 20)
  (`patches`: `spatial_generalization`).
- **Not a position cue:** the disk's mean position alone predicts direction with R² 0.20 (0.21 with cubic terms),
  far below the per-patch readout (`patches`: `position_baseline`).
- Transition by the same rule: index 1 for the per-patch mean (9,582 / 10,000 resamples), for patches the disk
  never came near, and for the other-half score; the per-patch mean reaches 80% of its rise at index 4
  (`patches`: `patch_bootstrap`, `patch_breakdown`, `spatial_generalization`).
- Test reproduces every one of these curves (`patches`: `patch_test_scores`); figure
  `results/patches/local_to_global.png` (`patches`: `figure_local_to_global`).

## Comparison with the physics paper

- Consistent with the paper's mean-pooled probes: speed and acceleration readable early, direction improving
  gradually.
- Different from the paper's per-patch result: the paper reports that direction becomes local-to-global and
  position-invariant around one third of the depth; here it is spread and position-invariant from the first block,
  and loses some invariance late. Possible reasons, not tested: our per-patch vectors are averaged over time; our
  stimuli are simpler (one disk on a flat background); we use 256-px input (the paper 224 px); the paper does not
  specify every detail of its protocol, so ours is an interpretation.
- Observation, not a pre-set test: the per-patch mean comes within about 0.01 R² of the mean-pooled probe from
  index 9 (depth 0.375) to 13.

## Kept on record

| Item | What it means |
|---|---|
| Per-patch probes: 20 of 256 probes at index 0 chose the smallest regularization on the grid (`patches`: `patch_probes` failed on this) | All at the patch embedding, near-empty outer patches; R² ≤ 0.035 either way. The explanation tested first (ties) did not hold (`patch_alpha_diagnostic`); a later explanation (a few clips whose disk passes an outer patch) is untested. Every transition margin exceeds the maximum possible effect |
| Shuffled labels, direction: single permutations reached circular MAE as low as 43° | The average over permutations is 87–92° at every site (the criterion); the pre-set explanation for the spread did not hold at 5 of 21 sites |
| Acceleration in these clips starts from rest | Acceleration equals distance travelled and mean speed here, so "acceleration" readout may be a distance readout (tested later) |

## Open items and what each affects

| Item | Affects |
|---|---|
| Per-patch vectors are time-averaged; a per-time-step version is the natural next step (future work) | The comparison with the paper's per-patch result |
| Why early patches with the disk score below average (hypothesis above) | Interpretation of local vs spread-out coding |
| Distance confound of acceleration | Interpretation of every acceleration result |
| Activations and per-patch arrays exist only locally (git-ignored, hashes recorded); re-extraction ~1 h 45 min + ~36 min | Every later analysis |