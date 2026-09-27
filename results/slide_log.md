# Slide log

Candidate slides, added as results come in. Each names its figure, its message and the saved evidence behind it.

## Where motion becomes readable across layers

- Figure: `results/layer_curves/layer_curves.png`
- Message: direction, speed and acceleration are linearly readable from the mean activation after the first block
  (index 1 of 24) and rise slowly to ≈ 0.99 R² around index 17–19; far above a pixel baseline, below a physics fit.
- Points to say:
  - Transition at index 1 in all 10,000 resamples, for all three variables; held-out values read as well as seen.
  - Pixel floor 0.48–0.63 R²; physics-fit ceiling ≥ 0.997; test agrees with validation.
  - Caveat: in these clips acceleration equals distance travelled (clips start from rest).
- Evidence: `probes`: `layer_curves`, `shuffled_labels`; `baselines`: `pixel_floor`, `physics_ceiling`;
  `layer_curves`: `bootstrap`, `test_scores`.

## Where inside the frame direction is readable

- Figure: `results/patches/local_to_global.png`
- Message: after one block, direction is readable from patches the disk never touched, and one probe trained on half
  the frame reads the other half almost as well; position-specific coding appears only late.
- Points to say:
  - Patches the disk never came near: R² 0.65 at index 1, 0.96 at index 6; disk position alone explains only 0.20.
  - Same vs other half: gap 0.003 at index 1, 0.13 at index 20.
  - Differs from the paper's one-third local-to-global transition; our per-patch vectors are time-averaged and the
    stimuli simpler (interpretation of the protocol, caveats stated).
- Evidence: `patches`: `patch_probes`, `patch_breakdown`, `spatial_generalization`, `position_baseline`,
  `patch_test_scores`.

## What the nullspace count measures

- Figure: `results/nullspace/nullspace_rounds.png`
- Message: removing each probe's directions and refitting erases the linear readout of every variable within 6–14
  rounds (6–16 dimensions), at every layer; removing the same number of random directions changes nothing, and
  removing the top principal components erases it fastest.
- Points to say:
  - Headline layer index 9: K = 6 / 7 / 7 (direction / speed / acceleration); test agrees within one round everywhere.
  - Random directions: R² drops by at most 0.014 even after 150 rounds; top principal components: K 2–7.
  - K stays 6–12 from index 3 to 24; the paper reports 40–136+ dimensions for direction and 400 at late layers.
- Evidence: `nullspace`: `nullspace_rounds`, `random_subspaces`, `pc_subspaces`, `depth_profile`,
  `nullspace_test_scores`, `figure_nullspace`.

## Linear erasure removes the readout, not the information

- Figure: `results/nullspace/erasure_and_procedure.png`
- Message: the nullspace count measures the probe procedure, not the representation — linear readout can be erased with
  as many dimensions as the target has (by construction), and each variable is still read nonlinearly afterwards.
- Points to say:
  - Same layer, same data, only the ridge penalty changed: K from 7 to more than 150 among probes that read well.
  - Removing the label-covariance direction(s) (1 dim, 2 for direction): a fresh linear probe fit on held-out clips is at
    chance; this is a property of linear regression, the empirical part is that the direction generalizes.
  - An RBF kernel probe still reads direction / speed / acceleration after erasure: test R² [0.72, 0.79] /
    [0.91, 0.94] / [0.92, 0.95] at index 9; shuffled-label control ≤ 0.005.
  - Caveats: linear erasure theory (LEACE) to be checked before citing; one kernel family; speed and acceleration labels
    equal distance travelled.
- Evidence: `nullspace`: `alpha_sweep`, `fresh_probe_erasure`, `kernel_erasure`, `kernel_shuffled_labels`,
  `kernel_hat_gap`, `nullspace_test_scores`, `figure_erasure`.

## Steering works at the steering layer, not nine blocks later

- Figure: `results/steering/steering_reduction.png`
- Message: an independent readout at the steering layer follows a multi-probe edit (error reduction 0.92–0.95), random
  edits do nothing — the paper's result — but a readout nine blocks later moves only 0.03 / 0.13 / 0.14.
- Points to say:
  - One probe is not enough even at the same layer (0.11 / 0.55 / 0.36); from two probes on it follows; the covariance
    direction alone does as well as five or six probes.
  - Headline uses K − 1 probes; all K push the activations off-distribution (1–4 × the clip-to-clip distance).
  - Caveat: same-layer readouts follow edits partly by construction; this is why we test downstream.
- Evidence: `steering`: `steering_setup`, `steering_cache`, `steer_*`, `steering_scores`, `figure_steering`.

## Where the edit is lost

- Figure: `results/steering/steering_propagation.png`
- Message: the effect collapses within three blocks; at index 18 the carried edit is only partly aligned with the
  later readout, and the block updates push back about half to three-quarters of what reaches it.
- Points to say:
  - Gain 0.96–0.99 at index 9 → 0.24–0.42 at index 12 → 0.07–0.14 at index 18; random ≤ 0.03.
  - A kernel readout at index 9 also follows the edit, so "nonlinear code intact at the steering layer" is not supported
    as tested; the uniform token edit is the leading open explanation.
  - Speed and acceleration co-move one-for-one in metres (both labels = distance).
- Evidence: `steering`: `steering_propagation`, `steering_kernel`, `steering_specificity`, `figure_steering`.


## Part 2 · The representation is a curved, low-dimensional manifold

- Figure: `results/manifolds/manifolds.png`
- Message: per-value activation centroids at the steering layer lie on smooth curves — a closed loop for direction, bent
  arcs for speed and acceleration — that predict held-out values far better than a straight line.
- Points to say:
  - Held-out centroid error, line → curve: 144 → 56 (direction), 87 → 45 (speed), 86 → 51 (acceleration); the curve
    wins at 12 / 12 validation-held-out values.
  - Mostly curvature, not uneven spacing (87 % / 77 % for speed / acceleration); direction needs harmonics beyond the
    ellipse.
  - Low-dimensional: 6–8 PCA dimensions (participation ratio 1.5–3.1); Goodfire-style exact interpolation is 1.2–2.2 ×
    worse on held-out centroids.
  - Speed and acceleration on a distance scale: two offset curves, not one.
- Evidence: `manifolds`: `manifold_loco`, `manifold_ladder`, `manifold_dimension`, `direction_harmonics`,
  `speed_acceleration_manifold`, `figure_manifolds`.

## Part 2 · Held-out design and isometry against a label null

- Figure: none (table from `results/spline_steering/comparison_table.md`, Q2 rows)
- Message: activation and behavior geometries correlate as Goodfire reports (0.89–0.99), but on a 1-D manifold both
  follow label order; with a label-distance null the activation geodesic adds nothing globally.
- Points to say:
  - Held out = clips never used (test), values with no centroid in the fit, readouts fit on other clips at later layers.
  - Split halves: activation and behavior curves from disjoint training clips (no shared noise).
  - Local test per unit label: speed's curves speed up and slow down together (r 0.59, ceiling 0.62); direction and
    acceleration do not.
  - Caveat: equal-width value bins make behavior distances inherit label spacing partly by construction.
- Evidence: `behavior`: `behavior_readouts`, `isometry`, `isometry_local`.

## Part 2 · Paths along the curve stay natural at the steering layer

- Figure: `results/spline_steering/spline_paths.png`
- Message: steering along the curve keeps intermediate states looking like real clips; the straight chord cuts through
  the loop and reads ambiguous — but only at the steering layer.
- Points to say:
  - Direction far-target chord midpoints bimodal in 47 %; readout length 0.62 vs 0.94; spline − chord naturalness
    −0.12 at index 9.
  - At index 18 every difference vanishes: every edit still reads the start.
  - Same-layer path results are close to by construction; only downstream readouts are evidence.
- Evidence: `spline_steering`: `spline_naturalness`, `spline_held_out`, `figure_spline_paths`.

## Part 2 · Downstream, every method collapses alike; only timing nudges it

- Figure: `results/spline_steering/spline_profile.png` (+ comparison table)
- Message: spline, covariance line and probes all fall from ≈ 1 at index 9 to 0.07–0.14 at index 18 (spline and
  covariance within 0.005); following the pooled-mean manifold doesn't change what reaches later blocks.
- Points to say:
  - The spline edit points 66–76 % differently from the covariance edit yet propagates identically.
  - All share the uniform token edit; a correctly timed per-step edit adds +0.01–0.03, reversed timing hurts; below
    the pre-declared 0.05 threshold, not confirmed on fresh clips.
  - Strengths / limitations / failure cases: spline needs labelled balanced values and no extrapolation; probes need no
    centroids; both equal at linear readouts by construction.
- Evidence: `spline_steering`: `spline_setup`, `spline_scores`, `comparison_table`, `figure_spline_profile`.