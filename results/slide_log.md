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