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