# Iterative nullspace probing — report

Date: 2026-09-26. Every claim points to a saved check result (`results/<subject>/checks.json`, key in backticks);
each entry carries its own provenance. The code behind every result was verified to be committed unchanged
(`evidence`: `code_hash_check`); two results saved before their code was committed were re-run from committed code
and reproduced exactly (`evidence`: `rerun_identical_nullspace`). Design decisions are in `docs/DECISIONS.md`.

**Verdict: nullspace probing complete.** Five checks failed or did not support their explanation and are kept on
record (see "Kept on record"); no reported number depends on them. Test was scored once, after every choice and
validation finding was fixed.

**Main message: the nullspace count measures the probe procedure, not the representation.** Linear readout can be
erased with as many dimensions as the target has (one for speed and acceleration, two for direction), and each
variable is still read by a nonlinear probe afterwards.

## Setup

- **Layers:** index 9 (after block 8) is the headline layer, with index 1 (where every variable becomes readable) and
  index 18 (the plateau) as contrasts. Index 9 is where the steering experiment will build its subspace from these
  probes, leaves downstream blocks for the later-layer readout, has all three variables strongly readable (val-seen
  R² 0.981 / 0.989 / 0.979), and is the depth comparable to the paper's "layer 8" (the paper numbers its layers
  0–23, i.e. blocks; evidence, not proof). It is not where variables emerge in our data (that is index 1).
- **Features:** the all-token mean at each layer, z-scored once with statistics of the training clips and never
  re-standardized.
- **Procedure:** each round fits a ridge probe on the training clips (regularization by exact leave-one-out inside the
  training clips), then removes its weight directions from the features (orthonormal basis, projection
  I − QQᵀ). Direction is one (sin θ, cos θ) probe per round, so it removes 2 dimensions per round; speed and
  acceleration remove 1.
- **Count:** K = the first round whose val-seen R² falls below 0.1 (the shuffled-label null measured in the
  layer-wise probing). All rounds were run up to the point where the training clips have no covariance with the
  label left (weights ≤ 1e-12 × round 1's; later rounds are undefined) or 150 rounds.
- **Saved for steering:** the scaler, the removed directions in order, each probe, and each probe's map in raw
  activation space; steering uses rounds 1…K only.
- **Test:** scored once from committed code, after the validation findings were recorded; nothing was selected on it.

## Nullspace counts

| K (dimensions removed before K) | index 1 | index 9 | index 18 |
|---|---|---|---|
| Direction, validation | 8 (14) | 6 (10) | 8 (14) |
| Speed, validation | 14 (13) | 7 (6) | 9 (8) |
| Acceleration, validation | 13 (12) | 7 (6) | 9 (8) |
| Test seen / unseen | 8/8, 14/14, 13/12 | 7/7, 7/7, 7/7 | 8/8, 9/9, 8/8 |

- Test agrees with validation within one round at every site; at index 9 the test K is 7 in 85–100 % of 10,000 clip
  resamples (6 otherwise) (`nullspace`: `nullspace_rounds`, `nullspace_test_scores`).
- The paper's thresholds read off the same curves: R² < 0.3 / 0.1 / 0.05 are crossed within one to two rounds of each
  other; the paper's secondary rules (direction circular MAE > 80°, speed MAE > 90 % of a mean prediction) are met
  within zero to two rounds of K.
- No rise after K: the largest R² after K is ≤ 0.03 at every site.
- **Controls** (`nullspace`: `random_subspaces`, `pc_subspaces`): removing the same number of random directions from
  the span of the training clips changes nothing (R² drops by at most 0.014 even after 150 rounds; five seeds);
  removing the top principal components erases the readout as fast or faster (K 2–7; for speed and acceleration a
  single top component can be enough).
- **Depth** (`nullspace`: `depth_profile`, every third index): K stays at 6–12 rounds from index 3 to 24 (highest at
  index 3, lowest at index 9, flat 8–9 late); nothing to count at the patch embedding.
- Figure: `results/nullspace/nullspace_rounds.png` (`nullspace`: `figure_nullspace`).

## What the count measures

1. **K depends on the probe's regularization** (`nullspace`: `alpha_sweep`, index 9, fixed ridge penalty). Among probes
   that read the variable well at round 1 (R² ≥ 0.93), K ranges from 7 to more than 150 (cap):

   | fixed alpha | 1e-3 | 1e-1 | 1e1 | 1e3 | 1e5 | 1e7 |
   |---|---|---|---|---|---|---|
   | round-1 R² | 0.94 | 0.95–0.96 | 0.98–0.99 | 0.97–0.98 | 0.64–0.68 | 0.02 |
   | K (direction / speed / acceleration) | 12 / 7 / 7 | >150 | 56 / 83 / 88 | 7 / 9 / 10 | 2 / 2 / 2 | 1 (probe already fails) |

   The headline procedure (alpha re-chosen each round) gives the smallest K among good probes (6 / 7 / 7), because its
   penalty rises as the signal weakens and its direction approaches the label covariance direction.
2. **Linear erasure needs only as many dimensions as the target** — a property of linear regression, not of V-JEPA:
   a linear probe's weights are zero whenever the features have no covariance with the label, and that covariance has
   one direction per target column. Empirically, the direction estimated on the training clips generalizes: after
   removing it, a fresh linear probe fit on the validation clips scores at chance (R² −0.045 … 0.001; direction
   circular MAE 90–92°), while removing random directions, even as many as the nullspace at K, changes nothing
   (`nullspace`: `fresh_probe_erasure`). On test at index 9 the fresh probe stays at chance (direction 95 % interval
   [−0.06, 0.01], speed [−0.02, 0.00]). "Erased" means undetectable by a linear probe fit on about 240 clips per fold.
3. **The information survives nonlinearly.** An RBF kernel probe on the same erased features still reads each variable
   (`nullspace`: `kernel_erasure`); at index 9 after removing the covariance direction(s), val-seen R² direction 0.753,
   speed 0.942, acceleration 0.928, and on test [0.72, 0.79], [0.91, 0.94], [0.92, 0.95] (95 % intervals, test seen).
   Across layers and arms, val-seen R² ≥ 0.66 under every regularization grid tried, while the linear probe is ≈ 0.
   The same pipeline on shuffled labels scores at most 0.005 (`nullspace`: `kernel_shuffled_labels`).
   Four kernel selections (all "nullspace at K" arms) were numerically unreliable and are quoted only as ranges
   (direction index 9: 0.66–0.69, index 18: 0.76–0.83; acceleration index 9: 0.92–0.93, index 18: 0.94)
   (`nullspace`: `kernel_hat_gap`).
4. **Redundancy:** 2–5 consecutive rounds keep R² ≥ 0.9 × round 1. Read as the procedure needing several steps to
   cover the covariance direction, not as several independent copies of the code.
- Figure: `results/nullspace/erasure_and_procedure.png` (`nullspace`: `figure_erasure`).

## Comparison with the physics paper

- The paper reports that direction needs many more dimensions (40–50 features, or 68 probes = 136 dimensions, at
  layer 8; 400 at layers 20–23, possibly a cap) and uses thresholds that differ between its text and figure captions.
  Our counts are much smaller (10–14 dimensions for direction) and do not grow with depth.
- Our regularization sweep shows counts of that size (56 to more than 150) for weakly regularized ridge probes that
  still read the variable well. Consistent with the paper's larger counts coming from its probes (trained with
  Adam and weight decay, no stated standardization) — a hypothesis; weight decay is not mapped to any ridge penalty.
- Our stimuli are simpler (one disk, flat background) and our input is 256 px (the paper 224 px).

## Kept on record

| Item | Reason / effect |
|---|---|
| `nullspace_rounds` failed "no leak into removed directions" (max 0.97–0.99999) | After the training clips' label covariance is exhausted, probe weights fall to rounding level and their direction is noise. Every round with signal is exact (leak ≤ 2.3e-14); K and every saved probe up to K are unaffected (`leak_diagnostic`, explanation holds) |
| Post-hoc leak criterion ("≤ 1e-8 before exhaustion") failed at 7 of 9 sites | By one round after K each (leaks 2.6e-8 … 6.3e-7); every direction removed before exhaustion has leak ≤ 6.3e-7 |
| `covariance_exhaustion`: pre-set rule did not hold | The covariance collapse is visible (it falls 10–10,000× per round), but my premise that the leak and the weight collapse start in the same round was wrong; exhaustion is defined by the weights |
| `kernel_erasure` failed "no alpha failure" (7 of 72 fits at the lowest penalty) | Scores are held-out; the 7 arms are flagged as under-resolved and not claimed as lower bounds |
| `kernel_grid_diagnostic`: rule did not hold | A finer grid moved 2 of 7 arms by more than 0.05, but its model selection was numerically invalid (rounding); those values appear only in the cross-grid ranges |
| Two results saved before their code was committed | Re-run from committed code, identical (`evidence`: `rerun_identical_nullspace`) |
| Acceleration's fresh linear probe after erasure is below chance on test seen (−0.20, interval [−0.54, 0.07]) | A probe fit to noise can do worse than a constant on another sample; observation |

## Open items and what each affects

| Item | Affects |
|---|---|
| Is the nonlinear code a curved low-dimensional manifold, and does direction's weaker recovery reflect higher harmonics? (hypotheses) | Spline steering and its dimensionality comparison |
| Probe steering may change the linear readout but leave the nonlinear code intact, so a downstream readout may only partly follow (hypothesis) | Steering evaluation (downstream readout) |
| The linear-erasure result matches LEACE (Belrose et al., 2023) as recalled; not yet checked against that paper | Any slide that cites it |
| Only linear and one kernel family tested, one sample size | Strength of "information survives" |
| Speed and acceleration labels equal distance travelled in these clips | Interpretation of every speed / acceleration count |
| Check script to be refactored (shared setup into the package) and affected keys re-run | Code submission |