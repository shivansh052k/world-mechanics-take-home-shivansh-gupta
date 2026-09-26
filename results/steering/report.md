# Multi-probe subspace steering — report

Date: 2026-09-26. Every claim points to a saved check result (`results/<subject>/checks.json`, key in backticks); each
entry carries its own provenance. The code behind every result was verified to be committed unchanged
(`evidence`: `code_hash_check`); two results saved before their code was committed were re-run from committed code and
reproduced exactly, including the files the steering runs read (`evidence`: `rerun_identical_steering`). Design
decisions are in `docs/DECISIONS.md` (D-50).

**Verdict: multi-probe steering complete.** Test clips were used once, from committed code, after every choice was
fixed. Four analyses were added after seeing the results; they are labelled post hoc, their rules were fixed before
computing, and nothing was selected on them.

**Main message: multi-probe steering reproduces the paper's same-layer result, but the edit barely reaches a readout
nine blocks later.** An independent readout at the steering layer follows the edit (error reduction 0.92–0.95 with
K − 1 probes), length-matched random edits do nothing, yet a readout at index 18 moves only 0.03 / 0.13 / 0.14 of the
way (direction / speed / acceleration). Under a uniform token edit, the linear code at the steering layer is readable
but largely not causally used downstream.

## Setup

- **Steering layer:** index 9 (output of block 8), the layer whose nullspace probe sequence was built in the previous
  step; probes 1…K with K = 6 / 7 / 7; **headline n = K − 1** (5 / 6 / 6: every probe that still reads the variable,
  val-seen R² ≥ 0.1). This was fixed from training-clip statistics before any steering run: round K carries 75–99 % of
  the edit's squared length (steering geometry, terminal check, F-129).
- **Edit:** the smallest shift, measured in the training-standardized space, that makes the first n probes all read the
  target — the paper's procedure (stack the probes, least squares, keep the orthogonal part) in our space. The shift lies
  exactly in the subspace the probes span. The same shift is added to all 2,048 tokens, which moves the pooled mean by
  exactly that amount.
- **Readouts:** ridge probes fit on the **validation** clips (never the steering probes' training data, never test):
  primary at **index 18** (output of block 17), reached by running blocks 9–17 on the edited activations; a
  same-layer readout at index 9 as a labelled control. Readout quality on the training clips: R² 0.969–0.987
  (`steering`: `steering_setup`). The partial forward equals the full model bit for bit on all 90 clips, with and
  without an edit (`steering_cache`).
- **Clips and targets (fixed before any model run):** per variable 15 test-seen + 15 test-unseen clips; 5 target
  values, 3 of them held-out test values (`steering_setup`). Before steering, the index-18 readout reads these clips'
  own labels with R² 0.980 / 0.989 / 0.991 (`steering_cache`).
- **Arms:** probe counts 1…K; the label-covariance direction(s) (1 dimension, 2 for direction); random directions of the
  same length as the real edit (3 seeds at n = 1, 3, K − 1, K); no edit. Six saved runs, 8,955 edited passes in total
  (`steer_*`).
- **Scores:** error to the target (circular degrees for direction) relative to the unedited clip ("error reduction":
  1 = the readout reaches the target, 0 = no change); 95 % intervals from 10,000 clip resamples (`steering_scores`).

## Headline

| error reduction, n = K − 1 (95 % CI) | direction | speed | acceleration |
|---|---|---|---|
| **readout at index 18** | **0.029** [0.026, 0.032] | **0.129** [0.124, 0.134] | **0.136** [0.127, 0.147] |
| independent readout at index 9 (same layer) | 0.95 | 0.95 | 0.92 |
| same layer, one probe only | 0.11 | 0.55 | 0.36 |
| covariance direction(s): index 18 / index 9 | 0.031 / 0.956 | 0.119 / 0.938 | 0.119 / 0.904 |
| random edits of the same length, index 18 | ≈ 0 | ≈ 0 | ≈ 0 |

- In absolute terms at index 18: direction error to the target 87.3° vs 90.0° without the edit; the readout still reads
  the clip's own direction almost as before (4.0° → 5.0°) (`steering_scores`).
- **One probe is not enough even at the same layer**; from two probes on, an independent readout follows almost fully.
  The covariance direction(s) alone do as well as K − 1 probes, so the "many directions" requirement belongs to the
  probe sequence, as in the nullspace count.
- **Held-out values:** test-seen and test-unseen clips, and seen and unseen targets, are reported separately in the
  saved scores (same pattern throughout).
- **All K probes** (n = K) push the activations 1–4 × the typical distance between two clips; for speed the readout of
  the clip's own value breaks (error 0.08 → 1.03 m/s) and a random edit of that size also degrades it — an
  off-distribution edit, not steering.
- Figure: `results/steering/steering_reduction.png` (`figure_steering`).

## Where the edit is lost (post hoc)

1. **The effect collapses within three blocks** (`steering_propagation`; validation-fit readouts at every index 9–18,
   gain = readout change / intended change in the readout's output space, (sin, cos) for direction). With K − 1
   probes the gain falls from 0.96–0.99 at index 9 to 0.24 / 0.42 / 0.25 at index 12 and 0.07 / 0.13 / 0.12 at index
   18. Random edits stay ≤ 0.03 at every index.
2. **Two reasons, measured separately at index 18.** The edit itself is carried unchanged by the skip connections; on
   its own it would move the index-18 readout by only 0.19 / 0.46 / 0.26 of the intended amount — the later readout
   is only partly aligned with the edited direction. The block updates then push back 64 / 73 / 48 % of that, leaving
   0.07 / 0.13 / 0.14 (95 % intervals in the saved result). The covariance arm behaves the same.
3. **A nonlinear readout does not see the original value either** (`steering_kernel`). The hypothesis was that the edit
   fools the linear readout but leaves a nonlinear code intact, which later blocks recover from. A validation-fit RBF
   kernel readout at index 9 follows the edit like the linear one (gain 0.96–0.99; nearest the target in 86–99 % of
   runs), so the mechanism is **not supported as tested**. Caveat: a full-feature kernel is dominated by the edited
   direction, and the edit leaves everything outside the probe subspace untouched — which the nullspace step showed
   still carries each variable nonlinearly. Whether later blocks use it would need further forward passes.
4. **Speed and acceleration steer together** (`steering_specificity`). In metres travelled over the clip, steering
   one moves the other's readout one-for-one (slopes 1.00 [0.97, 1.04] and 0.94 [0.89, 1.00] at index 18); random
   edits also move both (0.56–0.77). Expected, since both labels equal distance travelled in these clips; it cannot
   separate the two variables. Direction steering moves speed and acceleration no more than random edits do.
- Figure: `results/steering/steering_propagation.png` (`figure_steering`).

## Comparison with the physics paper

- The paper steers and evaluates at the same layer (its evaluation probe is trained on the test clips): error 82.9° →
  11.9° with 20 probes, 1–5 probes above 50°. We reproduce that pattern with an independent same-layer readout
  (validation-fit): one probe is weak, a few probes suffice, random edits do nothing.
- Evaluating nine blocks later, which the paper does not do, shows that the edit hardly propagates. A same-layer
  readout follows an edit along its own direction partly by construction, so a same-layer evaluation overstates what
  the steering does to the model's later computation.
- Our stimuli are simpler (one disk, flat background), input 256 px (paper 224 px), and the paper's exact least-squares
  details are undocumented; our edit is the minimum-norm shift in the standardized space.

## Kept on record

| Item | Reason / effect |
|---|---|
| `steering_setup` and `steering_cache` saved while their script was staged, not committed | Re-run from committed code; results and the SHA-256 of every file the steering runs read identical (`rerun_identical_steering`) |
| `steering_propagation` saved from uncommitted code | Its code was committed unchanged in the next commit (`code_hash_check`); `require_clean_code` now refuses such runs |
| Four analyses decided after seeing the results | Labelled post hoc; rules and predictions fixed before computing; nothing selected on them |
| n = K edits are off-distribution (1–4 × the clip distance) | Headline is n = K − 1, fixed before any steering outcome |

## Limitations and open items

| Item | Affects |
|---|---|
| Uniform token edit: the same shift on every token leaves all token-to-token differences unchanged (hypothesis: later blocks rebuild the variable from them) | Whether the collapse is about the representation or the edit; testable with a token-structured edit through the saved per-token cache |
| Same-layer readouts follow edits partly by construction | Weight of the same-layer result |
| One steering layer (index 9) and one readout layer (index 18) | Generality |
| 30 clips × 5 targets per variable | Precision (intervals reported) |
| Speed and acceleration labels equal distance travelled | Specificity; separated in the confound step |
| Kernel test on full features only | Strength of "H-12 not supported" |
| Predictor readout not tested | Whether the model's own forecast follows the edit |