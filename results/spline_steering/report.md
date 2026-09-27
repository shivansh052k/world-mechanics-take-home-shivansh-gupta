# Spline (manifold) steering vs multi-probe steering — report

Date: 2026-09-27. Every claim points to a saved check result (`results/<subject>/checks.json`, key in backticks);
each entry carries its own provenance. The code behind every result was verified to be committed unchanged
(`evidence`: `code_hash_check`, 68 keys, none to re-run). Design decisions are in `docs/DECISIONS.md` (D-51).

**Verdict: Part 2 complete.** Test clips were used once, from committed code, after every choice was fixed. Analyses
added after seeing results are labelled post hoc; their rules were fixed before computing.

**Main message.** Spline steering follows the curved pooled-mean manifold, and at the steering layer that matters:
straight chords cut through ambiguous states (direction far-target midpoints bimodal in 47 %; readout length 0.62 vs
0.94) while spline paths stay natural — largely by construction of a same-layer readout. Downstream it collapses like
the linear methods: at index 18 spline, covariance line and probes reach 0.07–0.14 of the intended change (spline and
covariance line within 0.005 of each other; probes within 0.02). All share the uniform token edit; only a
token-structured edit (correct per-time-step timing) changed propagation, slightly (+0.01–0.03; reversed timing hurts;
not confirmed on fresh clips).

## Setup

- **Four questions, in order:** Q1 geometry (what shape is each variable's representation?), Q2 isometry (does it
  match the geometry of a downstream readout?), Q3 held-out steering (spline vs Part 1's methods), Q4 token structure
  (does a time-structured edit propagate where the uniform edit collapses?).
- **Activation manifold (index 9, the steering layer):** per seen value, the mean of the training clips' pooled
  activations (train-standardized, as in Part 1); PCA refit inside every fold; a curve in the label value — a
  smoothing spline for speed / acceleration, a least-squares trig polynomial (closed loop) for direction. PCA dimension
  and smoothing chosen by leave-one-centroid-out (LOCO) on the training clips; the 4 validation-held-out values only
  confirm (`manifolds`: `manifold_loco`).
- **Behavior readout:** a 16-bin classifier of the label value fit on validation clips at indices 9–18; the behavior
  curve passes through the training clips' mean square-root probabilities per value (Goodfire-style)
  (`behavior`: `behavior_readouts`). For speed and acceleration this readout is blurry (top-1 0.28 / 0.26, within one
  bin 0.72 / 0.65); a two-stage readout was used only for isometry.
- **Steering:** Part 1's 90 test clips and 5 targets (3 of them test-unseen values with no centroid in the curve fit);
  edits at index 9, readouts at every index 9–18 through the same partial forward pass. Arms: spline path (fractions
  0.25–1), straight chord with the same endpoint, covariance line, free-spacing line, time-structured and
  time-reversed covariance, Part 1's probes and random edits. 6,240 edited passes (`spline_steering`:
  `spline_<variable>_<seen|unseen>`). Part 1's covariance edit, re-run inside the new code, reproduced its saved
  features bit for bit at all 450 clip × target pairs.
- **Scores:** gain = achieved ÷ intended change in the readout's output space (1 = reaches the target; direction on
  (sin, cos)); naturalness = the edited clip's Hellinger distance to the behavior curve minus the same clip's unedited
  distance; 95 % intervals from 10,000 clip resamples.

## Q1 — Geometry: curved for all three

| | direction | speed | acceleration |
|---|---|---|---|
| Held-out centroid error (LOCO), straight line → curve | 144.0 → 56.4 | 86.7 → 45.5 | 85.5 → 50.5 |
| line − curve [95 % CI] | 87.6 [70.8, 105.1] | 41.3 [30.9, 53.6] | 35.0 [21.4, 53.5] |
| share of that gain from uneven spacing along a straight line | — | 13 % | 23 % |
| PCA dimensions of the curve (k) / Part 1's nullspace count (K·m) | 8 / 12 | 6 / 7 | 6 / 7 |

- Nonlinear in the label for all three, held out at every validation-held-out value (curve beats line at 12 / 12).
  For speed / acceleration 87 % / 77 % of the gain is curvature, not uneven spacing; the best straight line with free
  spacing gives the same split (`manifold_ladder`). Direction needs harmonics beyond the first-harmonic ellipse:
  harmonics 1 and 2 hold 64 % and 23 % of the centroid variance (`direction_harmonics`).
- Goodfire-style exact interpolation is 1.2–2.2 × worse than the smoothed curve on held-out centroids (it fits the
  centroids' sampling noise).
- Low-dimensional: the fitted curve's participation ratio is 3.1 / 1.6 / 1.5; our direction curve needs 8 dimensions,
  not the paper's 40+ (`manifold_dimension`).
- Speed and acceleration on a shared distance-travelled scale lie on two offset, differently shaped curves with a
  shared main direction: the representation encodes motion profile beyond distance (observation;
  `speed_acceleration_manifold`). Whether the acceleration readout itself reduces to distance stays for Phase 7.
- Figure: `results/manifolds/manifolds.png` (`figure_manifolds`).

## Q2 — Isometry: dominated by label order

- Goodfire's pattern holds: activation and behavior geodesics correlate at 0.89–0.99, and the geodesic beats the
  straight chord (`behavior`: `isometry`; the two curves are built from disjoint halves of the training clips).
- But on a 1-D manifold parameterized by the same label both geometries follow label order: once a label-distance null
  is applied, the activation geodesic adds nothing globally (direction −0.001 [−0.004, 0.002]; speed −0.015; acceleration
  −0.105, both below the null). With equal-width value bins the behavior readout inherits label spacing partly by
  construction.
- Local test, per unit of label (post hoc, rule fixed first; `isometry_local`): speed's two curves speed up and slow
  down together beyond the label (r 0.59 [0.27, 0.77], near its split-half ceiling 0.62); direction and acceleration
  show no local agreement (intervals include 0). The behavior side's split-half reliability is low (0.17–0.41), which
  limits this test.

## Q3 — Held-out steering: spline vs Part 1

| gain at index 9 / 18 / mean 10–18 | direction | speed | acceleration |
|---|---|---|---|
| spline endpoint | 0.986 / 0.069 / 0.269 | 0.973 / 0.115 / 0.355 | 0.956 / 0.118 / 0.306 |
| covariance line | 0.986 / 0.069 / 0.268 | 0.977 / 0.119 / 0.357 | 0.957 / 0.119 / 0.308 |
| probes, K − 1 (Part 1) | 0.980 / 0.068 / 0.271 | 1.011 / 0.128 / 0.355 | 0.993 / 0.136 / 0.317 |
| random, same length | ≈ 0 | ≈ 0 | ≈ 0 |

- The spline edit points 66–76 % differently from the covariance edit (median ‖δ_spline − δ_cov‖ ÷ ‖δ_cov‖;
  `spline_setup`), yet the progress readouts cannot tell them apart: spline − covariance is within ±0.005 at index 18
  (`spline_scores`). At index 9 this is near by construction — a linear readout that decodes the centroids reads any
  move between curve points as that move in the label. Direction numbers are output-space gain.
- **Naturalness** (`spline_naturalness`): at the steering layer, spline midpoints are more natural than chord midpoints —
  spline − chord −0.124 [−0.138, −0.110] for direction (far targets −0.206), −0.046 for speed, −0.015 for acceleration.
  Direction chord midpoints are bimodal in 24 % of cases (47 % for far targets); spline waypoints are even more natural
  than the unedited clip. At index 18 the difference vanishes (all intervals include 0): every edit barely moves the
  readout there. Same-layer path results are close to by construction; only downstream readouts are evidence.
- **Held-out values** (`spline_held_out`): the spline reaches test-unseen targets (no centroid in the fit) as well as
  seen ones (index-9 gain 0.989 / 0.974 / 0.942 vs 0.983 / 0.971 / 0.987), and stays more natural than the covariance
  line and the probes there (all intervals below 0).
- Figure: `results/spline_steering/spline_paths.png` (`figure_spline_paths`): one direction clip steered 179° —
  the chord's midpoint splits between start and target at index 9; at index 18 every step still reads the start.

## Q4 — Token structure

- A time-structured covariance edit (each time step gets its own shift, same pooled mean) propagates slightly better
  than the uniform edit: mean gain over index 10–18 +0.030 [0.028, 0.032] / +0.013 / +0.017; the time-reversed control
  does worse (−0.026 / −0.007 / −0.007) (`spline_scores`). The model cares about when the change appears.
- Not confirmed: the pre-declared confirmation threshold (≥ 0.05) was not reached, so no fresh-clip run; the arms were
  designed after Part 1's results. A time-structured spline edit was not run (conditional on this threshold).
- Figure: `results/spline_steering/spline_profile.png` (`figure_spline_profile`); full table
  `results/spline_steering/comparison_table.md` (`comparison_table`).

## Comparison with the Goodfire paper

- They interpolate every centroid exactly and steer only between those centroids; we smooth by leave-one-out, report
  exact interpolation as a variant (worse on held-out centroids), and steer to values with no centroid.
- Their behavior distribution in the Mountain Car setting is a softmax over distances to the activation curve itself,
  so isometry is partly by construction; ours is a separate readout fit on other clips at later layers, and we add a
  label-distance null, which removes the global isometry.
- They keep 64 PCA dimensions; LOCO chose 6–8 here. Their manifold edit replaces the top PCA components; ours
  translates along the curve, so every method keeps the clip's own information and only the edit's shape differs.

## Strengths, limitations and failure cases

| | spline steering | multi-probe steering |
|---|---|---|
| **Strengths** | no procedure-dependent count (K); 6–8-dimensional curve; interpolates held-out values; natural paths at the steering layer (direction's circle) | needs no centroids; any target value; no curve to fit |
| **Limitations** | needs labelled, balanced values; no extrapolation beyond the fitted range; "behavior" is our validation-fit readout, blurry for speed / acceleration | count depends on regularization (Part 1); edits twice as long here |
| **Shared** | equal at linear readouts by construction; the same uniform token edit | |
| **Failure cases** | downstream collapse for every method; straight chords ambiguous on the circle; exact interpolation overfits (1.2–2.2 ×); speed and acceleration on two offset curves; isometry adds nothing beyond label order | |

## Kept on record

| Item | Reason / effect |
|---|---|
| `behavior_readouts` failed its own criterion "≥ 10 validation clips per bin" (direction, minimum 7) | Bins holding test-unseen values have fewer validation clips by design; the fit converged with an interior penalty |
| Harmonic permutation null was badly designed (own addition) | It tests "any structure", not "harmonic above noise"; no claim uses it; LOCO is the test |
| The independent-sampling noise estimate is an upper bound, not a floor | Held-out errors fall below it (hypothesis: nuisances balanced within each value by design); no noise-floor claim is made |
| The planner's prediction "spline and covariance differ at index 9" was wrong | Near by construction for a linear readout that decodes the centroids |
| Several analyses decided after results (local-speed isometry, held-out split of naturalness) | Labelled post hoc; rules fixed before computing |

## Limitations and open items

| Item | Affects |
|---|---|
| Every method edits all tokens the same way (per time step at most); real clips differ mainly in the disk tokens | Whether any pooled-space method can propagate; spatially structured edits need per-token training activations |
| One steering layer (index 9), one readout layer (index 18) | Generality |
| Behavior readout blurry for speed / acceleration | Their naturalness numbers |
| Speed and acceleration labels equal distance travelled | Interpretation; tested in Phase 7 |
| Future work: Goodfire's pullback check; 2-D joint (speed × direction) manifold; spatially structured edits | Talk |