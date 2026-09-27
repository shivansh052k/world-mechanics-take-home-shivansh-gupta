# V-JEPA 2 Physics Take-Home Assignment — Shivansh Gupta

**How a frozen V-JEPA 2 video encoder represents the direction, speed and acceleration of a moving object.**

Slides: [`slides.pdf`](slides.pdf) · Task: [`../README.md`](../README.md) · Data description: [`../DATA.md`](../DATA.md)

## TL;DR

I probed a frozen V-JEPA 2 ViT-L/16 on 4,572 synthetic clips of a moving disk, reproduced the physics paper's
probing → nullspace → steering progression, and extended it with Goodfire-style manifold (spline) steering.
- All three variables are **linearly readable after the first transformer block** and near-perfect (R² ≈ 0.99) by
  index 17–19 of 24.
- The nullspace "dimension count" **measures the probe procedure, not the representation**: it moves from 7 to more
  than 150 with the ridge penalty, and a nonlinear probe still reads the variable after linear erasure.
- Steering edits **work at the layer where they are made and fade within a few blocks**, for probe-subspace and
  spline steering alike; the representation is a curved, low-dimensional manifold, and following it keeps
  intermediate states natural but does not make the edit propagate.
- Within each dataset, speed and acceleration labels equal distance travelled: the linear probes read distance, yet
  the model still separates the two motion profiles at matched distance.

## Context

**Model.** V-JEPA 2 is a self-supervised video encoder trained to predict masked parts of a video in latent space.
We use the pretrained ViT-L/16 at 256 px (`facebook/vjepa2-vitl-fpc64-256`, pinned revision, weights hash-checked)
and keep it frozen: 16 frames become 2,048 tokens (8 time steps × 16 × 16 patches) passing through 24 blocks.

**The two papers.**
- *Joseph et al., Interpreting Physics in Video World Models:* train linear probes at every layer to find where a
  physical variable becomes readable; repeatedly fit a probe and remove its directions (iterative nullspace) to
  estimate how many dimensions carry it; steer the model by editing activations inside the subspace of several probes.
- *Goodfire (Wurgaft et al.), Manifold Steering:* representations of an ordered variable lie on a curved manifold;
  steering along a spline through the per-value centroids should move the model more naturally than a straight line.

**Data.** Three synthetic sets of 16-frame, 24 fps, 256 × 256 clips of one disk on a flat background:
direction (1,500 clips, 64 angles 5.625° apart, half constant velocity 1–7 m/s, half accelerating from rest
2–10 m/s²), speed (1,536 clips, constant velocity 0.25–4 m/s), acceleration (1,536 clips, from rest, 0.25–10 m/s²).

## Approach

The principles that make the numbers trustworthy:
1. **Audit before modelling.** Every clip was decoded, tracked and checked against its metadata (disk centre within
   0.8 px of the physics on every fully visible frame). The audit found an orange disk where DATA.md says blue,
   manifests sorted by label (so naive head/tail splits would be invalid), clips where the disk leaves the frame, and
   the distance confound.
2. **Held-out values, not just held-out clips.** 12 of the 64 values per set never appear in training, so every
   result can be checked on values the probes and curves never saw.
3. **Test data read once**, from committed code, after the validation findings were written down.
4. **A control or baseline for every claim:** shuffled labels, raw-pixel probes, a physics-fit ceiling, random-subspace
   removal, random steering edits of equal length, analytic chance levels.
5. **Every result saved with provenance** (commit, code hashes, package versions, time) and traceable to the exact
   code that produced it.

## What was done

| Step | Question | How |
|---|---|---|
| 1. Layer-wise probing | Where does each variable become readable? | Ridge probe at each of 26 sites (embedding, 24 blocks, final norm) on the mean activation; R² and (circular) MAE on validation; bootstrap transition point; one-time test. Per-patch probes for direction (local to global). |
| 2. Iterative nullspace | How many directions carry each variable? | At index 9 (plus 1 and 18): fit a probe, remove its directions, refit, for 150 rounds; K = the first round whose validation R² falls below 0.1. Controls: random and top-variance directions; ridge-penalty sweep; kernel probe after erasure. |
| 3. Multi-probe steering | Can we move the model's reading of a variable? | Smallest edit at index 9 that makes the first n probes read a target value, applied to held-out test clips with five target values, three of them never seen in training; read out at the same layer and nine blocks later by probes fit on other clips; random edits as control. |
| 4. Spline steering | Does following the representation's shape work better? | Curves through per-value training centroids (closed loop for direction), chosen by leaving one value out; edits along the curve vs. a straight line vs. the probe method, same clips and targets; naturalness of intermediate states. |
| 5. Confounds and robustness | Do the findings survive the data's known quirks? | Probes applied across sets at matched distance; motion-type transfer for direction; errors by clip flag; subspace overlap between variables; steering cross-talk. |

## Results

| | Finding | Figure |
|---|---|---|
| Probing | Readable after the first block (index 1 of 24) for all three variables, R² ≈ 0.99 by index 17–19; raw-pixel baseline 0.48–0.63, physics fit ≥ 0.997; test agrees. Direction is readable from patches the disk never came near (R² 0.65 at index 1). | [`layer_curves.png`](../results/layer_curves/layer_curves.png), [`local_to_global.png`](../results/patches/local_to_global.png) |
| Nullspace | K = 6 / 7 / 7 rounds (direction / speed / acceleration) at index 9; random removals change nothing. K ranges 7 to > 150 with the ridge penalty; an RBF kernel still reads each variable after erasure (test 95 % CIs between 0.72 and 0.95). | [`nullspace_rounds.png`](../results/nullspace/nullspace_rounds.png), [`erasure_and_procedure.png`](../results/nullspace/erasure_and_procedure.png) |
| Steering | Same-layer readout follows the edit (error reduction 0.92–0.95; random ≈ 0); nine blocks later only 0.03 / 0.13 / 0.14; the effect collapses within three blocks. | [`steering_reduction.png`](../results/steering/steering_reduction.png), [`steering_propagation.png`](../results/steering/steering_propagation.png) |
| Spline | Curved, 6–8-dimensional manifolds (held-out centroid error, line → curve: 144 → 56 / 87 → 45 / 86 → 51). Spline paths stay natural at the steering layer where straight chords cut through the direction loop; downstream every method reaches 0.07–0.14. | [`manifolds.png`](../results/manifolds/manifolds.png), [`spline_paths.png`](../results/spline_steering/spline_paths.png), [`spline_profile.png`](../results/spline_steering/spline_profile.png), [`comparison_table.md`](../results/spline_steering/comparison_table.md) |
| Confounds | Across sets both probes read distance (slope 1.00 / 0.99) while the motion profiles separate at matched distance (0.95 / 0.98 at index 9 / 18). Direction transfers across motion types; clip flags do not drive errors beyond a ~1° excess for clipped clips; direction is near-orthogonal to speed and acceleration. | [`confounds.png`](../results/confounds/confounds.png), [`tubelet_scatter.png`](../results/robustness/tubelet_scatter.png) |

**Relation to the papers.** Early readability of mean-pooled probes matches the physics paper; our per-patch
local-to-global transition is earlier than its one-third depth (our patch vectors are time-averaged and the stimuli
simpler). Our nullspace counts are far smaller than its tens to hundreds and depend on the ridge penalty. Same-layer
steering works as reported; the downstream fade is our addition. Direction's near-orthogonality to speed and
acceleration is in line with the chance-level overlaps the paper reports between its probe subspaces (App. C.4,
different variable pairs). Goodfire's representation–behavior isometry pattern holds, but on a one-dimensional
manifold it adds nothing beyond label order.

## Design choices at a glance

| Choice | Why |
|---|---|
| Mean over tokens as the probe feature | V-JEPA has no class token; the mean is layer-comparable and cheap. Per-patch probes cover the spatial question. |
| Our own forward hooks, not `hidden_states` | The last `hidden_states` entry means different things across transformers versions. |
| Resize and centre-crop off | The shipped processor would rescale our 256 px clips by ×1.14 and crop ~18 px per side. |
| Ridge, z-scored on train rows, leave-one-out penalty | Linear readout as in the paper; closed form, exact and deterministic. |
| Direction as (sin θ, cos θ) | No 0° / 360° jump (DATA.md's suggestion); scored by circular error. |
| Index 9 for nullspace and steering | All three variables are strongly readable there, and nine blocks remain for a downstream readout. |
| Downstream readout for steering | A readout at the edited layer follows the edit partly by construction; later layers show whether it propagates. |
| Readouts fit on validation, steering on test | The subspace (train), the readout (validation) and the evaluated clips (test) never overlap. |
| Splines through training centroids, chosen by leaving one value out | Each curve is judged on values it did not see; direction uses a periodic fit so the loop closes. |
| Speed vs. acceleration compared in metres | Within each set the label equals distance travelled; a shared distance scale separates the two readings. |

## How representations are extracted and pooled

- All 16 frames at native 256 × 256, decoded with PyAV; processor with resize and centre-crop off, rescale 1/255 and
  ImageNet normalization kept.
- Encoder frozen: fp32, eval mode, no gradients; the weights fingerprint is checked unchanged after forward passes,
  edits and extraction.
- Forward hooks at 26 sites: patch embedding, each of the 24 blocks, and the final LayerNorm output. Layer index:
  0 = patch embedding, 1–24 = blocks 1–24.
- Each site's 2,048 tokens are averaged over the 256 patches of each time step (8 × 1024 stored per site and clip);
  probes use the mean over all tokens. Per-patch vectors (direction only) are time-averaged per patch.
- Code: [`extraction.py`](../src/vjepa_physics/extraction.py), [`activations.py`](../src/vjepa_physics/activations.py),
  [`preprocess.py`](../src/vjepa_physics/preprocess.py).

## How fit and evaluation data are separated

- Per set: 12 of 64 values held out entirely (4 validation, 8 test); the other clips split train / val-seen /
  test-seen (speed and acceleration 16 / 4 / 4 per value; direction stratified by motion group × 45° octant).
  Built from metadata and a fixed seed: [`splits.csv`](../artifacts/manifests/splits.csv).
- Probes, nullspace rounds and spline curves: fit on train. Layer and model choices: validation. Steering readouts:
  fit on validation. Steered clips and target values: test, including unseen values.

## Quick start

```bash
python3.13 -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[dev]"
```

| Try | Command | Needs | Time | You should see |
|---|---|---|---|---|
| Unit tests | `python -m pytest -q` | nothing else | ~5 s | `29 passed` (metrics, kinematics, no leakage in probe fitting, nullspace erasure, minimum-norm steering, confound readings, subspace metrics) |
| The whole pipeline | `python scripts/run_pipeline.py --list`, then `--stage all --dry-run` | nothing else | 1 s | 9 stages, 141 steps, each with its recorded outcome |
| Video → physics | `python scripts/check_video_loader.py order`, then `figure` | `data/` | seconds | one clip's disk matches the physics within 1 px; wrong orders rejected; `results/video_loader/frames.png` |
| Data untouched | `python scripts/check_data_files.py fingerprint_final` | `data/` | 1–2 min | all 9,147 files identical to the fingerprint taken before any code ran |
| Frozen encoder | `python scripts/check_forward.py forward` | Apple Silicon, model download (~1.2 GB, once) | ~1 min | one clip through encoder and predictor, outputs (1, 2048, 1024), weights unchanged |
| Hooks and token layout | `python scripts/check_activations.py token_layout` | as above | ~1 min | the disk's token is found at the right time step, row and column |
| Edits are clean | `python scripts/check_intervention.py noop` | as above | 1–2 min | writing an activation back unchanged at any of 25 sites leaves every output bit-identical |

These are real checks: each prints its criteria and overwrites its saved record in `results/` (`git diff results/`
shows what changed).

## Repository map

| Path | Contents |
|---|---|
| [`src/vjepa_physics/`](../src/vjepa_physics) | The package. Data and video: `data`, `video`, `decoders`, `tracking`, `flags`, `geometry`, `splits`. Model and activations: `checkpoint`, `model`, `preprocess`, `activations`, `extraction`, `intervention`, `forecast`, `joined`. Analysis: `metrics`, `probes`, `baselines`, `curves`, `nullspace`, `steering`, `manifolds`, `behavior`, `confounds`, `robustness`. Infrastructure: `evidence` (provenance, hash-guarded artifacts), `reproducibility`, `plotting`. |
| [`scripts/`](../scripts) | `check_<subject>.py <key>`: each key is one saved experiment or verification. Verification: `video_loader`, `preprocessing`, `model`, `forward`, `activations`, `numerics`, `intervention`, `forecast`, `data_files`, `metadata`, `design`, `videos`, `tracking`, `evidence`. Experiments: `splits`, `extraction`, `joined` → `probes`, `baselines`, `layer_curves`, `patches` → `nullspace` → `steering` → `manifolds`, `behavior`, `spline_steering` → `confounds`, `robustness`. [`run_pipeline.py`](../scripts/run_pipeline.py) runs them in dependency order. |
| [`results/`](../results) | One folder per subject: `checks.json` (every saved result with its criteria and provenance) and the figures. |
| [`tests/`](../tests) | 29 unit tests of the maths; no data, model or network. |

Code comments and a few saved results cite IDs from my internal design log (D-, F-, H- numbers); the log itself is not
part of the submission.

## Full reproduction

```bash
python scripts/run_pipeline.py --stage all
```
- Run from a git clone: results record their commit and code hashes, so scripts need git and refuse uncommitted code.
- Supplied data in `data/`; about 45 GB free disk for activations; several hours in total (activation extraction
  alone ~1 h 45 min).
- Model steps use Apple MPS (results produced on an M3); other hardware needs the `DEVICE` constant changed in
  those scripts (untested).
- Nine checks failed on purpose when recorded (e.g. DATA.md's "blue disk" — the disk is orange); the runner expects
  them to fail again and flags only outcomes that differ from the record.

## Limitations and open questions

- Smaller, simpler stimuli than the paper; one checkpoint; parity with Meta's reference implementation is taken from
  Hugging Face's conversion check (atol 1e-3), not re-run here.
- Speed and acceleration labels equal distance travelled within each set; the linear probes read distance.
- Every steering method adds the same shift to every token at one layer — the leading candidate for why edits fade.
- Per-patch vectors are time-averaged; unspecified details of the paper's protocol are our interpretation.
- Open: why edits fade downstream; why direction edits nudge the speed and acceleration readouts slightly more than
  random edits.

## Use of AI tools

Developed with an AI coding assistant as a reviewer and guide; I wrote and ran all code and remain responsible for every choice and conclusion.