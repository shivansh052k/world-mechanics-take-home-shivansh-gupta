# Environment and model setup — report

Date: 2026-09-24. Every claim below points to a saved check result (`results/<subject>/checks.json`, key in
backticks); each saved entry carries its own provenance (commit, clean/dirty flag, code hashes, library versions).
Design decisions and their reasons are in `docs/DECISIONS.md`.

**Verdict: setup passed**, with the open items listed at the end, each with the step it blocks.

## Setup

- MacBook M3, 16 GB, macOS 27.0; Python 3.13.2 in `.venv/`; 53 pinned packages in `requirements.lock.txt`
  (torch 2.14.0, torchvision 0.29.0, transformers 5.17.0, PyAV 18.1.0, OpenCV 5.0.0.93). Compute on MPS, fp32.
- Model `facebook/vjepa2-vitl-fpc64-256` at revision `b3c1679b7c34d3255ef3547f27c7b226aefab26f`;
  `model.safetensors` SHA-256 `25466aef85727d16546c6cf8c99f12fcfad9cbca8225d45f23685e2e025b786b`.
- Supplied data: 9,147 files fingerprinted (`artifacts/manifests/data_fingerprint.sha256`), `data/` read-only.

## What was verified

| Area | Result | Evidence |
|---|---|---|
| Video decoding | 16 frames of 256×256 at exactly 1/24 s; deterministic; RGB order confirmed (disk orange, core ≈ (234, 114, 39)); disk centre within 0.654 px of the metadata in every frame, with reversed / shifted / unflipped alternatives rejected | `video_loader`: `inspect`, `load`, `repeat`, `colour`, `order`, `figure` |
| Decoder cross-check | PyAV vs OpenCV are not pixel-identical (every pixel within 3 levels, systematic +1 in R and B, from different bundled FFmpeg builds); under the per-clip tolerance criterion the test clip is "ok" | `video_loader`: `opencv` (failed, kept), `opencv_bicubic`, `opencv_diff_stats`, `opencv_tolerance` |
| Preprocessing | Resize and center-crop off; output equals manual normalisation (max diff 1.65e-7) and inverts to the decoded clip bit-for-bit; the shipped default would rescale ×1.14 and crop 18 px per side | `preprocessing`: `config`, `manual`, `identity`, `default` |
| Model loading | Config and structure as documented; weights hash matches; clean loading report; 325,971,328 parameters; fp32, eval, no gradients; in-memory weights fingerprint stable across loads; seeds reproduce | `model`: `config`, `load`, `fingerprint`, `seeds` |
| Result provenance | Dirty flag reacts to code and environment changes only, from any working directory; real repo clean at the re-run | `evidence`: `dirty_flag` |
| Forward pass | Encoder and predictor outputs (1, 2048, 1024), finite; weights unchanged by the pass | `forward`: `forward` |
| Activation capture | Own hooks equal transformers' `hidden_states` entry by entry (last entry is before the final LayerNorm in 5.17.0); token index = t·256 + row·16 + col, shown from the data | `activations`: `hidden_states`, `token_layout` |
| Numerics | Repeat and batch (size 2) bit-exact; MPS vs CPU relative error grows to ~1.2e-3 at the last layers, failing the pre-set criterion; against a float64 reference, MPS fp32 error ≤ CPU fp32 error at all 27 outputs (ratio 0.58–0.99), so the gap is fp32 rounding, not MPS; sdpa kept (eager ÷ sdpa error 0.93–1.04) | `numerics`: `repeat`, `batch`, `devices` (failed, kept), `precision`, `attention` |
| Activation write-back | Writing back an unchanged activation leaves all 27 outputs bit-identical at all 25 sites; a small random edit changes exactly the edited site and everything after it; the edit runs before earlier-registered hooks (negative control confirms the check can tell the orders apart) | `intervention`: `noop`, `positive_control`, `hook_order` |
| Predictor forecasting | Separate encode → predict calls reproduce the model's own forward bit-for-bit; forecasting time steps 4–7 from frames 0–7 beats copy-last-step and mean-context-token baselines in every dataset (96 clips, all 95% CIs below 0), but only ~3% better than the mean token | `forecast`: `predictor_path`, `forecast` |
| Speed and memory | ~0.84 s per clip (all 25 sites captured and pooled); no gain from batching (batched outputs bit-identical at 2/4/8); ~65 min to extract all 4,572 clips; MPS pool 2.06 GiB at batch 1 of 11.84 GiB recommended | `benchmark`: `benchmark` |

Weights provenance (source reading, not a check): the HF checkpoint holds the target (EMA) encoder, and the HF
conversion script asserts its outputs match Meta's implementation at `atol=1e-3`.

## Failures kept on record

- `video_loader` `opencv` (pixel-identical PyAV vs OpenCV): not achievable on this install. Replaced by the
  per-clip tolerance criterion (`opencv_tolerance`, passed). PyAV is the only decoder whose output the model sees.
- `numerics` `devices` (MPS-vs-CPU error ≤ 1e-3 × between-clip difference): failed at the last layers. Replaced,
  after the failure, by "MPS fp32 error vs float64 ≤ CPU fp32 error vs float64 at every output" (met, `precision`).

## Skipped

- Numerical parity with Meta's official implementation: not run by us (would need a ~5 GB download and a second
  model). Limitation: the conversion was checked by its authors (atol 1e-3), not independently by us.

## Clean re-run

All 33 check keys were re-run from a clean, committed tree: 31 passed or completed, and the two failures above
failed as expected. Compared with the previous committed results, every deterministic value is identical; only
provenance fields, timings and the recorded repo state changed, and both figures are byte-identical. Timing note:
the full encode took 0.93 s per clip in the forecast check's first run but 1.28 s when that check ran last, after
~15 minutes of continuous GPU load (thermal throttling is a hypothesis, not tested), so extraction may take ~90 min
under sustained load.

## Open items and the step each blocks

| Item | Blocks |
|---|---|
| Pooling method and layer-numbering convention not yet chosen | Activation extraction |
| Extraction batch size: 1 suggested; any size > 1 needs the batching check re-run at that size | Activation extraction |
| `artifacts/` ignore rule untested until the first file is written there | First artifact write |
| Data facts from scouting (value grids, frame exits, tiny motion, colour, mapping) not yet reproduced on all clips | Splits |
| Decoder tolerance not yet applied to all clips | Data audit sign-off |
| Which later encoder layer serves as the steering readout | Steering evaluation |
| Whether the predictor readout carries motion information (small forecast margin; the all-token metric is likely background-dominated) | Only the optional predictor-readout extension |
| Pretraining `out_layers` setting and `loss_exp` value not read | Nothing; affects only the claim that the forecast target matches training exactly |
| Early-token difference between 8-frame and full-clip encodings (0.72 relative, one clip; cause not separated) | Nothing; informs pooling and readout choices |
| Possible slowdown under sustained GPU load | Nothing; affects the time budget |

Remaining design questions for later steps (splits, probes, nullspace, steering, splines, confounds, presentation)
are tracked in `docs/DECISIONS.md`.