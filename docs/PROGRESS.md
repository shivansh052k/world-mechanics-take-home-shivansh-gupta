# PROGRESS.md — where we are right now

Last updated: 2026-09-24.

This file is the live status tracker. Update it at the end of every phase (and any time work
pauses mid-phase), so this chat, Claude Code, and any future session can pick up instantly
without re-reading everything else.

---

## Current status

**Planning: complete** (including two review rounds). `CLAUDE.md`, `DECISIONS.md`, and
`EXECUTION_PLAN.md` are finalized and in place.

**Execution: in progress.** Steps 0.1–0.8 done (guided in Claude Code since 0.8d, D-21/D-22). Loader
`src/vjepa_physics/video.py` (`load_clip`) verified by checks `inspect`, `load`, `repeat`, `colour`,
`order`, `figure`; `opencv` failed (PyAV vs OpenCV not pixel-identical, F-49) → D-05 criterion is with the
planning chat (affects step 1.8, not 0.8). **0.9 done:** `src/vjepa_physics/preprocess.py` (`preprocess_clip`,
checkpoint processor with resize/crop off) verified by `config`, `manual`, `identity`, `default` (F-50).
**0.10 done:** `src/vjepa_physics/model.py` (`load_model`, `weights_fingerprint`) and `reproducibility.py`
(`set_seeds`) verified by `config`, `load`, `fingerprint`, `seeds` (F-51, D-26).
**0.11 done:** full forward pass on MPS verified by `scripts/check_forward.py forward` (F-52).
**0.12 done:** `src/vjepa_physics/activations.py` (`capture_encoder`, `GRID`, `as_grid`) verified by
`hidden_states` and `token_layout` (F-53).
**0.13 done:** `scripts/check_numerics.py` — bit-exact repeat and batch; MPS vs CPU recorded; float64 reference
shows MPS fp32 ≥ as accurate as CPU fp32; sdpa kept (F-54, F-55). MPS-vs-CPU criterion (O-19) with the planning
chat. **0.14 done:** `src/vjepa_physics/intervention.py` (`edit_encoder`, `encoder_sites`) verified by
`scripts/check_intervention.py` `noop`, `positive_control`, `hook_order` (F-56). **0.15 done:** `src/vjepa_physics/forecast.py` (`encode`, `predict`,
`training_target`) verified by `scripts/check_forecast.py` `predictor_path` and `forecast` (F-57); O-18 answered from
source (F-58, user confirmation pending); O-20 (is the predictor readout informative about motion?) with the planning
chat. **Next: 0.16 (speed and memory benchmark)** — waiting for the user's go.

---

## Phase status

| Phase | Status | Notes |
|---|---|---|
| 0 — Environment and model setup | 🟨 In progress | 0.1–0.15 done; 0.16 next |
| 1 — Data audit | ⬜ Not started | |
| 2 — Splits and activation extraction | ⬜ Not started | |
| 3 — Part 1a: Layer-wise probing | ⬜ Not started | |
| 4 — Part 1b: Iterative nullspace probing | ⬜ Not started | |
| 5 — Part 1c: Multi-probe subspace steering | ⬜ Not started | |
| 6 — Part 2: Spline steering | ⬜ Not started | |
| 7 — Confounds and robustness | ⬜ Not started | |
| 8 — Presentation and final delivery | ⬜ Not started | |

Status legend: ⬜ Not started · 🟨 In progress · ✅ Passed gate · ⚠️ Blocked (see Open issues below)

---

## Key numbers so far

| Metric | Value | Source |
|---|---|---|
| Supplied data files (excl. `.DS_Store`) | 9,147 | F-45 |
| Model revision | `b3c1679b7c34d3255ef3547f27c7b226aefab26f` | F-44 |
| `model.safetensors` SHA-256 | `25466aef85727d16546c6cf8c99f12fcfad9cbca8225d45f23685e2e025b786b` | F-44 |
| Conv3d MPS vs CPU relative difference | 2.5e-7 | F-43 |
| Locked packages | 53 (torchvision 0.29.0 added) | F-42, D-25 |
| Test clip frame timing (PyAV) | time_base 1/12288, pts step 512 = 1/24 s | F-46 |
| Parameters (encoder + predictor) | 303,885,312 + 22,086,016 = 325,971,328 | F-51 |
| In-memory weights fingerprint (reference for "model unchanged") | `c865f524c1376e4452943b208d7d50ba588be9490604f235a3d9c9dc80804ede` | F-51 |
| Median time per clip on MPS (full encode / 8-frame encode / predictor) | 0.93 / 0.42 / 0.13 s | F-57 |

---

## Saved evidence

Every saved check result (D-24). Each entry: file, key, what it proves, status.

| File | Key | Proves | Status | Source |
|---|---|---|---|---|
| `results/video_loader/checks.json` | `inspect` | Test clip `data/speed/videos/scene_1000/video.mp4` (PyAV 18.1.0): 16 frames decoded and reported by the container; every frame 256×256; pts × time_base rises by exactly 1/24 s (exact fractions, 0 → 5/8 s). Codec `mpeg4`, `yuv420p` recorded as observations | ✅ passed (explicit criteria since re-run) | F-46; `scripts/check_video_loader.py inspect` |
| `results/video_loader/checks.json` | `load` | `load_clip` gives (16, 256, 256, 3) uint8, C-contiguous; raises `ValueError` on a wrong frame count (15) or size (224) instead of padding/truncating/resizing. Min 0 / max 249 explained by `colour` | ✅ passed | `scripts/check_video_loader.py load` |
| `results/video_loader/checks.json` | `repeat` | Two loads bit-identical; decoded-pixel SHA-256 `03285f4cf9ffc6ecf90541b47e7f08c217eab4e4809d42f7042baf29ab9d95c0` for cross-run comparison | ✅ passed | `scripts/check_video_loader.py repeat` |
| `results/video_loader/checks.json` | `colour` | RGB order: disk core (234.0, 114.4, 39.2) orange, background (29, 32, 29); blue = 0 only in a ring 0–2.2 px outside the disk | ✅ passed | F-47; `scripts/check_video_loader.py colour` |
| `results/video_loader/checks.json` | `order` | Frame content in time order, geometry exact: max 0.654 px vs metadata, no bias; reversed / +1 frame / no y flip rejected (54.1 / 4.2 / 73.7 px) | ✅ passed | F-48; `scripts/check_video_loader.py order` |
| `results/video_loader/checks.json` | `opencv` | PyAV vs OpenCV (FFmpeg backend) pixel-identical | ❌ failed: all pixels differ by ≤ 3, systematic +1 in R and B; D-05 criterion under review | F-49; `scripts/check_video_loader.py opencv` |
| `results/video_loader/checks.json` | `opencv_bicubic` | Diagnostic: swscale flag is not the cause (PyAV BILINEAR = BICUBIC); frame colour tags unspecified | ℹ️ diagnostic | F-49; `scripts/check_video_loader.py opencv_bicubic` |
| `results/video_loader/checks.json` | `opencv_diff_stats` | Diagnostic: signed PyAV − OpenCV histogram and means by region (+1.01 / +0.01 / +1.00 overall) | ℹ️ diagnostic | F-49; `scripts/check_video_loader.py opencv_diff_stats` |

| `results/video_loader/frames.png` + `checks.json` | `figure` | Visual evidence: frames 0/5/10/15 with predicted (+) and measured (×) disk centre and predicted path; disk orange, moving down-left (θ 230.6°); errors at shown frames 0.319 / 0.42 / 0.11 / 0.407 px (= `order`) | ℹ️ visual (reviewed) | `scripts/check_video_loader.py figure` |

| `results/preprocessing/checks.json` | `config` | Our processor: resize and crop off; rescale 1/255 and ImageNet normalization as shipped. Default: shortest edge 292 + crop 256 | ✅ passed | F-50; `scripts/check_preprocessing.py config` |
| `results/preprocessing/checks.json` | `manual` | Output (16, 3, 256, 256) float32 CPU = manual (x/255 − mean)/std, max abs diff 1.65e-7 | ✅ passed | F-50; `scripts/check_preprocessing.py manual` |
| `results/preprocessing/checks.json` | `identity` | Un-normalize + round = decoded uint8 clip bit-for-bit (max 9.4e-6 levels) → no spatial change | ✅ passed | F-50; `scripts/check_preprocessing.py identity` |
| `results/preprocessing/checks.json` + `default_vs_ours.png` | `default` | Shipped default scales ×1.1406 (slope 1.1393/1.1405) and crops 18 px/side (offset −17.93 px); disk 350 → 455 px; 36.5 px/m; figure reviewed | ✅ passed | F-50; `scripts/check_preprocessing.py default` |

| `results/model/checks.json` | `config` | Config and built structure = F-10: 24 encoder / 12 predictor blocks, patch Conv3d (1024, 3, 2, 16, 16), dropout 0 | ✅ passed | F-51; `scripts/check_model.py config` |
| `results/model/checks.json` | `load` | Weights SHA-256 = pinned; loading report empty (587 tensors); 303.9M + 22.1M params; fp32, eval, no grad, sdpa; fingerprint recorded | ✅ passed | F-51; `scripts/check_model.py load` |
| `results/model/checks.json` | `fingerprint` | Two independent loads → identical in-memory weights fingerprint `c865f524…04ede` | ✅ passed | F-51; `scripts/check_model.py fingerprint` |
| `results/model/checks.json` | `seeds` | `set_seeds(0)` reproduces Python / NumPy / torch CPU / torch MPS draws; seed 1 changes all | ✅ passed | F-51; `scripts/check_model.py seeds` |

| `results/evidence/checks.json` | `dirty_flag` | In a scratch git repo: `git_dirty` False when committed and for `results/`- or `docs/`-only changes; True for any edit or new file under `src/`, `scripts/`, `pyproject.toml`, `requirements.lock.txt`; same answer from repo root, subfolder and outside. Real repo root found from anywhere. Real-repo confirmation after commit `c39e0a8`: `False []` | ✅ passed | D-27; `scripts/check_evidence.py dirty_flag` |

| `results/forward/checks.json` | `forward` | One clip through encoder + predictor on MPS: all outputs (1, 2048, 1024), finite; predictor target = encoder output; weights fingerprint unchanged by the forward pass | ✅ passed | F-52; `scripts/check_forward.py forward` |

| `results/activations/checks.json` | `hidden_states` | Own hooks vs `hidden_states` in one pass: entry 0 = embedding, entry i = block i−1, entry 24 = block 23 pre-LayerNorm; `last_hidden_state` = LN(block 23) exactly; our hooks removed (48 transformers hooks before and after). Proves index alignment and semantics, not run-to-run determinism (0.13) | ✅ passed | F-53; `scripts/check_activations.py hidden_states` |
| `results/activations/checks.json` | `token_layout` | Token index = t·256 + row·16 + col shown from the data: disk patch is the most deviant embedding token in 8/8 time steps; swapped rows/cols 3/8, reversed time 4/8 rejected. Limit: one clip whose path crosses near the row = col diagonal | ✅ passed | F-53; `scripts/check_activations.py token_layout` |

| `results/numerics/checks.json` | `repeat` | Same input twice: bit-exact at all 27 outputs on MPS and on CPU (re-run after fixing the MPS float64 conversion, F-55) | ✅ passed | F-54; `scripts/check_numerics.py repeat` |
| `results/numerics/checks.json` | `batch` | Batch of 2 = each clip alone, bit-exact on MPS (batch size 2 only) | ✅ passed | F-54; `scripts/check_numerics.py batch` |
| `results/numerics/checks.json` | `devices` | MPS vs CPU per layer: rel. error 6e-7 → 1.2e-3 with depth, token cosine ≥ 0.9999; default fp32 tolerances fail everywhere (hypothesis confirmed) | ❌ failed own criterion at blocks 19–23 + final norm (≤ 1.23e-3 vs 1e-3); explained by `precision` (fp32 limit, not MPS); criterion → O-19 | F-54; `scripts/check_numerics.py devices` |
| `results/numerics/checks.json` | `precision` | Diagnostic vs CPU float64: MPS fp32 error ≤ CPU fp32 error at all 27 outputs (ratio 0.58–0.99) | ℹ️ diagnostic | F-54; `scripts/check_numerics.py precision` |
| `results/numerics/checks.json` | `attention` | Diagnostic: eager ÷ sdpa error vs float64 = 0.93–1.03 → keep sdpa (pre-stated rule) | ℹ️ diagnostic | F-54, D-26; `scripts/check_numerics.py attention` |

| `results/intervention/checks.json` | `noop` | Writing back an unchanged activation (`clone`, `+ 0`) at each of 25 sites leaves all 27 outputs bit-identical on MPS; hook fired once per run; hooks removed (0/0); fingerprint unchanged | ✅ passed | F-56; `scripts/check_intervention.py noop` |
| `results/intervention/checks.json` | `positive_control` | Adding a seeded random δ at each site: outputs before it bit-identical, site = baseline + δ exactly, every output after it changed (incl. final norm, predictor); change sizes recorded as observations | ✅ passed | F-56; `scripts/check_intervention.py positive_control` |
| `results/intervention/checks.json` | `hook_order` | With transformers' `hidden_states` hooks and our capture registered before the edit, both see baseline + δ at every site (prepend works); negative control (appended edit) leaves capture unedited; hooks 48/48 | ✅ passed | F-56; `scripts/check_intervention.py hook_order` |

| `results/forecast/checks.json` | `predictor_path` | Separate `encode` → `predict` = combined forward bit-for-bit; reversed forecast targets return reversed rows; forecast shapes (1, 1024, 1024); context-vs-full-clip difference 0.72 recorded as an observation | ✅ passed | F-57; `scripts/check_forecast.py predictor_path` |
| `results/forecast/checks.json` | `forecast` | 96 clips (32 per dataset, ids saved): predictor beats copy-last-step and mean-context-token baselines in every dataset (all 95% CIs < 0); margin over the mean token ~3% of L1 (→ O-20); H-06 fields, per-step L1s, timings recorded | ✅ passed | F-57; `scripts/check_forecast.py forecast` |

All check scripts save through `src/vjepa_physics/evidence.py` (`save_result`). Provenance (D-27): `git_dirty`
= uncommitted or untracked changes in code/environment paths only, `git_dirty_paths`, `code` (SHA-256 per file
and combined), `versions` (from package metadata). **Entries saved before commit `c39e0a8` keep the old meaning**
(`git_dirty` = any change anywhere in the tree).

Still pending: `opencv` re-scored once the planning chat settles D-05's criterion. For slides, a zoomed
crop of `frames.png` would read better (disk ≈ 21 px of 256).

---

## Open issues / blockers

*(Anything stopping progress goes here, with enough context to resume without re-deriving it.)*

No blockers.

Awaiting the planning chat:
- **D-05 cross-check criterion.** Pixel-identical PyAV vs OpenCV is not achievable on our install (F-49:
  systematic +1 R/B offset, max 3 levels; likely FFmpeg 8 vs 7). Proposed: (a) tolerance band per clip
  (max |diff| ≤ 3, mean signed diff near (+1, 0, +1), outliers flagged), PyAV the only decoder for model
  inputs; optional (b) textbook BT.601 reference from PyAV's raw YUV. Affects the `opencv` check and step 1.8.
- **O-19, MPS-vs-CPU criterion.** Pre-set criterion failed at the last layers; float64 reference shows fp32 itself
  (CPU too) cannot meet it and MPS is at least as accurate as CPU (F-54). Proposed: keep D-06; criterion "MPS fp32
  error vs float64 ≤ CPU fp32 error" (met); probe-level MPS-vs-CPU comparison at step 2.6.
- **O-20, is the predictor readout informative about motion?** 0.15 passed its pre-set rule (F-57), but the margin over
  the mean-token baseline is ~3% of L1, the all-token metric is likely background-dominated, and the predictor's shrunk
  scale may explain part of the margin. Proposed: disk-patch-restricted forecast comparison before step 5.2 relies on
  the predictor readout.

To confirm by the user:
- **O-18 / F-58:** check in the browser that `checkpoint_key="target_encoder"` (vjepa2 `src/hub/backbones.py`), the
  `torch.hub.load(HUB_REPO, "vjepa2_" + model_name, ...)` call (transformers `convert_vjepa2_to_hf.py`) and
  `F.layer_norm(hi, (hi.size(-1),))` (vjepa2 `app/vjepa/train.py`) read as quoted; then mark F-58 verified and close O-18.

To confirm later:
- **Phase 0 gate (0.18):** re-run every check once from a clean, committed tree, so the whole evidence set has
  clean provenance (D-27).
- **Batch size at extraction (2.4):** `batch` proved bit-exactness for batch size 2 only; re-run it with the
  extraction's real batch size.
- **D-05 criterion must be settled in the planning chat before step 1.8** (current "pixel-identical" wording
  cannot pass, F-49).
- **`.gitignore` for `artifacts/`** — the dry-run proved `artifacts/manifests/` is committed; that other
  files in `artifacts/` are ignored is untested until the first one is written. Check with
  `git status` / `git check-ignore -v` then.

Known decision points the plan cannot remove in advance (each has a planned fallback):
- **O-18** — which encoder weights HF ships: answered from source at 0.15 (target encoder, F-58), user confirmation pending.
- **Phase 1 may overturn scouting facts** (F-21–F-36); D-14's split counts are provisional until step 1.12.
- **0.15 may fail** → step 5.2 falls back to a later-layer readout, and Part 2's behavior manifold (D-17) moves with it.
- **Phase 3 may show no clean transition** → O-15/O-16 become documented judgment calls.
- **Downstream steering effects may wash out** → a finding, not a failure; the same-layer control keeps it interpretable.
- **MPS operator gaps and the compute budget** are unknown until steps 0.6 and 0.16.

---

## Log

Newest entry on top. One entry per work session: what was done, what passed, what didn't,
what's next.

### 2026-09-24 — Step 0.8d started in Claude Code
- Claude Code permissions set: `Bash`, `NotebookEdit` denied; edits denied everywhere except `docs/` and
  `CLAUDE.md`, which always prompt (D-22). `CLAUDE.md` §1 updated to match.
- Working-style rules added: no plan IDs in code or outputs (D-23); checks are scripts saving evidence to
  `results/` (D-24).
- Planned 0.8d checks on test clip `speed/scene_1000` (2.69 m/s, θ 230.625°, ~54 px motion, stays in frame):
  `inspect`, `load`, `repeat`, `colour`, `order`, `opencv`, `figure`. Full-data versions stay in 1.7–1.9.
- `scripts/check_video_loader.py` created (user-written); `inspect` passed → F-46, saved to
  `results/video_loader/checks.json` (see "Saved evidence"). Committed and pushed (`cd2458c`).
- `src/vjepa_physics/video.py` written: `load_clip(path, n_frames=16, size=256)` → (16, 256, 256, 3) uint8 RGB
  via PyAV `to_ndarray(format="rgb24")`; raises `ValueError` on missing/non-increasing pts, wrong frame count
  or shape (no padding/cropping/resizing).
- Checks `load`, `repeat`, `colour` (F-47), `order` (F-48) passed. `opencv` failed: PyAV vs OpenCV differ by
  ≤ 3 levels on every pixel, systematic +1 in R and B (F-49); diagnostics `opencv_bicubic`,
  `opencv_diff_stats` rule out the swscale flag and colour tags. D-05 marked under review; note sent to the
  planning chat. `cv2` now imported lazily (objc duplicate-class warning from two bundled libavdevice builds).
- `figure` check: `results/video_loader/frames.png` reviewed (direction, colour, order, markers agree with
  `order`). No objc warning without `cv2` → lazy import works. **Step 0.8 done.**
- **Next:** 0.9 (preprocessing); re-score `opencv` once the planning chat settles D-05.
- 0.9 started: `src/vjepa_physics/checkpoint.py` (MODEL_ID, full REVISION, WEIGHTS_SHA256). Found that transformers'
  video processor requires torchvision (not installed) → D-25: `torchvision 0.29.0` installed with the lock as
  constraint (dry run: only torchvision; `pip check` clean; torch 2.14.0 unchanged); `requirements.lock.txt`
  regenerated (`pip freeze --exclude-editable`), diff = one added line → 53 packages.
- `src/vjepa_physics/preprocess.py` (`load_processor`, `preprocess_clip`) and `src/vjepa_physics/evidence.py` (shared
  `save_result`) added; `scripts/check_preprocessing.py` checks `config`, `manual`, `identity`, `default` all passed
  (F-50); figure `results/preprocessing/default_vs_ours.png` reviewed (title fixed). F-36's scale/crop verified.
  **Step 0.9 done.**
- `src/vjepa_physics/model.py` (`load_model`: file SHA-256 check, offline pinned load, fp32, sdpa, clean loading
  report required, eval, no grad; `weights_fingerprint`) and `reproducibility.py` (`set_seeds`, `SEED = 0`) added;
  `scripts/check_model.py` checks `config`, `load`, `fingerprint`, `seeds` all passed (F-51, D-26). **Step 0.10 done.**
- Planning-chat review, two changes before 0.11: (1) `evidence.py` provenance rewritten — code-only dirty flag via
  `git -C <repo root>`, `git_dirty_paths`, `code` hashes (D-27); proved by `scripts/check_evidence.py dirty_flag`
  in a scratch repo, confirmed on the real repo (`False []` at `c39e0a8`); `check_video_loader.py` switched to the
  shared `save_result` (duplicate import and whitespace cleaned; `repeat` re-run, same pixel hash). (2) `inspect`
  given explicit criteria and a verdict; re-run: passed (F-46).
- `scripts/check_forward.py forward`: first full forward pass on MPS passed (F-52); predictor vs encoder output
  scale difference recorded as hypothesis H-06 (tested at 0.15). **Step 0.11 done.**
- `src/vjepa_physics/activations.py` (`capture_encoder`: own read-only hooks on the patch embedding and 24 blocks,
  removed on exit; `GRID`, `as_grid`) and `scripts/check_activations.py`: `hidden_states` and `token_layout` passed
  (F-53); F-15 verified on transformers 5.17.0. **Step 0.12 done.**
- `scripts/check_numerics.py`: first `repeat` run was a false pass (MPS → CPU float64 conversion gave all-zero tensors,
  F-55) → fixed with two-step conversion + guards, re-run: bit-exact. `batch` bit-exact (size 2). `devices` failed the
  pre-set criterion at the last layers; `precision` (float64 reference) shows fp32 itself is the limit and MPS is at
  least as accurate as CPU; `attention` → keep sdpa. `load_model` gained diagnostic-only `dtype` / `attn_implementation`.
  Planning-chat note sent (O-19). **Step 0.13 done.**
- `src/vjepa_physics/intervention.py` (`edit_encoder`: prepended write-back hook; `encoder_sites`) and
  `scripts/check_intervention.py`: `noop`, `positive_control`, `hook_order` all passed (F-56). `hook_order` added after
  noticing `positive_control` could not test `prepend` (its capture hook is registered after the edit anyway).
  **Step 0.14 done.**
- `src/vjepa_physics/forecast.py` (`encode`, `predict`, `training_target`) and `scripts/check_forecast.py`:
  `predictor_path` passed (bit-exact vs the combined forward); `forecast` passed on 96 clips, clip draw and pass rule
  (per-dataset CI < 0 vs both baselines) set by the planning chat before the run (F-57). H-06 supported. O-18 answered
  from source (F-58, confirmation pending); O-20 raised (small margin, background-dominated metric). **Step 0.15 done.**
- **Next:** 0.16 (speed and memory benchmark) — waiting for the user's go.

### 2026-09-24 — Steps 0.2–0.7 done, 0.8 started; checkpoint before Claude Code
- 0.2: 9,147 files fingerprinted (SHA-256, sorted, `.DS_Store` excluded), `shasum -c` passes, `data/` read-only; fingerprint committed.
- 0.3: `.venv` from Homebrew Python 3.13.2 (first attempt interrupted during pip setup → rebuilt with `--clear`); `python@3.13` pinned.
- 0.4: all packages installed and checked; OpenCV as `opencv-python-headless`; `pip check` clean.
- 0.5: `requirements.lock.txt` (52 packages) committed; F-38/F-39 re-confirmed on installed versions.
- 0.6: MPS works; patch-embedding-shaped Conv3d agrees with CPU (rel. diff 2.5e-7).
- 0.7: pinned checkpoint downloaded (4 files, no `original/`); weights SHA-256 matches the Hub.
- 0.8a–c: `pyproject.toml`, `src/vjepa_physics/__init__.py`, editable install verified; `*.egg-info/` git-ignored.
- Docs checkpoint: F-41–F-45, D-20, D-21; CLAUDE.md updated with the working style and environment.
- **Next:** 0.8d (video loader), in Claude Code.

### 2026-09-24 — Step 0.1 done (project setup)
- Renamed the repo folder to `world-mechanics-take-home-shivansh-gupta` (before any venv, which stores
  absolute paths).
- Created `src/`, `scripts/`, `artifacts/manifests/`, `results/`, `slides/` with `.gitkeep` placeholders;
  wrote `.gitignore` (`data/`, `paper/`, `artifacts/*` except `manifests/`, venv, caches, `.DS_Store`).
- `git add --dry-run` listed exactly the 12 expected files; first commit `8171aa3`; `git ls-files` confirmed.
  Repo-local git email set.
- Private GitHub repo created; new passphrase-protected SSH key (`~/.ssh/id_ed25519_github_personal`,
  used only for github.com via `~/.ssh/config`); GitHub host key verified against GitHub's published
  fingerprint; authenticated as `shivansh052k`; pushed (13 objects, consistent with the 12 files).
- Found `data/.DS_Store` (Finder, 2026-09-24 00:52) → D-19. Added D-18 for the repo decisions.
- **Next:** Step 0.2 (Data fingerprint and protection).

### 2026-09-24 — Plan review, round 2 (researched fixes)
- Verified the round-1 D-07 fix; found remaining leak and underspecification points plus blockers.
- Researched fixes against primary sources (physics paper App. B/C.11/C.12, Goodfire §5, V-JEPA 2
  training code, sklearn and PyTorch source), then a self-review corrected 8 flaws in the first draft.
- `DECISIONS.md`: added F-37–F-40, D-14–D-17, O-18; narrowed O-01/O-05/O-10/O-11; clarified P-06.
- `EXECUTION_PLAN.md`: reworded 28 steps; still **107 steps**, none added or removed.
- **Next:** Step 0.1 (Project setup).

### 2026-09-24 — Planning complete, then corrected
- Finalized `CLAUDE.md`, `DECISIONS.md`, `EXECUTION_PLAN.md` (**107 steps** across Phases 0–8;
  originally miscounted as 117 — corrected).
- Files placed in the repo: `CLAUDE.md` at root, `DECISIONS.md` / `EXECUTION_PLAN.md` / this file in `docs/`.
- Repo folders still needed: `src/`, `scripts/`, `artifacts/`, `results/`, `slides/`.
- A fresh review caught a real methodology bug (Phase 3/4 wording had layer-selection and
  nullspace stopping using test data, contradicting D-07) and some stale cross-references;
  both fixed. See `DECISIONS.md` change log for details.
- **Next:** Step 0.1 (Project setup).
