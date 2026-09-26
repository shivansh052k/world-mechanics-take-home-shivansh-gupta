# PROGRESS.md — where we are right now

Last updated: 2026-09-25.

This file is the live status tracker. Update it at the end of every phase (and any time work
pauses mid-phase), so this chat, Claude Code, and any future session can pick up instantly
without re-reading everything else.

---

## Current status

**Planning: complete** (including two review rounds). `CLAUDE.md`, `DECISIONS.md`, and
`EXECUTION_PLAN.md` are finalized and in place.

**Execution: in progress.** Steps 0.1–0.8 done (guided in Claude Code since 0.8d, D-21/D-22). Loader
`src/vjepa_physics/video.py` (`load_clip`) verified by checks `inspect`, `load`, `repeat`, `colour`,
`order`, `figure`; `opencv` failed (PyAV vs OpenCV not pixel-identical, F-49) → D-05 now has a tolerance
criterion (test clip re-scored as `opencv_tolerance` at 0.18; full data at step 1.8). **0.9 done:** `src/vjepa_physics/preprocess.py` (`preprocess_clip`,
checkpoint processor with resize/crop off) verified by `config`, `manual`, `identity`, `default` (F-50).
**0.10 done:** `src/vjepa_physics/model.py` (`load_model`, `weights_fingerprint`) and `reproducibility.py`
(`set_seeds`) verified by `config`, `load`, `fingerprint`, `seeds` (F-51, D-26).
**0.11 done:** full forward pass on MPS verified by `scripts/check_forward.py forward` (F-52).
**0.12 done:** `src/vjepa_physics/activations.py` (`capture_encoder`, `GRID`, `as_grid`) verified by
`hidden_states` and `token_layout` (F-53).
**0.13 done:** `scripts/check_numerics.py` — bit-exact repeat and batch; MPS vs CPU recorded; float64 reference
shows MPS fp32 ≥ as accurate as CPU fp32; sdpa kept (F-54, F-55); MPS fp32 accepted with the float64-relative
criterion (D-28). **0.14 done:** `src/vjepa_physics/intervention.py` (`edit_encoder`, `encoder_sites`) verified by
`scripts/check_intervention.py` `noop`, `positive_control`, `hook_order` (F-56). **0.15 done:** `src/vjepa_physics/forecast.py` (`encode`, `predict`,
`training_target`) verified by `scripts/check_forecast.py` `predictor_path` and `forecast` (F-57); HF ships the target
(EMA) encoder (F-58, verified; O-18 closed); primary steering readout = later-layer probe on the full clip, predictor
readout optional (D-30). **0.16 done:** `scripts/check_benchmark.py benchmark` (F-59): ~0.84 s per
clip, batching gives no speed-up (batch size 1 suggested), ~65 min to extract all clips, memory well within limits.
**0.17 skipped** (D-29). **0.18 done — Phase 0 passed:** `opencv_tolerance` passed; clean re-run of all 33 checks
matches the committed results; O-18 closed (F-58); report `results/setup/report.md` lists every open item with the
step it blocks. **Phase 1 started. 1.1 done:** `src/vjepa_physics/data.py` (`DATASETS`, `read_manifest`,
`resolve`) and `scripts/check_data_files.py manifests` passed on all data (F-60, D-31). **1.2 done:**
`read_metadata` + `scripts/check_metadata.py`: `consistency` passed on all 4,572 clips; `documented_fields` failed as
predicted (direction has no `primary_label`; kept on record, note for the planning chat); `sorted_by_label`: all three
manifests sorted by label (F-61, D-32). **1.3 done:** `check_data_files.py fingerprint` passed — all 9,147 hashes
match (Python and `shasum`), no files added/removed, `data/` still read-only (F-62). **1.4 done:** `load_dataset`,
`LABEL_FIELD` in `data.py`; `scripts/check_design.py`: `value_grids` passed (64 values, DATA.md ranges, direction
5.625° apart), `design_balance` diagnostic (angles shared by id across speed/acceleration, direction groups balanced
across angles and octants) (F-63). **1.5 done:** `start_positions` and `label_independence` diagnostics (flag rule
p < 1e-4, D-33): start positions distinct and uniform; 0 of 70 label-dependence tests flagged (F-64). **1.6 done:**
`src/vjepa_physics/geometry.py` (`frame_times`, `distance_travelled`, `speed_at`) and `distance_confound`: labels
exactly proportional to distance within speed/acceleration sets; overlap window 0.156–1.953 m (1,176 / 1,440 clips);
direction motion type vs distance r = −0.565 (F-65). **1.7 run:** `probe_clip` in `video.py`; pixel mapping
(`world_to_pixel`, `disk_centres`, `distance_outside_image`) in `geometry.py`; `scripts/check_videos.py`: `format`
failed only on `no_uniform_frames` (185 uniform exit frames in 52 direction clips; kept on record), `uniform_frames`
diagnostic shows they are clean exit frames but its pre-stated rule failed narrowly on 3 frames (kept on record),
`duplicates` passed (no duplicate clips; `artifacts/` ignore rule verified) (F-66, F-67, D-34). **1.8 done:** `check_videos.py decoders` — PyAV vs OpenCV verdict ok on
all 4,572 clips under D-05 (F-68). **1.9 done:** `src/vjepa_physics/tracking.py` (`disk_mask`, `count_objects`,
`track_disk`) and `scripts/check_tracking.py`: `track` passed (one disk per frame; positions in
`artifacts/tracking/tracked_disk.npz`), `mapping` passed (≤ 0.783 px on all fully visible frames; scale 32 px/m and
origin 128 fitted from data; frame 0 = start everywhere), `documented_colour` failed as predicted (orange, not blue)
(F-69–F-71, D-35). **1.10 done:** `src/vjepa_physics/flags.py` (`clip_flags`) and `check_tracking.py flags`: all six
integrity criteria passed; flags table `results/tracking/clip_flags.csv` committed (F-72, D-36). **1.11 done:**
`src/vjepa_physics/plotting.py`; figures `results/videos/contact_sheet.png`, `results/design/design.png`,
`results/tracking/tracking.png`, each reviewed and fixed (F-73). **1.12 done:** every scouting fact F-21–F-36 has a
final status (summary under DECISIONS §1.3): 10 verified as stated, 4 verified with precision-level corrections, 2 partly
verified, no material contradiction; D-14's counts confirmed. **1.13 done — Phase 1 passed (2026-09-25):** clean re-run
of all 21 audit keys reproduced every committed result (`rerun_identical`, F-74); report `results/data_audit/report.md`.
**Phase 2 started (2026-09-25).** Step 2.1 decision brief (O-01–O-03, plus D-37 scope and the 2.3 items O-04, O-06,
sites, batch size, dtype, gate re-run cost) sent to the planning chat. Meanwhile `motion_group` and `angle_octant` added
to `src/vjepa_physics/data.py`; smoke test reproduced the saved `design_balance` group counts and octant × group table
exactly (F-75). **2.1 and 2.3 settled by the planning chat** (D-38 splits and 2.2 criteria, D-39 keep and flag, D-40
extraction settings, D-41 layout and gate; D-37 extended to Phase 2). Hash-guard helper `verified_artifact` added to
`evidence.py` (F-76). **2.2a done:** `src/vjepa_physics/splits.py` gives D-38's counts exactly (direction 813 / 203 / 94 /
203 / 187; speed and acceleration 832 / 208 / 96 / 208 / 192), shared speed/acceleration assignment, deterministic,
byte-identical writes (F-77). **2.2b done:** `scripts/check_splits.py build` passed; splits saved to
`artifacts/manifests/splits.csv` (SHA-256 `bb64b6ae…`, F-78). **2.2c done:** `balance` diagnostics (F-79); two notes for
the planning chat (one seen direction angle without a test_seen clip; direction exit clips val_seen 8 vs test_seen 20).
**2.4a done:** `src/vjepa_physics/extraction.py` (26 sites, per-time-step pooling; test clip within 5.3e-8 of a float64
pool, repeat bit-exact, F-80). **2.4b done:** `scripts/check_extraction.py` committed, `pipeline` passed (F-81).
**2.5 done:** all 4,572 clips extracted, every criterion passed, test clip reproduced bit for bit, ~106 min (F-82).
**2.6 done:** `verify` passed — hash-guarded, ids = splits, 48 seeded clips re-extracted live bit-identical (F-83).
**2.7a done:** `src/vjepa_physics/joined.py` (`build_table`, `load_joined`); in-memory tables reproduce F-72's flag
counts and F-70's pixel errors (F-84). **2.7b done:** `scripts/check_joined.py build` passed; tables in
`artifacts/joined/` (F-85). **2.8 done:** `storage` passed — 10/10 artifacts match, `load_joined` round trip, 3.628 GiB
on disk, peak RSS 2.43 GB with a full array in RAM (F-86). **2.9 gate:** clean re-run identical (F-87); report
`results/splits_and_extraction/report.md` written by the user from a Claude Code draft. **Phase 2 passed (2026-09-25).**
**Phase 3 started (2026-09-25).** Step 3.1 brief (O-05, feature entry F1 all-token mean / F2 flattened / F3 per-step /
F4 PCA; recommendation closed-form ridge + all-token mean for selection) prepared for the planning chat with the pending
Phase 2 notes and the later Phase 3 choices (3.3 permutations, 3.4 floor/ceiling details, 3.5 bootstrap, 3.7 transition
rule, bin edges, 3.8, gate determinism). `evidence.PACKAGES` now records scikit-learn (F-88, commit `0e2f572`).
**3.2a done:** `src/vjepa_physics/metrics.py` (R², MAE, circular error) matches sklearn and hand-worked cases (F-89).
**3.4a done:** `baselines.physics_fit` + `geometry.pixel_to_world`; exact on metadata positions, tracked ceiling
speed 0.033 (quadratic) / 0.009 (linear) m/s, acceleration 0.115 m/s², direction 0.30° (F-90).
**3.1 settled by the planning chat:** D-42 (RidgeCV on the all-token mean, alpha rule; O-05 closed), D-43 (Phase 3
settings; 3.8 skipped), D-44 (Phase 2 items closed; D-37 extended to Phase 3). Three own-addition follow-ups sent back
(time-averaged pixel floor; scope of 3.3's circular-MAE criterion; transition-index interval).
**3.2b done:** `src/vjepa_physics/probes.py`; leave-one-out = brute force to 1e-12; block_11 already R² ≈ 0.985–0.989
on val-seen for all three variables (F-91).
**3.2c done:** `scripts/check_probes.py layer_curves` passed (F-92): val-seen R² jumps from the embedding to block_0
(speed 0.00 → 0.98, acceleration 0.00 → 0.98, direction 0.12 → 0.85), maxima ≈ 0.99 around index 18–19; transition at
index 1 for all three by D-43's rule (hypothesis H-07). Test scores (end of 3.2) wait until 3.7 is frozen.
**Planning chat:** the idx-1 result matches the paper for mean-pooled probes (F-93); D-45 (O-15 narrowed, no saturation
rule, time-averaged floor, 3.3 criterion, transition bootstrap); D-46 (3.8 repurposed as a direction local-to-global
test; design brief first).
**3.3 done:** `shuffled_labels` passed — max shuffled R² ≤ 0.087, direction mean circular MAE 87–92° at every site
(F-94). Low per-fit direction minima: H-08's pre-set symmetry rule failed at 5 of 21 sites (not supported as stated;
post-hoc refinement H-08b untested; does not affect the pass).
**3.4b done:** exact pixel Gram and Gram-based ridge in `baselines.py`, identical to sklearn RidgeCV (F-95).
**3.4c done:** `check_baselines.py pixel_grams` passed; six exact Grams saved, decoded pixels = audit hashes (F-96).
**3.4d done:** `pixel_floor` passed; full-RGB floor R² 0.48 / 0.63 / 0.49 (direction / speed / acceleration),
time-averaged 0.20 / 0.23 / 0.16, all far below block_0 (F-97; supports H-07; H-09 new).
**3.4e done:** `physics_ceiling` passed; ceiling R² ≥ 0.997 (headline quadratic), design-informed variants tighter
(F-98). **Step 3.4 complete.**
**3.5a done:** bootstrap helpers in `metrics.py`; a constant-target guard bug (Claude Code's) found and fixed (F-99).
**3.5b-1 done:** `src/vjepa_physics/curves.py` (D-43 rules; saved curves → transition index 1 for all three, F-100).
**3.5b-2 done:** `check_layer_curves.py bootstrap` passed (F-101): transition index 1 in every resample, paired
adjacent changes, direction with/without exit nearly identical.
**3.6 done:** `results/layer_curves/layer_curves.png` reviewed and fixed (F-102).
**3.7 done:** transition = index 1 (depth 1/24) for all three, robust; layer choices frozen (F-103).
**Test scores done (once, committed code):** all criteria passed; test agrees with validation (F-104). 3.2 ticked.
**3.8 design settled (D-47, per-patch).** **3.8a done:** `pool_patches`, `patch_activations` (F-105).
**Lean mode adopted (D-48):** hash-check gates, commit before each saved run, no full re-runs, docs once per phase
(one log line per step), Phase 4–8 designs and cuts fixed; Goodfire quotations F-106.
**Phase 3 passed (F-115).** **Phase 4 passed (2026-09-26, F-128):** D-49 with addenda (a)–(f); F-116–F-128; report
`results/nullspace/report.md`. Headline: K at idx 9 = 6 / 7 / 7, procedure-dependent (7 to > 150 across ridge
penalties); m-dim linear erasure; nonlinear recovery after erasure. **Next:** Phase 5 design brief (multi-probe
steering at idx 9; readout idx 18; covariance-direction arm; H-12).

---

## Phase status

| Phase | Status | Notes |
|---|---|---|
| 0 — Environment and model setup | ✅ Passed gate | 0.1–0.16, 0.18 done; 0.17 skipped (D-29); report `results/setup/report.md` (open items listed there) |
| 1 — Data audit | ✅ Passed gate | 1.1–1.13 done; report `results/data_audit/report.md`; clean re-run identical (`documented_fields`, `documented_colour` failed as predicted; `format` / `uniform_frames` failed, closed without re-score, D-34) |
| 2 — Splits and activation extraction | ✅ Passed gate | 2.1–2.9 done (D-38–D-41, F-75–F-87); report `results/splits_and_extraction/report.md`; clean re-run identical; balance notes closed (D-44) |
| 3 — Part 1a: Layer-wise probing | ✅ Passed gate (2026-09-25) | 3.1–3.9 done (F-88–F-115; D-42–D-48); transition idx 1 for all three (mean-pooled) and for per-patch direction; test once (F-104, F-113); hash-check gate (F-115); report `results/layer_probing/report.md`, slide log `results/slide_log.md`, talk outline `slides/talk_outline.md` |
| 4 — Part 1b: Iterative nullspace probing | ✅ Passed gate (2026-09-26) | 4.1–4.8 done (D-49 + addenda; F-116–F-128); K idx 9 = 6 / 7 / 7, procedure-dependent (F-120); linear erasure with m dims, nonlinear recovery (F-119–F-123); test once (F-127); hash-check gate, 2 keys re-run identical (F-128); five failures kept on record; report `results/nullspace/report.md`, figures `nullspace_rounds.png`, `erasure_and_procedure.png` |
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
| Extraction stand-in per clip (all 25 sites, mean-pooled), batch 1 / 8 | 0.84 / 0.92 s; batch size 1 suggested | F-59 |
| Estimated extraction time, all 4,572 clips | ~65 min | F-59 |
| MPS memory: weights / pool at batch 1 / Metal recommended max | 1.215 / 2.06 / 11.84 GiB | F-59 |
| Pooled storage fp32 (mean over tokens / per time step) | 0.44 / 3.5 GiB | F-59 |

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
| `results/video_loader/checks.json` | `opencv` | PyAV vs OpenCV (FFmpeg backend) pixel-identical | ❌ failed: all pixels differ by ≤ 3, systematic +1 in R and B; kept as failed; re-scored under `opencv_tolerance` (D-05) | F-49; `scripts/check_video_loader.py opencv` |
| `results/video_loader/checks.json` | `opencv_bicubic` | Diagnostic: swscale flag is not the cause (PyAV BILINEAR = BICUBIC); frame colour tags unspecified | ℹ️ diagnostic | F-49; `scripts/check_video_loader.py opencv_bicubic` |
| `results/video_loader/checks.json` | `opencv_diff_stats` | Diagnostic: signed PyAV − OpenCV histogram and means by region (+1.01 / +0.01 / +1.00 overall) | ℹ️ diagnostic | F-49; `scripts/check_video_loader.py opencv_diff_stats` |
| `results/video_loader/checks.json` | `opencv_tolerance` | PyAV vs OpenCV under D-05's per-clip tolerance (`vjepa_physics.decoders.compare_decoders`): verdict ok, no reasons; max \|diff\| 3/2/2, mean (+1.010, +0.012, +1.002), disk mean (+1.69, +0.62, +1.39) over 5,593 disk pixels | ✅ passed | D-05, F-49; `scripts/check_video_loader.py opencv_tolerance` |
| `results/video_loader/frames.png` + `checks.json` | `figure` | Visual evidence: frames 0/5/10/15 with predicted (+) and measured (×) disk centre and predicted path; disk orange, moving down-left (θ 230.6°); errors at shown frames 0.319 / 0.42 / 0.11 / 0.407 px (= `order`) | ℹ️ visual (reviewed) | `scripts/check_video_loader.py figure` |
| `results/preprocessing/checks.json` | `config` | Our processor: resize and crop off; rescale 1/255 and ImageNet normalization as shipped. Default: shortest edge 292 + crop 256 | ✅ passed | F-50; `scripts/check_preprocessing.py config` |
| `results/preprocessing/checks.json` | `manual` | Output (16, 3, 256, 256) float32 CPU = manual (x/255 − mean)/std, max abs diff 1.65e-7 | ✅ passed | F-50; `scripts/check_preprocessing.py manual` |
| `results/preprocessing/checks.json` | `identity` | Un-normalize + round = decoded uint8 clip bit-for-bit (max 9.4e-6 levels) → no spatial change | ✅ passed | F-50; `scripts/check_preprocessing.py identity` |
| `results/preprocessing/checks.json` + `default_vs_ours.png` | `default` | Shipped default scales ×1.1406 (slope 1.1393/1.1405) and crops 18 px/side (offset −17.93 px); disk 350 → 455 px; 36.5 px/m; figure reviewed | ✅ passed | F-50; `scripts/check_preprocessing.py default` |
| `results/model/checks.json` | `config` | Config and built structure = F-10: 24 encoder / 12 predictor blocks, patch Conv3d (1024, 3, 2, 16, 16), dropout 0 | ✅ passed | F-51; `scripts/check_model.py config` |
| `results/model/checks.json` | `load` | Weights SHA-256 = pinned; loading report empty (587 tensors); 303.9M + 22.1M params; fp32, eval, no grad, sdpa; fingerprint recorded | ✅ passed | F-51; `scripts/check_model.py load` |
| `results/model/checks.json` | `fingerprint` | Two independent loads → identical in-memory weights fingerprint `c865f524…04ede` | ✅ passed | F-51; `scripts/check_model.py fingerprint` |
| `results/model/checks.json` | `seeds` | `set_seeds(0)` reproduces Python / NumPy / torch CPU / torch MPS draws; seed 1 changes all | ✅ passed | F-51; `scripts/check_model.py seeds` |
| `results/evidence/checks.json` | `dirty_flag` | In a scratch git repo: `git_dirty` False when committed and for `results/`- or `docs/`-only changes; True for any edit or new file under `src/`, `scripts/`, `pyproject.toml`, `requirements.lock.txt`; same answer from repo root, subfolder and outside. Real repo root found from anywhere. Real-repo confirmation after commit `c39e0a8`: `False []` (seen in the terminal, not saved) | ✅ passed | D-27; `scripts/check_evidence.py dirty_flag` |
| `results/forward/checks.json` | `forward` | One clip through encoder + predictor on MPS: all outputs (1, 2048, 1024), finite; predictor target = encoder output; weights fingerprint unchanged by the forward pass | ✅ passed | F-52; `scripts/check_forward.py forward` |
| `results/activations/checks.json` | `hidden_states` | Own hooks vs `hidden_states` in one pass: entry 0 = embedding, entry i = block i−1, entry 24 = block 23 pre-LayerNorm; `last_hidden_state` = LN(block 23) exactly; our hooks removed (48 transformers hooks before and after). Proves index alignment and semantics, not run-to-run determinism (0.13) | ✅ passed | F-53; `scripts/check_activations.py hidden_states` |
| `results/activations/checks.json` | `token_layout` | Token index = t·256 + row·16 + col shown from the data: disk patch is the most deviant embedding token in 8/8 time steps; swapped rows/cols 3/8, reversed time 4/8 rejected. Limit: one clip whose path crosses near the row = col diagonal | ✅ passed | F-53; `scripts/check_activations.py token_layout` |
| `results/numerics/checks.json` | `repeat` | Same input twice: bit-exact at all 27 outputs on MPS and on CPU (re-run after fixing the MPS float64 conversion, F-55) | ✅ passed | F-54; `scripts/check_numerics.py repeat` |
| `results/numerics/checks.json` | `batch` | Batch of 2 = each clip alone, bit-exact on MPS (batch size 2 only) | ✅ passed | F-54; `scripts/check_numerics.py batch` |
| `results/numerics/checks.json` | `devices` | MPS vs CPU per layer: rel. error 6e-7 → 1.2e-3 with depth, token cosine ≥ 0.9999; default fp32 tolerances fail everywhere (hypothesis confirmed) | ❌ failed own criterion at blocks 19–23 + final norm (≤ 1.23e-3 vs 1e-3); explained by `precision` (fp32 limit, not MPS); kept as failed; replacement criterion D-28 (met) | F-54; `scripts/check_numerics.py devices` |
| `results/numerics/checks.json` | `precision` | Diagnostic vs CPU float64: MPS fp32 error ≤ CPU fp32 error at all 27 outputs (ratio 0.58–0.99) | ℹ️ diagnostic | F-54; `scripts/check_numerics.py precision` |
| `results/numerics/checks.json` | `attention` | Diagnostic: eager ÷ sdpa error vs float64 = 0.93–1.04 → keep sdpa (pre-stated rule) | ℹ️ diagnostic | F-54, D-26; `scripts/check_numerics.py attention` |
| `results/intervention/checks.json` | `noop` | Writing back an unchanged activation (`clone`, `+ 0`) at each of 25 sites leaves all 27 outputs bit-identical on MPS; hook fired once per run; hooks removed (0/0); fingerprint unchanged | ✅ passed | F-56; `scripts/check_intervention.py noop` |
| `results/intervention/checks.json` | `positive_control` | Adding a seeded random δ at each site: outputs before it bit-identical, site = baseline + δ exactly, every output after it changed (incl. final norm, predictor); change sizes recorded as observations | ✅ passed | F-56; `scripts/check_intervention.py positive_control` |
| `results/intervention/checks.json` | `hook_order` | With transformers' `hidden_states` hooks and our capture registered before the edit, both see baseline + δ at every site (prepend works); negative control (appended edit) leaves capture unedited; hooks 48/48 | ✅ passed | F-56; `scripts/check_intervention.py hook_order` |
| `results/forecast/checks.json` | `predictor_path` | Separate `encode` → `predict` = combined forward bit-for-bit; reversed forecast targets return reversed rows; forecast shapes (1, 1024, 1024); context-vs-full-clip difference 0.72 recorded as an observation | ✅ passed | F-57; `scripts/check_forecast.py predictor_path` |
| `results/forecast/checks.json` | `forecast` | 96 clips (32 per dataset, ids saved): predictor beats copy-last-step and mean-context-token baselines in every dataset (all 95% CIs < 0); margin over the mean token ~3% of L1 (→ D-30); H-06 fields, per-step L1s, timings recorded | ✅ passed | F-57; `scripts/check_forecast.py forecast` |
| `results/benchmark/checks.json` | `benchmark` | Time and memory per clip on MPS at batch sizes 1/2/4/8 (no batching gain; batch 1 suggested by the pre-set rule); batched pooled outputs = single bit-for-bit at 2/4/8; extraction ~65 min; storage estimates | ℹ️ diagnostic | F-59; `scripts/check_benchmark.py benchmark` |
| `results/setup/report.md` | — | Setup report: what was verified (with check keys), failures kept on record, skipped parity, clean re-run and diff, open items with the step each blocks | ✅ gate passed | 0.18; user-written from saved evidence |
| `results/data_files/checks.json` | `manifests` | All 3 manifests: lines parse, rows 1,500 / 1,536 / 1,536 = DATA.md, ids 0 … N − 1, paths inside `data/<dataset>/`, distinct, non-empty regular files, video + metadata same folder, no orphan files/folders; 9,147 files = fingerprint count, file set = manifests ∪ referenced. Diagnostics: `scene_{id:04d}` naming everywhere, line order = id order. Saved with `git_dirty` true (new code uncommitted); clean re-run identical at the Phase 1 gate (F-74) | ✅ passed | F-60, D-31; `scripts/check_data_files.py manifests` |
| `results/metadata/checks.json` | `consistency` | All 4,572 metadata files: strict parse, id = manifest id, fps 24 / frames 16, finite typed fields, speed/acceleration ≥ 0, direction θ in [0, 360), `primary_label`/`magnitude` per DATA.md in speed/acceleration, motion consistent with values and dataset. Diagnostics: acceleration set all from rest; direction 750 + 750; start ranges ±1.2 / ±2 m | ✅ passed | F-61, D-32; `scripts/check_metadata.py consistency` |
| `results/metadata/checks.json` | `documented_fields` | Every file has DATA.md's documented fields | ❌ failed as predicted: all 1,500 direction files lack `primary_label` (speed/acceleration complete); kept on record; labels come from the dataset name | F-61, D-32, F-25; `scripts/check_metadata.py documented_fields` |
| `results/metadata/checks.json` | `sorted_by_label` | Diagnostic: label never decreases with id in all three manifests; 64 runs (64 × 24; direction 36 × 23 + 28 × 24); Spearman 0.999878 | ℹ️ diagnostic | F-61, F-22; `scripts/check_metadata.py sorted_by_label` |
| `results/data_files/checks.json` | `fingerprint` | Reference tracked and unchanged; 9,147 lines well-formed; file set unchanged; all hashes match (Python + `shasum -c`); nothing under `data/` writable. Diagnostic: only byte-identical files = speed/acceleration manifests | ✅ passed | F-62; `scripts/check_data_files.py fingerprint` |
| `results/design/checks.json` | `value_grids` | 64 distinct values per dataset; speed 0.25–4.0, acceleration 0.25–10.0; direction gaps all 5.625° around the circle; 4,572 clips. Diagnostics: linear grids (full grids saved for the split), 24 clips per value, direction 24 clips at the first 28 angles and 23 at the rest | ✅ passed | F-63; `scripts/check_design.py value_grids` |
| `results/design/checks.json` | `design_balance` | Diagnostic: speed/acceleration unique (value, θ) pairs, same θ by id in both sets, corr with cos/sin θ ≈ −0.001; direction 12 motion groups (108/107 velocity, 150 acceleration), angle × group cells 1–3, Cramér's V 0.073, octant × group table (13–20 per cell) saved | ℹ️ diagnostic | F-63; `scripts/check_design.py design_balance` |
| `results/design/checks.json` | `start_positions` | Diagnostic: start positions distinct in every dataset, within ±2 m (direction) / ±1.2 m (speed, acceleration), uniform by KS (p ≥ 0.26), x–y uncorrelated, quadrants even; no flags | ℹ️ diagnostic | F-64, D-33; `scripts/check_design.py start_positions` |
| `results/design/checks.json` | `label_independence` | Diagnostic: 0 of 70 tests flagged (p < 1e-4) — start position vs magnitude and cos/sin θ, along/across-motion projections (no starts placed behind the motion), ANOVA across label values | ℹ️ diagnostic | F-64, D-33; `scripts/check_design.py label_independence` |
| `results/design/checks.json` | `distance_confound` | Diagnostic (from metadata): distance = 0.625 × speed and 0.1953 × acceleration exactly, corr with distance / mean / final speed 1.0; overlap window [0.15625, 1.953125] m, 1,176 speed + 1,440 acceleration clips, nearest-value distance gap ≤ 0.018 m (0.58 px); direction distance per group, motion type vs distance r = −0.565 | ℹ️ diagnostic | F-65; `scripts/check_design.py distance_confound` |
| `results/videos/checks.json` | `format` | All clips decode; 256×256, 24 fps, constant 1/24 s step (from t = 0); no black frames; frame medians all (29, 32, 29); 113 direction clips with disk-less frames | ❌ failed: `no_uniform_frames` — 185 uniform frames in 52 direction clips (disk has left); kept on record; re-score pending the planning chat | F-66, D-34; `scripts/check_videos.py format` |
| `results/videos/checks.json` | `uniform_frames` | Diagnostic of the `format` failure: all 185 uniform frames = background, no disk pixels, at clip end; no disk pixels anywhere with the disk predicted fully outside; 149 exit frames keep faint residue | ⚠️ `explanation_holds: false` — pre-stated rule (c) missed 3 frames by 0.39–0.55 px (within mapping error + pixel-centre offset); kept on record | F-66; `scripts/check_videos.py uniform_frames` |
| `results/videos/checks.json` | `duplicates` | 4,572 distinct whole-clip hashes across datasets; test clip hash = Phase 0 `repeat`; per-clip hashes in `artifacts/videos/decoded_hashes.csv` (git-ignored, verified) | ✅ passed | F-67, D-34; `scripts/check_videos.py duplicates` |
| `results/videos/checks.json` | `decoders` | PyAV vs OpenCV under D-05 on all 4,572 clips: all ok (0 flag, 0 fail); max \|diff\| R 3 / G ≤ 3 / B ≤ 3; means (+1.00–1.02, +0.01–0.02, +1.00); per-clip stats in `artifacts/videos/decoder_comparison.csv` | ✅ passed | F-68, D-05; `scripts/check_videos.py decoders` |
| `results/tracking/checks.json` | `track` | Disk tracked in all 73,152 frames; one piece in every frame with disk pixels; direction 347 disk-less / 466 border frames (199 clips), ≥ 6 fully visible frames per clip; area 346–353 px; positions + core colour in `artifacts/tracking/tracked_disk.npz` (git-ignored, verified) | ✅ passed | F-69, D-35; `scripts/check_tracking.py track` |
| `results/tracking/checks.json` | `mapping` | Tracked centre within 1 px of metadata in all 72,339 fully visible frames (max 0.783); frame 0 = start in every clip; fitted ±32.00 px/m, origin 128.00; alternatives rejected per dataset; angle median error 0.3–0.5° | ✅ passed | F-70, D-35; `scripts/check_tracking.py mapping` |
| `results/tracking/checks.json` | `documented_colour` | DATA.md's "blue disk" | ❌ failed as predicted: red brightest and R > G > B in all 4,572 clips, core (234.2, 114.5, 39.2); kept on record | F-71, D-35; `scripts/check_tracking.py documented_colour` |
| `results/tracking/checks.json` + `clip_flags.csv` | `flags` | Flags for all 4,572 clips; exit (113) and disk-less frames (347) = `format`; every exit clip clipped; tracked vs predicted displacement ≤ 1.27 px on 4,373 clean clips. Direction exit 113 / clipped 199 / sub-patch 150 / frozen 92; speed sub-patch 240 / frozen 1; acceleration sub-patch 360 / frozen 267 | ✅ passed | F-72, D-36; `scripts/check_tracking.py flags` |
| `results/videos/contact_sheet.png` + `checks.json` | `figure_contact_sheet` | 6 example clips (seeded picks from the flags table; exit = most residue, id 1152) with predicted/tracked centres and a residue column | ℹ️ visual (reviewed) | F-73; `scripts/check_videos.py figure_contact_sheet` |
| `results/design/design.png` + `checks.json` | `figure_design` | Distance vs label (speed, acceleration) with overlap window; direction octant × group heatmap | ℹ️ visual (reviewed) | F-73; `scripts/check_design.py figure_design` |
| `results/tracking/tracking.png` + `checks.json` | `figure_tracking` | Tracked-vs-predicted distance histograms (max ≤ 0.783 px); flag counts per dataset | ℹ️ visual (reviewed) | F-73; `scripts/check_tracking.py figure_tracking` |
| `results/evidence/checks.json` | `rerun_identical` | All 21 audit keys re-run from clean HEAD `7869969`; every result identical to the committed one; flags table and PNGs unchanged | ✅ passed | F-74; `scripts/check_evidence.py rerun_identical` |
| `results/splits/checks.json` + `artifacts/manifests/splits.csv` | `build` | Splits from metadata + seed 0 (D-38): every clip once, counts = D-38, unseen values as decided and absent from seen roles, 16/4/4 per seen value, speed = acceleration roles by id, rebuild byte-identical; file SHA-256 `bb64b6ae…` | ✅ passed (saved with `git_dirty` true; clean re-run identical, F-87) | F-78; `scripts/check_splits.py build` |
| `results/extraction/checks.json` | `pipeline` | Test clip through the extraction code: (1, 26, 8, 1024) fp32, repeat bit-exact, 5.3e-8 vs float64 pool, weights unchanged; pooled SHA-256 `29997d28…` for the full run to reproduce | ✅ passed | F-81; `scripts/check_extraction.py pipeline` |
| `results/extraction/checks.json` + `artifacts/activations/*.npy` | `extract_direction`, `extract_speed`, `extract_acceleration` | Pooled activations (clips, 26, 8, 1024) fp32 for all 4,572 clips; read-back = computed, finite, non-zero, distinct, weights unchanged, speed test clip = `pipeline`; SHA-256 direction `0d36d025…`, speed `0c57023e…`, acceleration `067aad46…` (git-ignored, verified) | ✅ passed | F-82; `scripts/check_extraction.py extract_<dataset>` |
| `results/extraction/checks.json` | `verify` | Arrays, ids, split file hash-guarded; shapes; ids = split ids in order; 16 seeded clips per dataset re-extracted live bit-identical (also the gate test, D-41); site scale and nearest-pair diagnostics | ✅ passed | F-83; `scripts/check_extraction.py verify` |
| `results/joined/checks.json` + `artifacts/joined/*.npz` | `build` | Joined tables per variable from hash-guarded sources: all clips, one entry per clip, role counts = D-38, flag counts = F-72, tracking ≤ 1 px by id, rebuild byte-identical; SHA-256 direction `3a3790ac…`, speed `1f81b80e…`, acceleration `9e191b64…` (git-ignored) | ✅ passed | F-85; `scripts/check_joined.py build` |
| `results/joined/checks.json` | `storage` | 10/10 artifacts match their hashes; `load_joined` round trip (memory-mapped, aligned); activation files exact size; 3.628 GiB on disk; ≥ 10 GB free; full largest array in RAM → peak RSS 2.43 GB | ✅ passed | F-86; `scripts/check_joined.py storage` |
| `results/evidence/checks.json` | `rerun_identical_splits_extraction` | 6 split/extraction/joined keys re-run from clean HEAD `910a798`, results identical (only `storage` machine-state fields excluded); `extract_*` untouched (D-41); `verify` re-run; split file unchanged | ✅ passed | F-87; `scripts/check_evidence.py rerun_identical_splits_extraction` |
| `results/splits_and_extraction/report.md` | — | Splits and extraction report: splits, how representations are extracted and pooled, joined tables, storage, clean re-run, notes, open items | ✅ gate passed | F-87; user-written from saved evidence |
| `results/splits/checks.json` | `balance` | Flags, mean cos/sin θ per role; direction group × octant per role; clips per seen angle (test_seen: one angle with 0) | ℹ️ diagnostic | F-79; `scripts/check_splits.py balance` |
| `results/probes/checks.json` + `artifacts/probes/layer_predictions.npz` | `layer_curves` | Ridge probe per site (26) and variable on train, scored on val_seen / val_unseen; n_fit 813 / 832; alpha rule; only validation rows predicted; refit identical; predictions saved (SHA-256 `b7261ed0…`, git-ignored) | ✅ passed (saved with `git_dirty` true; hash-checked, F-115) | F-92; `scripts/check_probes.py layer_curves` |
| `results/probes/checks.json` | `shuffled_labels` | 20 train-label permutations × 26 sites × 3 variables, scored on val_seen with true labels: max R² ≤ 0.087 < 0.1; direction mean circular MAE 87–92° > 80; no alpha failure; n_fit exact | ✅ passed (saved with `git_dirty` true; hash-checked, F-115) | F-94; `scripts/check_probes.py shuffled_labels` |
| `results/baselines/checks.json` + `artifacts/baselines/pixel_gram_*.npy` | `pixel_grams` | Exact pixel Gram matrices per variable (full RGB, time-averaged); decoded pixels = audit hashes for all 4,572 clips; symmetric, integer < 2^53, diagonal = int64 sum of squares; 6 file hashes recorded (git-ignored) | ✅ passed (saved with `git_dirty` true; hash-checked, F-115) | F-96; `scripts/check_baselines.py pixel_grams` |
| `results/baselines/checks.json` + `artifacts/baselines/floor_predictions.npz` | `pixel_floor` | Ridge on raw pixels (full RGB headline, time-averaged secondary) via the saved Grams; train-only, alpha rule, validation rows only; val-seen R² full 0.48 / 0.63 / 0.49, time-averaged 0.20 / 0.23 / 0.16 | ✅ passed (saved with `git_dirty` true; hash-checked, F-115) | F-97; `scripts/check_baselines.py pixel_floor` |
| `results/baselines/checks.json` + `artifacts/baselines/ceiling_estimates.npz` | `physics_ceiling` | Per-clip physics fit on tracked positions, validation rows only; exact on metadata positions (≤ 1.7e-13); headline quadratic R² ≥ 0.997; design-informed constant-velocity (speed) and from-rest (acceleration) reported labelled | ✅ passed (saved with `git_dirty` true; hash-checked, F-115) | F-98; `scripts/check_baselines.py physics_ceiling` |
| `results/layer_curves/checks.json` | `bootstrap` | 10,000-resample clip bootstrap of probes, floors, ceiling on val_seen / val_unseen (direction also without exit); 178 point estimates = saved scores; transition index 1 in every resample; paired adjacent changes; per-value errors on val_unseen | ✅ passed (saved with `git_dirty` true; hash-checked, F-115) | F-101; `scripts/check_layer_curves.py bootstrap` |
| `results/layer_curves/layer_curves.png` + `checks.json` | `figure_layer_curves` | R² and MAE / circular MAE per layer index with 95% bands, val-unseen, pixel floors, physics-fit ceiling, transition line | ℹ️ visual (reviewed) | F-102; `scripts/check_layer_curves.py figure_layer_curves` |
| `results/layer_curves/checks.json` + `artifacts/probes/test_predictions.npz` | `test_scores` | One-time test scores (test_seen / test_unseen, direction also without exit) for all probe sites, floors, ceiling, with 95% CIs and per-value errors; 84 refits reproduce saved alphas and validation predictions; code committed | ✅ passed (from committed code) | F-104; `scripts/check_layer_curves.py test_scores` |
| `results/patches/checks.json` + `artifacts/patches/direction.npy` | `extract_direction`, `verify` | Per-patch time-averaged direction activations (1500, 25, 256, 1024), 39.3 GB, patch mean = all-token mean ≤ 6.5e-8; 16 clips re-extracted bit-identical | ✅ passed | F-107 |
| `results/patches/checks.json` + `patch_probes.npz` | `patch_probes` | 6,400 per-patch probes; mean per-patch R² curve; fitted probes saved | ❌ failed `no_alpha_failure` (20 lower-edge alphas at idx 0), kept on record; no transition result depends on it | F-108 |
| `results/patches/checks.json` | `patch_alpha_diagnostic` | Tie explanation for the 20 failures (rule fixed first) | ⚠️ diagnostic, `explanation_holds: false` (H-10 not supported) | F-108 |
| `results/patches/checks.json` + `patch_breakdown.npz` | `patch_breakdown` | On/off-path breakdown; off-path idx 0 ≈ 0 (negative control), 0.648 at idx 1 | ✅ passed | F-109 |
| `results/patches/checks.json` | `patch_bootstrap` | Headline CIs, transition distribution, gap to mean-pooled | ✅ passed | F-110 |
| `results/patches/checks.json` + `spatial_generalization.npz` | `spatial_generalization` | Shared probe per half, clip-grouped CV; same vs across, gap | ✅ passed | F-111 |
| `results/patches/checks.json` | `position_baseline` | Position-only baseline, linear 0.197 / cubic 0.211 < 0.5 | ✅ passed | F-112 |
| `results/patches/checks.json` + `patch_test_scores.npz` | `patch_test_scores` | One-time test of per-patch mean, off-path, across-half (saved probes, exact reproduction) | ✅ passed (committed code) | F-113 |
| `results/patches/local_to_global.png` + `checks.json` | `figure_local_to_global` | Heatmaps + local-to-global curves | ℹ️ visual (reviewed) | F-114 |
| `results/evidence/checks.json` | `code_hash_check`, `rerun_identical_probing` | Phase 3 gate: 18 keys matched to commits holding their exact code; 5 re-runs identical | ✅ passed | F-115 |
| `results/layer_probing/report.md` | — | Layer-wise probing report: setup, mean-pooled curves and test, controls and references, per-patch local-to-global results, comparison with the paper, items kept on record, open items | ✅ gate passed | F-115; user-written from saved evidence |
| `results/slide_log.md`, `slides/talk_outline.md` | — | Slide log (4 slides so far) and talk outline, updated per phase | ℹ️ presentation drafts | D-48 |
| `results/nullspace/checks.json` + `artifacts/nullspace/rounds.npz` | `nullspace_rounds` | Iterative nullspace at idx 1/9/18, 150 rounds; K, crossings, secondary rules, redundancy; probe sequence and composite maps for Phase 5 (`25ed92ab…`) | ❌ failed `no_leak_into_removed_directions` (kept; explained F-117/F-118); other 11 passed | F-116 |
| `results/nullspace/checks.json` | `leak_diagnostic` | Leaks only in no-signal rounds; exact on every round with signal | ℹ️ diagnostic, explanation holds | F-117 |
| `results/nullspace/checks.json` | `covariance_exhaustion` | Guard reproduces saved rounds; covariance collapse per round | ⚠️ diagnostic, explanation_holds false (kept) | F-118 |
| `results/nullspace/checks.json` | `fresh_probe_erasure` | Fresh validation-fit probe after removal: covariance arm at chance, random = none | ✅ passed | F-119 |
| `results/nullspace/checks.json` | `alpha_sweep` | K vs fixed ridge alpha at idx 9: 7 to > 150 among good probes (re-run clean, F-128) | ✅ passed | F-120 |
| `results/nullspace/checks.json` | `kernel_erasure` | RBF kernel after erasure: strong nonlinear recovery | ❌ failed `no_alpha_failure` (7 lower edges, kept) | F-121 |
| `results/nullspace/checks.json` | `kernel_shuffled_labels` | Kernel negative control: max R² ≤ 0.005 | ✅ passed | F-122 |
| `results/nullspace/checks.json` | `kernel_grid_diagnostic` | Extended grid: 2/7 arms moved > 0.05; LOO invalid at extended alphas | ⚠️ diagnostic, explanation_holds false (kept) | F-123 |
| `results/nullspace/checks.json` | `kernel_hat_gap` | Numerical soundness per kernel selection; 4 / 72 quoted as ranges (re-run clean, F-128) | ℹ️ observation | F-123 |
| `results/nullspace/checks.json` + `artifacts/nullspace/random_subspaces.npz`, `pc_subspaces.npz` | `random_subspaces`, `pc_subspaces` | Random train-span removal changes nothing; top PCs erase fastest | ✅ passed | F-124 |
| `results/nullspace/checks.json` | `depth_profile` | K at every third index: 6–12 from idx 3 to 24 | ✅ passed | F-125 |
| `results/nullspace/checks.json` + `artifacts/nullspace/test_predictions.npz` | `nullspace_test_scores` | One-time test (F-126 scope); findings hold; headline CIs | ✅ passed (committed code) | F-127 |
| `results/nullspace/nullspace_rounds.png`, `erasure_and_procedure.png` + `checks.json` | `figure_nullspace`, `figure_erasure` | Rounds grid; procedure and erasure panels | ℹ️ visual (reviewed) | D-49 f |
| `results/evidence/checks.json` | `code_hash_check`, `rerun_identical_nullspace` | Phase 4 gate: 33 / 33 keys matched; 2 re-runs identical | ✅ passed | F-128 |
| `results/nullspace/report.md` | — | Nullspace report: setup, counts, what the count measures, paper comparison, kept on record, open items | ✅ gate passed | F-128; user-written from Claude Code's draft |
| `results/data_audit/report.md` | — | Data audit report: what was verified (with check keys), failures kept on record, open items and what each affects | ✅ gate passed | F-74; user-written from saved evidence |

All check scripts save through `src/vjepa_physics/evidence.py` (`save_result`). Provenance (D-27): `git_dirty`
= uncommitted or untracked changes in code/environment paths only, `git_dirty_paths`, `code` (SHA-256 per file
and combined), `versions` (from package metadata). **Entries without a `git_dirty_paths` field keep the old
meaning** (`git_dirty` = any change anywhere in the tree).

`opencv_tolerance` (test clip re-scored under D-05's tolerance) passed at step 0.18. For slides, a zoomed
crop of `frames.png` would read better (disk ≈ 21 px of 256).

---

## Open issues / blockers

*(Anything stopping progress goes here, with enough context to resume without re-deriving it.)*

No blockers.

**Settled by the planning chat (2026-09-25):** 2.1 and 2.3 — D-38 (held-out values, shared 16/4/4 assignment, direction
stratification, 2.2 pass criteria), D-39 (keep and flag; direction headlines with and without exit clips), D-40
(per-time-step pooling, 26 sites, index convention, batch 1, fp32, free disk ≥ 10 GB), D-41 (layout; gate = seeded
16-clip-per-dataset re-extraction, extraction code committed before the real run); D-37 extended to Phase 2. Joined
files stay small, pinning the activation `.npy` hash and memory-mapping it (Claude Code design, within D-41).

Settled by the planning chat (2026-09-24): `documented_fields` stays failed on record, labels from the dataset name
(D-32); D-29's limitation narrowed (F-58); `format` / `uniform_frames` closed without re-score, failures kept on record,
149 residue frames → O-02 at 2.1 (D-34); 1.10 flag definitions (D-36); Claude Code may set Phase 1 criteria taken
directly from DATA.md or the plan without asking (D-37).

Known decision points the plan cannot remove in advance (each has a planned fallback):
- **Phase 1 may overturn scouting facts** (F-21–F-36); D-14's split counts are provisional until step 1.12.
- **Steering readout** — 0.15 passed, but D-30 makes a later-layer probe on the full clip the primary readout (the
  predictor readout is optional); which layer is settled at step 5.2 (O-07), and Part 2's behavior manifold (D-17)
  moves with it.
- **Phase 3 may show no clean transition** → O-15/O-16 become documented judgment calls.
- **Downstream steering effects may wash out** → a finding, not a failure; the same-layer control keeps it interpretable.
- **MPS operator gaps and the compute budget:** no operator gaps so far (fallback never enabled, F-43); budget
  measured at 0.16 (F-59: ~0.9 s per forward pass, ~65 min for full extraction).

---

## Log

Newest entry on top. One entry per work session: what was done, what passed, what didn't,
what's next.

### 2026-09-26 — Phase 5 started (multi-probe subspace steering)
- Phase 5 layout (`steering.py`, partial forward in `intervention.py`, `check_steering.py`) and design brief drafted
  (steer idx 9 / read idx 18; min-norm δ over maps 1…K; zero-edit and two-path checks; arms 1 / 3 / K, covariance,
  norm-matched random; 3 planner questions: norm space of the min-norm solve, random-arm definition, round-K probe
  dominating ‖δ‖). **Next:** brief to the planning chat; first step = probe-sequence loader + δ solver, no model.
- 5.1a: `src/vjepa_physics/steering.py` (`ProbeSequence`, `load_probe_sequence` hash-guarded, K from `nullspace_rounds`
  and exhaustion round from `covariance_exhaustion` (first version read a field the pre-guard result lacks — Claude
  Code's slip, KeyError, nothing saved), `shift_matrix`, `min_norm_shift` standardized / raw). Smoke test (terminal, not
  saved; 50 seeded train rows, no test rows): K 6 / 7 / 7; composite = saved val predictions ≤ 9.3e-15; rounds
  orthogonal in the standardized space (off-block ≤ 2e-17); all probes hit target ≤ 3e-14 in both spaces;
  standardized δ inside span(Q) ≤ 2.1e-15, raw δ 0.38–0.42 outside; median ‖δ_z‖ n = 1 / 3 / K: direction
  4.6 / 19.3 / 45.3, speed 3.9 / 7.6 / 195.0, acceleration 2.7 / 7.9 / 72.4 vs median train clip-to-clip z-distance
  44.5 / 46.6 / 46.4; round K's share of ‖δ_z‖² 0.75 / 0.99 / 0.89. Raw-space solve: z-length +8–9 %. Claude Code's
  guesses: K/1 ratio 10–1000× met for speed (50×) and acceleration (27×), missed for direction (9.8×); last-two-rounds
  share ≥ 0.8 met (0.80 / 0.99 / 0.95); train z-distance 20–45 missed for speed, acceleration (46.6, 46.4).
  **Step 5.1a done.**
- Planning chat approved the brief with changes (→ D-50 at the phase docs pass; paper App. C.12 checked there: probe
  weights stacked, QR → V, c* by least squares so all probes read θ*, x⊥ kept; eval probe on test; 8 directions →
  90°; MAE 82.9° → 11.9° with 20 of 25 probes; no random baseline): Q1 standardized solve (= the paper's procedure in
  our space; raw minimum norm not used); Q2 random = unit direction in the standardized train span, length-matched per
  clip / target / n, 3 seeds at n ∈ {1, 3, K−1, K}; Q3 counts n = 1…K (full curve), **headline n = K−1** (5 / 6 / 6,
  all probes with val-seen R² ≥ 0.1; fixed from train-only norms before any outcome), n = K alongside with per-round
  shares and ‖δ‖ / clip distance, CIs at both, no cap; covariance arm δ_z = Σ_zy Σ_yy⁻¹ (t − ŷ), ŷ = round-1 reading
  of the unsteered clip; idx-18 error to the original label added; specificity = change vs unsteered only;
  validation smoke run = mechanics only; H-12 RBF kernel readout (validation-fit, Phase 4 pipeline) at idx 9 and 18,
  binned by ‖δ‖ / clip distance, labelled addition; random seeds → 2 if the cache timing projects > 30 min.
  **Next:** 5.1b partial forward + terminal smoke test.
- 5.1b: `intervention.run_blocks` (blocks first…last, `layer(h, None)[0]`). Smoke test (terminal, not saved; speed/1000,
  role train; MPS): stored pooled = recomputed at block_8 and block_17 (bit-exact); partial path = full forward at
  blocks 9–17, zero edit too (bit-exact); seeded δ (1e-2 × site std): partial = `edit_encoder` full pass at blocks 9–17
  (bit-exact); block_17 relative change 1.19e-2. Partial forward + pooling: median 0.759 s, max 3.528 s (20 runs).
  Claude Code's time guess 0.3–0.6 s wrong (higher) → 3,000 runs ≈ 38 min per variable at 3 seeds. **Step 5.1b done.**
- Planning chat (→ D-50): option A — each variable split into test-seen / test-unseen saved runs
  (`steer_<variable>_seen` / `_unseen`, 6 runs), 3 random seeds kept; random draws keyed by a SeedSequence of (SEED,
  variable, clip id, target index, n, seed index), order-independent; `steering_scores` combines both halves and checks
  they came from the same committed code; a half projecting > 30 min goes back to the planning chat. 5.1c approved
  with its smoke-test contents.
- 5.1c: `probes.fit_probe(fit_roles=…)`; `steering.py` clips / targets / covariance / random arms (`keyed_rng`,
  `spread_counts`, `steering_clips`, `steering_targets`, `covariance_map`, `covariance_shift`, `random_probe_counts`,
  `random_key`, `random_shift`). Two paste slips (code placed in `probes.py`, steering constants overwritten; NameError,
  nothing ran) fixed by the user. Smoke test (terminal, not saved) all hard expectations met: 15 / 15 clips per
  variable, roles ok, spread as specified, repeatable; targets as stated (direction 39.375 / 95.625 / 163.125 /
  253.125 / 326.25°, unseen T F T T F; speed / acceleration indices 4 / 16 / 34 / 46 / 59, unseen T F T F T); B =
  LinearRegression ≤ 3.1e-15, in the covariance span; random length error ≤ 2.1e-16, in span ≤ 4.2e-15, order-independent
  and distinct; default `fit_probe` = `layer_curves` bit for bit at block_8 and block_17; readout n_fit 297 / 304 / 304,
  interior alphas, train R² 0.969–0.987 (≥ 0.9 criterion previewed). Observations: no exit clip among the 30 selected
  direction clips (with/without-exit reporting identical for steering); covariance median ‖δ_z‖ 9.13 / 2.53 / 1.97 (Claude
  Code's 3–15 guess missed for speed, acceleration); round-1 reading after the covariance edit within ~0.005 of the target.
  **Step 5.1c done.** Committed and pushed (user said).
- 5.2a: `scripts/check_steering.py steering_setup` **passed** (10/10; committed code, no model): clips, targets and readouts
  = the 5.1c smoke test; readout raw maps = probe ≤ 1.5e-14; `artifacts/steering/setup.npz` SHA-256 `abcc701b…`. Chosen
  clips' flags (test-seen / test-unseen): direction clipped 1 / 0, sub-patch 3 / 4, frozen 1 / 3; speed sub-patch 1 / 1;
  acceleration sub-patch 4 / 3, frozen 3 / 2; no exit clip anywhere. Claude Code's flag guesses: sub-patch 2–6 per
  variable missed for acceleration (7). Two seen-target cases equal a clip's own label (direction 418 at 95.625°, speed
  407 at 1.202 m/s; kept, D-50). **Step 5.2a done.**
- 5.2b: `steering_cache` **passed** (8/8; committed code, MPS, 90 clips): on every clip stored pooled = recomputed at
  block_8 / block_17, zero edit = full pass at blocks 9–17, seeded δ (1e-2 × site std): partial path = `edit_encoder`
  full pass and block_17 changed — all bit-exact; cache re-read = computed; hooks 0 / 0; fingerprint = reference. Caches
  (git-ignored) direction `2f8244ef…`, speed `b127ca6c…`, acceleration `f9c3e763…`. Unsteered idx-18 readout on the
  steered test clips (own labels): R² 0.980 / 0.989 / 0.991, direction circular MAE 3.98°, speed MAE 0.083, acceleration
  0.210; seen vs unseen within 0.009 R². Partial forward median 0.659 s (max 1.24) → 1,440 / 1,515 / 1,515 runs per
  half ≈ 15.8 / 16.6 / 16.6 min, all within budget. Claude Code's guesses: time 0.7–0.9 s missed (faster); R² 0.95–0.99
  missed narrowly for acceleration (0.991). **Step 5.2b done.**
- 5.2c: `steering.arm_table`, `arm_shifts`, `steered_features`. Mechanics smoke test (terminal, not saved; one seeded
  val_seen clip per variable, 2 targets; no readout computed): 19 / 20 / 20 arms; zero shift = stored features at idx 9
  and 18 bit for bit; features finite; float64 probe hit ≤ 1.9e-14; after the fp32 edit ≤ 3.5e-7 (target units);
  idx-9 site vs stored + shift ≤ 1.8e-5 abs; random length error ≤ 2.1e-16; 0.67–0.73 s per arm. Claude Code's guesses:
  site diff 1e-6–1e-5 missed (1.8e-5), fp32 hit 1e-6–1e-3 missed (smaller). Tolerances for check (iv) fixed now, before
  any test run (Claude Code's own addition, ~30× / ~5× the measured maxima): probe hit ≤ 1e-5, site diff ≤ 1e-4.
  **Step 5.2c done.**
- 5.2d: six saved runs `steer_<variable>_<seen|unseen>` (one function, `functools.partial`) **all passed** (11/11 each;
  clean commit `80a2670`, chained with `&&` under `caffeinate`). Two starts stopped at NameErrors before any model load
  or save (missing `partial`, `train_span` imports; Claude Code's step listed them, not added). Runs 1,440 / 1,515 /
  1,515 per half; 10.5–12.5 min each (~72 min total; Claude Code's 16–17 min guess wrong, faster); unedited = stored on
  all 90 clips; max site diff 2.2–2.8e-5 (≤ 1e-4), max probe hit 1.8e-7–7.2e-7 (≤ 1e-5), random length ≤ 2.6e-16;
  weights, hooks unchanged; npz saved = computed. Claude Code's guesses missed: site ≤ 2e-5, hit ≤ 4e-7 (acceleration).
  **Process slip (F-128 again):** `steering_setup` and `steering_cache` were saved with `scripts/check_steering.py`
  staged, not committed (`git_dirty`, commit `cda056e`) → at the gate: `code_hash_check`, re-run flagged keys from
  committed code (results committed first), `compare_reruns`; the steering runs read both through their hashes.
  **Step 5.2d done.**
- 5.3a: `steering.readout_values`, `label_difference`; `steering_scores` **passed** (8/8, clean commit `f1b1889`; both halves
  same clean commit and code; unedited readouts = setup maps ≤ 7.2e-16; unedited error to own label = cache MAE exactly).
  **Surprise: steering at idx 9 barely reaches the idx-18 readout.** Headline n = K−1 (95 % CI), idx-18 error reduction:
  direction 0.029 [0.026, 0.032] (error to target 87.3° vs unedited 90.0°), speed 0.129 [0.124, 0.134], acceleration
  0.136 [0.127, 0.147]; gain 0.022 / 0.127 / 0.136. Same-layer idx-9 validation-fit readout follows: reduction 0.95 /
  0.95 / 0.92, gain 0.95 / 0.99 / 0.99 — but at n = 1 only 0.11 / 0.55 / 0.36 (n ≥ 2 ≈ 0.86–0.96). Random arms ≈ 0
  (|reduction| ≤ 0.007 at 1…K−1); covariance ≈ probes (0.031 / 0.119 / 0.119). n = K: 0.024 / 0.005 [−0.117, 0.128] /
  0.120; speed n = K shift 4.1 × clip distance, error to own label 0.083 → 1.03 m/s, its random arm reduction −0.16
  (off-distribution damage). Error to own label rises little at K−1 (direction 3.98 → 4.99°, speed 0.083 → 0.19,
  acceleration 0.21 → 0.54). Shift / clip distance at K−1 0.52 / 0.52 / 0.53; round-K share at n = K 0.73 / 0.99 / 0.89.
  Specificity (idx 18, K−1, mean |change|): speed steering moves acceleration 0.53 m/s² (random 0.16), acceleration
  steering moves speed 0.14 m/s (random 0.036) — both ≈ the own change in distance units (≈ 0.10 / 0.09 m; observation,
  sign not measured); direction steering moves speed / acceleration ≈ random. Claude Code's predictions: same-layer < 30 %
  of unedited ✓; idx-18 reduction 30–80 % and direction 10–40° **wrong** (far smaller); n = K no better ✓; random gain
  < 0.1 ✓; covariance highest gain per length **wrong** (probes 1–3 higher); shift / distance 0.4–1.0 ✓; round-K share
  ≥ 0.75 missed for direction (0.73). Pipeline checks rule out a wiring bug (edits reach block_17 bit-exact, same-layer
  readout moves, unedited readouts exact). → planning chat. **Step 5.3a done.**
- Planning chat on the results (→ D-50 addendum at the phase docs pass): no re-run; mechanism open — (i) blocks 9–17
  cancel the edit, (ii) idx-18 readout ≈ orthogonal to δ (skip connections carry δ), (iii) uniform token mapping (O-08)
  unlike real clip differences (limitation, future work); no "absorbs" wording until the decomposition and the
  propagation profile are in. Accepted, all post hoc on saved test outputs (rules and predictions fixed first, nothing
  selected): (1) H-12 kernel readout (read at idx 9, idx 18 consistency); (2) decomposition of the readout change into
  direct (readout map on δ) + block updates (map on Δmean − δ); (3) signed specificity in distance units, predicted
  slope ≈ +1 under the shared-distance reading, clip-bootstrap CIs; (4) propagation profile: validation-fit ridge
  readouts at every idx 9–18 on the saved steered means, gain / reduction vs depth per arm (n = 1, K−1, K, covariance,
  random), decomposition per index; (5) gain vs ‖δ‖ / clip distance over all arms, large edits labelled
  off-distribution; (6) covariance arm's idx-9 reduction next to K−1's. Provisional headline: same-layer result
  reproduced (0.92–0.95 at K−1, 0.11–0.55 at n = 1, random nothing), downstream barely reached (0.03 / 0.13 / 0.14);
  the paper tests the same layer only (C.12, P-06). Phase 6 note: score both methods over the idx 9 → 18 profile.
- 5.3b `steering_propagation` (post hoc, no model) **passed** (6/6) but saved **dirty** (`plot_index` import fix not
  committed before the run; `M scripts/check_steering.py`, commit `03ad13c`) → re-run at the gate. Profile readouts at
  idx 9–18: interior alphas, train R² 0.969–0.987; idx-9/18 refits = setup maps; readouts from saved fp32 features =
  runs; direct + blocks = total. **Decomposition at idx 18, n = K−1 (output-space gain, 95 % CI):** direction total
  0.068 = direct 0.189 [0.158, 0.220] + blocks −0.122 [−0.151, −0.091]; speed 0.128 = 0.463 [0.435, 0.493] − 0.336
  [−0.365, −0.307]; acceleration 0.136 = 0.262 [0.251, 0.274] − 0.126 [−0.138, −0.113]; covariance alike (direct 0.18 /
  0.29 / 0.24, blocks −0.11 / −0.17 / −0.12) → **both mechanisms**: the carried δ reaches the idx-18 readout only
  partly (ii), and block updates cancel about half to two-thirds of that (i). **Profile (total gain, K−1, idx 9 → 18):**
  direction 0.98 / 0.90 / 0.69 / 0.24 / 0.11 / 0.14 / 0.12 / 0.10 / 0.075 / 0.068; speed 0.99 / 0.94 / 0.78 / 0.42 /
  0.28 / 0.24 / 0.19 / 0.19 / 0.13 / 0.13; acceleration 0.96 / 0.86 / 0.62 / 0.25 / 0.24 / 0.20 / 0.13 / 0.16 / 0.13 /
  0.12; the drop happens in blocks 9–11 (idx 10–12), where blocks start cancelling (e.g. speed idx 12 direct 0.79,
  blocks −0.37). Random K−1 |gain| ≤ 0.03 everywhere. Idx-9 reduction covariance vs K−1: 0.956 / 0.955, 0.938 / 0.940,
  0.904 / 0.918 (comparable → the "many directions" need is a property of the probe sequence). Gain vs shift size
  (probes + covariance, in-distribution bins): direction 0.016–0.071, speed 0.075–0.124, acceleration 0.061–0.139,
  rising from the smallest bin; off-distribution speed bin ≥ 2 gain 0.58 with direct −2.25 / blocks +2.79 (n = K
  breakage). Claude Code's predictions: hard expectations ✓; direct 0.3–1.0 and blocks −0.2…−0.9 met for speed only
  (direction, acceleration smaller); most of the drop by idx 12 ✓; random < 0.05 ✓ (K−1 arms checked); flat within
  ±0.05 ✓ (with a rising trend). **Step 5.3b done.**

### 2026-09-25 — Phase 4 started (nullspace probing)
- Working style reconfirmed (user said): Claude Code edits only `docs/` and `CLAUDE.md` (automatically), asks before
  every step, reads files only for bugs or suspicious results. Phase 4 layout (`nullspace.py`, `check_nullspace.py`)
  and design brief drafted (layers idx 1 / 9 / 18; 3 planner questions: non-monotone curve, cap hit, alpha rule near
  0.1). **Next:** brief to the planning chat; 4.3a core `nullspace.py` + terminal smoke test.
- Planning chat approved the brief with changes (→ D-49 at the phase docs pass): headline layer idx 9 (contrasts
  idx 1, 18); all 150 rounds; K = first val-seen R² < 0.1, cap → "> 150" / 150 probes; paper thresholds and secondary
  rules reported; alpha rule: upper edge with R² < 0.1 = `no_signal`; random control in the train span; top-PC
  control; K-vs-depth profile at idx 0–24 (stop R² < 0.05 or 150); report caveat revised. Doc cleanup applied
  (Phase 2 re-run rows cite F-87, not F-115; report line 29 already fixed).
- 4.3a: `src/vjepa_physics/nullspace.py` (`train_scaler`, `project_out`, `extend_basis`, `run_rounds`,
  `composite_maps`, `train_span`, `random_span_basis`, `nullspace_alpha_verdict`). Smoke test at block_8 (terminal, not
  saved): round 1 = `layer_curves` bit-identical (alpha 56.23 speed / 17.78 direction); orthonormal err ≤ 6.7e-16; leak
  ≤ 2.8e-15; composite = round-by-round ≤ 4.7e-15; span rank 831 / 812; random and PC round 1 = real; 0.13–0.15 s per
  fit. Val-seen R² real speed 0.989 → 0.420 (5 rounds), direction 0.981 → 0.926 (3); random flat; **top PCs drop
  fastest** (speed 0.132 after 2 PCs, direction 0.071 after 4). Claude Code's predictions wrong on speed redundancy
  (> 0.9 at round 5) and on PC ordering (predicted between random and real). **Step 4.3a done.**
- Plan 4.1 and 4.2 ticked: both are decision steps, settled by the brief + planning chat (D-49) before any code.
- D-49 written into DECISIONS (O-15 closed). 4.3b: `nullspace.py` curve helpers (`round_scores`, `first_true`,
  `curve_summary`, `redundancy_counts`); `scripts/check_nullspace.py nullspace_rounds` (~3 min) **failed
  `no_leak_into_removed_directions`** (saved max leak 0.97–0.99999 at all 9 sites; smoke test rounds 1–5 were ≤ 2.8e-15);
  the other 11 criteria passed (round 1 bit-identical everywhere). K (val-seen R² < 0.1): direction 8 / 6 / 8, speed
  14 / 7 / 9, acceleration 13 / 7 / 9 (idx 1 / 9 / 18); 135–143 `no_signal` rounds per site; no rise after K > 0.03.
  Artifact `rounds.npz` `25ed92ab…`. Failure kept on record; hypothesis: rounding-dominated weights in no-signal
  rounds. Diagnostic `leak_diagnostic` rule fixed before looking: (a) leak > 1e-8 only at `no_signal` rounds, (b) every
  `ok` round leak ≤ 1e-8.
- `leak_diagnostic` (committed code): **explanation holds** at all 9 sites — leaking rounds all `no_signal`; `ok`
  rounds max leak 2.3e-14; first leaking round = last `ok` + 3–4, always after K; |W| 0.25–8 → 1e-18–1e-19, leaked part
  ≈ all of |W|. Claude Code's analysis: at alpha 1e7 ridge ≈ covariance direction; removing it exhausts the train
  covariance (Xᵀy = 0 ⇒ weights 0 for any alpha); consequence: m covariance directions suffice to erase linear
  readout (LEACE, from memory) → K is procedure-dependent. Failure + proposals (re-scored leak criterion before
  "covariance exhausted", covariance-direction control, mechanism check) → planning chat.
- Planning chat (→ D-49 addendum at the phase docs pass): leak failure kept; re-scored post hoc = criterion before
  covariance exhaustion (|W_k| ≤ 1e-12·|W_1|), later rounds "undefined"; no re-run; code guard (commit first).
  Covariance control rejected → fresh-probe erasure test (new ridge fit on validation, clip-grouped 5-fold CV; arms:
  none / nullspace at K / covariance dirs / random m and K·m, 5 seeds; idx 1/9/18). Mechanism check accepted. New:
  alpha sweep at idx 9 (6 fixed alphas, guard, cap 150). Interpretation held; LEACE to be verified before any slide.
  Order: mechanism → erasure → alpha sweep → controls / profile → test. Claude Code's details: nested CV in the
  erasure test; sweep grid 1e-3, 1e-1, 1e1, 1e3, 1e5, 1e7.
- 4.3c: guard in `run_rounds` (`EXHAUSTION_RATIO` 1e-12, `exhausted_round`); `nullspace_rounds` kept runnable under
  the guard; `covariance_exhaustion` (committed code) **explanation_holds false**: scaler rebuilt = saved, guarded run
  stops at weight collapse and reproduces saved rounds bit for bit (9/9); **failed (a)** covariance ratio at first
  leaking round 8e-11–1.2e-8 (not ≤ 1e-12), **(b)** weight collapse one round after first leak at 7/9 sites (Claude
  Code's claim to the planner was wrong). Mechanism qualitatively supported (covariance −1e-1…−1e-4 per round from
  the last ok round; leak rises ~1e3 per round ∝ 1/|W|). Re-scored leak criterion fails at 7/9 sites by one no-signal
  round after K (leaks 2.6e-8…6.3e-7); every direction added under the guard has leak ≤ 6.3e-7 → planning chat.
- Planning chat (→ D-49 addendum): proposal accepted; all three failures kept, no threshold changed, no re-run; report
  line on leaks fixed; guard = definition of exhaustion; Phase 5 uses only Q[:, :K·m] and maps 1…K (loader enforces);
  rounds of record = up to the guard round, later "undefined (train covariance exhausted)"; pre-stated check for
  profile / sweep / controls: guard round > K everywhere; sweep: no edge rule, cap without exhaustion expected at small
  alpha; diagnostic rules only on established facts (uncertain premises → observations). Next: fresh-probe erasure test.
- 4.3d: `probes.nested_cv_predictions` (+ `NestedCV`), `nullspace.train_ridge`; `fresh_probe_erasure` **passed** (7/7,
  ~8.5 min; committed code). Fresh out-of-fold probe on validation (nested grouped CV): `none` 0.72 (direction idx 1) /
  0.947–0.987; random m and K·m = `none` to ~0.001 (5 seeds); **`covariance` (m = 1 or 2 dims) −0.045…0.001** and
  `nullspace_k` −0.012…0.021, direction circular MAE 78–92°. Claude Code's prediction (fresh R² ≥ 0.8 after covariance
  removal) **wrong**: β = Σₓₓ⁻¹Σₓᵧ = 0 when Σₓᵧ = 0, and the train cross-covariance ≈ the population one → linear
  readout erased with m dims for held-out fresh probes too (LEACE, from memory). K counts procedure steps, not
  dimensionality (interpretation → planning chat together with the alpha sweep).
- 4.3e `alpha_sweep` **passed** (4/4; first run failed at argparse: `CHECKS` entry missing, fixed and re-committed;
  nothing saved by the failed call). idx 9, fixed alpha 1e-3 / 1e-1 / 1e1 / 1e3 / 1e5 / 1e7: round-1 R² 0.94 / 0.95–0.96 /
  0.98–0.99 / 0.97–0.98 / 0.64–0.68 / 0.02; K direction 12 / >150 / 56 / 7 / 2 / 1, speed 7 / >150 / 83 / 9 / 2 / 1,
  acceleration 7 / >150 / 88 / 10 / 2 / 1 (LOO 6 / 7 / 7); guard never ≤ K. K spans 7 to >150 among good probes.
  Claude Code's predictions: 3 met, 3 wrong (monotonicity; 1e-3 vs LOO; LOO between 1e1 and 1e3). Non-monotone at 1e-3
  and pre-guard leaks up to 5.5e-2 at 1e3: observations. Erasure + sweep interpretation → planning chat.
- Planning chat (→ D-49 addendum): m-dim linear erasure = math property (cross-covariance rank m), never "stored in m
  dims"; empirical content = (a) train covariance direction generalizes (fresh probes at chance, random removal incl.
  K·m changes nothing), (b) K 7 to >150 across well-fitting ridge probes → procedure-dependent; paper's counts
  consistent with weak regularization (hypothesis); "erased" = undetectable by a probe on ~240 clips/fold; 1e-3
  non-monotonicity = observation. Redundancy = procedure steps. **New key:** RBF kernel ridge after erasure
  (`kernel_ridge`, exact LOO; gamma grid around median heuristic, alpha relative grid; arms none / covariance /
  nullspace K / random m ×5; idx 1/9/18; val-seen). H-01 reframed (manifold PCA dim vs K and m). Test adds the erasure
  headline. Depth profile every third index (or dropped, user decides); random control kept, PC optional. One
  multi-panel figure; message "the nullspace count measures the probe procedure, not the representation". Phase 5 K
  6 / 7 / 7 unchanged; notes for Phase 5 (covariance steering arm) and 7.5 (covariance principal angles).
- User: keep depth profile (every third index) and PC control. 4.3f: `KernelFit.loo_mse`, `squared_distances`,
  `median_gamma`, `rbf_kernel_ridge` (`baselines.py`), `nullspace.covariance_basis`; two script bugs (missing import,
  `GAMMA_FACTORS_USED`, Claude Code's slip) caught by review before running. `kernel_erasure` **failed
  `no_alpha_failure`** (7 / 72 fits at the lower relative-alpha edge, all erased arms at idx 9/18); other 5 passed (RBF
  Gram = sklearn ≤ 3.3e-16; linear-Gram kernel_ridge = RidgeCV). Kernel val-seen R² after erasure: covariance
  0.71–0.93 (direction), 0.93–0.96 (speed), 0.86–0.96 (acceleration); nullspace_k 0.69–0.94; random = none. Claude
  Code's recovery prediction (0.2–0.8) wrong (stronger). Failure + options (A keep / B extended-grid diagnostic) →
  planning chat.
- Planning chat (→ D-49 addendum): failure kept, 7 arms flagged "alpha at lower grid edge (under-resolved)", scores
  held out, not a lower bound. Required before any slide: kernel negative control (covariance removed at idx 9, 5
  shuffled-label permutations, pass max R² < 0.1 and direction mean circular MAE > 80°). Modified B diagnostic (7 arms,
  alpha to 1e-9 relative, gamma 0.125–8; holds if every refit within 0.05 of saved; edge = recorded; report
  min(1 − h_ii), flag < 1e-6). Hypothesis-level reading (nonlinear/curved encoding, H-01; direction weaker ↔ H-02);
  kernel arms added to the test key; new H-12 (probe steering leaves the nonlinear code intact). Order: negative
  control → B → controls / profile → test.
- 4.3g: `probes.label_permutations` (= F-94's first 5); `kernel_shuffled_labels` **passed** (4/4, ~30 s): covariance
  removed at idx 9, 5 shuffled-label kernel fits: max val-seen R² 0.005 / 0.001 / 0.005 (real arm 0.753 / 0.942 /
  0.928); direction mean circular MAE 89.5° (min 86.8°); labels moved 97.7–99.0 %; no lower-edge alpha (6 / 15 upper
  edge, rest interior). Claude Code's "most upper edge" prediction wrong; gamma always on a grid edge (observation).
  Kernel slide requirement cleared.
- 4.3h: `KernelFit.min_one_minus_hat` (first placed in `RBFFit` by mistake → TypeError, fixed; nothing saved by the
  failed calls). `kernel_grid_diagnostic` (committed code) **explanation_holds false**: original-grid refits = saved;
  2/7 arms moved > 0.05 (direction idx 18 nullspace_k −0.073, speed idx 18 nullspace_k −0.094), 6/7 downward; all 7 at
  the new lower edge (~1e-10); **min(1 − h_ii) negative in all 7 (−1.2e-8 … −1.5e-6) → LOO invalid at the extended
  alphas** (rounding breakdown). Extended val-seen R² all ≥ 0.66 (observation). Claude Code's predictions: change
  < 0.05 wrong at 2 arms; hat gap 1e-4–1e-7 wrong (negative). Proposal (keep failure; rule-free hat-gap observation
  over all 72 original fits; grid-dependent wording) → planning chat.
- Planning chat (→ D-49 addendum): diagnostic failure kept, no re-score / grid extension; extended fits' scores only in
  the cross-grid range. Observation key over all 72 original fits: min(1 − h_ii) ≥ 1e-6 → point value (edge flag
  kept), < 1e-6 or negative → range only. Headline = idx 9 covariance arm; summary line "val-seen R² ≥ 0.66 under
  every grid tried; linear ≈ 0; shuffled ≤ 0.005". Test: kernel arms from original-grid refits (must reproduce saved
  val-seen), unreliable arms labelled. Kernel thread closes after the observation key. Next: random (+ PC, user
  decision) control, depth profile, test, report and slides; each Phase 4 failure one line in "kept on record".
- 4.3i `kernel_hat_gap` (observation; refits = saved, 72/72): 4 / 72 selections unreliable (min(1 − h_ii) < 1e-6), all
  `nullspace_k` at the lower edge → quoted as ranges: direction idx 9 0.664–0.692, idx 18 0.760–0.834; acceleration
  idx 9 0.922–0.929, idx 18 0.937–0.939. Headline idx 9 covariance arm sound → points: direction 0.753, speed 0.942,
  acceleration 0.928 (gap 1.0e-6, marginal, edge). Unerased / random gaps ≥ 9e-3. **Kernel thread closed.**
- 4.4 `random_subspaces` (5 seeds, ~14 min) and `pc_subspaces` (~3 min) **passed** (7/7 each; round 1 = real
  bit-identical; bases orthonormal, in train span). Random: never below 0.1 in 150 rounds, max drop 0.014; at real K
  still 0.85–0.99. Top PCs: K ≤ real K everywhere — direction 7 / 3 / 6, speed 2 / 5 / 2, acceleration 2 / 7 / 3
  (idx 1 / 9 / 18); one top axis erases speed (idx 1, 18) and acceleration (idx 1) (observation). Claude Code's
  "PC K ≤ 5 at idx 9" wrong for acceleration (7). Artifacts `random_subspaces.npz` `75795bef…`, `pc_subspaces.npz`
  `9e1807e0…`. **Step 4.4 done.**
- 4.3j `depth_profile` **passed** (6/6, ~1–2 min; idx 9 and 18 bit-identical to the full runs; no guard ≤ K). K (R² <
  0.1) at idx 0/3/6/9/12/15/18/21/24: direction 2/9/7/6/7/6/8/8/8, speed 1/12/8/7/8/9/9/9/9, acceleration
  1/11/9/7/8/9/9/9/9 (idx 0 = no signal); 0.05 stop adds ≤ 1 round. Mild U (max idx 3, min idx 9), no growth with depth
  (paper: 40–136+ / 400, P-02). Procedure count. **Step 4.3 done** (validation).
- Validation findings frozen in DECISIONS §1.10 before any test row is read: F-116–F-125, test scope F-126; H-12
  added, H-01 reframed. Next: `nullspace_test_scores` (once, committed code).
- Test once: `nullspace_test_scores` **passed** (8/8, committed code; round 1 = F-104). Test K within ±1 of validation
  everywhere (idx 9: 7 / 7 / 7); fresh linear after covariance removal ≈ 0 (acceleration below chance on test-seen,
  −0.195 idx 9, wide CI: observation); kernel after removal within 0.039 of val-seen (idx 9 CI direction [0.72, 0.79],
  speed [0.91, 0.94], acceleration [0.92, 0.95]). F-127. Claude Code's "fresh ≤ 0.05" wrong for acceleration. Script
  size noted: refactor shared arm setup into the package at 8.6 (user said), re-run affected keys then.
- 4.5 pass 1: `figure_nullspace` → `results/nullspace/nullspace_rounds.png` (3 × 3, rounds 1–20; validation, frozen test,
  random mean + band, top PCs, stop 0.1, paper thresholds in the legend; K with dims in each panel; dims not on a second
  axis). Claude Code's reviews: (1) invisible random band, K labels in its path, paper labels crowding, legend gap →
  fixed; (2) paper labels crossing the PC line → moved to the legend; (3) clean.
- 4.5 pass 2: `figure_erasure` → `results/nullspace/erasure_and_procedure.png` (K vs fixed alpha, K vs depth, erasure
  bars at idx 9: linear train-fit / fresh linear / RBF kernel, test-seen dots with CIs, grid-range whiskers). Second PNG
  instead of a fourth grid row (slide-sized; layout Claude Code's call, noted for the report). Review (1): hidden
  off-scale test point (acceleration fresh covariance −0.195), clipped y-label, overlapping idx-0 markers, crowded tick
  labels, legend gap → fixed; review (2) clean. **Step 4.5 done.**
- 4.8a gate: `code_hash_check` (+ `nullspace`) flagged `alpha_sweep`, `kernel_hat_gap` (run before their commit; next
  commits bundled more code; Claude Code's "no key flagged" prediction wrong). Results committed (`5d33503`), both re-run
  clean: `rerun_identical_nullspace` passed (identical, `seconds` excluded); `code_hash_check` re-run passed, 33 / 33 keys,
  `rerun_needed` empty (F-128). Lesson: run only after the commit command has finished.
- 4.8b–d: report `results/nullspace/report.md` (user-written from Claude Code's draft), slide log +2 slides, talk outline
  item 5 done; full docs pass (D-49 addenda a–f, O-16 fixed, change log, plan 4.3–4.8 ticked, phase table, Saved
  evidence, CLAUDE.md). **Step 4.8 done — Phase 4 passed.** Next: gate commit + push, then the Phase 5 design brief.

### 2026-09-25 — Phase 3 started (layer-wise probing)
- Working style reconfirmed (user said): Claude Code edits only `docs/` and `CLAUDE.md` (automatically after each step),
  asks before every step, and reads files only when there is a bug or suspicious result.
- Phase 3 layout proposed: package `probes.py`, `metrics.py`, `baselines.py`; scripts `check_probes.py` (3.2, 3.3,
  test scores), `check_baselines.py` (3.4), `check_layer_curves.py` (3.5–3.7, figure); gate key
  `rerun_identical_probing` in `check_evidence.py`.
- Step 3.1 decision brief prepared for the planning chat, plus pending Phase 2 notes and later Phase 3 choices.
  The physics paper PDF could not be read in Claude Code (no `pdftoppm`); paper details come from DECISIONS only.
- `scikit-learn` added to `evidence.PACKAGES`; check printed `1.9.1 9` as predicted; committed `0e2f572` (F-88).
- Push rule (user said): commit per step, push only after a substantial step (recorded in `CLAUDE.md`).
- 3.2a: `src/vjepa_physics/metrics.py`; smoke test (terminal, not saved) exactly as predicted (F-89). **Step 3.2a done.**
- 3.4a: `physics_fit` in new `baselines.py`, `pixel_to_world` in `geometry.py`; smoke test (terminal, not saved): exact
  on metadata positions; tracked ceiling values recorded; two Claude Code predictions wrong (acceleration MAE lower than
  predicted, exit clips better not worse) (F-90). **Step 3.4a done.**
- Planning-chat decisions recorded: D-42, D-43, D-44; O-05 closed; plan 3.1 ticked, 3.8 marked skipped. Claude Code's
  review: adopt; three own-addition follow-ups sent back (none blocks 3.2b).
- 3.2b: `probes.py`; smoke test (terminal, not saved): all hard expectations met; no numeric score prediction was made
  (F-91). **Step 3.2b done.**
- 3.2c: `scripts/check_probes.py layer_curves` passed, smoke-test points reproduced exactly (F-92); early transition
  (index 1) recorded with hypothesis H-07 and a note for the planning chat. **Step 3.2 done for validation** (test at 3.7).
- Planning-chat decisions recorded: F-93, D-45, D-46; plan 3.8 repurposed (un-ticked).
- 3.3: `check_probes.py shuffled_labels` passed (F-94); Claude Code's predictions for max R² and per-fit direction
  minima were wrong; H-08 with a symmetry rule fixed before reading. **Step 3.3 done.**
- H-08 read-out (terminal, from the saved result): symmetry rule failed at block_0, 7, 14, 16, 17; kept on record;
  H-08b noted (post hoc). Claude Code's symmetry premise was wrong (reflections give ≈ 90°, not > 90°).
- 3.4b: `pixel_matrix`, `exact_gram`, `kernel_ridge` in `baselines.py`; smoke test exact as predicted (F-95).
  **Step 3.4b done.**
- Committed (user said). 3.4c: `scripts/check_baselines.py pixel_grams` passed (F-96); Claude Code's time and memory
  guesses were high (22 s per full Gram, 5.38 GB). **Step 3.4c done.**
- 3.4d: `check_baselines.py pixel_floor` passed (F-97); one Claude Code prediction wrong (time-averaged direction floor
  above chance → H-09). **Step 3.4d done.**
- 3.4e: `from_rest` model in `physics_fit`; `check_baselines.py physics_ceiling` passed as predicted (F-98).
  **Step 3.4 done.**
- Committed and pushed (user said). 3.5a: bootstrap helpers; smoke test caught a guard bug in `r2`/`resampled_r2`
  (Claude Code's), fixed and re-tested (F-99). **Step 3.5a done.**
- 3.5b-1: `curves.py`; unit tests and saved curves as worked out by hand (F-100). **Step 3.5b-1 done.**
- 3.5b-2: `scripts/check_layer_curves.py bootstrap`; first run crashed in the print function (Claude Code's bug, nothing
  saved), fixed; passed, predictions met (F-101). **Step 3.5 done (validation).**
- 3.6: `figure_layer_curves` (dataviz method; project palette); Claude Code reviewed the PNG, found three defects
  (label collision, log ticks, spine colour), fixed, re-reviewed clean (F-102). **Step 3.6 done.**
- Committed and pushed (user said). 3.7 recorded from saved evidence (F-103); plan ticked. **Step 3.7 done.**
- `test_scores` added and committed before running; run once, passed (F-104). Identical speed-embedding val/test
  scores checked and explained. **Step 3.2 done (test).**
- 3.8 brief sent; planning chat revised the design to per-patch (D-47); H-08b closed as an observation.
- 3.8a: per-patch pooling in `extraction.py`; smoke test as predicted (F-105). **Step 3.8a done.**
- 3.8b: `scripts/check_patches.py` given; `extract_direction` started (~40 min).
- Lean-mode review: Claude Code found the literal hash rule would fail every key; precise rule, Phase 5 clip counts,
  fallback layer, centroid clips, Phase 8 list and doc cadence settled with the planning chat (D-48, F-106).
- 3.8b `extract_direction` passed: 39,321,600,128 B, patch mean = all-token mean ≤ 6.5e-8, 36.1 min, ids = F-82's
  (SHA-256 `bc414b66…`); code committed (user said; hash rule, D-48).
- 3.8b `verify` passed: 16 seeded clips re-extracted bit-identical, ids = joined table, weights unchanged.
- 3.8c `patch_probes` **failed `no_alpha_failure`** (20 lower-edge alphas, all at index 0; all other criteria passed;
  indices 1–24 all ok). Mean per-patch val-seen R² 0.006 → 0.535 (idx 1) → 0.951 (idx 6) → peak 0.977 (idx 13) →
  0.932 (idx 24); transition idx 1, 80% rise idx 4. `patch_alpha_diagnostic`: pre-set tie rule **failed** (LOO spread
  0.2–1.2%, 4–21 distinct train vectors) → H-10 not supported; post-hoc H-10b (few-clip support at outer patches;
  optimum below the grid). Failure kept on record; note to the planning chat.
- Planning chat: option A (failure and failed diagnostic kept on record; index 0 reported with caveat; H-10b stays a
  hypothesis; no extended-grid refit). Robustness from saved numbers (Claude Code's arithmetic): 80 % threshold
  0.782755; idx 3 margin −0.0614, idx 4 +0.0129; 50 % threshold 0.491420, idx 1 +0.0439; the 20 failing probes shift
  idx 0's mean by ≤ 0.0027 → threshold by ≤ 0.0005. All margins > 0.005 → no transition result depends on the failure.
- 3.8d-1: `geometry.distance_to_patches`, `PATCH_PX`, `PATCH_GRID`, `DISK_RADIUS_PX` (= √(350/π) = 10.555 px);
  hand-worked cases matched (Claude Code's expected radius 10.563 was an arithmetic slip; code right).
- 3.8d-2 `patch_breakdown` passed (7/7): off-path mean R² (256 patches) −0.003 → 0.648 (idx 1) → 0.959 (idx 6) →
  0.979 (idx 13) → 0.932; on-path (55 central patches) −0.112 → −0.580 (idx 1) → 0.134 (idx 5) → 0.842 (idx 6) →
  0.945 (idx 9) → 0.911; off-path transition idx 1, 80 % rise idx 2. Checked as suspicious (layout, masks, counts):
  consistent. Hypothesis H-11: one probe per patch learns the spread-out code from the off-path majority; on-path
  tokens early are dominated by local disk appearance. `patch_bootstrap` running (Claude Code's 3–5 min estimate too low).
- 3.8d-2 `patch_bootstrap` passed (4/4; points = saved bit-exact): per-patch mean-curve transition idx 1 in 9,582 /
  10,000 resamples (294 at 2, 124 at 3); 80 % rise idx 4 in 8,606 (1,383 at 5, 11 at 3); gap mean-pooled − per-patch
  > 0 at every index (CI excl. 0): ~0.32–0.33 at idx 1–2, ~0.01 at idx 9–13, 0.057 at idx 24. Idx 2 CI wide
  [0.40, 0.73]: heavy-tailed mean (median 0.733). Observation (post hoc, labelled): per-patch ≈ mean-pooled from idx 9
  (depth 0.375) to 13.
- 3.8e-1: `probes.clip_folds`, `grouped_cv_ridge` (exact grouped K-fold ridge via per-fold sums + one eigh per fold);
  vs sklearn fold-by-fold `Ridge`: CV MSE 3.8e-16 rel, same alpha (31.6, interior), coef/intercept/predict ≤ 1.3e-14;
  each clip in one fold (12 per fold); seeded folds repeatable.
- 3.8e-2 `spatial_generalization` passed (9/9, 7.9 min; read from the saved file): same / across / gap val-seen R²
  (mean of 4 halves) idx 0 0.002 / −0.006 / 0.008; idx 1 0.814 / 0.811 / 0.003; idx 9 0.974 / 0.957 (across peak) /
  0.018; idx 13 0.980 (same peak) / 0.945 / 0.034; idx 20 0.948 / 0.819 / 0.129; idx 24 0.932 / 0.829 / 0.103.
  Across-curve transition idx 1 in 10,000 / 10,000 resamples, 80 % rise idx 1. Claude Code's hypothesis (early gap
  that closes) was wrong: generalization is near-perfect from block_0 and degrades from idx 11. Differs from the
  paper's reported one-third (F-93) under our time-averaged interpretation (D-47) → planning chat.
  `patch_alpha_diagnostic` saved with git_dirty true → hash check at 3.9.
- Planning chat on 3.8 findings: report as a finding with caveats (time-averaging, simpler stimuli, 256 vs 224 px P-09,
  unspecified protocol); H-11 hypothesis; "per-patch ≈ pooled from ~⅓ depth" labelled post hoc; per-time-step
  per-patch re-extraction = future work. Two checks first: (1) negative control, off-path val-seen R² at idx 0 ≈ 0 —
  already in `patch_breakdown`: mean −0.0034, median −0.0017 (256 patches) ✓; (2) position-only baseline (tracked mean
  disk position → (sin, cos), train fit, val-seen score; ≥ 0.5 → planning chat); Claude Code adds a cubic variant
  (own addition, reported alongside). Then test once (per-patch mean, off-path, across-half; saved probes), 3.9 gate.
- 3.8f `position_baseline` passed (4/4): tracked mean position → (sin, cos), OLS on train (no alpha: n ≫ p); val-seen
  R² linear 0.197 (circ MAE 63.0°), cubic 0.211; val-unseen 0.233 / 0.255; far below off-path 0.648 (idx 1) / 0.959
  (idx 6) → off-path readout is not explained by position. Also H-09's proposed test: position alone ≈ the
  time-averaged pixel floor (0.197 vs 0.202, F-97) → supports H-09 (not proof).
- 3.8g `patch_test_scores` passed (8/8, 2.5 min, committed code `43ee0b6`; read from the saved file): saved probes
  reproduce validation outputs exactly (difference 0.0); count bootstrap = `resampled_r2`. Test-seen / test-unseen:
  transitions idx 1 for per-patch mean, off-path, across-half; 80 % rise idx 4 / 2 / 1 (= validation); peaks 0.974 /
  0.975, 0.976 / 0.977, 0.955 / 0.953; gap ≤ ~0.03 to idx 10, 0.133 / 0.130 at idx 20, ~0.10 at idx 24. 3.8 findings
  hold on test.
- 3.8h `figure_local_to_global`: `results/patches/local_to_global.png` (heatmaps at idx 0/1/2/4/6/13/24; mean-pooled
  vs per-patch mean vs off/on-path vs position-only; same vs other half with gap). Claude Code's first review: legend
  collided with the position line, uneven heatmap spacing, tall colour bar → fixed; second review clean.
  **Step 3.8 done.**
- 3.9a `code_hash_check`: 18 keys, 5 flagged (as predicted; Claude Code's "~25 keys" count was wrong). Lost
  `pixel_grams` entry in `check_baselines.py` `CHECKS` restored. 3.9b: 5 re-runs from `8121797`. 3.9c
  `rerun_identical_probing` passed (all identical); `code_hash_check` re-run passed (`rerun_needed` empty) (F-115).
  Full Phase 3 docs pass done (F-107–F-115, H-10/H-10b/H-11).
- 3.9d: report `results/layer_probing/report.md` (user-written from Claude Code's draft; mean-pooled direction values
  at idx 6/13/24 confirmed from the saved result), slide log `results/slide_log.md` (2 slides), talk outline
  `slides/talk_outline.md` (8.3 started); committed and pushed (user said). Pending wording fix: report line 29
  "after block 1" → "after the first block" (next commit). **Step 3.9 done — Phase 3 passed.**
- **Next:** Phase 4 design brief (nullspace probing) for the planning chat.

### 2026-09-25 — Phase 2 started (splits and extraction)
- Working style reconfirmed (user said): Claude Code edits only `docs/` and `CLAUDE.md` (updated directly after each
  step), asks before every step, and does not re-read files unless there is a bug or suspicious result.
- Step 2.1 decision brief drafted and sent to the planning chat, together with the D-37 scope question and the 2.3 items
  (see "Open issues").
- `motion_group` and `angle_octant` added to `src/vjepa_physics/data.py` (Phase 1 scripts keep their local copies, so
  their code hashes stay unchanged). Smoke test (terminal, not saved): group counts and octant × group table equal
  `design_balance` exactly, cells 13–20, octant edges correct, θ = 360 rejected (F-75).
- Planning-chat decisions recorded: D-37 extended; D-38–D-41; plan steps 2.1 and 2.3 ticked.
- `verified_artifact` (hash-guarded artifact read) added to `evidence.py`; `file_sha256` chunked (same digests). Smoke
  test (terminal, not saved): tracking npz and flags table pass, wrong hash and missing key rejected (F-76).
- 2.2a: `src/vjepa_physics/splits.py` (D-38 rules as pure functions, CSV write/read). Smoke test (terminal, not saved):
  every count, the shared assignment, 16/4/4, rebuild and byte-identical writes as predicted (F-77). **Step 2.2a done.**
- 2.2b: `scripts/check_splits.py build` passed, all values as predicted; `artifacts/manifests/splits.csv` written and
  recorded (F-78). Own-addition criteria `labels_match_metadata`, `value_index_follows_label` noted for the planning
  chat (D-37). **Step 2.2b done.**
- 2.2c: `balance` diagnostic (split file and flags table hash-guarded). Exact predictions met (sub-patch counts per role,
  no speed/acceleration exits, identical val/test seen tables). Notes: one seen direction angle has no test_seen clip;
  exit clips val_seen 8 vs test_seen 20 (chance level). Recommendation: keep splits; notes to the planning chat (F-79).
  **Step 2.2 done.**
- 2.4a: `src/vjepa_physics/extraction.py`; smoke test on the test clip: shape/dtype/device right, repeat bit-exact,
  rel. error 5.29e-8 vs a float64 pool, derived mean 5.13e-8, final_norm ≠ block_23, plot indices 0/1/24 (F-80).
  **Step 2.4a done.**
- 2.4b: `scripts/check_extraction.py` (`pipeline`, `extract_<dataset>`) committed before any run (D-41); `pipeline`
  passed, errors identical to the smoke test, pooled SHA-256 `29997d28…` recorded (F-81). **Step 2.4 done.**
- 2.5: `extract_direction` / `extract_speed` / `extract_acceleration` all passed (31.8 / 36.0 / 38.3 min; median
  1.25 → 1.55 s per clip, slowing under sustained load); file sizes as predicted; speed test clip = `pipeline`;
  speed/acceleration id files identical (same ids, expected) (F-82). **Step 2.5 done.** Tip given: back up
  `artifacts/activations/` (git-ignored; only recovery is a ~1 h 45 min re-extraction; hashes prove a restored copy).
- 2.6: `verify` passed (48/48 live clips bit-identical, ids = splits, weights unchanged). Claude Code's prediction that
  site std grows with depth was wrong: it peaks at block_8 (≈ 8.1) and falls to 3.3 at block_23 (F-83). **Step 2.6 done.**
- 2.7a: `src/vjepa_physics/joined.py`; smoke test (terminal, not saved): all three tables built from hash-guarded
  sources, flag counts = F-72, tracked-vs-predicted max px = F-70 per dataset (F-84). **Step 2.7a done.**
- 2.7b: `scripts/check_joined.py build` passed, all values as predicted (F-85). **Step 2.7 done.**
- 2.8: `storage` passed (F-86). Claude Code's peak-RSS prediction (1.5–2 GB) was low: 2.43 GB observed, still 14% of
  physical memory. **Step 2.8 done.**
- 2.9a: `check_evidence.py rerun_identical_splits_extraction` added (existing `rerun_identical` untouched, so F-74 still
  describes unchanged code): `extract_*` keys must stay byte-identical including provenance (not re-run, D-41);
  `verify` must be re-run; only `storage`'s `free_disk_bytes` and `memory` are excluded as machine state. Committed
  `910a798`, tree clean.
- 2.9b: the 6 re-runnable keys re-run from clean HEAD `910a798`, all passed (user said; terminal output not available);
  `rerun_identical_splits_extraction` passed, read by Claude Code from the saved result (F-87).
- 2.9c: report `results/splits_and_extraction/report.md` written by the user from Claude Code's draft (saved evidence
  only; documents how representations are extracted and pooled, as the README asks). **Step 2.9 done — Phase 2 passed.**
- **Next:** Phase 3, step 3.1 (O-05 probe family) with the planning chat; pending notes for it: two split-balance notes
  (F-79) and the own-addition criteria listed in F-78, F-81, F-83, F-86.

### 2026-09-24 — Phase 1 started (data audit)
- Working style reconfirmed (user said): Claude Code edits only `docs/` and `CLAUDE.md` (directly, after each step);
  asks before every step or sub-step; does not re-read user-saved files unless there is an error or suspicious result.
- Phase 1 script layout proposed: `check_data_files.py` (1.1, 1.3), `check_metadata.py` (1.2), `check_design.py`
  (1.4–1.6), `check_videos.py` (1.7, 1.8), `check_tracking.py` (1.9, 1.10); figures as `figure_*` checks; package
  gains `data.py` now, `geometry.py` / `tracking.py` at 1.9.
- 1.1: criteria fixed before the run (D-31; folder naming made a diagnostic, same-folder and inside-the-dataset
  rules added by the user). `src/vjepa_physics/data.py` (`DATASETS`, `read_manifest`, `resolve`);
  `scripts/check_data_files.py manifests` passed on all data (F-60). **Step 1.1 done.**
- 1.2: criteria re-reviewed before the run — speed/acceleration ≥ 0 and motion consistency added, because the first
  set checked presence but not label meaning (D-32). `read_metadata` added to `data.py`; `scripts/check_metadata.py`:
  `consistency` passed; `documented_fields` failed as predicted (direction lacks `primary_label`; note drafted for the
  planning chat); `sorted_by_label` shows all three manifests sorted by label (F-61). **Step 1.2 done.**
- 1.3: `fingerprint` check added to `check_data_files.py` (reference tracked/unchanged, format, file set, hashes via
  Python and `shasum -c`, write bits); passed (F-62). **Step 1.3 done.**
- 1.4: `load_dataset` + `LABEL_FIELD` added to `data.py`; `scripts/check_design.py`: `value_grids` passed (criteria
  from DATA.md only), `design_balance` recorded as a diagnostic (F-63). Exact-zero deviation from a linear grid checked
  and explained (values bit-identical to `linspace`); identical label–angle correlations in speed and acceleration
  explained (same θ by id, labels affine in the value index). **Step 1.4 done.**
- 1.5: diagnostics only, flag rule p < 1e-4 fixed before the runs (D-33); `start_positions` and `label_independence`
  added to `check_design.py`; nothing flagged (F-64). F-26's "< 0.05" slightly exceeded in direction subsets
  (\|r\| ≤ 0.08, n = 750), chance-level. **Step 1.5 done.**
- 1.6: `src/vjepa_physics/geometry.py` (motion formula; functions typed to always return ndarrays after a Pylance
  return-type warning); `distance_confound` diagnostic added to `check_design.py` (F-65). All predictions matched.
  **Step 1.6 done.**
- 1.7: criteria adjusted by the user before the run (D-34). `probe_clip` added (first attempt went into `model.py` by
  mistake and was removed; `model.py` verified clean). `format` failed on `no_uniform_frames`: Claude Code's premise
  (noisy background) was wrong — the background is flat, so exit frames are uniform. Pixel mapping added to
  `geometry.py`; `uniform_frames` diagnostic (rule fixed before the run) narrowly failed on 3 frames; exit-frame residue
  found. `duplicates` passed; first `artifacts/` write confirmed git-ignored. Planning-chat note drafted (re-score
  proposal, residue → O-02). **Step 1.7 run; re-score pending.**
- 1.8: `decoders` check (reuses `vjepa_physics.decoders`, D-05 unchanged): all 4,572 clips ok; the whole-clip "no disk
  pixels" flag never triggers because exit clips show the disk in early frames (Claude Code's earlier worry was wrong)
  (F-68). **Step 1.8 done.** Closes the Phase 0 report item "decoder tolerance not yet applied to all clips".
- 1.9: `tracking.py` added (reproduces F-47/F-48 on the test clip); `track` (one decoding pass, npz artifact with core
  colour), `mapping` (hash-guarded read of the npz; fitted scale/origin; alternatives; angle), `documented_colour`
  (pass/fail by the user's choice; failed as predicted). Claude Code's predicted max error (≈ 0.707 px) was slightly
  low (observed 0.783). **Step 1.9 done.**
- Planning-chat decisions recorded (D-29 narrowed, D-32 confirmed, D-34 closed, D-36 flags, D-37 criteria authority).
- 1.10: `flags.py` (`clip_flags`, reproduces the test clip) and `flags` check; all integrity criteria passed; flag counts
  equal the metadata predictions; table committed (F-72). **Step 1.10 done.**
- 1.11: `plotting.py` (validated palette, shared axis style); three figure checks. Reviews by Claude Code: contact-sheet
  exit example showed no residue → rule changed to "most residue" (id 1152, caveat recorded); design heatmap tick labels
  collided and "m/s^2" → fixed; tracking figure first run failed on a mask-shape bug (Claude Code's) → fixed. **Step 1.11
  done.**
- 1.12 (docs only, Claude Code): F-21–F-36 reviewed one by one against F-60–F-73; F-33's "1–7 frames, 40 ≥ 4" counted
  in `clip_flags.csv` (Grep on the committed table); F-34's two threshold counts derived from the verified formula;
  unverified remnants recorded (F-34 "≤ 3 distinct positions", F-36 default-preprocessing counts); status tags in
  DECISIONS §1.3/§1.4 and CLAUDE.md §5 updated; D-14 annotated. **Step 1.12 done.**
- 1.13: `rerun_identical` added to `check_evidence.py` and committed before the re-run; all 21 keys re-run from clean
  HEAD `7869969` (18 exit 0, the 3 recorded failures exit 1); `rerun_identical` passed (21 keys identical, outputs
  unchanged); npz determinism confirmed from numpy/CPython source; report `results/data_audit/report.md` written by the
  user from a Claude Code draft. **Step 1.13 done — Phase 1 passed.** Work paused at the user's request.
- **Next:** Phase 2, step 2.1 — settle the split details with the planning chat (inputs listed under "Open issues").

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
- `scripts/check_benchmark.py benchmark` (diagnostic): batching gives no speed-up on MPS (0.84 s per clip at batch 1),
  batched pooled outputs bit-identical at 2/4/8, extraction ~65 min, memory well within limits (F-59). **Step 0.16 done.**
- Doc workflow changed (user said): doc updates now applied directly after each step, no chat preview (D-22 amended).
- Planning-chat decisions applied: D-28 (MPS fp32 accepted, closes O-19), D-29 (0.17 skipped, closes O-14), D-30
  (primary steering readout = later-layer probe; predictor readout optional; closes O-20, narrows O-07); D-05 tolerance
  criterion for step 1.8. Corrections: F-54 ranges, D-27 old-semantics wording, `dirty_flag` terminal-only result,
  evidence table blank lines removed, CLAUDE.md §5 (predictor line, decoder agreement).
- 0.18 in progress: `opencv_tolerance` passed (verdict ok); everything committed (`5dd1385`, tree clean; this also
  added `intervention.py` / `check_intervention.py`, which the 0.14 commit had missed). Clean re-run of all 33 check
  keys: 31 exit 0, `opencv` and `devices` exit 1 as expected. `git diff results/` vs `5dd1385`: every deterministic
  value identical; only provenance (incl. old-format `versions` blocks), timings and `real_repo_code_changes_now` →
  `[]` (now saved) changed; both figures byte-identical. Observation: `forecast` full-encode median 0.93 → 1.28 s
  (+38%) when run last after ~15 min of GPU load, while `benchmark` earlier in the loop stayed at 0.85 s (thermal
  throttling is a hypothesis, untested) → extraction may take ~90 min under sustained load, not ~65.
- Re-run results committed. O-18 confirmed on source files the user downloaded from GitHub `main` (F-58 verified:
  HF ships the target/EMA encoder; the conversion script itself asserts parity with the original at atol 1e-3). O-18
  closed.
- Setup report written to `results/setup/report.md` (named without "phase" per D-23). **Step 0.18 done; Phase 0
  passed** with open items listed in the report.
- **Next:** Phase 1 (data audit), step 1.1 — waiting for the user's go.

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
