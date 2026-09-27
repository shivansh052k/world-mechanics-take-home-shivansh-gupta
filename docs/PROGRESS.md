# PROGRESS.md — where we are right now

Last updated: 2026-09-27.

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
penalties); m-dim linear erasure; nonlinear recovery after erasure. **Phase 5 (multi-probe steering), 5.1–5.10 done
and gate evidence complete (2026-09-26):** D-50, F-129–F-138. Headline: an independent readout at the steering layer
(idx 9) follows the edit (error reduction 0.95 / 0.95 / 0.92 at n = K−1; 0.11 / 0.55 / 0.36 at n = 1), random
edits do nothing, but a readout nine blocks later (idx 18) moves only 0.029 / 0.129 / 0.136; the effect collapses within
three blocks, the carried edit is only partly aligned with the later readout and block updates push back 48–73 % of
it; H-12's mechanism not supported as tested (kernel readout follows at idx 9). **Phase 5 passed (2026-09-26):**
report `results/steering/report.md`. **Phase 6 (spline steering, Part 2) passed (2026-09-27):** D-51, F-139–F-148;
report `results/spline_steering/report.md`. Headline: the representation is a curved, low-dimensional manifold (curve
beats line on held-out values for all three; curvature 77–87 % of the gain; 6–8 dims); isometry holds as Goodfire
reports but adds nothing beyond label order (speed agrees locally); spline steering stays natural at the steering
layer (direction chord midpoints bimodal in 47 % of far targets) but collapses downstream exactly like the covariance
line and probes (idx 18: 0.07–0.14, spline and covariance within 0.005); a time-structured edit helps slightly
(+0.01–0.03), not confirmed. **Phase 7 (confounds and robustness) passed (2026-09-27):** D-52, F-149–F-157; report
`results/confounds/report.md`. Headline: the linear speed / acceleration probes read distance travelled (cross-applied
slopes ≈ 1; H-03 supported for the readouts) while the representation separates the motion profiles at matched distance
(0.95 / 0.98 at idx 9 / 18); direction shared across motion types from idx 9 (test agrees); H-04 not supported beyond a
~1° clipped-clip excess; H-05's binary reading rejected; speed–acceleration subspaces aligned, direction at chance (the
paper's C.4 result); small direction → speed / acceleration steering cross-talk. **Next:** Phase 8 (presentation and
final delivery).

---

## Phase status

| Phase | Status | Notes |
|---|---|---|
| 0 — Environment and model setup | ✅ Passed gate | 0.1–0.16, 0.18 done; 0.17 skipped (D-29); report `results/setup/report.md` (open items listed there) |
| 1 — Data audit | ✅ Passed gate | 1.1–1.13 done; report `results/data_audit/report.md`; clean re-run identical (`documented_fields`, `documented_colour` failed as predicted; `format` / `uniform_frames` failed, closed without re-score, D-34) |
| 2 — Splits and activation extraction | ✅ Passed gate | 2.1–2.9 done (D-38–D-41, F-75–F-87); report `results/splits_and_extraction/report.md`; clean re-run identical; balance notes closed (D-44) |
| 3 — Part 1a: Layer-wise probing | ✅ Passed gate (2026-09-25) | 3.1–3.9 done (F-88–F-115; D-42–D-48); transition idx 1 for all three (mean-pooled) and for per-patch direction; test once (F-104, F-113); hash-check gate (F-115); report `results/layer_probing/report.md`, slide log `results/slide_log.md`, talk outline `slides/talk_outline.md` |
| 4 — Part 1b: Iterative nullspace probing | ✅ Passed gate (2026-09-26) | 4.1–4.8 done (D-49 + addenda; F-116–F-128); K idx 9 = 6 / 7 / 7, procedure-dependent (F-120); linear erasure with m dims, nonlinear recovery (F-119–F-123); test once (F-127); hash-check gate, 2 keys re-run identical (F-128); five failures kept on record; report `results/nullspace/report.md`, figures `nullspace_rounds.png`, `erasure_and_procedure.png` |
| 5 — Part 1c: Multi-probe subspace steering | ✅ Passed gate (2026-09-26) | 5.1–5.10 done (D-50; F-129–F-138); steer idx 9, read idx 18 via bit-exact partial forward; headline K−1 idx-18 reduction 0.029 / 0.129 / 0.136 vs same-layer 0.95 / 0.95 / 0.92; post-hoc profile, decomposition, specificity, kernel; gate: 46 keys traced, setup and cache re-run identical (F-138); figures `steering_reduction.png`, `steering_propagation.png`; report `results/steering/report.md`; slide log +2; talk outline item 6 done |
| 6 — Part 2: Spline steering | ✅ Passed gate (2026-09-27) | 6.1–6.17 done (6.10, 6.11 skipped; D-51; F-139–F-148); Q1 curved (LOCO, ladder), Q2 isometry vs label null, Q3 spline vs covariance vs probes over idx 9–18 + naturalness + held-out targets, Q4 time-structured nudge (not confirmed); gate: 68 keys, none to re-run; `comparison_table.md`, figures `manifolds.png`, `spline_paths.png`, `spline_profile.png`; report `results/spline_steering/report.md`; slide log +4; talk outline item 7 done |
| 7 — Confounds and robustness | ✅ Passed gate (2026-09-27) | 7.1–7.7 done (D-52; F-149–F-157); no new forward passes; test once for 7.2 and 7.3 after freezes (F-152, F-155); 7.3–7.5 criteria re-drafted (brief text lost), approved before runs; gate: `code_hash_check` 77 keys, none to re-run; figures `confounds.png`, `tubelet_scatter.png`; report `results/confounds/report.md`; slide log +1 main, +3 backups; talk outline item 8 done |
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
| `results/videos/checks.json` | `format` | All clips decode; 256×256, 24 fps, constant 1/24 s step (from t = 0); no black frames; frame medians all (29, 32, 29); 113 direction clips with disk-less frames | ❌ failed: `no_uniform_frames` — 185 uniform frames in 52 direction clips (disk has left); closed without re-score, failure kept on record (D-34) | F-66, D-34; `scripts/check_videos.py format` |
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
| `results/steering/checks.json` + `artifacts/steering/setup.npz` | `steering_setup` | Clips (15 + 15 per variable, seeded, spread), targets (5, 3 unseen), validation-fit readouts idx 9 / 18 (train R² 0.969–0.987); `abcc701b…` | ✅ passed (saved dirty; re-run identical, F-138) | F-131 |
| `results/steering/checks.json` + `artifacts/steering/cache_*.npy` | `steering_cache` | Per-token block_8 caches of the 90 clips; stored = recomputed, zero edit and two-path bit-exact; unsteered idx-18 R² 0.980 / 0.989 / 0.991; timing 0.66 s | ✅ passed (saved dirty; re-run identical, F-138) | F-130, F-132 |
| `results/steering/checks.json` + `artifacts/steering/runs_*.npz` | `steer_{direction,speed,acceleration}_{seen,unseen}` | Six test runs (1,440 / 1,515 passes each): shifts, fp32 features idx 9–18, all readouts; check (iv) within tolerances | ✅ passed (clean, `80a2670`) | F-133 |
| `results/steering/checks.json` | `steering_scores` | Headline: idx-18 reduction at K−1 0.029 / 0.129 / 0.136 (CIs), same-layer 0.95 / 0.95 / 0.92, n = 1, covariance, random, n = K, specificity, shift sizes | ✅ passed (test) | F-134 |
| `results/steering/checks.json` | `steering_propagation` | Post hoc: readouts idx 9–18, gain profile, direct / block decomposition with CIs, gain vs shift size | ✅ passed (saved dirty; code matched `e6aebfc`) | F-135 |
| `results/steering/checks.json` | `steering_specificity` | Post hoc: signed speed ↔ acceleration slopes in metres 1.004 / 0.940 at idx 18 | ✅ passed | F-136 |
| `results/steering/checks.json` | `steering_kernel` | Post hoc: H-12 RBF readout at idx 9 / 18 — "not supported" (kernel follows the edit at idx 9) | ✅ passed | F-137 |
| `results/steering/steering_reduction.png`, `steering_propagation.png` + `checks.json` | `figure_steering` | Reduction vs n; propagation profile and decomposition | ℹ️ visual (reviewed) | F-138 |
| `results/evidence/checks.json` | `code_hash_check`, `rerun_identical_steering` | Phase 5 gate: 46 keys, 44 matched; setup and cache re-run identical incl. artifact hashes | ✅ passed | F-138 |
| `results/manifolds/checks.json` + `artifacts/manifolds/curves.npz` | `manifold_loco` | Q1: LOCO selection at idx 1 / 9 / 18; curve beats line (CIs > 0), 12 / 12 val-unseen; line = covariance map ≤ 7e-15; `b1a00661…` | ✅ passed | F-139 |
| `results/manifolds/checks.json` | `speed_acceleration_manifold` | Speed vs acceleration on a distance scale: two offset, differently shaped curves | ℹ️ observation (passed) | F-141 |
| `results/manifolds/checks.json` | `manifold_ladder` | Line → free-spacing (PC1 / B) → curve; curvature 77–87 % | ✅ passed | F-140 |
| `results/manifolds/checks.json` | `manifold_dimension`, `direction_harmonics` | H-01 k vs K·m, participation ratio; H-02 harmonic spectrum and round blocks (permutation null flawed, kept on record) | ℹ️ observation (passed) | F-142 |
| `results/manifolds/manifolds.png` + `checks.json` | `figure_manifolds` | Curves in PC view + Q1 ladder | ℹ️ visual (reviewed) | F-148 |
| `results/behavior/checks.json` + `artifacts/behavior/readouts.npz` | `behavior_readouts` | 16-bin readouts idx 9–18, behavior curves, natural distances, two-stage readout; `d038fbea…` | ❌ failed own criterion "≥ 10 validation clips per bin" (direction min 7, cause recorded); other 10 passed | F-143 |
| `results/behavior/checks.json` | `isometry`, `isometry_local` | Q2: split-half isometry vs label null; local speed (post hoc) | ✅ passed | F-144 |
| `results/spline_steering/checks.json` + `artifacts/spline_steering/setup.npz` | `spline_setup` | All Phase 6 edits; covariance f = 1 = Phase 5 bit for bit; `cf0a2d19…` | ✅ passed | F-145 |
| `results/spline_steering/checks.json` + `artifacts/spline_steering/runs_*.npz` | `spline_{direction,speed,acceleration}_{seen,unseen}` | Six runs (6,240 passes); Phase 5 covariance replicated bit for bit | ✅ passed (clean, `48308a0`) | F-145 |
| `results/spline_steering/checks.json` | `spline_scores` | Q3 progress idx 9–18, headline pairs, Q4, C11; Phase 5 reproduced to 4.4e-16 | ✅ passed | F-146 |
| `results/spline_steering/checks.json` | `spline_naturalness`, `spline_held_out` | Q3 naturalness (idx 9 / 18); seen vs unseen targets and clips | ✅ passed | F-147 |
| `results/spline_steering/comparison_table.md` + `checks.json` | `comparison_table` | 6.12–6.14 table from 11 clean sources | ✅ passed | F-148 |
| `results/spline_steering/spline_paths.png`, `spline_profile.png` + `checks.json` | `figure_spline_paths`, `figure_spline_profile` | Path figure (Q3), propagation profile + Q4 | ℹ️ visual (reviewed) | F-148 |
| `results/evidence/checks.json` | `code_hash_check` | Phase 6 gate: 68 keys, all matched, none to re-run | ✅ passed | F-148 |
| `results/spline_steering/report.md` | — | Part 2 report: Q1–Q4, Goodfire comparison, strengths / limitations / failures, kept on record | ✅ gate passed | F-148; user-written from Claude Code's draft |
| `results/confounds/checks.json` | `cross_applied_probes` | Speed / acceleration probes applied across sets in the F-65 window: slopes ≈ 1 on true distance → "reads a speed / distance quantity"; τ/T ≈ 0.5 | ✅ passed | F-149 |
| `results/confounds/checks.json` | `matched_distance_classifier` | Set classifier at matched distance: control 0.500; idx 9 0.947, idx 18 0.983 → profile separable | ✅ passed | F-150 |
| `results/confounds/confounds.png` + `checks.json` | `figure_confounds` | Main-slide figure (cross-applied readings; classifier by layer); refits reproduce saved predictions and slopes | ℹ️ visual (reviewed) | D-52 |
| `results/robustness/checks.json` + `artifacts/robustness/motion_type_predictions.npz` | `motion_type_transfer` | Direction transfer between motion types, validation; readings per idx 1 / 9 / 18 | ✅ passed | F-151 |
| `results/robustness/checks.json` | `motion_type_test` | One-time test of the transfer; validation reproduced bit for bit; readings = validation | ✅ passed (committed code) | F-153 |
| `results/robustness/checks.json` | `flag_breakdown` | Stratified flag errors and sub-patch trend on validation (idx 1 / 9 / 18) | ✅ passed | F-154 |
| `results/robustness/tubelet_scatter.png` + `checks.json` | `figure_tubelet` | Error vs mean within-tubelet px, idx 9; Spearman observations | ℹ️ visual (reviewed) | F-154 |
| `results/robustness/checks.json` | `flag_test` | One-time test read of the flag breakdown + Phase 5 steering by flag; validation reproduced exactly first | ✅ passed (committed code) | F-156 |
| `results/robustness/checks.json` | `subspace_overlap` | C.4 subspace metrics with both nulls (common + raw space), reverse validity, direction specificity | ✅ passed | F-157 |
| `results/evidence/checks.json` | `code_hash_check` | Phase 7 gate: 77 keys, all matched, none to re-run | ✅ passed | D-52 |
| `results/confounds/report.md` | — | Confounds and robustness report: 7.1–7.5, strengths / limitations, kept on record, open items | ✅ gate passed | D-52; user-created from Claude Code's draft |
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

Known decision points the plan cannot remove in advance:
- (i) Uniform edits collapse by idx 12 (F-135), so Phase 6 is scored over idx 9→18, not idx 18 alone.
- (ii) Speed/acceleration centroid curves may be near-linear, making spline ≈ covariance line (predicted on train
  before any test run). **Resolved (6a-2):** the planner's prediction was wrong (it was conditional on a near-linear
  LOCO result, not met) — all three are nonlinear in the label; the line → free-spacing line → curve ladder (6a-4)
  splits uneven spacing from curvature; spline and covariance endpoints are expected to differ.
- (iii) Time reserved for slides + two rehearsals (8.13).
- (iv) One-line note for the planning chat (not blocking): direction → speed / acceleration steering cross-talk beyond a
  random edit (F-157), small; no decision affected.

---

## Log

Newest entry on top. One entry per work session: what was done, what passed, what didn't,
what's next.

### 2026-09-27 — Phase 8 started (presentation and final delivery)
- Order proposed (8.1 → 8.9 → 8.3 → 8.2/8.4 → 8.6 → 8.7 → 8.10 → 8.11/8.12 → 8.13 → 8.15 → 8.14, user sends); open
  choices: slide tool, README placement, deadline.
- 8.1: `check_data_files.py fingerprint_final` (new key = the same `check_fingerprint`, keeps F-62's record;
  `require_clean_code` added) **passed** 6/6: 9,147 files, every SHA-256 and `shasum -c` match, none added / removed,
  `data/` read-only; only byte-identical pair = the speed / acceleration manifests (= F-62). **Step 8.1 done.**
- 8.9: requirements checklist written under F-01–F-09 in DECISIONS §1.1 (README checked line by line): every analysis
  requirement met with evidence; open items only in Phase 8 (Meta-repo parity skip to state, D-29; extraction / pooling
  slide + README section; slides, rehearsal, sending; code README and packaging). **Step 8.9 done.**
- 8.3: `slides/talk_outline.md` replaced with the final slide-by-slide plan (15 main slides, 14:15 + 0:45 buffer;
  isometry → backup; new protocol slide for extraction / pooling; conclusions + limitations; 6 backups; full limitations
  list for Q&A). **Step 8.3 done.**
- User choices (2026-09-27): slides in **PowerPoint** (PDF export, D-48); **≤ 1 day** to submission → lean Phase 8
  (one rehearsal, condensed defence prep; 8.6 limited to what the README needs).
- Code submission plan (user approved, 2026-09-27; slides paused until it is done): **minimal submission**. Task
  `README.md` / `DATA.md` untouched at the root; `submission/` holds only our one-page `README.md` and `slides.pdf`;
  code stays in place (moving it would break the recorded artifact paths and code hashes). Sent: `src/`, `scripts/`
  (+ new `run_pipeline.py`), new `tests/`, `results/` (figures, `checks.json`, `clip_flags.csv`),
  `artifacts/manifests/`, `pyproject.toml`, `requirements.lock.txt`. **Kept local, untracked:** `docs/`, `CLAUDE.md`,
  `.claude/`, all `results/*/report.md`, `results/slide_log.md`, `slides/talk_outline.md` (prep material for the slides
  and the next round). Steps C1 pipeline map → C2 runner → C3 light cleanup + identical-results re-check → C4
  dependencies → C6 tests → C5 `submission/README.md` → C7 packaging → slides (8.4).
- C1: pipeline map approved after two review passes. 147 `CHECKS` keys = 141 in 9 stages (setup 32, audit 21,
  extraction 9, probing 18, nullspace 15, steering 13, spline 22, confounds 9, final 2) + 6 excluded (`benchmark`,
  5 `rerun_identical*`); every cross-key read (`verified_artifact` and saved-JSON reads) traced to an earlier producer.
  Runner rules: continue past failures, compare each new `passed` with the committed record, stop only on a crash (no
  new record); scripts need a git clone (the zip is for browsing); setup-stage flags expected on other hardware;
  `figures` = 15 keys (14 `figure*` + preprocessing `default`), most need stage 1–2 artifacts. **Step C1 done.**
- C2: `scripts/run_pipeline.py` (stages as data; `--list`, `--stage … | all | figures`, `--dry-run`; expected outcome
  = committed `passed`; crash = no new UTC record → stop). Smoke test (dry run, nothing written): stage counts as
  predicted; 0 keys "not recorded" (no typos); first dry run exposed a bug (`all` swallowed `figures`), fixed →
  156 steps = 119 passed / 28 no verdict / 9 failed on record: video_loader opencv, numerics devices, metadata
  documented_fields, videos format, tracking documented_colour, patches patch_probes, nullspace nullspace_rounds,
  nullspace kernel_erasure, behavior behavior_readouts (the last missed in Claude Code's from-memory list; F-143).
  **Step C2 done** (committed with C3).
- C3: `ruff check src scripts --select F` (pyflakes only; ruff installed into `.venv`, not in the lock): 2 unused
  imports (`check_behavior.py` PERIOD, `check_manifolds.py` GRID) + duplicate `PAIR` in `confounds.py` removed; rerun
  proof replaced by import smoke test (edits provably behaviour-neutral): ruff clean, all four scripts import. Commit
  `065a052` (runner + cleanup). **Step C3 done.**
- C4: `pyproject.toml` declares the 10 direct dependencies pinned to the result-producing versions (= provenance
  versions) + `dev` extra (pytest, ruff); `pip install -e ".[dev]" --dry-run` would add only vjepa-physics, pytest,
  iniconfig, pluggy (every pin already satisfied); `pip check` clean. **Step C4 done** (committed with C6).
- C6 batch 1: `tests/test_metrics.py`, `test_geometry.py`, `test_confounds.py`, `test_robustness.py` (sklearn, hand
  cases, kinematics, analytic k/d null) + pytest `testpaths`; **20 passed in 3.96 s** (as predicted).

### 2026-09-27 — Phase 7 (confounds and robustness) — passed
- Gate steps 4–5: slide log +1 main ("The probes read distance, yet the motion profiles stay separable") +3 backups;
  talk outline item 8 done, limitations extended; full docs pass (D-52, plan 7.1–7.7 ticked, O-12 closed, H-03 / H-04 /
  H-05 statuses, phase table, Saved evidence, CLAUDE.md §5 / §7). **Phase 7 passed.** Next: Phase 8.

### 2026-09-27 — Phase 7 started (confounds and robustness)
- Phase 7 layout (`confounds.py`, `robustness.py`; `check_confounds.py`, `check_robustness.py`; subjects `confounds`,
  `robustness`; CPU only on stored pooled activations, saved predictions and Phase 5 runs, ≲ 15 min total) and one
  design brief for 7.1, 7.2, 7.3 + 7.4, 7.5 drafted (pre-stated criteria and reading rules; 5 planner questions).
  **Next:** brief to the planning chat; first step = `confounds.py` part 1 (distance scale, F-65 window) + smoke test.
- Planning chat approved the brief with changes (→ D-52 at the phase docs pass): 7.1a slope rule replaced — speed-at-τ
  readings (acceleration probe on speed clips slope T/(2τ): CI < 0.25 "reads acceleration", CI > 0.5 "reads a
  speed/distance quantity", else "mixed"; speed probe on acceleration clips τ/T = slope/2, no verdict; slope 1 = distance
  = mean = mid-clip speed); 7.1b control (i) criterion, idx 0 = observation; 7.2 band "partly shared" (0.5–0.9), test
  once idx 1/9/18; 7.3 test breakdown from F-104 as confirmation, H-05 scatter with/without exit; 7.5 paper C.4 metrics
  (mean principal angle, projection overlap, Grassmann distance), both nulls (train-span, k_A/d), weights with weights,
  patterns with patterns, raw-space angles too, reverse validity check; slides: one main + backups. Claude Code note: a
  pure final-speed reading (slope exactly 0.5) falls in "mixed" under the CI > 0.5 rule (user: keep the rule, τ/T
  reported alongside).
- 7.0a: `src/vjepa_physics/confounds.py` (`CLIP_SECONDS`, `clip_distance`, `overlap_window`, `in_window`). Smoke test
  (terminal, not saved): T 0.625; distance = label × per-unit bit for bit (speed, acceleration); window [0.15625,
  1.953125] = F-65; in window speed 1,176 clips / 49 values (train 640, val_seen 160, val_unseen 72, test_seen 160,
  test_unseen 144), acceleration 1,440 / 60 (768 / 192 / 96 / 192 / 192); direction distance 0.390625–4.375 = F-65. All
  predictions met. **Step 7.0a done.**
- 7.0b: `confounds.py` part 2 (`DISTANCE_PER_UNIT`, `READS_ACCELERATION_BELOW` 0.25 / `READS_SPEED_ABOVE` 0.5,
  `as_distance`, `expected_slope`, `tau_fraction_from_slope`, `reading_verdict`, `matched_value_pairs`). Smoke test
  (terminal, not saved): hand cases exact; synthetic speed-at-τ readings on real clips give the predicted slopes (1 / 0.5
  acceleration probe, 0 / 1 / 2 speed probe) within 2e-15; verdict rule as stated; window pairing 49 speed values ↔ 49 of
  60 acceleration values, one to one, max gap 0.0144 m (≤ half an acceleration step 0.0151), median 0.0074, 11 unpaired.
  All predictions met. **Step 7.0b done.**
- 7.1a: `scripts/check_confounds.py cross_applied_probes` **passed** (8/8, clean commit; refits = `layer_curves` alpha
  and validation predictions exactly). Window validation clips 232 speed / 288 acceleration. Slope of cross-predicted on
  true distance (m) [95 % CI], idx 9 / 18: **acceleration probe on speed clips 0.959 [0.940, 0.978] / 1.004 [0.988,
  1.020] → "reads a speed / distance quantity"** (τ = T / (2 slope) ≈ mid-clip; R² vs distance 0.977 / 0.981; intercept
  0.042 / −0.024 m); **speed probe on acceleration clips 1.016 / 0.991 → τ/T 0.508 [0.499, 0.517] / 0.495 [0.486,
  0.504]** (R² 0.973 / 0.940; intercept 0.016 / 0.115 m). Own-set slopes in the window 0.987–1.017. Idx 1: acceleration probe on speed
  clips 1.016, speed probe on acceleration clips 0.991 (same as 9 / 18). Both probes, applied out of distribution, read distance = mean = mid-clip speed → H-03 supported for
  the linear readouts. Claude Code's predictions: criteria ✓, verdict ✓, slope 0.6–1.3 ✓, τ/T 0.3–0.7 ✓, own slope
  0.95–1.02 ✓; "idx 1 closest to 1" not borne out (all ≈ 1). **Step 7.1a done.**
- 7.1b-1: `confounds.py` part 3 (`clip_pairs`, `shared_pair_rows`, `pair_weights`, `weighted_balanced_accuracy`). Smoke
  test (terminal, not saved): weighted balanced accuracy = sklearn to 5.6e-17; `roc_auc_score` takes `sample_weight`; 49
  pairs, 1,176 + 1,176 clips; train 32 pairs, 512 + 512 clips, weight sums 32 / 32, mean-distance gap 0.0003 m;
  validation 37 pairs, 208 + 188 clips, weight sums 37 / 37, gap raw 0.037 → weighted 0.0004 m. All predictions met.
  **Step 7.1b-1 done.**
- 7.1b-2: `check_confounds.py matched_distance_classifier` **passed** (6/6, clean commit). Distance-only control:
  upper-edge alpha (`no_signal`), balanced accuracy 0.500 (constant prediction), AUC 0.4996 → matching works. Weighted
  balanced accuracy by index: 0 0.509, 1 0.765, 2 0.855, 3 0.90, 6 0.93, 9 0.947, 12 0.98, 15–18 0.983–0.986, 24 0.969,
  final norm 0.973. **Idx 9: 0.947 [0.920, 0.971], AUC 0.990 [0.983, 0.996]; idx 18: 0.983 [0.967, 0.997], AUC 0.999
  [0.997, 1.0] → "motion profile linearly separable at matched distance"** (with 7.1a: the linear speed / acceleration
  probes read distance, while the representation also carries the profile; F-141 confirmed per clip). Idx 0 ≈ chance
  (observation). Claude Code's predictions: criteria ✓, control 0.45–0.58 ✓, idx 9 / 18 ✓, idx 0 ✓; idx 1 ≥ 0.8 **wrong**
  (0.765: profile separability grows with depth, unlike the probes' idx-1 transition). **Step 7.1b done.**
- 7.2a: `src/vjepa_physics/robustness.py` (`MOTION_TYPES`, `motion_masks`, `distance_overlap` = F-65's rule via
  `confounds.overlap_window`, replacing the brief's hand-picked groups). Smoke test (terminal, not saved): 750 / 750;
  roles velocity train 397 / val_seen 107 / val_unseen 48 / test_seen 107 / test_unseen 91, acceleration 416 / 96 / 46 /
  96 / 96; exit 113 / 0; overlap [0.625, 1.953] m = velocity 1–3 m/s (322) + acceleration 4–10 m/s² (600), 2 m/s²
  excluded, no exit clip. Prediction "≈ 406 train each" missed: velocity cells are smaller (13–14 vs 18–20 clips, F-63),
  so D-38's per-cell floor + remainder allocation gives them a larger validation / test share (explanation by
  arithmetic, not re-derived). **Step 7.2a done.**
- 7.2b: `scripts/check_robustness.py motion_type_transfer` **passed** (7/7, clean commit; artifact
  `motion_type_predictions.npz`). Across R² v → a / a → v: idx 1 0.710 / 0.607 ("partly shared" both); idx 9 0.931
  "shared, partly type-specific" / 0.962 "shared"; idx 18 0.972 / 0.951 "shared, partly type-specific" (a → v gap 3.6°,
  CI [2.6, 4.6]). Distance overlap removes idx 1's v → a gap, not idx 18's a → v gap. Claude Code's predictions: criteria
  ✓, idx 9 / 18 across 0.93–0.98 ✓, idx 1 0.6–0.9 ✓, overlap gap shrinks ✓ (mostly); idx 9 a → v "shared" (predicted
  partly type-specific; flagged as uncertain). Validation findings frozen as F-149–F-152 in DECISIONS §1.13 (early, for
  the test-once rule). **Step 7.2b done.**
- 7.2c: `check_robustness.py motion_type_test` (one-time test, F-152 scope) **passed** (4/4; validation reproduced bit
  for bit). All six readings = validation; idx 18 a → v gap 4.1° [3.2, 4.9], survives distance matching; idx 1 a → v
  across R² 0.505 (validation 0.607). Claude Code's predictions: criteria ✓, readings ✓ (6/6), idx 9 / 18 within ±0.03 ✓,
  idx 1 within ±0.08 missed for a → v (−0.10). F-153 added. **Step 7.2 done.**
- 7.3 criteria (re-drafted; the original brief text was lost; Claude Code's own additions, D-37; user approved; fixed
  before any run). `sub_patch_motion` is label-determined (metadata, `flags.py`), `frozen_start` is not (tracked centre).
  Correctness: artifacts via `verified_artifact`, ids aligned; `within_tubelet_px` = flags table columns to its 4-decimal rounding (≤ 5e-5 px); recomputed
  errors = saved `layer_curves` scores ≤ 1e-12 (field confirmed at 7.3b); flag counts per role = joined table.
  (A) stratified flagged − unflagged error (10,000 resamples, idx 1 / 9 / 18, validation): direction exit / clipped /
  frozen_start within motion group; acceleration frozen_start within label value; CI low > 0 "excess error beyond
  stratum", CI ∋ 0 "no excess", CI high < 0 "lower"; raw difference and n per stratum observations. (B) H-05: speed /
  acceleration sub-patch clips, `within_range_trend`: slope CI > 0 and Spearman CI > 0 "continuous inside the sub-patch
  range", slope CI ∋ 0 "consistent with a binary detector"; full-range slope, per-value errors, relative error flagged
  vs unflagged ("confounded with label value") observations. (C) figure: error vs mean tubelet px at idx 9, direction
  with / without exit; Spearman on absolute and relative error observations. (D) Phase 5 per-clip reduction by flag,
  observation (Phase 6 omitted: spline = covariance within 0.005 downstream). Test later: same A / B on F-104's saved
  test predictions, labelled confirmation after the validation F-entry, no selection. No planner questions.
- 7.3a: `robustness.py` part 2 (`clip_errors`, `within_tubelet_px`, `stratified_difference`, `within_range_trend`).
  Smoke test (terminal, not saved): tubelet px = flags table ≤ 4.97e-5 (CSV rounding), ids match; hand case 5.0, strata
  A / B, n 3 / 3, no-flag case raises; slope = polyfit 1.8e-15, Spearman = scipy 1.1e-16; flat case CIs contain 0;
  clip errors = `circular_mae` / abs exactly. Prediction "other seed differs" missed: the hand case's resampled
  statistic takes 7 discrete values and both seeds hit the extremes, CI [4, 6] = 2/3 · {1, 4} + 10/3 exactly
  (arithmetic; not a bug). **Step 7.3a done.**
- 7.3b criteria refined before the run (Claude Code, stated to user): "flag counts per role = joined table" was circular
  → flag totals per dataset = F-72's counts; criterion 3 field confirmed (`layer_curves` saves `r2` + `mae` /
  `circular_mae` per role; recomputed and per-clip-error means ≤ 1e-12); trend outcomes outside both rules (e.g. slope
  CI > 0 but Spearman CI ∋ 0, or slope CI < 0) read "neither rule met".
- 7.3b: `check_robustness.py flag_breakdown` **passed** (5/5, clean commit `f0cde42`). Validation flagged: direction exit
  17 / clipped 41 / frozen 15; acceleration frozen 56 (40 vs 60 in eligible values). (A) stratified difference [95 % CI],
  idx 1 / 9 / 18: direction exit +3.07° [−1.83, 8.40] / −0.62 / −0.53 → "no excess" at all three; clipped +4.06
  [0.85, 7.48] "excess" / +0.72 "no excess" / **+1.24 [0.33, 2.22] "excess"**; frozen −12.8 [−26.2, −0.47] "lower"
  (raw +4.45; 15 vs 13 clips, fragile) / "no excess" / "no excess"; acceleration frozen within value "no excess" at all
  three (−0.05 to +0.02 m/s²). (B) sub-patch trend "continuous inside the sub-patch range" for speed and acceleration
  at all three (slope inside / full: speed 0.68 / 1.01, 0.87 / 1.00, 0.95 / 1.00; acceleration 0.58 / 1.00, 0.86 /
  0.99, 0.98 / 0.99); relative error sub-patch vs rest 3–6× (speed 0.18 / 0.058 at idx 1 → 0.13 / 0.032 at idx 18;
  acceleration 0.39 / 0.064 → 0.16 / 0.038; confounded with label value); acceleration frozen-start trend slope 0.86 /
  0.97 / 0.98 (observation). H-04 not supported for exit (validation n = 17); clipped carries a small excess at idx 1
  and 18; H-05's binary-detector alternative rejected. Claude Code's predictions: criteria ✓, trend ✓, acceleration
  frozen ✓, idx 18 exit ✓; idx 1 exit excess ✗ (CI wide), direction frozen "no excess" ✗ at idx 1, clipped excess at
  idx 18 not predicted. **Step 7.3b done.**
- 7.3 order changed (user approved; no criterion changed): Phase 5 steering runs are on test clips (`HALVES`), so D
  moves after the freeze: 7.3c figure (validation) → freeze F-154 → 7.3d one test run from committed code (E: A / B on
  F-104's saved test predictions, labelled confirmation; D: Phase 5 per-clip reduction by flag, post-hoc observation).
  D rule (Claude Code's own addition): stratified difference with CI only if ≥ 5 flagged and ≥ 5 unflagged clips in
  eligible strata, else counts and means, "too few to read".
- 7.3c: `check_robustness.py figure_tubelet` **passed** (3/3, clean commit; re-rendered once with headroom after review:
  ρ note overlapped acceleration points, legend overlapped a direction point). `results/robustness/tubelet_scatter.png`
  reviewed, clean. Validation idx 9, Spearman error vs mean within-tubelet px (observations, confounded with label):
  direction −0.25 (without exit −0.23); speed −0.05, relative −0.55; acceleration +0.22, relative −0.44. Predictions ✓
  (speed "stripes" blur into a continuum: 64 values, 0.33–5.33 px). **Step 7.3c done.**
- Freeze: F-154 (7.3b + 7.3c validation findings) and F-155 (7.3d scope: E on F-104's saved test predictions, D Phase 5
  steering by flag with the ≥ 5 / ≥ 5 rule) written to DECISIONS §1.13 before any test row is read for 7.3.
- 7.3d: `check_robustness.py flag_test` (one-time test read, F-155 scope) **passed** (6/6; validation reproduced exactly
  first). B continuous 6 / 6 = validation; A: clipped excess at idx 18 on both splits (+0.72° test); exit excess only at
  test idx 9 (+2.59°), acceleration frozen excess only at test idx 1 (+0.10 m/s²), direction frozen idx-1 "lower" not
  confirmed → no consistent flag effect beyond clipped; D all "too few to read" / no stratum, reductions flagged ≈
  unflagged. F-156 added. Claude Code's predictions: criteria ✓, B ✓, D ✓, exit idx 18 ✓; exit idx 9 "no excess" ✗;
  test exit count 25–30 ✗ (35). **Step 7.3 (+ 7.4) done.**
- 7.5 criteria (re-drafted, Claude Code's own additions except where planner; user approved; fixed before any run).
  Headline idx 9 (idx 1 / 18 observations). Subspaces (train-fit): weights = Q via `load_probe_sequence`, n = 1 and
  n = K−1 (10 / 6 / 6 dims); patterns = `covariance_basis` (m = 2 / 1 / 1); weights with weights, patterns with
  patterns. Spaces: common standardized (pooled train scaler of all three sets at the site; headline) and raw
  (paper-comparable); re-expression weights × σ_c/σ_own, patterns × σ_own/σ_c, then QR. Pairs speed–acceleration,
  speed–direction, acceleration–direction. Metrics (C.4): mean principal angle, overlap ‖Q_AᵀQ_B‖²_F / dim(B) both
  orders, Grassmann distance. Nulls: k_A/d (d = 1024); train-span null = 1,000 seeded random subspaces of B's size in
  B's own standardized train span, passed through the same transform (corrected in the review pass: a common-space span
  would be unfair). Reading on overlap: > null 97.5th pct "aligned beyond chance", within "at chance", < 2.5th "less
  aligned than chance". Specificity (Phase 5 test runs, by design): direction ↔ speed / acceleration, per-clip mean
  |cross change| idx 18, probes K−1 − random K−1, paired clip bootstrap; CI low > 0 "cross-talk beyond a random edit",
  else "no cross-talk beyond a random edit"; read only if the validity check passes. Validity (validation clips):
  speed / acceleration readouts on direction velocity clips ≤ 4 m/s vs clip distance in metres (planner), direction
  readout on speed / acceleration clips vs θ (own addition); R² ≥ 0.5 "valid". Criteria: hashes; Q orthonormal ≤ 1e-10;
  principal angles from `scipy.linalg.subspace_angles`, SVD-cosine formula agrees ≤ 1e-7 rad (arccos loses precision
  near 0; revised from 1e-10 when writing 7.5a); round-1 prediction invariant under re-expression ≤ 1e-10; covariance
  direct = transformed ≤ 1e-10; null draws in span ≤ 1e-10; full-space draws = k_A/d within 3 SE; specificity =
  `steering_scores` ≤ 1e-12; finite. Tables only (backup slide). No planner questions (pooled scaler = implementation
  detail, both spaces reported).
- 7.5a: `robustness.py` part 3 (`orthonormal`, `rescale`, `re_express`, `principal_angles`, `subspace_metrics`,
  `null_metrics`, `null_reading`). Smoke test (terminal, not saved): hand case exact (angles 0 / 30°, overlap 0.875,
  Grassmann π/6); scipy vs SVD-cosine 2.2e-16 rad; pooled train rows 2,477; K / m 6 / 2, 7 / 1, 7 / 1; Q orthonormal
  ≤ 1.4e-15; round-1 prediction invariant ≤ 1.3e-15; covariance direct = transformed ≤ 4.0e-17; span rank 831, draw
  residual 5.0e-16; full-space overlap 0.00585 vs k_A/d 0.00586 (0.19 SE). Preview (not a result): speed–acceleration
  weights K−1 idx 9 overlap 0.371 (mean angle 54.3°) vs null median 0.0069 → "aligned beyond chance". All predictions
  met. **Step 7.5a done.**
- 7.5b: `check_robustness.py subspace_overlap` **passed** (8/8, clean commit; ~3–5 min CPU). idx 9: speed–acceleration
  aligned beyond chance in every subspace and space (weights K−1 0.371 / 54°, patterns 0.71 / 33°); direction pairs at
  chance for weights (angles 84–89°, = paper C.4), speed–direction patterns beyond chance but tiny (83°); readouts valid
  on the other sets (R² 0.93–0.98); direction steering moves speed / acceleration readouts slightly more than random
  (+0.012 m/s, +0.055 m/s², CIs > 0), speed / acceleration steering does not move direction. F-157 added. Claude Code's
  predictions: criteria ✓, speed–acceleration ✓ (patterns stronger ✓), direction weights at chance ✓, validity ✓;
  speed–direction patterns beyond chance ✗, direction → speed / acceleration cross-talk ✗. **Step 7.5 done.**
- Gate step 1: `HASH_CHECK_SUBJECTS` += `confounds`, `robustness`; `check_evidence.py code_hash_check` **passed** (HEAD
  `24d262f`): 76 keys, all 8 Phase 7 keys match their own clean run commit, `rerun_needed` empty. Predictions ✓.
  Planning-chat note pending (one line): direction → speed / acceleration cross-talk (F-157), small, no decision affected.
- Gate step 2: `check_confounds.py figure_confounds` **passed** (4/4: idx-18 refits reproduce saved validation
  predictions bit for bit and the saved slopes; clean commit). `results/confounds/confounds.png` (main-slide figure:
  cross-applied readings vs true distance, slopes 1.00 / 0.99; set classifier by layer, 0.51 → 0.95 idx 9 → 0.98 idx 18,
  control 0.50) re-rendered twice after review (legend overlapped points; now below the axes), clean.
- Gate step 3: report `results/confounds/report.md` (Claude Code draft, user-created; covers 7.1–7.5, kept on record,
  open items). `code_hash_check` re-run (HEAD `4e1f8a2`): **77 keys, `rerun_needed` empty, passed** (adds
  `figure_confounds`).

### 2026-09-26 — Phase 6 started (spline steering)
- Phase 6 layout (`manifolds.py`, `behavior.py`, spline arms in `steering.py`; `check_manifolds.py`,
  `check_spline_steering.py`) and design brief drafted (idx 9 / idx 18 as Phase 5; spline edit = translate along the
  train-centroid curve; Phase 5 arms re-scored from saved runs; path arms curve vs chord; token-weighted variant for
  H-13; 3 planner questions). scipy 1.18.1 read: `CubicSpline` / `make_interp_spline` periodic = exact interpolation
  only; `make_smoothing_spline` has no periodic option; `make_splprep(bc_type="periodic", s=…)` is the periodic
  smoother; legacy `splprep` limited to ≤ 10 dims. **Next:** brief to the planning chat; first step = scipy spline
  smoke test (terminal, no files).
- Strategy memo → planning chat kickoff (→ D-51 at the phase docs pass; closes O-09, O-11): four questions Q1 geometry,
  Q2 isometry, Q3 held-out steering (spline vs K−1 probes vs covariance line), Q4 H-13 (time-structured edit);
  design C1–C14. Section A doc fixes applied (O-07 cleanup, `format` row, known decision points, plan date, D-48
  superseded note, D-13 cache exception, LEACE check in D-49 c). **Next:** implementation plan (section E).
- Implementation plan approved with five changes (→ D-51): λ = ∞ = explicit least-squares line (line = B ≤ 1e-10
  criterion); waypoint metric = gain with intended_f = f × Phase 5's intended (direction angle gain on the shorter
  arc + output-space gain at f = 1), reduction at f = 1 only; ŷ clamped to the curve range, results with / without;
  isometry on seeded split halves (activation vs behavior curves from disjoint train clips), adjacent-segment lengths
  + all-pairs with label null, chord vs geodesic on all pairs; val-unseen in-sample for the 16-bin readout (behavior
  quality from train LOCO only). Own trig polynomial for direction accepted. Order 6a → Q1 line-vs-curve + C4a to the
  planning chat before 6c. **Next:** 6a-1 `manifolds.py` part 1 + terminal smoke test.
- 6a-1: `src/vjepa_physics/manifolds.py` (`value_centroids`, `centroid_pca`, `Curve`, `trig_design`, open / loop
  curves: LINE = polyfit, EXACT = natural / periodic cubic, `make_smoothing_spline` lam, trig polynomial H;
  `loco_predictions`, `loco_errors`). Smoke test (terminal, not saved; idx 9, train rows): 52 values each, counts 16 /
  16, direction 11–20; centroids = manual 0.0; exact interpolation ≤ 1.3e-14, 51 axes; **line = covariance map B
  ≤ 7.0e-15** (planner criterion 1e-10); line = full-space polyfit ≤ 7.1e-15; batched smoothing = per column 0.0;
  loop wraps ≤ 1.4e-15; trig H = 3 = LinearRegression 1.3e-15; LOCO = brute force 0.0, left-out centroid never seen
  0.0. Observation: median LOCO error (k = 4; line / H = 2) 7.66 / 9.29 / 8.49 vs median adjacent centroid spacing
  9.74 / 8.70 / 9.38 (ratio 0.79 / 1.07 / 0.90; Claude Code's 0.5–1.5 ✓) → centroid noise is of the order of one
  value step. All predictions met. **Step 6a-1 done.**
- 6a-2: `manifolds.py` (`centroid_noise`, `check_values`, `curve_from_pca`, `scored_folds`, `smoothing_grid`,
  `loco_grid`, `select_setting`; `value_centroids(role=…)`; grid k 1…16 + all, open LINE + 13 lam = f × range³,
  loop H 1…12; mean squared LOCO; smallest k then smoothest within 1 %); `scripts/check_manifolds.py manifold_loco`
  **passed** (8/8, clean commit, 26 s CPU; scaler = Phase 5's bit for bit; speed / acceleration line = covariance map
  ≤ 7.0e-15 at all three sites; direction H = 1 vs covariance 2.7–3.4 %). `curves.npz` `b1a00661…`. **Idx 9
  selection:** direction k 8, H 4; speed k 6, lam 0.030; acceleration k 6, lam 0.52 (no grid-end choice). **Line vs
  curve (mean squared LOCO, paired value-bootstrap CI):** direction 144.0 vs 56.4, gap 87.6 [70.8, 105.1]; speed 86.7
  vs 45.5, gap 41.3 [30.9, 53.6]; acceleration 85.5 vs 50.5, gap 35.0 [21.4, 53.5] — **curve beats line for all
  three** (idx 1 and 18 same sign, CIs above 0). Val-unseen (fit on train, 4 held-out values, idx 9): curve beats line
  at 12 / 12 values (direction 27–33 vs 127–166; speed 16–33 vs 38–88; acceleration 15–39 vs 28–101). Exact variants
  1.2–2.2 × the selected LOCO MSE. **Surprise:** selected LOCO MSE below the iid centroid-noise estimate (excess −8.8 /
  −5.1 speed / acceleration idx 9; val-unseen too, where the prediction is independent) → `centroid_noise` (σ²/n) is
  an upper bound, not a floor; hypothesis: nuisances (angle in speed / acceleration, magnitude in direction) are
  balanced within each value by design, so centroids err less than iid sampling implies (observation, untested).
  Claude Code's predictions: criteria ✓; noise 55–90 missed (38–62); speed / acceleration small gap **wrong** (clear);
  direction H 2–4 ✓, k 4–8 ✓; excess |< 30 %| ✓ but sign negative; exact 1.3–2.5 × ✓ (1.2 at speed idx 18 just
  below); val-unseen ≤ 2 × noise ✓. Contradicts planner prediction (ii) (speed / acceleration near-linear).
- 6a-3: `manifolds.unit_tangents`; `speed_acceleration_manifold` (C4a observation, clean `56f119a`) **passed** (4/4):
  window = F-65's [0.15625, 1.953125] m exactly; shared scaler on both sets' window train clips (640 + 768); curves
  in metres re-selected by LOCO: speed 40 values, k 3; acceleration 48 values, k 4 (line vs curve gap in the window
  41.2 [31.1, 52.5] / 12.3 [6.4, 20.5]). **Two separate curves:** centroids vs the other set's curve at the same
  distance MSE 161 / 155 vs own LOCO 51 / 47 (gap 110 [94, 126] / 108 [89, 127]); after removing the mean offset
  (norm 8.2 ≈ one adjacent spacing 9.1) still 94 / 89 (gap 43 [34, 52] / 42 [34, 50]) → offset explains ~60 % of
  the excess, shape differs too. Directions: straight-line slopes 31° apart, tangents at matched distance median 51°
  (19–68°), principal angles 17 / 40 / 80°; each curve turns 102° / 90° from start to end tangent; arc length over the
  common range 56.5 (speed) vs 37.9 (acceleration). Claude Code's predictions: counts ✓, raw gap > 0 ✓, aligned
  shrink ≥ 50 % and > 0 ✓, slope 20–45° ✓, largest principal ≥ 60° ✓; tangent median ≤ 40° **wrong** (51°).
  Q1 + C4a → planning chat before 6c.
- Planning chat on Q1 / C4a (→ D-51): "nonlinear in the label" ≠ "curved" → new no-model key, ladder line →
  free-spacing line (c̄ + u·p(v), u = covariance direction, p by the LOCO rule; nests the covariance arm) → curve, with
  val-unseen; direction ladder ellipse (H = 1) → H = 4, wording "needs harmonics beyond the first-harmonic ellipse";
  final Q1 wording with [X] spacing / [Y] curvature shares. Prediction (ii) recorded as wrong. C4a accepted as an
  observation ("encodes motion profile beyond distance"; never "H-03 does not hold"; H-03 stays for 7.1). Noise:
  no balance diagnostic, stratification = untested hypothesis, iid estimate only as an upper bound, no noise-floor
  claims anywhere. Proceed to 6b; 6c adds endpoint arm (h) free-spacing line (speed / acceleration, +75 passes per
  half). Planner predictions for 6c: (a) − (c) endpoint difference non-zero at idx 9; downstream share still
  collapses by idx 12 unless the spline edit is better aligned with what blocks 9–17 use (H-12, Phase 6 form); no
  confident sign. Planner on the rungs: PC1 free-spacing line (best-fitting straight line) = conservative curvature
  estimate, used for "curved"; B line = steering-relevant split; same LOCO rule on every rung; direction magnitudes only.
- 6a-4: `manifolds.covariance_axis`, `fit_spacing_line`, `loco_spacing_grid` (u, c̄, p refit per fold);
  `manifold_ladder` **passed** (5/5, clean commit; selection and k = 1 row = `manifold_loco`'s; B line with linear
  p = covariance map; spacing grid = brute force). Idx 9, mean squared LOCO line → PC1 spacing / B spacing → curve:
  speed 86.7 → 81.4 / 81.2 → 45.5; acceleration 85.5 → 77.1 / 77.5 → 50.5; direction ellipse 144.0 → H = 4 56.4
  (gap 87.6 [70.8, 105.1]). **Uneven-spacing share** (B split / PC1 split, value-bootstrap CI): speed 0.13 [0.10, 0.18]
  / 0.13 [0.02, 0.22]; acceleration 0.23 [0.18, 0.31] / 0.24 [0.00, 0.39] → **curvature 77–87 %** (PC1 lower bound
  0.78 / 0.61). Spacing lines − curve: speed 35.7 [25.9, 47.3] / 35.9 [27.9, 44.8]; acceleration 27.0 [15.0, 43.5] /
  26.6 [17.8, 36.7]. Val-unseen: curve beats both spacing lines at 4 / 4 values for speed and acceleration. Speed's
  PC1 and B lines nearly coincide (same lam, MSE 81.4 vs 81.2; observation). Q1 wording filled: [X] 13 % / 23 % from
  uneven spacing along the covariance direction, [Y] 87 % / 77 % from curvature. Claude Code's predictions all met.
- 6a-5: `manifolds.participation_ratio`, `harmonic_coefficients`, `harmonic_power`, `principal_cosines`;
  `manifold_dimension` and `direction_harmonics` **passed** (3/3 each; observation keys; rebuilt selections =
  `manifold_loco`'s; saved scalers = Phase 4's at idx 1 / 9 / 18, so round blocks share our space). **C4b (H-01), idx 9
  (direction / speed / acceleration):** selected k 8 / 6 / 6 vs K·m 12 / 7 / 7 (m 2 / 1 / 1); centroid variance in k
  0.93 / 0.88 / 0.86; axes for 90 % 6 / 8 / 10; participation ratio centroids 4.0 / 2.5 / 2.5, fitted curve 3.1 /
  1.6 / 1.5. Idx 1: speed / acceleration nearly 1-D (90 % in one axis, curve PR 1.04 / 1.02), direction PR 4.4 / 3.2;
  idx 18: k 16 / 3 / 3, curve PR 3.4 / 1.5 / 1.4. Reading (observation): low effective dimension (curve PR 1.5–3.4),
  a handful of axes for the curvature; k is of the order of K·m; the paper's 40+ is not reproduced. **C4c (H-02):**
  harmonic power share at idx 1 / 9 / 18: h1 0.50 / 0.64 / 0.57, h2 0.24 / 0.23 / 0.22, h3 + h4 0.04 / 0.03 / 0.07,
  Parseval ratio 0.88 / 0.94 / 0.93. The angle-shuffle null (95th percentile ≈ 0.08 per harmonic) passes only h1–h2:
  **own-addition design flaw, kept on record** — shuffling spreads the whole (h1-dominated) variance over all
  harmonics, so it tests "any structure", not "harmonic h above noise"; LOCO (H = 4 chosen over 2–3) is the
  predictive test; no claim rests on the null. Nullspace rounds at idx 9 (share of the 2-dim block in harmonic planes):
  round 1 only 0.056 in h1 (0.07 in the whole harmonic span), round 2 0.51 in h1, round 3 0.25, rounds 5–6 closest to
  h3 / h5 with shares ≤ 0.09; same pattern at idx 1 / 18 → rounds do not step through successive harmonics (H-02 not
  supported as tested; observation). Explanation for round 1's small share (hypothesis): probe weights are decoding
  directions (∝ Σ⁻¹ × encoding pattern), harmonic planes are encoding patterns; they differ by the within-value
  covariance. Claude Code's predictions: K ✓, k ✓, curve PR ✓; centroid PR 4–15 **wrong** (2.5–4); variance in k
  0.6–0.85 missed (0.86–0.93); h1 ≥ 60 % met at idx 9 only; harmonics 1–4 above null **wrong**; Parseval ✓; round 1
  share ≥ 0.5 **wrong**; later rounds toward h2–3 **wrong**. **Step 6a done except the figure (after 6b).**
- 6b-1: `src/vjepa_physics/behavior.py` (`value_bins`, `BinReadout`, `fit_bin_readout` = validation z-score +
  `LogisticRegressionCV` L2 / lbfgs / multinomial, shuffled stratified 5-fold log-loss, Cs 1e-4…1e4; `hellinger`,
  `bhattacharyya`, `sphere_log` / `sphere_exp`); sklearn 1.9.1 source read first (`l1_ratios`, `scoring`,
  `use_legacy_attributes` set explicitly). First run: ModuleNotFoundError (file saved under `scripts/`; moved).
  Smoke test (terminal, not saved; idx 18): numerics all ≤ 8.3e-16 (rows sum to 1, Hellinger² = 1 − BC, log / exp
  round trip, tangents ⟂ base, |log| = angle); fits 1.3–2.7 s, converged, C interior (0.1 / 0.01 / 0.01). **Suspicious:
  speed / acceleration readout weak** — out-of-sample on train top-1 0.28 / 0.26, within one bin 0.72 / 0.65,
  log-loss 2.12 / 2.22 (log 16 = 2.77), although the idx-18 ridge readout has MAE 0.07–0.08 m/s (bin width 0.24 m/s);
  direction 0.64 / 0.99 / 0.96. Validation clips per bin: direction min 7 → the plan's own criterion "≥ 10 per bin"
  would fail (bins holding test-unseen values). Claude Code's predictions: top-1 0.6–0.85 / within-one ≥ 0.97 /
  log-loss 0.4–0.9 **wrong** for speed / acceleration, met for direction except log-loss (0.96); ≥ 2 train values per
  bin ✓ (≥ 3). Next: PCA-dimension diagnostic (terminal), then planning chat.
- 6b-1 diagnostic (terminal, not saved; validation-PCA inputs k 8…128 / all, selection by validation CV log-loss):
  **PCA does not fix it.** Best val CV log-loss speed 1.60 (k 8) vs 1.69 (all), acceleration 1.71 vs 1.77; train
  top-1 ≤ 0.36, within one bin ≤ 0.83. Direction best at k 8 (CV 0.82 vs 0.97). Claude Code's prediction (k 16–32,
  top-1 0.4–0.6) **wrong**. New hypothesis: a softmax with logits linear in the features can only make peaked
  middle bins of a 1-D ordered variable with very large weights (tangent-line construction), which L2 forbids;
  direction escapes because logits linear in (sin, cos) are von Mises bumps. Next: quadratic-feature diagnostic.
- 6b-1 diagnostic 2 (terminal, not saved): quadratic features of the top-k validation PCs (k 2–16) do **not** help
  speed / acceleration (train top-1 ≤ 0.27, within one ≤ 0.66; direction worse than raw) → the hypothesis is only
  half right: the top unsupervised PCs are dominated by nuisance (e.g. motion angle in the speed set), so they miss the
  speed direction. Two-stage readout (validation-fit ridge readout ŷ, then multinomial on ŷ and its squares): train
  (out of sample) top-1 0.78 / 0.63 / 0.59, within one bin 0.99 / 0.99 / 0.98, log-loss 1.47 / 1.33 / 1.79. **Its
  validation CV log-loss (0.03 / 0.22 / 0.26, C at the upper edge) is leaked** — ŷ was fit on all validation clips,
  including the CV folds (Claude Code's diagnostic slip; a real version needs cross-fitting). Reading: the
  information is there (ridge direction), but a 16-way linear classifier on 1024 raw dims with ~300 clips cannot find
  and sharpen it; a supervised 1-D readout is accurate but makes every distribution a function of ŷ alone
  (naturalness near-degenerate). Claude Code's predictions: ridge + square best ✓ (accuracy), quadratic PCs close
  **wrong**. → planning chat (readout choice, bin-count criterion).
- Planning chat (→ D-51): option C with roles reassigned. Progress = Phase 5's validation-fit ridge readouts at idx
  9–18 (unchanged). Naturalness = raw multinomial (A) for all three; direction = headline naturalness case; speed /
  acceleration labelled "blurry readout (top-1 0.28 / 0.26, within one bin 0.72 / 0.65)" on every number; naturalness
  reported as excess Hellinger over the same clip's unedited distance to the behavior curve + position within the
  natural (unedited) clips' distances, always paired with progress. (B) cross-fitted two-stage (out-of-fold ridge ŷ on
  validation, 5 folds, multinomial on [ŷ, ŷ²] / direction [sin, cos] + squares; applied via the all-validation ridge,
  stacking) only for speed / acceleration isometry alongside (A), labelled "discriminability of the idx-18 ridge
  readout, close to label distance by construction"; (B) naturalness not computed (degenerate). No further tuning of
  (A). Bin criterion "≥ 10 validation clips per bin" kept, failure recorded with cause (minimum 7); observation:
  per-bin validation counts and per-bin train accuracy. Claude Code's wording correction: those bins have *fewer*
  validation clips (only their seen values), not none.
- 6b-2: `behavior.py` (`softmax`, `readout_map`, `map_probabilities`, `two_stage_features`, `BehaviorCurve`,
  `behavior_curve`, `curve_grid`, `nearest_on_curve`); `scripts/check_behavior.py behavior_readouts` (clean commit,
  60 s CPU) **failed only the kept own criterion** "≥ 10 validation clips per bin" (direction; cause recorded:
  bins holding test-unseen values have fewer validation clips by design, D-38); the other 10 criteria passed (fit on
  exactly the validation clips, C interior, converged, raw-space map = sklearn, rows sum to 1, curves interpolate,
  two-stage C interior / converged, finite, saved = computed). `readouts.npz` `d038fbea…`. Readout (A) on train clips,
  idx 9 → 18: direction top-1 0.52 → 0.64, within one bin 0.96–0.99, C 0.01 (idx 9–12) / 0.1; speed top-1 0.25–0.29,
  within one 0.65–0.72; acceleration 0.21–0.28, within one 0.57–0.67 (C 0.01 throughout). Natural train-clip
  Hellinger distance to the behavior curve, median idx 9 / 18: direction 0.133 / 0.140, speed 0.197 / 0.187,
  acceleration 0.219 / 0.199; nearest-curve-point value error (median) direction 3.6° / 3.9°, speed 0.14 / 0.12 m/s,
  acceleration 0.39 / 0.30 m/s². Readout (B) at idx 18 on train clips: top-1 0.80 / 0.60 / 0.51, within one bin
  1.00 / 0.97 / 0.96, C 100 / 1000 / 1000 (interior). Claude Code's predictions: criteria pattern ✓; direction top-1
  0.55–0.75 met from idx 13 on (idx 9–12 0.49–0.54); speed / acceleration ✓ (acceleration within-one at idx 9–11
  0.57–0.58 just below 0.6); natural distance direction 0.2–0.35 **wrong** (0.13–0.14), speed / acceleration ✓;
  (B) ✓.
- 6b-3: `manifolds.path_positions`, `geodesic_distances` (first placed in `behavior.py` → ImportError at import time,
  nothing saved; moved and re-committed); `check_behavior.py isometry` **passed** (4/4: halves disjoint and cover
  train, both hold every value, behavior curves interpolate, finite). Split halves: direction min 5 / 6 clips per
  value, speed / acceleration 8 / 8. Idx 9 activation curve vs behavior curve (readout A idx 18 / A idx 9 same-layer /
  B idx 18), 2,000 bootstraps. **All pairs:** behavior geodesic vs activation geodesic r direction 0.990 / 0.986 /
  0.992, speed 0.968 / 0.959 / 0.897, acceleration 0.890 / 0.894 / 0.910; vs label distance 0.991–0.997 everywhere →
  **activation geodesic never beats label distance** (A idx 18: direction −0.001 [−0.004, 0.002], speed −0.015
  [−0.024, −0.006], acceleration −0.105 [−0.131, −0.074]); geodesic beats activation chord (A idx 18: +0.139 [0.105,
  0.173] / +0.015 [0.004, 0.025] / +0.032 [0.010, 0.052]). **Adjacent segments** (activation vs behavior arc length):
  A idx 18 direction 0.63 [0.42, 0.79] (label-step null 0.65), speed 0.73 [0.50, 0.86] (null 0.47 [0.19, 0.70]),
  acceleration 0.55 [0.13, 0.78] (null 0.49); same-layer A idx 9: 0.69 / 0.74 / 0.23 [−0.04, 0.45]; B idx 18: 0.59 /
  0.26 [0.07, 0.46] (null 0.67) / 0.42. No paired CI for segment r − null (not computed). Claude Code's predictions:
  criteria ✓; all-pairs ≥ 0.9 and label almost as high ✓ (acceleration 0.89–0.91); geodesic − label within ±0.05
  **wrong** for acceleration (−0.10); chord below geodesic for direction ✓; segments direction 0.2–0.5 and speed /
  acceleration 0–0.4 **wrong** (higher); B higher than A **wrong**; same-layer higher **wrong** (mixed); wide CIs ✓.
- Planning chat on Q2 (→ D-51): segment test confounded by grid gaps (label-step null = gap indicator) → local speed
  = segment length ÷ label step in both spaces, split-half reliability as ceiling, post hoc, rule fixed first ("beats
  the label locally" only if the CI of r(local speed) > 0), applied to A idx 18, A idx 9, B. By-construction note:
  equal-width bins make behavior distances inherit label spacing (limitation, no fix). Wording: Goodfire's isometry
  pattern holds (0.89–0.99; geodesic beats chord) but label order dominates; with the label null the activation
  geodesic adds nothing globally; never "not reproduced".
- 6b-4: `isometry_local` (post hoc, clean commit) **passed** (3/3; label steps exactly 1 / 2 grid steps). r(activation
  local speed, behavior local speed), A idx 18: direction 0.09 [−0.19, 0.40], **speed 0.59 [0.27, 0.77]** (Spearman
  0.36; ceiling 0.62), acceleration 0.41 [−0.10, 0.65] (Spearman −0.09) → by the rule **speed beats the label
  locally; direction and acceleration do not**. Same-layer A idx 9: 0.27 [−0.02, 0.54] / 0.56 [0.26, 0.73] / 0.08.
  B idx 18: −0.06 / −0.04 / 0.34 [0.07, 0.56] (acceleration passes the rule but its behavior reliability CI includes
  0: fragile). Split-half reliability of local speed: activation 0.84 / 0.98 / 0.98 (smoothed curves), behavior (A)
  0.17 / 0.39 / 0.41 (exact interpolation of noisy centroids) → power limited by the behavior side (observation).
  Speed's Pearson > Spearman (0.59 vs 0.36): driven by a few large segments (observation). Filled wording: "Local test
  (per-unit-label speed): speed's activation and behavior curves speed up and slow down together beyond the label
  (r 0.59 [0.27, 0.77], near its split-half ceiling 0.62); direction and acceleration show no local agreement
  (intervals include 0); the behavior side's split-half reliability is low (0.17–0.41), which limits this test."
  Claude Code's predictions: criteria ✓; direction 0.1–0.4 ~ (0.09); speed 0.2–0.5 **wrong** (0.59); acceleration
  −0.1–0.3 **wrong** (0.41, CI incl. 0); verdict speed only ✓; activation reliability 0.3–0.6 **wrong** (0.84–0.98);
  behavior 0.2–0.5 ✓ (0.17 direction just below); B ≈ 0 ✓ except acceleration.
- 6a-6: `check_manifolds.py figure_manifolds` → `results/manifolds/manifolds.png` (row 1: idx-9 curves in the first
  two centroid PCs with straight reference, train and val-unseen centroids, label marks; row 2: paired per-value
  squared LOCO errors by curve family with means and the spacing / curvature shares). Criteria passed each time
  (ladder means = `manifold_ladder`, exact = `manifold_loco`, ≤ 1e-12). Claude Code's reviews: (1) colliding row-2
  tick labels, "90°" on a held-out ring, labels on data / the dashed line, legend gap → fixed; (2) remaining tick
  collision, legend gap from the subplot top margin, label boxes hiding the curve → fixed (radial labels, top 0.92);
  (3) first-point labels across the y-axis → fixed; (4) clean. **Steps 6a and 6b done.**
- Plan mapping line added under the Phase 6 heading in EXECUTION_PLAN (6a = 6.1–6.3, 6.7, 6.12, 6.13; 6b = 6.4–6.6;
  6c = 6.8–6.9; 6d = 6.14).
- 6.8 part 1: `steering.py` (`PATH_FRACTIONS`; `steered_features` accepts (A, T, d) per-time-step shifts, (A, d) path
  unchanged; `curve_parameter`, `clamp_to_curve`, `path_values`, `spline_path_shifts`, `chord_shifts`,
  `covariance_path_shifts`, `time_covariance_maps`, `time_structured_shift`). Smoke test (terminal, not saved; 3 seeded
  val-seen clips per variable × 5 targets, no model): mean over steps of B_s = B ≤ 2.2e-15; spline endpoint = S(t)
  ≤ 6.1e-16; chord end = spline end ≤ 4.1e-16; time-structured mean (forward and reversed) = covariance shift
  ≤ 1.5e-15; covariance f = 1 = Phase 5's shift bit for bit; reversed = flipped; 0 clamped starts. **Pre-test
  prediction (validation clips):** |spline end − covariance| / |covariance| median 0.74 / 0.84 / 0.70 (range 0.10–4.26;
  large ratios where the covariance shift is small, target near the clip's reading); |spline| / |covariance| 1.29 /
  1.09 / 0.93; chord vs arc at f = 0.5 (÷ endpoint length) median 0.45 / 0.26 / 0.18 (max 0.65 / 0.42 / 0.40); free-
  spacing line (h) vs covariance 0.25 / 0.39. Claude Code's predictions: exact checks ✓, clamps ✓, endpoint difference
  0.3–0.9 ✓ (medians), length ratio ✓, (h) ✓; chord-vs-arc maxima above the guesses (0.65 direction, 0.42 / 0.40).
- 6.8 part 2: `steering.spline_arm_table`, `spline_arm_shifts`; `scripts/check_spline_steering.py spline_setup`
  **passed** (8/8, clean commit, 3.8 s CPU, no model, no readout): clip ids = Phase 5's runs; covariance f = 1 =
  Phase 5's saved shift bit for bit at all 450 clip × target pairs; spline endpoint = S(t) ≤ 1.5e-15, chord end =
  spline end ≤ 5.9e-14, time-structured means = covariance ≤ 1.5e-15, reversed = flipped. `setup.npz` `cf0a2d19…`.
  0 clamped starts. Median shift / train clip distance (direction / speed / acceleration): spline endpoint 0.58 /
  0.31 / 0.26; covariance f = 1 0.47 / 0.23 / 0.23; per-step time-structured 0.58 / 0.26 / 0.28; free-spacing line
  — / 0.23 / 0.22 (Phase 5's K − 1 probe edit: 0.52 / 0.52 / 0.53). Spline vs covariance endpoint (relative, median
  [IQR]) on test clips: 0.75 [0.42, 1.14] / 0.76 [0.49, 0.93] / 0.66 [0.50, 0.89]; seen ≈ unseen targets. Claude
  Code's predictions: criteria ✓, clamps ✓, endpoint difference ✓ (acceleration 0.66 just below 0.7), per-step ratio
  1–1.6 ✓ (1.15–1.24); spline 0.4–0.8 of clip distance **wrong** for speed / acceleration (0.31 / 0.26). **Step 6.8
  done.**
- 6.9 part 1: validation smoke run (terminal, not saved; MPS; one seeded val-seen clip per variable — ids 1148 /
  1217 / 1048 — × 2 targets × all 12–13 arms, no readout): unedited partial pass = stored features at idx 9 and 18
  bit for bit (3 / 3); max site difference 1.25–1.33e-5 (uniform and timed means); max per-step difference (timed arms,
  each step's pooled mean vs stored per-step mean + δ_s) 3.5–5.2e-5 → token mapping correct; finite. 0.305 s per pass
  (max 0.308) → 915 / 990 / 990 passes per half ≈ 4.7 / 5.0 / 5.0 min. Tolerances fixed now, before any test run
  (Claude Code's own addition): site ≤ 1e-4 (Phase 5's value, ~7.5× measured), per-step ≤ 3e-4 (~6× measured).
  Claude Code's predictions: unedited ✓, site ✓, per-step ~1e-5 **wrong** (3.5–5.2e-5), time 0.45–0.75 s **wrong**
  (faster, 0.31), within budget ✓.
- 6.9 part 2: `spline_run` (+ Phase 5 covariance f = 1 replication per clip × target, Claude Code's own addition) —
  six saved test runs `spline_<variable>_<seen|unseen>` **all passed** (10/10 each; one clean commit `48308a0`,
  chained under `caffeinate`): 990 / 990 / 1,065 / 1,065 / 1,065 / 1,065 passes, 6.7–9.9 min each (~52 min total,
  0.41–0.56 s per pass); unedited pass = stored and = Phase 5's saved unedited features bit for bit (90 / 90); **Phase
  5's covariance arm re-run reproduces Phase 5's saved steered features bit for bit at all 450 clip × target pairs,
  idx 9–18**; max site difference 2.2–2.8e-5 (≤ 1e-4), max per-step 5.2–7.6e-5 (≤ 3e-4); ids, hooks, weights
  unchanged; finite; saved = computed. Claude Code's predictions: criteria ✓; time 5–6 min per half **wrong** (6.7–9.9).
  **Step 6.9 done.**
- 6.14 part 1: `steering.PROFILE_SITES`, `READOUT_ROLES`, `ridge_readout_map`; `spline_scores` **passed** (5/5, clean
  commit; profile readouts interior alpha, train R² ≥ 0.9; same ids; **Phase 5 arms reproduce `steering_propagation`
  to 4.4e-16**; finite). Output-space gain, idx 9 / idx 18 / mean idx 10–18 (direction; speed; acceleration):
  spline endpoint 0.986 / 0.069 / 0.269; 0.973 / 0.115 / 0.355; 0.956 / 0.118 / 0.306 — covariance line 0.986 /
  0.069 / 0.268; 0.977 / 0.119 / 0.357; 0.957 / 0.119 / 0.308 — Phase 5 K−1 probes (n = 5 / 6 / 6) 0.980 / 0.068 /
  0.271; 1.011 / 0.128 / 0.355; 0.993 / 0.136 / 0.317 — free-spacing line (h) 0.972 /
  0.118 / 0.355; 0.961 / 0.118 / 0.308; random ≈ 0. **Headline (paired clip bootstrap):** spline − covariance at idx 9
  +0.001 [−0.013, 0.016] / −0.004 [−0.008, 0.001] / −0.001 [−0.009, 0.007]; at idx 18 −0.0004 [−0.002, 0.001] /
  −0.005 [−0.006, −0.003] / −0.0003 [−0.004, 0.004] → **no difference** despite 66–76 % different edit directions
  (by construction at idx 9 for a linear readout that decodes the centroids: W·S(v) ≈ v; the downstream collapse is
  the same). **H-13:** time-structured − covariance, mean idx 10–18: +0.030 [0.028, 0.032] / +0.013 [0.012, 0.015] /
  +0.017 [0.015, 0.019]; reversed − covariance −0.026 / −0.007 / −0.007 (CIs below 0); structured − reversed +0.056 /
  +0.020 / +0.024 → a small but real effect of time structure, with the correct time order helping and the reversed
  order hurting; **C11 not triggered** (point < 0.05 everywhere). Chord and covariance gains are identical across f
  (linear readout, by construction); spline path gain varies with f (direction 0.80 at f = 0.25). Planner's
  prediction "(a) − (c) non-zero at idx 9" **not met**; collapse ✓. Claude Code's predictions: criteria ✓, covariance
  idx 9 ≈ 0.95 ✓, spline lower at idx 9 **wrong** (equal), idx 18 small and no difference ✓, (e) − (c) small positive
  ✓, (g) ≈ (e) **wrong** (order matters), C11 not triggered ✓.
- 6.14 part 2: `spline_naturalness` **passed** (4/4, clean commit; saved readouts / curves reproduce the saved natural
  train distances exactly, 0.0; same ids; sums to 1; finite). Readout (A), excess Hellinger over the clip's unedited
  distance to the behavior curve, paired with behavior progress (gain of the nearest curve value). **Direction, idx 9
  (same layer, labelled):** spline waypoints excess −0.018 to −0.028 (more natural than the unedited clip; natural
  percentile 0.52–0.56 vs unedited 0.70), behavior gain 0.77–0.81, bimodal 0–0.7 %; chord f = 0.5 excess +0.103
  (percentile 0.95, 52 % above natural 95th), bimodal 24 % (far targets 47 %), ridge (sin, cos) readout length 0.62
  vs 0.94 unedited; covariance f = 0.5 alike (+0.113); endpoints: spline −0.026 (gain 0.81), covariance +0.062 (gain
  0.65), Phase 5 K − 1 +0.042 (gain 0.70), random +0.006; time-structured = covariance at idx 9 (by construction).
  **Headline, midpoint spline − chord (paired clip bootstrap):** idx 9 direction −0.124 [−0.138, −0.110] (far targets
  −0.206, near −0.047), speed −0.046 [−0.055, −0.037], acceleration −0.015 [−0.025, −0.004]; **idx 18 0.000
  [−0.004, 0.004] / −0.001 [−0.003, 0.000] / −0.001 [−0.004, 0.001]** → the spline's naturalness advantage exists at
  the steering layer only (close to by construction there, C9) and vanishes downstream, where every edit's behavior
  progress is ≤ 0.05 (direction). Claude Code's predictions: criteria ✓; unedited percentile 0.4–0.6 **wrong** for
  direction (0.70 / 0.75); endpoint |excess| < 0.05 **wrong** for covariance (0.062) and K − 1 not the largest;
  random ≈ 0 ✓; chord less natural, larger for far targets ✓; planner's bimodal chord midpoints ✓ (far 47 %);
  readout length shorter ✓; idx 18 shrinks ✓; speed / acceleration ≈ 0 at idx 9 **wrong** (small but CI < 0).
  Q3 + Q4 → planning chat.
- Planning chat on Q3 / Q4 (→ D-51): accepted with wording fixes ("follows the curved pooled-mean manifold"; collapse
  refutes the off-manifold explanation for the pooled mean only, points at H-13, which Q4 weakly supports; "within
  0.005" not "indistinguishable"; metrics named: 66–76 % = median ‖δ_spline − δ_cov‖ / ‖δ_cov‖ on test clip × target
  pairs, direction idx-18 numbers = output-space gain, Q4 not confirmed on fresh clips). Planner's prediction "(a) −
  (c) non-zero at idx 9" recorded as wrong (near by construction). Same-layer observation kept: direction endpoints
  differ only for the 16-bin readout (progress 0.81 vs 0.65; excess −0.026 vs +0.062). Before figures: (1) seen vs
  test-unseen targets for spline and probes, progress and naturalness; (2) filled Q1 / Q2 numbers; (3) 6.12–6.14
  table incl. H-01 / H-02, ‖δ‖ / clip distance per arm, random as floor; (4) time-structured spline (f) not run
  (C11 threshold not met); (5) strengths / limitations / failure-cases table. Part 2 gets four slides (exception to
  D-48's 1–2). Then gate (new subjects), report, talk outline, full docs pass. Claude Code's correction sent back:
  "within 0.005" holds for spline vs covariance only; the K − 1 probes differ from both by up to 0.018 at idx 18
  (speed 0.128 vs 0.115, acceleration 0.136 vs 0.118).
- 6.14 part 3: `spline_held_out` (C7) **passed** (3/3; reproduces `spline_scores` / `spline_naturalness` to 1.8e-15).
  Seen vs test-unseen targets (unseen = no centroid in the curve fit): spline gain idx 9 0.983 vs 0.989 (direction),
  0.971 vs 0.974 (speed), 0.987 vs 0.942 (acceleration); idx 18 unchanged (0.069 / 0.068; 0.103 / 0.119; 0.119 /
  0.118) → the spline interpolates held-out values as well as seen ones, except a −0.045 idx-9 dip for acceleration.
  Paired on unseen targets: spline − covariance gain idx 18 −0.002 / 0.000 / −0.001 (CIs include 0), downstream mean
  ≈ 0 except acceleration −0.006 [−0.009, −0.004]; spline − K − 1 probes idx 18 −0.002 / −0.009 [−0.014, −0.005] /
  −0.021 [−0.026, −0.016] (probe edits are twice as long); excess idx 9 spline − covariance −0.083 / −0.047 / −0.034,
  spline − probes −0.057 / −0.083 / −0.044 (all CIs < 0: the spline endpoint stays more natural at held-out values).
  Claude Code's predictions: criteria ✓, spline unseen idx 9 ≥ 0.9 ✓, unseen ≈ seen within ±0.03 **wrong** for
  acceleration (−0.045), spline − covariance CIs incl. 0 ✓ (except acceleration downstream), spline − probes
  −0.01 to −0.02 ✓, excess spline − covariance ≈ 0 for speed / acceleration **wrong** (negative).
- 6.14 part 4: `comparison_table` **passed** (4/4: 11 sources clean, none failed except `behavior_readouts`' recorded
  bin criterion; K − 1 probe edit size computed from Phase 5's saved shifts 0.52 / 0.52 / 0.53 = F-134; saved =
  rendered) → `results/spline_steering/comparison_table.md` (24 rows: Q1, H-01, H-02, Q2, Q3 incl. random floor and
  held-out targets, Q4, with notes and source commits). Claude Code's review: numbers match; two cosmetic defects
  ("-0.000", CI decimals in the local-speed row) → formatting fix and re-run (passed 4/4; review clean).
- 6.14 part 5: `figure_spline_paths` → `results/spline_steering/spline_paths.png` (passed 3/3; example by a fixed
  geometry rule: largest start–target angle, direction clip 318, 72° → 253.1°, 179.2°). Rows 1–2: 16-bin readout
  along the spline path vs the chord at idx 9 / 18; row 3: midpoint spline − chord excess with CIs, all variables.
  Idx 9: the spline path goes the short way round through 0°; the chord midpoint is bimodal (start and target
  peaks). Idx 18: all steps sit on the start (collapse). Claude Code's reviews: (1) legend covering data, "start"
  label on the idx-18 peak, top gap → figure-level legend, labels dropped (named in the title), top margin; (2) clean.
- 6.14 part 6: `figure_spline_profile` → `results/spline_steering/spline_profile.png` (passed 3/3; covariance
  replication profile = Phase 5's covariance profile at every index). Row 1: gain idx 9 → 18 for spline (hollow,
  on top), covariance line, time-structured, K − 1 probes, random floor, idx-18 values in the panel titles; row 2:
  Q4 differences with CIs and the C11 threshold. Claude Code's reviews: (1) idx-18 text over the idx-9 points,
  invisible CIs → text moved, title notes CIs narrower than markers; (2) edit not saved (line 1019 unchanged) →
  redone; (3) text across the steep drop → moved into titles; (4) old `set_title` overwrote the new one (duplicate)
  → removed; (5) titles too wide, running into each other → shorter wording, font 9.5, wspace 0.2; (6) clean.
  **Step 6.14 figures done.**
- 6.17 gate: `check_evidence.py` `HASH_CHECK_SUBJECTS` += `manifolds`, `behavior`, `spline_steering`;
  `code_hash_check` **passed**: 68 keys checked (46 from Phases 3–5 + 22 from Phase 6), every key matches a commit
  holding exactly its recorded code, `rerun_needed` empty (no re-runs needed this phase: every Phase 6 run went through
  `require_clean_code()`). Claude Code's predictions: 68 keys ✓, none flagged ✓. **Gate evidence complete.**
- 6.17: report `results/spline_steering/report.md` (user-written from Claude Code's draft), slide log +4 (Part 2
  exception to D-48), talk outline item 7 done (3 min; trim at rehearsal). Phase 6 docs pass: DECISIONS §1.12
  F-139–F-148, D-51 (closes O-09, O-11), H-01 / H-02 / H-12 / H-13 statuses, P-12 extended, change log; EXECUTION_PLAN
  6.1–6.17 ticked (6.6, 6.8, 6.12 reworded); PROGRESS status, phase table, Saved evidence; CLAUDE.md. **Phase 6
  passed.** Next session: Phase 7 (confounds and robustness).

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
- Guard: `evidence.require_clean_code()` (same `code_changes` test as provenance's `git_dirty`), called in
  `check_steering.py main()` after argument parsing; refused a run while uncommitted as expected, nothing saved; committed.
  After three dirty saves (`steering_setup`, `steering_cache`, `steering_propagation`), every new script calls it.
- 5.3c `steering_specificity` (post hoc, no model; clean) **passed** (3/3). Signed cross-readout slope in metres (speed ×
  0.625 s, acceleration × 0.195 s²), through the origin, 95 % clip-bootstrap CI. **Idx 18, n = K−1:** speed →
  acceleration **1.004 [0.969, 1.043]**, acceleration → speed **0.940 [0.885, 0.997]** (planner's prediction +1: met for
  speed → acceleration, just below for the reverse); probes 1 / 3 / covariance 0.91–1.09; idx 9: 0.89–0.93 (speed →
  acceleration) and 1.01–1.03 (reverse), CIs narrow (the same-layer readouts are linear in δ). Slope on intended at idx
  18 0.129 / 0.128. Mean |change| at idx 18 K−1: own 0.102 / 0.094 m, cross 0.104 / 0.089 m (= scores key). **Random
  arms also give positive slopes** (0.56–0.77, CIs above 0): any edit moves the two readouts together, only less
  exactly than probe edits → the speed and acceleration readouts nearly share one direction in distance units
  (observation; both labels = distance in these datasets, F-65, H-03). Claude Code's guesses met: 0.8–1.2 at idx 18,
  0.3–1.2 at idx 9, 0.10–0.15 on intended. **Step 5.3c done.**
- 5.3d `steering_kernel` (post hoc, no model; clean, `2e95a6c`) **passed** (3/3): validation-fit RBF kernel readouts
  (Phase 4 pipeline) at idx 9 / 18; train R² 0.972–0.988, unedited steered clips 0.974–0.990; no lower-edge alpha;
  min(1 − h_ii) ≥ 9.6e-6 (no flag); gamma at the lower factor edge in 4 / 6 fits (widest kernel; observation).
  **Pre-stated H-12 reading: "not supported" for all three.** Idx 9, probes K−1, in distribution: kernel gain 0.99 /
  0.98 / 0.96 vs linear 0.98 / 1.01 / 0.99; nearest target 0.99 / 0.95 / 0.86, nearest original 0.01 / 0.01 / 0.07 —
  the nonlinear readout follows the edit as fully as the linear one. Idx 18: kernel ≈ linear (K−1 gains 0.065 / 0.120 /
  0.126 vs 0.068 / 0.128 / 0.136), nearest original 0.77–0.96. Random arms: nearest original 0.81–0.96, kernel gain
  ≤ 0.13. Same pattern at n = 1 (kernel ≈ linear, 0.15–0.47 at idx 9). Caveat for the reading: an RBF readout on the full
  features is dominated by the high-variance probe direction; the minimum-norm edit leaves the complement of
  span(Q[:, :n·m]) untouched by construction, and Phase 4 showed that complement still carries the variable
  nonlinearly (F-121) — so H-12 is not supported *as tested by a full-feature kernel readout*. Claude Code's predictions:
  train R² 0.95–0.99 ✓, linear K−1 at idx 9 ✓, H-12 "supports" for speed / acceleration **wrong**, idx 18 kernel within
  ±0.1 of linear ✓, random nearest original ≥ 0.9 missed (0.81–0.96), speed n = K nearest mean largest at idx 9 **wrong**
  (target 0.48, mean 0.27). **Step 5.3d done.** Post-hoc set (5.3b–d) → planning chat.
- Planning chat on 5.3b–d (→ D-50 addendum at the phase docs pass): no further analysis. Mechanism clause: readable at
  idx 9 (linear and kernel), effect collapses within three blocks (idx 9 → 12: 0.96–0.99 → 0.24–0.42); at idx 18 the
  directly carried edit alone gives 0.19–0.46 of full gain and block updates "push back" part of it (not "actively
  cancel"); direction metric named for every number (output-space 0.068 vs angle 0.02), one metric for the headline.
  Headline adds covariance ≈ K−1 at idx 9 (many-directions need belongs to the probe sequence) with the D-16
  same-layer caveat (why P-06's same-layer evaluation is weak). Uniform-token hypothesis = labelled limitation.
  Specificity: "consistent with F-65; cannot separate speed from acceleration"; H-03 stays for 7.1. H-12: prediction
  observed, proposed mechanism not supported as tested, collapse equally consistent with the uniform-token
  hypothesis; complement caveat kept. Figures: (a) reduction vs n at idx 9 / 18 with covariance and random, (b)
  profile, (c) decomposition bars (two PNGs fine); 5.10 limitations as one-liners; P-06 extended. Phase 6 note:
  token-structured edit feasible with the cache + partial forward. Claude Code's corrections sent back: pushed-back
  fraction is 48–73 % (acceleration 0.48), not 50–75 %; specificity slopes are expected from F-65, not guaranteed; the
  gate must also cover `steering_setup` and `steering_cache` (saved dirty), not only `steering_propagation`.
- 5.4a `figure_steering` (from saved results only): `results/steering/steering_reduction.png` (error reduction vs n at
  idx 9 / 18, covariance, random, CIs) and `steering_propagation.png` (profile idx 9–18; idx-18 direct / block bars with
  CIs, n = K omitted). Claude Code's review (1): covariance line hidden under the K−1 line, subtitle "1–4 ×" → "≈ 1–4 ×"
  (direction n = K 0.99) → fixed (covariance as hollow diamonds on top); review (2) clean. **Step 5.4a done.**
- 5.11a gate: `check_evidence.py` — `steering` added to `HASH_CHECK_SUBJECTS`, `require_clean_code` in `main()`.
  `code_hash_check` (HEAD `37b04d0`): 46 keys, 44 match a commit holding exactly their code (all 33 Phase 3–4 keys; 11
  steering keys, incl. `steering_propagation` via its fix commit `e6aebfc`); **`rerun_needed`: `steering_setup`,
  `steering_cache`** (both saved at `cda056e` with the script staged). Claude Code's predictions: setup flagged ✓;
  cache "probably matches" wrong; propagation left open (it matched).
- 5.11b gate: `STEERING_RERUNS`, timing fields of `steering_cache` excluded in `without_machine_state`,
  `rerun_identical_steering`. `steering_setup` and `steering_cache` re-run from clean commit `f121ee8` (chained under
  `caffeinate`); **`rerun_identical_steering` passed** (3/3): both results identical to the committed ones, incl. the
  SHA-256 of `setup.npz` and the three caches the steering runs read; re-runs clean and at a new commit. All 46 keys now
  trace to committed code. **Gate evidence complete.**
- Phase 5 docs pass: DECISIONS §1.11 F-129–F-138, D-50 (closes O-07, O-08, O-10, O-16), H-12 status, H-13, P-06
  extended, change log; EXECUTION_PLAN 5.1–5.10 ticked (5.11 after the report); PROGRESS status, phase table, Saved
  evidence (10 steering / gate rows); CLAUDE.md header, §5 Phase 5 facts and guard rule, §7 layout. **Next:** report.
- 5.11: report `results/steering/report.md` (user-written from Claude Code's draft), slide log +2 ("Steering works at the
  steering layer, not nine blocks later", "Where the edit is lost"), talk outline item 6 done; plan 5.11 ticked.
  **Phase 5 passed.** Next session: Phase 6 design brief.

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
