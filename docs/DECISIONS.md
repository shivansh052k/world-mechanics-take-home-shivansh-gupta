# DECISIONS.md — facts, decisions, open questions, hypotheses

Last updated: 2026-09-24. Source of truth for *what we know* and *what we decided*.

## How to read this file

- **Facts (F-xx):** things that are true, with a source. No interpretation.
- **Decisions (D-xx):** choices we agreed on, with why and the evidence behind them.
- **Open decisions (O-xx):** choices not yet made. Never decided silently; settled in the planning chat.
- **Hypotheses (H-xx):** ideas to test. **Never state these as findings.**
- **Paper notes (P-xx):** inconsistencies and weak spots in the two papers (interview material).

Status tags:
**[verified]** checked against a primary source ·
**[scouting]** from planning-chat analysis; must be reproduced in the data audit phase ·
**[untested]** verified by reading code, not yet run ·
**[open]** not decided.

Scouting source: planning-chat sandbox scripts (not in the repo), run on 2026-09-24 over all
4,572 `metadata.json` files, the 3 manifests, and 14 selected videos.

---

## 1. Facts

### 1.1 Task (source: README.md, email from World Mechanics)

| ID | Fact |
|---|---|
| F-01 | Model: pretrained **V-JEPA 2 ViT-L/16, 256-res**, `facebook/vjepa2-vitl-fpc64-256`; reference implementation `vjepa2_vit_large` in Meta's official repo. (README) |
| F-02 | The encoder must stay **frozen**. We must document how representations are extracted and pooled. (README) |
| F-03 | Data used to fit probes/manifolds must be clearly separated from evaluation data. Probe fitting, layer selection, nullspace construction, and spline construction must not use the final held-out examples. (README, DATA.md) |
| F-04 | Do not modify `data/`. All derived artifacts go elsewhere. (README, DATA.md) |
| F-05 | Part 1 = layer-wise probing → iterative nullspace probing → multi-probe subspace steering on held-out data. Goal: reproduce methodology and qualitative findings, not exact numbers. (README) |
| F-06 | Part 2 = spline/manifold steering for speed, acceleration, and direction (at minimum); handle direction's circular structure; design a meaningful held-out steering evaluation; compare with Part 1 (strengths, limitations, failure cases). (README) |
| F-07 | Deliverable: presentation (~15 min, basis for open discussion) + source code, emailed to constantin@worldmechanics.ai. They value approach, experiment design, and interpretation over "perfect" answers. (README, email) |
| F-08 | AI tools are allowed; the candidate remains responsible for understanding everything. (README) |
| F-09 | Direction target suggested as (sin θ, cos θ), converted back to an angle for circular error. Suggested metrics: circular MAE and R² on sin/cos (direction); MAE and R² (speed, acceleration). (DATA.md) |

### 1.2 Model and libraries

| ID | Status | Fact | Source |
|---|---|---|---|
| F-10 | [verified] | 24 layers, hidden 1024, 16 heads, patch 16, tubelet 2, crop 256; predictor 12 layers, hidden 384; fp32. Saved with transformers 4.53.0.dev0. | checkpoint `config.json` |
| F-11 | [verified] | Checkpoint revision `b3c1679`. `model.safetensors` = 1.3 GB (≈325M params = encoder ≈303M + predictor ≈22M). Full repo 6.43 GB due to an `original/` folder we do not need. | HF repo file listing |
| F-12 | [verified] | `frames_per_clip = 64` "does not impact inference". Positions use RoPE computed from token indices (no fixed position table). | transformers `configuration_vjepa2.py`, `modeling_vjepa2.py` |
| F-13 | [verified] | 16 frames at 256 px → 8 × 16 × 16 = **2048 tokens**, no CLS token. Token `i` → time `i // 256`, row `(i % 256) // 16`, col `i % 16`. | `modeling_vjepa2.py` (Conv3d patch embed + RoPE position ids) |
| F-14 | [verified] | Shipped video processor: resize shortest edge to **292**, **center-crop 256**, rescale 1/255, ImageNet mean [0.485, 0.456, 0.406] / std [0.229, 0.224, 0.225], bilinear. | `video_preprocessor_config.json` |
| F-15 | [verified] | `hidden_states` returns 25 entries (patch-embedding output + 24 block outputs). Last entry is **post**-final-LayerNorm in v4.53.x but **pre**-LayerNorm in v5.x (`tie_last_hidden_states=False`). | transformers source, v4.53.3 vs main |
| F-16 | [verified] | Predictor default: context = all tokens, target = all tokens (reconstruction). The encoder always runs on the full input before masks are applied. Target tokens are placed at the positions given in `target_mask`. | `modeling_vjepa2.py` |
| F-17 | [verified] | transformers v5.17.0 is the latest release (2026-09-09); requires Python 3.10–3.14 and torch ≥ 2.5. V-JEPA 2 first appears in v4.53.0. | GitHub releases, `setup.py` |
| F-18 | [verified] | Wheels: torch 2.14.0 (cp313, macOS 14 arm64); PyAV 18.1.0 (Python ≥ 3.11, macOS 14 arm64, bundles FFmpeg); torchcodec 0.16.0 (needs a separately installed FFmpeg); opencv-python 5.0.0.93 (macOS 13 arm64). | PyPI, torchcodec README |
| F-19 | [verified] | Conv3D on MPS was added to PyTorch in PR #114183 (Dec 2023); requires macOS ≥ 13.2. | pytorch/pytorch PR #114183 |
| F-20 | [verified] | Meta's official `torch.hub` `vjepa2_vit_large` returns (encoder, predictor). The official README points to the HF checkpoints as the same models. | facebookresearch/vjepa2 README, `hubconf.py`, `src/hub/backbones.py` |

### 1.3 Data — manifests and metadata [scouting]

| ID | Fact |
|---|---|
| F-21 | Integrity: every manifest line parses; counts 1,500 / 1,536 / 1,536; ids unique, contiguous, match `scene_XXXX`; no missing, orphan, or duplicate files; `frames` = 16 and `fps` = 24 everywhere. |
| F-22 | **All three manifests are sorted by label** (speed/acceleration: corr(id, label) = 0.9999; direction: sorted by angle). |
| F-23 | Speed set: 64 values, 0.25–4.0 m/s, linear step 0.0595; 24 clips per value; acceleration = 0; each clip has a unique (speed, θ) pair; θ from the 64-angle grid, 24 clips per angle; corr(speed, cos θ / sin θ) ≈ −0.001. |
| F-24 | Acceleration set: 64 values, 0.25–10.0 m/s², linear step 0.1548; 24 clips per value; **all start from rest** (`speed_mps` = 0); same θ sequence as the speed set. |
| F-25 | Direction set: 64 angles, step 5.625°; 23 clips for 36 angles, 24 for 28 angles. **No `primary_label` or `magnitude` fields** (contradicts DATA.md). 750 constant-velocity clips (speeds 1–7 m/s, integers) + 750 accelerating-from-rest clips (2, 4, 6, 8, 10 m/s²; 150 each). Motion type and magnitude balanced across angles. |
| F-26 | Start positions: all unique, roughly uniform; within ±1.2 m (speed, acceleration) and ±2 m (direction); \|corr\| with every label < 0.05. |
| F-27 | **Confound:** within the acceleration set, acceleration correlates r = 1.000 with distance travelled and with mean speed. Within the speed set, speed is proportional to distance. Distance-travelled overlap window between the two sets: 0.156–1.953 m (1,176 speed clips, 1,440 acceleration clips inside). |

### 1.4 Data — videos [scouting]

| ID | Fact |
|---|---|
| F-28 | Files: MPEG-4 Part 2 (Simple Profile), yuv420p, 256×256, 24 fps, exactly 16 frames, ~4 KB each (14 clips checked). |
| F-29 | **The disk is orange** (mean ≈ RGB 227, 113, 44), not blue as DATA.md states. Background ≈ RGB 29, 31, 28. Two independent decoders (ffmpeg, OpenCV) agree pixel for pixel. |
| F-30 | Flat top-down rendering: disk area 341–350 px in every frame (radius ≈ 10.5 px ≈ 0.33 m, diameter ≈ 21 px ≈ about 2×2 patches); positions snap to whole pixels; exactly one object per frame. |
| F-31 | **Exact mapping:** pixel x = 128 + 32·x_world, pixel row = 128 − 32·y_world (32 px/m, visible area ±4 m, y flipped). Frame k is at t = k/24 s, so a clip spans 15/24 s. Fit residual ≤ 0.64 px; validated on all 14 clips (≤ 0.66 px). The alternative t = (k+1)/24 gives errors up to ~6 px. |
| F-32 | Angle convention: 0° = right, **90° = up on screen**, 180° = left, 270° = down. |
| F-33 | **Frame exits (direction set only):** 199 clips (13.3%) have the disk clipped or gone in ≥ 1 frame; 113 (7.5%) lose it entirely for 1–7 frames (40 for ≥ 4 frames). Driven by speed: 59/107 clips at 7 m/s, 32/107 at 6, 21/107 at 5, 1/107 at 4. Speed and acceleration sets: 0 clips. (Predicted from metadata + F-31; matched all 14 real videos exactly.) |
| F-34 | **Tiny motion:** acceleration set: minimum total movement 1.56 px; 48 clips < 3 px; 20 clips show ≤ 3 distinct positions; 12 clips frozen ≥ 8 frames after frame 0 (max 11). Speed set: minimum 5.0 px (the 24 clips at 0.25 m/s); 120 clips < 10 px. Direction set: minimum 12.5 px. |
| F-35 | Label → pixel units: speed = 1.333·v px/frame (4 m/s ≈ 5.3 px/frame); acceleration = 0.0556·a px/frame² (10 m/s² ≈ 0.56 px/frame²). |
| F-36 | **Default preprocessing damage** (derived from F-14 + F-31): resize 292 + crop 256 keeps only original pixels 15.78–240.22 → visible world ±3.51 m, scale 36.5 px/m. Clipped-or-gone clips would rise from 199 → 302 (direction), 0 → 22 (speed), 0 → 0 (acceleration). **Scale and crop verified on 1 clip (F-50); clip counts not yet re-checked.** |

### 1.5 Added during the plan review (2026-09-24)

| ID | Status | Fact | Source |
|---|---|---|---|
| F-37 | [verified] | V-JEPA 2 pretraining: the predictor's targets are the **target (EMA) encoder's** output passed through a **non-affine `F.layer_norm`**; loss = mean \|z − h\|^`loss_exp` over masked tokens (L1 in the papers). The context encoder sees only context tokens; targets come from the full, unmasked clip. | facebookresearch/vjepa2 `app/vjepa/train.py` L429–447 (main, read 2026-09-24); V-JEPA paper Eq. 2; V-JEPA 2 Fig. 2 |
| F-38 | [verified] | scikit-learn: `RidgeCV` default `cv=None` = efficient leave-one-out; `Ridge` (`cholesky`) solves the dual/kernel form when `n_features > n_samples`; `KernelRidge` has **no `fit_intercept`** (center manually); stratified splitters raise an error when a class has only 1 member. **Confirmed on the installed sklearn 1.9.1** (step 0.5b). | sklearn `main`: `linear_model/_ridge.py`, `kernel_ridge.py`, `model_selection/_split.py` (read 2026-09-24) |
| F-39 | [verified] | `torch.testing.assert_close` default fp32 tolerances: rtol 1.3e-6, atol 1e-5. **Confirmed on the installed torch 2.14.0** (step 0.5b). | pytorch `main`: `torch/testing/_comparison.py` (read 2026-09-24) |
| F-40 | [verified] | Physics paper's held-out steering: steering probes trained on train (70%, 240 videos); the eval probe is trained on **test activations only**, deliberately, so it has never seen the steering probes or the subspace-building activations. | Physics paper App. C.12 |

### 1.6 Environment as built (Phase 0, 2026-09-24)

| ID | Status | Fact | Source |
|---|---|---|---|
| F-41 | [verified] | Python **3.13.2** from Homebrew; venv at `.venv/` (`include-system-site-packages = false`); Homebrew formula `python@3.13` **pinned**, because the venv's `home` is Homebrew's version-floating `opt/` path. | `.venv/pyvenv.cfg`; `brew list --pinned` |
| F-42 | [verified] | Installed: torch 2.14.0 (arm64), transformers 5.17.0, huggingface_hub 1.32.0, av 18.1.0 (bundled FFmpeg: libavcodec 62.28.102, libavformat 62.12.102), opencv-python-headless 5.0.0.93 (`FFMPEG: YES`), numpy 2.5.3, scikit-learn 1.9.1, scipy 1.18.1, matplotlib 3.11.2, pandas 3.0.6, tqdm 4.70.1; pip 26.2.1. 52 packages in `requirements.lock.txt`; `pip check` clean; numpy↔torch round-trip exact. `VJEPA2Model` is exported by transformers 5.17.0. **torchvision 0.29.0** added 2026-09-24 (step 0.9): required by transformers' video processors (`video_processing_utils.py:79`, `@requires(backends=("vision", "torchvision"))`); installed with `-c requirements.lock.txt` (dry run listed only torchvision); `pip check` clean; torch still 2.14.0; lock now **53** packages. | install output; `requirements.lock.txt`; transformers v5.17.0 `modeling_vjepa2.py` `__all__` |
| F-43 | [verified] | MPS available (macOS 27.0), `PYTORCH_ENABLE_MPS_FALLBACK` unset. A Conv3d shaped like the patch embedding (3→1024, kernel = stride = (2,16,16)) on a (1,3,16,256,256) input gives (1,1024,8,16,16); MPS vs CPU relative difference 2.5e-7. | step 0.6a output |
| F-44 | [verified] | Checkpoint full revision **`b3c1679b7c34d3255ef3547f27c7b226aefab26f`**. Downloaded only `config.json`, `model.safetensors` (1,303.9 MB), `video_preprocessor_config.json`, `README.md` to the HF cache (`~/.cache/huggingface/hub/`). `model.safetensors` SHA-256 **`25466aef85727d16546c6cf8c99f12fcfad9cbca8225d45f23685e2e025b786b`** = the Hub's published LFS checksum. | `HfApi().model_info`; step 0.7c output |
| F-45 | [verified] | `data/` holds **9,147** files excluding `.DS_Store` (4,572 clips × 2 + 3 manifests — confirms F-21's file count). Fingerprint `artifacts/manifests/data_fingerprint.sha256` passes `shasum -a 256 -c` (exit 0); `data/` is read-only (0 writable entries). | steps 0.2a–0.2d output |
| F-46 | [verified] | On `data/speed/videos/scene_1000/video.mp4`, PyAV 18.1.0 reports codec `mpeg4`, `yuv420p`, 256×256, 16 frames (container count = decoded count), average rate 24, time_base 1/12288, `pts` = 0, 512, …, 7680 (step 512 = 1/24 s). Confirms F-28 on our install, and frame k at t = k/24 from container timestamps (F-31's timing). Re-run with explicit criteria (frames 16 decoded / 16 reported, every frame 256×256, pts × time_base step exactly 1/24 s as exact fractions, 0 → 5/8 s): passed. | `results/video_loader/checks.json`, key `inspect` (step 0.8d) |
| F-47 | [verified, 1 clip] | Colour on the test clip (PyAV `load_clip`, RGB order confirmed): disk core mean (234.0, 114.4, 39.2) (edge ring excluded; R > G > B = orange); background mean (29.0, 32.0, 29.0), and pixels > 16 px from the disk still range 21–37 (not flat; compression noise is a hypothesis). Clip min 0 occurs **only in blue**, 467 px, all 0–2.24 px outside the disk edge (median 1 px); max 249 only in red, 2 px, on the disk. Disk mask (brightest channel > 128): 349–351 px per frame (F-30: 341–350). | `results/video_loader/checks.json`, key `colour` (step 0.8d) |
| F-48 | [verified, 1 clip] | Geometry and frame order on the test clip: disk centroid vs metadata (t = k/24, 32 px/m, origin 128, y flipped) max error **0.654 px** over 16 frames, mean residual (col, row) = (+0.033, −0.008) px (no half-pixel bias). Alternatives rejected: frames reversed 54.1 px, time shifted +1 frame 4.2 px, no y flip 73.7 px. Confirms F-31 and D-09 on our loader. | `results/video_loader/checks.json`, key `order` (step 0.8d) |
| F-49 | [verified, 1 clip] | **PyAV and OpenCV are not pixel-identical.** PyAV 18.1.0 (bundles FFmpeg 8.x, libavdevice 62) vs OpenCV 5.0.0.93 (forced `CAP_FFMPEG`, bundles FFmpeg 7.x, libavdevice 61): same shape, frames and geometry, but 1,048,570 / 1,048,576 pixels differ. PyAV − OpenCV: R 0…+3 (99.0 % = +1), G −1…+2 (98.9 % = 0), B 0…+2 (99.6 % = +1); mean (+1.01, +0.01, +1.00) overall, (+1.00, 0.00, +1.00) far background, (+1.69, +0.62, +1.39) on the disk. **Not the cause:** swscale flag (PyAV BILINEAR and BICUBIC give identical output; PyAV defaults to BILINEAR, OpenCV uses BICUBIC); frame tags (colorspace 2 / range 0 = unspecified). **Hypothesis (untested):** FFmpeg 8 vs 7 swscale coefficients/rounding. F-29's pixel-identical agreement was FFmpeg CLI vs OpenCV, not PyAV. Loading both libraries in one process prints a harmless-looking objc duplicate-class warning (both bundle libavdevice); `cv2` is imported only inside the OpenCV checks. | `results/video_loader/checks.json`, keys `opencv`, `opencv_bicubic`, `opencv_diff_stats`; PyAV `av/video/reformatter.py` (default `SWS_BILINEAR`); OpenCV `modules/videoio/src/cap_ffmpeg_impl.hpp` (5.x, `SWS_BICUBIC`) |
| F-50 | [verified, 1 clip] | **Preprocessing** (`preprocess_clip`, the checkpoint's `VJEPA2VideoProcessor` with `do_resize=False`, `do_center_crop=False`, loaded offline at the pinned revision): output (16, 3, 256, 256) float32 on CPU. `config`: rescale 1/255 and ImageNet mean/std as shipped; default = shortest edge 292 + crop 256. `manual`: max abs diff **1.65e-7** vs (x/255 − mean)/std in float64 (processor fuses it as (x − 255·mean)/(255·std) in float32). `identity`: un-normalizing and rounding recovers the uint8 clip **bit-for-bit** (max 9.4e-6 levels from integers) → no resize, crop, shift or channel swap. `default` (shipped settings): disk centroids follow col' = s·(col + 0.5) − 0.5 − 18 with fitted slope 1.1393 / 1.1405 (s = 1.140625) and offset −17.929 / −17.938 px at slope s (predicted −17.930); disk area 350 → 455 px (× s²); only original pixels 15.78–240.22 kept, 36.5 px/m. Verifies F-36's scale and crop. | `results/preprocessing/checks.json`, keys `config`, `manual`, `identity`, `default`; `results/preprocessing/default_vs_ours.png`; transformers `video_processing_utils.py:228–345`, `image_processing_backends.py:299–334` |
| F-51 | [verified] | **Model loading** (`load_model`: `VJEPA2Model.from_pretrained`, offline, pinned revision, `dtype=float32`, `attn_implementation="sdpa"`): weights file SHA-256 = pinned (F-44); loading report empty (no missing/unexpected/mismatched keys, no errors; 587 state-dict tensors); **303,885,312** encoder + **22,086,016** predictor = **325,971,328** parameters (F-11); config and built structure = F-10 (24 encoder / 12 predictor blocks, patch Conv3d weight (1024, 3, 2, 16, 16), all dropout and drop-path 0.0); fp32, eval mode, no parameter requires grad. In-memory weights fingerprint **`c865f524c1376e4452943b208d7d50ba588be9490604f235a3d9c9dc80804ede`**, identical across two independent loads (nothing randomly initialized). `set_seeds(0)` reproduces draws from Python `random`, NumPy, torch CPU and torch MPS, and seed 1 changes all four; `torch.manual_seed` also seeds MPS. | `results/model/checks.json`, keys `config`, `load`, `fingerprint`, `seeds`; transformers `modeling_utils.py:4065, 4351`, `utils/loading_report.py:156–194`; torch `random.py:69–72` |
| F-52 | [verified, 1 clip] | **First full forward pass** (MPS, fp32, `torch.inference_mode`, default masks): input (1, 16, 3, 256, 256), mean −1.456 / std 0.200 (mostly background, consistent with F-47); encoder `last_hidden_state` (after the final LayerNorm, `modeling_vjepa2.py:457`) (1, 2048, 1024), mean 0.0367 / std 3.117; predictor output (1, 2048, 1024), mean 0.0001 / std 0.650; predictor target = encoder output (all tokens are targets by default, `modeling_vjepa2.py:934–949`); every value finite; weights fingerprint unchanged by the forward pass (= F-51 reference). | `results/forward/checks.json`, key `forward` |

---

## 2. Decisions made

| ID | Decision | Why | Evidence |
|---|---|---|---|
| D-01 | Use the HF `transformers` checkpoint `facebook/vjepa2-vitl-fpc64-256`, pinned to revision `b3c1679`. Encoder frozen, `eval()` mode, no gradients; checksum weights before and after runs. | README names it; reproducibility; proves "frozen". | F-01, F-02, F-11 |
| D-02 | **Preprocessing: resize OFF, center-crop OFF.** Keep rescale (1/255) and ImageNet normalization (the model was trained with it). | Default would distort scale and crop the disk out of many clips. | F-14, F-36 |
| D-03 | Feed all **16 frames at native 256×256**. | Matches the data; frame count does not affect inference. | F-12, F-28 |
| D-04 | **Capture activations with our own forward hooks** on each encoder block (plus the patch-embedding output). Pin the exact transformers version in a lock file. | `hidden_states` semantics differ between versions; we need the same hooks for steering anyway. | F-15 |
| D-05 | Video decoding: **PyAV** primary, **OpenCV** (`opencv-python-headless`: same `cv2`, no GUI, which we never use) as an independent cross-check (pixel-identical comparison). | PyAV bundles FFmpeg on macOS; torchcodec needs a separate FFmpeg; two decoders catch decoding bugs. **Under review (2026-09-24):** the pixel-identical criterion is not achievable for PyAV vs OpenCV on our install (F-49); a replacement criterion (e.g. tolerance band) awaits the planning chat. PyAV stays the only decoder for any data the model sees. | F-18, F-29, F-49 |
| D-06 | Environment: fresh **Python 3.13.2** venv, **MPS**. Fallback to a 3.12 venv only if a package fails on 3.13. Default dtype **fp32** until step 0.13 (MPS vs CPU) justifies anything else. Download only weights + configs (~1.3 GB). Built from Homebrew Python with `python@3.13` pinned (F-41); always install with `python -m pip`. | All key wheels verified for 3.13 / macOS ≥ 14; fp32 is the checkpoint's native dtype, so half precision must earn its place with evidence; saves ~5 GB. | F-17, F-18, F-19, F-11 |
| D-07 | Splits: **never head/tail.** Use a **three-way train / validation / test** split; test is never touched for fitting or selection. Exact scheme: see O-01. | Manifests are sorted by label; README requires fit/eval separation. | F-03, F-22 |
| D-08 | Direction targets = **(sin θ, cos θ)**; report circular MAE and R² on sin/cos. | DATA.md recommendation; handles wrap-around. | F-09 |
| D-09 | Angle convention: math convention (0° right, 90° up); **flip y** when converting between image and world coordinates. | Confirmed on real clips. | F-32 |
| D-10 | If the predictor is used for forecasting: encode frames 0–7 only, then call the predictor separately with context positions 0–1023 and targets 1024–2047. Must pass step 0.15 (beats a trivial baseline) before any result relies on it. | Default path is reconstruction with future leakage. | F-16 |
| D-11 | The **full data audit is reproduced by the user in code** over all 4,572 clips (Phase 1, steps 1.1–1.13 in `EXECUTION_PLAN.md`), outputs saved, key figures in the presentation. | Our decisions rest on it; reviewers must see how conclusions were reached. | user said |
| D-12 | Workflow: planning chat = architect and main guide; Claude Code = guide only; the **user writes all code**. Source of truth = `CLAUDE.md` + `docs/`. | Learning goal; interview readiness; avoids conflicting decisions. | user said |
| D-13 | Phase 2 stores only **pooled** activations for all clips, never bulk per-token activations. Steering (Phases 5–6) computes per-token activations **live**, via forward hooks on real clips, only for the specific clips involved in a given experiment. | Resolves O-13: bulk per-token storage would cost ≈19 GB per layer in fp16 for all clips, but nothing downstream actually needs it stored — steering re-runs the model on demand. | reasoning during Phase 2 planning |
| D-14 | **Two-level split**, per dataset. (1) **Value level:** hold out whole label values: ~8 **test-unseen** and ~4 **val-unseen**, evenly spaced, never the range endpoints (speed, acceleration; direction is circular, so no endpoints). (2) **Clip level:** within the remaining ~52 values, split clips ~16/4/4 per value into **train / val-seen / test-seen**. Provisional counts (speed, acceleration): train 832; val 304 (208 seen + 96 unseen); test 400 (208 seen + 192 unseen). Direction: value level by angle; clip level stratified on motion group (12) × angle octant (8), marginals verified with a balance table. **Selection** (layer curve, nullspace stopping) uses **val-seen**; val-unseen is reported alongside, never used for selection. Exact indices and counts are fixed at step 2.1, after Phase 1 confirms the counts. | Gives separate "unseen clips" and "unseen values" tests; makes steering to unseen values a real held-out test (fixes P-12). A full angle × motion × magnitude cross is infeasible (768 cells, ~2 clips each) and single-member classes break stratified splitters. Trade-off: train (832) < d (1024); ridge handles it. | F-03, F-22, F-23, F-24, F-25, F-38, P-12; DATA.md ("fair test of generalization"); README |
| D-15 | **Fitting discipline.** Probe regularization (alpha) is chosen by efficient leave-one-out **inside train**. Scalers and PCA are fit on **train only**, everywhere. Spline hyperparameters are chosen by **leave-one-centroid-out within train values**, with val-unseen centroids as confirmation only. Validation is reserved for structural choices (layer, nullspace stopping); test is for reporting only. | Each split has one job; leave-one-out is free for ridge; 4 val-unseen centroids are too few to tune splines on. | F-38, D-07 |
| D-16 | The **steering readout / eval probe is fit on validation** (val-seen + val-unseen): disjoint from the train data that builds the steering subspace, and never fit on test. **Caveat:** data disjointness removes overfitting coupling, but a *same-layer* readout will still tend to share directions with the steering probes. Real independence comes from a downstream readout (O-07 b/c), which step 5.2 prefers; any same-layer result is labeled with this limitation. | Keeps the paper's independence goal (F-40) without its fit-on-test flaw (P-06). | F-40, P-06 |
| D-17 | The **behavior manifold is built downstream of the steering site**: activation manifold at layer L; behavior readout at the readout location (a later layer or the predictor output, per O-07). The same-layer softmax-over-centroids version is reported only as a labeled **by-construction control**. | In Goodfire's Mountain Car setup the behavior distribution is a softmax over distances to the same activation centroids, so isometry is partly by construction; a downstream readout makes it depend on the network's own computation. | P-10; Goodfire §5, Eq. 9 |
| D-18 | **Version control and naming.** Local repo folder: `world-mechanics-take-home-shivansh-gupta`. Remote: **private** GitHub repo `shivansh052k/world-mechanics-take-home-shivansh-gupta`, pushed over SSH with a dedicated personal key (separate from an older school key). Full title "World Mechanics Take Home Assignment (Shivansh Gupta)" goes in the GitHub description, the code README (8.7), and the title slide. `paper/` (third-party PDFs) is git-ignored. Final delivery method is still settled at 8.10. | Off-site backup; a plain name avoids shell quoting; not public because the repo holds World Mechanics' task text (publishing needs their okay). | user said (2026-09-24) |
| D-19 | **`.DS_Store` is not supplied data.** It is excluded from the data fingerprint (0.2, and so 1.3 / 8.1) and from Phase 1 integrity checks (1.1). It is never deleted from `data/`, since that would itself be a write. | One exists, `data/.DS_Store`, dated 2026-09-24 00:52 — after the data folders (2026-09-23 23:22), so written by Finder. Without the exclusion, any Finder rewrite would make 1.3 falsely report "data changed". | `ls -la data` and `find data -name .DS_Store` output, 2026-09-24 |
| D-20 | **Code layout: installable package.** Code lives in the package `src/vjepa_physics/` (src layout), described by `pyproject.toml` at the repo root and installed once with `python -m pip install -e .`. Scripts, notebooks, and checks import it (`from vjepa_physics.<module> import …`). `pyproject.toml` declares **no dependencies**: `requirements.lock.txt` is the single source of truth for versions, so installing the package can never change the environment. | Imports work from any folder; conventional layout reviewers expect; no `sys.path` hacks. | user said (2026-09-24) |
| D-21 | **Working style (all sessions, both tools).** One small step at a time, waiting for the user's output before the next. Files are given as **exact path + full content** (or the exact lines to add/replace); the user creates and edits them himself — never via terminal commands. Terminal commands are only for running and checking; any command that writes files as a side effect (downloads, `pip freeze >`, installs) is flagged, with where it writes, before it is run. From step 0.8d, coding is guided in Claude Code, which runs read-only by default (project-local settings); design decisions stay in the planning chat (D-12). | Learning goal; the user must be able to defend every line; prevents unreviewed writes. | user said (2026-09-24) |
| D-22 | **Claude Code may edit `docs/` and `CLAUDE.md` only**, after showing the exact change in chat and getting the user's explicit yes to that specific change. Enforced by `.claude/settings.local.json`: path-scoped `Edit` deny on every other repo path (incl. `.claude/`), `ask` on `Edit(/docs/**)` and `Edit(/CLAUDE.md)`, `Bash` and `NotebookEdit` denied. All code is still written by the user (D-21). | Keeps the source-of-truth docs current across sessions; code stays user-written. | user said (2026-09-24); Claude Code permissions docs (rules evaluated deny → ask → allow; `Edit(path)` rules cover all built-in file-editing tools) |
| D-23 | **No plan IDs in code or outputs.** Step numbers, F-/D-/O-/H- tags and "phase" stay in `docs/`; code, file names, docstrings and saved outputs use descriptive names (e.g. `scripts/check_video_loader.py`, not `step_0_8_…`). | The plan is internal; the submitted code follows normal development standards for reviewers. | user said (2026-09-24) |
| D-24 | **Checks are scripts that save their evidence.** One `scripts/check_<subject>.py` per subject, one function per check, run by name (`python scripts/check_<subject>.py <check>`); each result is saved with provenance (UTC time, git commit, dirty flag, library versions) under its own key in `results/<subject>/checks.json`. Saved results are listed in `PROGRESS.md` ("Saved evidence"). First: `scripts/check_video_loader.py` → `results/video_loader/checks.json`. | Reproducible, committable proof for the reports and the presentation (D-11); REPL output is lost. | user said (2026-09-24) |
| D-25 | **Add torchvision 0.29.0** so the checkpoint's own `VJEPA2VideoProcessor` runs, with resize and center-crop off (D-02). Installed under the lock constraint (`-c requirements.lock.txt`), so no pinned package changed. | Gives an independent reference for the manual-normalization check and the *real* shipped default for the comparison figure; the alternative (own normalization only) would make 0.9's check self-referential and the default an approximation. | F-42; `video_processing_utils.py:79–80`; user said (2026-09-24) |
| D-26 | **Model loading settings.** `dtype=torch.float32` explicit (D-06); `attn_implementation="sdpa"` explicit, so it is pinned rather than implied (eager vs sdpa numerics to be measured in 0.13); weights file hash-checked and loading report required empty before use; `eval()` + `requires_grad_(False)`. Seeds: `SEED = 0` via `set_seeds` (Python, NumPy global, torch incl. MPS); library-level generators (sklearn `random_state`, `np.random.default_rng`) get `SEED` explicitly where used. The fingerprint in F-51 is the reference for "model unchanged" (D-01). | Every setting that could silently differ between runs or versions is stated, not defaulted; the global seed does not reach library-level generators. | F-51; transformers `modeling_utils.py:3946, 4068` (`dtype` default `"auto"`); `modeling_vjepa2.py:855` |
| D-27 | **Result provenance** (`evidence.py`). `git_dirty` = uncommitted or untracked changes **only** in `src/`, `scripts/`, `pyproject.toml`, `requirements.lock.txt`, with git run against the repo root found from the package's own location (`git -C`), so the answer is the same from any working directory; `git_dirty_paths` lists them. A `code` block records the SHA-256 of the running script and every `.py` in `src/vjepa_physics/` (sorted by path) plus one combined hash. Library versions come from package metadata, never imports. All check scripts use it. Entries saved before commit `c39e0a8` keep the old semantics (`git_dirty` = any change in the tree). | `results/` and doc edits made every entry dirty, so the recorded commit never identified the code that ran. | planning-chat review (2026-09-24); `results/evidence/checks.json`, key `dirty_flag` |

---

## 3. Open decisions [open]

| ID | Question | Options on the table |
|---|---|---|
| O-01 | Split details (structure settled by D-14) | Exact held-out value indices and final counts per dataset, confirmed after Phase 1 (step 2.1). |
| O-02 | Direction clips where the disk exits/clips (F-33) | Keep all + flag; exclude; report with and without. |
| O-03 | Low-motion acceleration clips (F-34) | Keep + flag; report error by magnitude bin; exclude below a movement threshold. |
| O-04 | Pooling | Mean over all 2048 tokens (paper's main choice); per-time-step pooling; per-patch / attentive probes as a secondary analysis. |
| O-05 | Probe family (alpha selection and standardization settled by D-15) | Closed-form ridge vs SGD (paper used Adam + weight-decay sweep). |
| O-06 | What "layer 0" means | Patch-embedding output vs first block output; indexing convention for all plots. |
| O-07 | Steering "behavior" readout | (a) probe at the same layer; (b) steer at layer L, run the remaining layers, read at a later layer; (c) forecast via the predictor (needs D-10). |
| O-08 | Mapping a pooled-space steering vector onto 2048 tokens | Add the same δ to every token (shifts the mean by exactly δ); per-token alternatives. |
| O-09 | Spline construction | Exact interpolation vs smoothing spline; PCA dimension; parameterization (label value vs arc length); periodic spline for direction; intrinsic angle from labels vs `atan2` of PCs. |
| O-10 | Held-out steering details (structure settled by D-14, D-16) | Range of target values per variable; number of source clips per target; reporting test-seen and test-unseen separately; fair comparison in the same subspace. |
| O-11 | Behavior readout form and naturalness metric (readout location settled by D-17) | Readout form: multinomial logistic over value bins vs softmax-over-centroids at the readout location; temperature. Metric: Hellinger/Bhattacharyya (Goodfire style); distance to the activation manifold; nearest-neighbour retrieval. |
| O-12 | Confound controls (F-27) | Velocity-vs-acceleration classifier at matched distance; speed probe applied to acceleration clips; displacement probe. |
| O-14 | HF vs official implementation parity check | Run it (rigor, both named in README) or skip (cost, time). |
| O-15 | Which layer(s) for nullspace probing (Phase 4) | 2–3 layers spanning before/at/after the emergence transition found in Phase 3 (paper's by-layer figure style). |
| O-16 | Which single layer to steer at (Phase 5) | One of Phase 4's candidate layers, or a new choice with reasoning; distinct from O-15 since nullspace probing studies several layers but steering needs exactly one. |
| O-17 | Presentation medium (Phase 8) | Slides; a notebook; a PDF; some combination. |
| O-18 | Which encoder weights the HF checkpoint ships (target/EMA vs context encoder) | Check the HF conversion script and model card in step 0.15; affects how 0.15's result is interpreted (F-37). |

---

## 4. Hypotheses (to test — not findings)

| ID | Hypothesis | How we would test it |
|---|---|---|
| H-01 | The physics paper's "direction needs 40+ linear dimensions" is at least partly what one **curved, closed 1-D manifold** looks like to linear tools. | Fit a periodic spline; measure how many PCA dims it needs; compare spline steering in that small space vs multi-probe steering. |
| H-02 | The nullspace **sawtooth** pattern relates to sin/cos feature pairs at several frequencies (harmonics) of that curve. | Fourier analysis of activations vs θ; compare sawtooth before/after removing harmonic pairs. |
| H-03 | "Acceleration" decoding within the acceleration set may reduce to **distance travelled / mean speed**. | O-12 controls. |
| H-04 | Direction errors concentrate in **exit/clipped clips**; low-end acceleration errors concentrate in **tiny-motion clips**. | Error breakdown by flag and magnitude bin. |
| H-05 | Slow motion is hard partly because a slow disk often does not move within one 2-frame tubelet (F-30, F-34, F-13). | Error vs per-tubelet pixel motion. |
| H-06 | The predictor output (std 0.65) and the encoder `last_hidden_state` (std 3.12) differ in scale because the predictor was trained against a **non-affine** layer norm of the target encoder's output (F-37), while `last_hidden_state` has passed the final LayerNorm **with** its learned scale and shift (F-52). So they are only comparable after `layer_norm` on the encoder side. | Step 0.15: compare predictor output with `F.layer_norm(encoder output)` vs the raw output; per-token std of each. |

---

## 5. Paper notes (inconsistencies and weak spots)

### Physics paper — Joseph et al., *Interpreting Physics in Video World Models*

| ID | Note | Where |
|---|---|---|
| P-01 | Nullspace stopping thresholds disagree: direction R² < 0.1, speed R² < 0.05 vs direction R² < 0.3, speed R² < 0.1. | App. C.11 vs Fig. 22 caption |
| P-02 | Direction dimensionality at layer 8 disagrees: "40–50 features" vs 136 dims (68 probes) vs 25 probes until R² < 0.1. Layers 20–23 all show exactly 400 (possible cap). | §7.2, Table 3, App. C.12 |
| P-03 | Steering error: "< 0.5°" vs held-out 11.9° (the former is likely measured with the steering probes themselves). | §7.2 vs App. C.12 |
| P-04 | Held-out split lists 240 + 103 = 343 videos; the velocity dataset has 392. | App. C.12 vs A.1.2 |
| P-05 | Units: "pixels per frame" vs m/s. | §5.1 vs A.1.2 |
| P-06 | Steering evaluation is weak: edits and reads the same layer (no downstream propagation); single target angle (90°); 8 directions; the "held-out" eval probe is trained on the same test videos it evaluates (deliberately, for independence from the steering probes, F-40; our alternative is D-16). | App. C.12 |
| P-07 | Acceleration clips start from rest, so acceleration is confounded with speed/distance — yet §5.2 argues acceleration does not rely on velocity. | A.1.2, §5.2 |
| P-08 | Undocumented: what "grouped" CV grouped by; whether layer 0 = embeddings or first block; how the least-squares step applies probes trained on already-projected activations. | App. B, C.11, C.12 |
| P-09 | Uses 224-px input (14×14 = 196 patches, 1,568 tokens); we use native 256 (2,048 tokens). | App. C.6 |

### Goodfire paper — Wurgaft et al., *Manifold Steering…*

| ID | Note | Where |
|---|---|---|
| P-10 | Mountain Car "behavior" F is a softmax over distances to centroids sampled from the activation spline itself → activation–behavior isometry (r = 0.996) is partly by construction. | §5, App. B.1, C.3 |
| P-11 | Unequal interventions: linear steering replaces the entire residual stream; manifold steering replaces only the top-64 PCA components. | App. A.6 |
| P-12 | No held-out evaluation: splines interpolate all centroids exactly; steering always runs between those same centroids. | App. A.3, A.4, A.6 |
| P-13 | Unablated choices: 64 PCA dims, layer 28, τ = 0.5, exact interpolation vs smoothing. | §2.2, App. A.2, B.1 |

---

## Change log

- 2026-09-24 — initial version (facts from README, DATA.md, email, both papers, library/model sources, and scouting analysis).
- 2026-09-24 — final consistency pass while completing EXECUTION_PLAN.md: resolved O-13 as D-13; added O-15 (Phase 4 layer selection), O-16 (Phase 5 steering-layer selection), O-17 (Phase 8 presentation medium).
- 2026-09-24 — corrections caught by a fresh review of the full plan: fixed a real methodology bug where Phase 3/4's layer-selection curve and nullspace stopping criterion were worded as using test data, contradicting D-07 (fixed in EXECUTION_PLAN.md steps 3.2, 3.7, 4.2, 4.3, 5.4); remapped stale old-checklist references in D-06/D-10/D-11 to actual step IDs; corrected a step-count error (107, not 117) and a wording slip in PROGRESS.md.
- 2026-09-24 — plan review, round 2 (researched fixes): added F-37–F-40 (V-JEPA training target, sklearn ridge/split behavior, torch tolerances, the paper's held-out protocol); added D-14 (two-level split), D-15 (fitting discipline), D-16 (readout fit on validation, with caveat), D-17 (downstream behavior readout); narrowed O-01, O-05, O-10, O-11; added O-18; clarified P-06. Matching step edits in EXECUTION_PLAN.md.
- 2026-09-24 — after step 0.1: added D-18 (private GitHub remote, folder name, `paper/` ignored) and D-19 (`.DS_Store` excluded from fingerprint and integrity checks). Matching wording in EXECUTION_PLAN.md steps 0.1, 0.2, 1.1, 8.7, 8.10.
- 2026-09-24 — Phase 0 checkpoint before moving coding to Claude Code: added F-41–F-45 (environment as built, checkpoint hash and weights checksum, data fingerprint); F-38/F-39 confirmed on installed versions; D-05 → opencv-python-headless; D-06 → Homebrew pin and `python -m pip`; added D-20 (installable package) and D-21 (working style).
- 2026-09-24 — step 0.8d (Claude Code): added F-46 (`inspect` check on our install confirms F-28 and frame timing); added D-22 (Claude Code may edit docs only, with approval), D-23 (no plan IDs in code or outputs), D-24 (checks as scripts saving evidence to `results/`).
- 2026-09-24 — step 0.8d checks: added F-47 (colour), F-48 (geometry and frame order), F-49 (PyAV vs OpenCV not pixel-identical: systematic +1 R/B offset); marked D-05's pixel-identical criterion as under review (planning chat).
- 2026-09-24 — step 0.9 start: F-42 updated (torchvision 0.29.0, lock 53 packages); added D-25 (torchvision for the HF video processor).
- 2026-09-24 — step 0.9 done: added F-50 (preprocessing checks); F-36's scale and crop marked verified on 1 clip.
- 2026-09-24 — step 0.10 done: added F-51 (model loading, checksum, fingerprint, seeds) and D-26 (loading settings and seeds).
- 2026-09-24 — planning-chat review: added D-27 (code-only dirty flag, code hashes); F-46 now has explicit pass criteria.
- 2026-09-24 — step 0.11 done: added F-52 (first forward pass) and H-06 (predictor vs encoder output scale).
