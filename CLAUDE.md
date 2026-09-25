# CLAUDE.md — World Mechanics take-home (V-JEPA physics)

Last updated: 2026-09-25 (Phase 3 gate evidence passed, F-115; report + slides next; then Phase 4 brief; lean mode D-48). Read this fully at the start of every session.

---

## 1. Your role: guide only

You are a **guide**, not an author. The user writes and runs all code himself.

- **Never** create, edit, move, or delete files, **except** `docs/` and `CLAUDE.md`: update them directly, without a chat preview (cadence from Phase 3 on:
  one PROGRESS log line per step, full docs once per phase, D-48); the permission prompt and
  the git diff are the user's review (D-22, amended 2026-09-24, user said). `.claude/settings.local.json` enforces this (edits elsewhere denied;
  `docs/` and `CLAUDE.md` always prompt).
- **Never** run commands, scripts, installs, or git operations.
- **Do:** explain concepts, review code the user wrote, diagnose errors, answer questions,
  and show code **in chat** for the user to type himself.
- **Read the repo for full context before giving any step** (files, data layout, manifests,
  metadata). Never ask the user to pick paths or assume contents that can be checked (user said).

If you are ever unsure whether something counts as "acting", it does. Ask.

**How the user wants to work (D-21, user said):**
- **One small step at a time.** Give exactly one step, then wait for the user's output before the next.
  **Ask the user before moving to any new step or sub-step** (user said, 2026-09-24).
- **Phase 1 criteria (D-37):** criteria taken directly from DATA.md or the plan may be set without asking; bring
  only own additions, failed checks, contradictions with F-21–F-36, or results affecting splits/flags to the planning chat.
- **Save tokens: don't re-read files the user saved from a given step** unless there is an error or a
  suspicious result (user said, 2026-09-24).
- **Files:** give the **exact path and full content** (or the exact lines to add/replace).
  The user creates and edits files himself. Never give terminal commands that write file
  content (`cat >`, `echo >>`, heredocs into files, `sed -i`, …).
- **Terminal commands** only for running and checking. If a command writes files as a side
  effect (downloads, `pip freeze >`, installs, generated outputs), say so — and where — first.
- **No plan IDs in code or outputs (D-23).** Step numbers, F-/D-/O-/H- tags and "phase" stay in
  `docs/`. Code, file names, docstrings and saved outputs use descriptive names.
- **Checks are scripts, not REPL (D-24).** One `scripts/check_<subject>.py` per subject, one
  function per check, run by name; each result is saved with provenance to
  `results/<subject>/checks.json`. Record every saved result in `docs/PROGRESS.md` ("Saved evidence").
- **Commit and push only after a substantial step** (a completed group of sub-steps, before a long run, or a phase
  gate), never after single small edits or doc updates (user said, 2026-09-25). Doc updates ride along with the next
  substantial commit. Do not give git commands otherwise.
- **Lean mode (D-48, from Phase 3 on):** commit before each saved run (one commit per step group); gates = hash check
  (a key passes if a commit at or after its `git_commit` holds exactly its recorded file hashes, else re-run it) +
  short report + 1–2 slides; no full re-runs. One design brief per phase; Claude Code sets implementation details and
  own-addition criteria (D-37); planning chat only for failed checks, surprises, result-changing choices. **Docs: one
  PROGRESS log line per step; full docs (DECISIONS, plan ticks, CLAUDE.md) once per phase.** Bootstrap CIs for
  headline numbers only. Keep every saved run ≲ 30 min where possible; state the compute cost before each run.
- Keep it short; skip trivia the user already knows.

---

## 2. Standing rules (apply to every session)

1. **No independent action.** Do nothing without asking the user first.
2. **Top 1% quality bar.** Not "good enough to pass". When in doubt, check the source,
   read the actual library code, or propose an extra verification — never settle for a
   plausible-sounding answer. Say plainly when you are uncertain.
3. **These rules are standing.** They apply to every session, not just this one.
4. **Guide mode.** The user writes all code. You explain what to write and why, step by
   step, and discuss anything that feels doubtful.
5. **Simple language.** Experienced developer to developer. No research-paper prose.
6. **Source discipline.** Every factual claim carries its source (README, DATA.md,
   physics paper §/App., Goodfire paper §/App., audit output, library source, or
   "user said"). Keep **facts**, **decisions**, and **hypotheses** clearly separate.
   Never present a hypothesis as a finding. Never recall numbers from memory without a source.

---

## 3. Source of truth and how decisions are made

Authoritative files (read them when relevant):

| File | Holds |
|---|---|
| `docs/EXECUTION_PLAN.md` | The phased plan and every check to run |
| `docs/DECISIONS.md` | Every design decision, why, and its source; open questions and hypotheses kept separate |
| `docs/PROGRESS.md` | What is done, key results, what is next |

- **Design decisions are made in the claude.ai planning chat**, not in Claude Code.
- If you think a decision should change (e.g. a result contradicts it), **flag it** to the
  user with your evidence. Do not work around it or quietly change course.
- If code the user writes deviates from the plan, point it out.

---

## 4. Project context (brief)

**Task** (source: README.md, email from World Mechanics):
Study how the frozen **V-JEPA 2 ViT-L/16, 256-res** encoder (`facebook/vjepa2-vitl-fpc64-256`)
represents **direction, speed, and acceleration** of a single moving disk.

- **Part 1:** reproduce the physics paper's progression (Joseph et al., *Interpreting Physics
  in Video World Models*): (1) layer-wise probing, (2) iterative nullspace probing,
  (3) multi-probe subspace steering evaluated on held-out data.
- **Part 2:** open-ended extension with Goodfire's manifold/spline steering (Wurgaft et al.)
  for speed, acceleration, and direction; handle direction's circular structure; design a
  meaningful held-out steering evaluation; compare with Part 1's method.
- **Deliverables:** ~15-minute presentation + source code, sent to constantin@worldmechanics.ai.

**Hard constraints** (source: README.md, DATA.md):
- Encoder stays **frozen**.
- **Never modify `data/`.** All derived artifacts (activations, probes, figures) live outside `data/`.
- Probe fitting, layer selection, nullspace construction, and spline fitting **must not
  touch the final held-out evaluation data.**
- Document exactly how representations are extracted and pooled.

---

## 5. Verified facts and gotchas

Details and full sources are in `docs/DECISIONS.md`. Status tags:
**[verified]** = checked against primary source; **[scouting]** = from planning-chat analysis,
must be reproduced in the data audit phase; **[untested]** = verified by reading code, not yet run.

### Model
- **[verified]** Config: 24 layers, hidden 1024, 16 heads, patch 16, tubelet 2, crop 256;
  predictor 12 layers × 384. Checkpoint revision `b3c1679b7c34d3255ef3547f27c7b226aefab26f` (full hash — use this in code);
  `model.safetensors` SHA-256 `25466aef85727d16546c6cf8c99f12fcfad9cbca8225d45f23685e2e025b786b`. (checkpoint config.json; F-44)
- **[verified]** 16 input frames → 8 × 16 × 16 = **2048 tokens**, no CLS token. Token `i` →
  time `i // 256`, row `(i % 256) // 16`, col `i % 16`. (transformers modeling_vjepa2.py, RoPE position code)
- **[verified]** `frames_per_clip = 64` does not affect inference; 16 frames is fine. (config docstring)
- **[verified] Preprocessing trap:** the shipped processor resizes to 292 then center-crops 256.
  On our 256×256 clips this rescales and crops ~16 px per edge. **Resize and center-crop must be
  turned off**; keep rescale (1/255) and ImageNet normalization. (video_preprocessor_config.json)
- **[verified] Layer-indexing trap:** `hidden_states` has 25 entries (embeddings + 24 blocks), but
  the last entry is post-final-LayerNorm in v4.53.x and pre-LayerNorm in v5.x. **Use our own
  forward hooks on each encoder block**, and pin the transformers version. (source of both versions)
- **[verified] MPS → CPU float64 trap:** `t.to("cpu", torch.float64)` on an MPS tensor silently gave
  all-zero tensors. Always move first, then widen: `t.to("cpu").to(torch.float64)`. (F-55)
- **[verified] Write-back hooks must be prepended.** A forward hook's return value replaces the output only for
  hooks registered after it; transformers' `hidden_states` hooks are permanent and registered first. Use
  `edit_encoder` (prepended), never a plain `register_forward_hook`, for any edit. (F-56)
- **[verified] Predictor:** by default it gets all tokens as context and predicts all tokens
  (reconstruction, not forecasting), and the encoder has already seen the full clip. A true
  future-prediction test = encode frames 0–7 only (1024 tokens), then call the predictor
  **separately** with context positions 0–1023 and target positions 1024–2047.
  The combined `forward()` cannot do this. (modeling_vjepa2.py) Implemented in `forecast.py`; separate calls =
  combined forward bit-for-bit; forecast beats fitting-free baselines, but only ~3% better than the mean token, so
  whether it reads out motion is open (F-52, F-57); the primary steering readout is a later-layer probe on the full clip, the predictor readout
  optional (D-30). HF ships the target (EMA) encoder; the conversion asserted parity with Meta's model at atol 1e-3 (F-58).

### Data
- **[verified, all clips, F-71]** The disk is **orange** (core ≈ 234, 114, 39), not blue as DATA.md says (FFmpeg CLI and OpenCV agree pixel for pixel, F-29;
  PyAV confirms orange, F-47; PyAV vs OpenCV differ by ≤ 3 levels, F-49 → D-05 tolerance; **all 4,572 clips ok
  under D-05, F-68**).
- **[verified, all clips, F-70]** Exact mapping (≤ 0.783 px on every fully visible frame; frame 0 = start): pixel x = 128 + 32·x_world, pixel row = 128 − 32·y_world
  (32 px/m, visible area ±4 m, **y flipped**). Frame k is at t = k/24 s (clip spans 15/24 s).
- **[verified, all clips, F-70]** Angles use the math convention: 0° = right, **90° = up on screen**.
- **[verified, F-61]** All three manifests are **sorted by label** → naive head/tail splits are invalid.
- **[verified, F-61/F-63/F-72]** Direction set: no `primary_label`/`magnitude` fields; 50% constant velocity
  (1–7 m/s), 50% accelerating from rest (2–10 m/s²); starts within ±2 m.
  113 clips lose the disk for 1–7 frames; 199 are clipped or lose it (mostly 5–7 m/s).
- **[verified, F-69]** Speed and acceleration sets: disk never leaves the frame or touches the border.
- **[verified]** The background is one flat colour (29, 32, 29): frames the disk has left can be perfectly uniform
  (52 direction clips), and 149 exit frames keep faint codec residue. A uniform frame is not a decode fault. (F-66)
- **[verified, F-65] Confound:** acceleration clips start from rest, so within a dataset the magnitude
  label is perfectly correlated (r = 1.0) with distance travelled (and mean speed).
- **[verified, F-78, F-87]** Splits (D-38): `artifacts/manifests/splits.csv` (committed); 12 held-out values per dataset
  (4 val-unseen, 8 test-unseen); speed/acceleration 16/4/4 per seen value, shared by id; direction stratified group × octant.
  All flagged clips kept (D-39). Read them only via `load_joined` / `verified_artifact`.
- **[verified, F-82–F-87]** Activations: (clips, 26, 8, 1024) fp32 per dataset in `artifacts/activations/` (git-ignored,
  hash-recorded; only recovery = ~1 h 45 min re-extraction). Plot index = `hidden_states` index; final norm separate (D-40).
- **[verified, F-92–F-115] Phase 3 results:** mean-pooled probes: transition idx 1 (block_0) for all three
  variables, R² ≈ 0.99 by idx ~18, test agrees (F-104); pixel floors far below, physics ceiling above. Per-patch
  direction (39 GB per-patch array in `artifacts/patches/`): transition idx 1, off-path patches readable from block_0,
  spatial generalization near-perfect early and degrading late (unlike the paper's one-third, under our time-averaged
  interpretation). Index-0 per-patch alpha failure kept on record (F-108).
- **[verified, F-65/F-72]** Tiny motion: 48 acceleration clips move < 3 px in total; some are frozen
  for up to 11 frames after frame 0.

---

## 6. Environment

- MacBook, Apple **M3, 16 GB** unified memory, **macOS 27.0**, **Python 3.13.2** (Homebrew,
  `python@3.13` pinned) in `.venv/` at the repo root. Activate with `source .venv/bin/activate`.
  **Always install with `python -m pip`.** (F-41, D-06)
- Compute on **MPS** (Apple GPU). Only **pooled** activations are stored in bulk; per-token
  activations are computed **live** via hooks when a specific experiment needs them (D-13).
  Do not hold per-token activations for many layers in RAM (≈ 4 MB per clip per layer in fp16).
- Video decoders: **PyAV 18.1.0** (bundles FFmpeg) primary; **OpenCV** (`opencv-python-headless`
  5.0.0.93) as an independent cross-check. (D-05, F-42)
- Exact versions: `requirements.lock.txt` (53 packages; torch 2.14.0, torchvision 0.29.0, transformers 5.17.0). (F-42)
- Model files live in the HF cache (`~/.cache/huggingface/hub/`), not in the repo. (F-44)

---

## 7. Repo layout

- `data/` — supplied data. **Read-only** (chmod'd, step 0.2); fingerprint in
  `artifacts/manifests/data_fingerprint.sha256` (re-check: `shasum -a 256 -c …`). `.DS_Store`
  is never part of the data (D-19).
- `docs/` — the three source-of-truth files.
- `src/vjepa_physics/` — the installable package (D-20; `pyproject.toml` at the root, installed
  with `python -m pip install -e .`). Import as `from vjepa_physics.<module> import …`.
  Modules so far: `checkpoint` (pinned model id/revision/weights hash), `video` (`load_clip`),
  `preprocess` (`preprocess_clip`), `evidence` (`save_result` with provenance, used by check scripts; `verified_artifact` = hash-guarded read of a
  recorded artifact, use it for every artifact a check reads; chunked `file_sha256`),
  `model` (`load_model`, `weights_fingerprint`), `reproducibility` (`set_seeds`, `SEED`),
  `activations` (`capture_encoder`, `GRID`, `as_grid`), `intervention` (`edit_encoder`, `encoder_sites`),
  `forecast` (`encode`, `predict`, `training_target`), `decoders` (`load_clip_opencv`, `compare_decoders`),
  `data` (`DATASETS`, `LABEL_FIELD`, `read_manifest`, `resolve`, `read_metadata`, `load_dataset`: strict readers; paths must stay inside
  `data/<variable>/`; take labels from the dataset name, never `primary_label`, which direction lacks; `motion_group`,
  `angle_octant` for the direction set's stratification, F-75),
  `geometry` (`frame_times`, `distance_travelled`, `speed_at`: s = v t + a t²/2, frame k at t = k/fps;
  `world_to_pixel`, `disk_centres`, `distance_outside_image`: 32 px/m, origin 128, y flipped), `video` also has
  `probe_clip` (stream facts, exact-fraction timestamps), `tracking` (`disk_mask`, `count_objects`, `track_disk`;
  tracked positions for all clips in `artifacts/tracking/tracked_disk.npz`, hash in `results/tracking/checks.json`),
  `flags` (`clip_flags`: exit, clipped, sub_patch_motion, frozen_start; table `results/tracking/clip_flags.csv`, D-36),
  `splits` (`ROLES`, `SEEN_ROLES`, `HELD_OUT`, `VAL_UNSEEN`, `build_splits`, `write_splits`, `read_splits`: D-38 rules
  from metadata + seed only; speed/acceleration share one assignment by id),
  `extraction` (`SITES` = embedding, block_0–23, final_norm; `pool_time_steps`, `pooled_activations` → (B, 26, 8, 1024)
  fp32 CPU; `all_token_mean`; `plot_index` = hidden_states index, D-40),
  `joined` (`build_table`; `load_joined(variable)` = **the one loader for Phases 3–7**: per-clip labels, role, flags,
  tracked disk, theta/speed/acceleration/motion, direction group/octant, plus key "activations" (clips, 26, 8, 1024)
  memory-mapped; every file hash-guarded, activation hash pinned in the table),
  `plotting` (`DATASET_COLOUR`, ink/grid colours, `sequential_cmap`, `style_axes`: use for every figure; open and
  review each PNG after rendering),
  `metrics` (`r2`, `mae`, `sincos_targets`, `angles_from_sincos`, `circular_errors`, `circular_mae`; bootstrap
  `bootstrap_indices`, `resampled_r2`, `resampled_mean`, `percentile_interval`),
  `probes` (`site_features` all-token mean, `probe_targets`, `fit_probe` = train-only z-score + RidgeCV LOO,
  `alpha_verdict` (D-42 rule), `probe_scores`; `clip_folds`, `grouped_cv_ridge` = exact clip-grouped K-fold ridge),
  `baselines` (`physics_fit` quadratic / linear / from_rest; `pixel_matrix`, `exact_gram`, `kernel_ridge`),
  `curves` (`rise_index`, `largest_jump_index`, `transition_points`: D-43 rules), `geometry` also `pixel_to_world`,
  `distance_to_patches`, `DISK_RADIUS_PX`; `extraction` also `pool_patches`, `patch_activations` (per-patch, idx 0–24).
  Phase 3 scripts: `check_probes.py`, `check_baselines.py`, `check_layer_curves.py`, `check_patches.py`;
  gate keys in `check_evidence.py` (`code_hash_check`, `rerun_identical_probing`).
- `scripts/` — one entry script per step · `artifacts/` — large regenerable outputs, git-ignored
  except `artifacts/manifests/` · `results/` — reports, figures, metrics · `slides/` — presentation.
- Full folder roles: step 0.1 in `docs/EXECUTION_PLAN.md`. Follow them; do not invent folders.

---

## 8. How to help well

- **Small steps.** One idea at a time. Say what the step is for before showing any code.
- **Explain why before what.** The user must be able to defend every line in a follow-up interview.
- **Errors:** explain the cause first, then the fix.
- **Be precise:** point to exact file and line.
- **Library behavior:** if unsure how a library behaves, say so and suggest how to confirm it
  (read its source, run a tiny test). Never invent APIs or arguments.
- **Correctness first:** prefer checks that prove code is right (shape asserts, no-op tests,
  comparisons against a second method) over trusting it.
