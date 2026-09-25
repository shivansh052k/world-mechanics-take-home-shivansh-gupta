# Splits and activation extraction — report

Date: 2026-09-25. Every claim points to a saved check result (`results/<subject>/checks.json`, key in backticks);
each entry carries its own provenance. The re-runnable checks were re-run from a clean, committed tree and reproduced
the committed results (`results/evidence/checks.json`, key `rerun_identical_splits_extraction`). Design decisions and
their reasons are in `docs/DECISIONS.md`.

**Verdict: splits and activation extraction passed**, with two balance notes kept on record and the open items listed
at the end.

## Splits

Built from metadata and a fixed seed (0) only; no pixels or model outputs are used. Two levels per dataset:

- **Unseen values:** 12 of the 64 label values are held out whole. Speed and acceleration: every 5th value from
  index 4 (never an endpoint); direction: 12 angles spaced evenly around the circle, none cardinal or diagonal.
  Four of them form val-unseen, eight test-unseen.
- **Seen values:** the remaining 52 values' clips are split into train / val-seen / test-seen. Speed and
  acceleration: 16 / 4 / 4 clips of every seen value, with one assignment shared by both sets (they give each id the
  same angle and value). Direction: stratified on motion group × 45° octant, val-seen and test-seen one sixth each.

| Dataset | Train | Val-seen | Val-unseen | Test-seen | Test-unseen |
|---|---:|---:|---:|---:|---:|
| Direction | 813 | 203 | 94 | 203 | 187 |
| Speed | 832 | 208 | 96 | 208 | 192 |
| Acceleration | 832 | 208 | 96 | 208 | 192 |

Held-out values — val-unseen: speed 0.786 / 1.679 / 2.571 / 3.464 m/s; acceleration 1.643 / 3.964 / 6.286 /
8.607 m/s²; direction 11.25 / 101.25 / 191.25 / 281.25°.

All clips are kept, including those flagged by the data audit (exit, clipped, sub-patch motion, frozen start);
results will be broken down by flag and magnitude bin, and direction headline numbers given with and without exit
clips.

| Checked | Result | Evidence |
|---|---|---|
| Split file | Every clip exactly once; labels equal the metadata; counts as above; unseen values exactly as decided and absent from every seen role; 16/4/4 per seen value; speed and acceleration roles identical by id; a rebuild is byte-identical. Saved to `artifacts/manifests/splits.csv` (committed) | `splits`: `build` |
| Balance | Flags, motion groups, octants and angles spread over the roles (diagnostic); speed and acceleration sub-patch clips per role exactly as the design implies | `splits`: `balance` |

## How representations are extracted and pooled

- Frozen V-JEPA 2 ViT-L/16 (`facebook/vjepa2-vitl-fpc64-256`, pinned revision, weights hash-checked, fp32, eval,
  no gradients, sdpa attention), on MPS, one clip at a time.
- Input: all 16 frames at native 256×256, decoded with PyAV; the checkpoint's processor with resize and center-crop
  turned off (rescale 1/255 and ImageNet normalization kept).
- Captured with our own forward hooks at 26 sites: the patch embedding, each of the 24 blocks, and the encoder's
  final LayerNorm output.
- Pooled per time step: each site's 2,048 tokens (8 time steps × 16 × 16 patches) are averaged over the 256 patches
  of each time step, giving 8 × 1024 values per site and clip. The mean over all tokens is derived from these.
- Layer numbering for plots: 0 = patch embedding, 1–24 = blocks 1–24 (transformers' `hidden_states` index); depth
  fraction = index / 24; the final LayerNorm output is shown separately.

| Checked | Result | Evidence |
|---|---|---|
| Pooling code | On a test clip: right shape; repeat bit-exact; within 5.3e-8 (relative) of a float64 pool of the same tokens, and so is the derived all-token mean; model weights unchanged | `extraction`: `pipeline` |
| Full extraction | All 4,572 clips; every stored row read back equals what was computed; all finite, none all-zero, all distinct; weights unchanged; the test clip reproduces the standalone result bit for bit. Arrays in `artifacts/activations/` (git-ignored, hashes recorded) | `extraction`: `extract_direction`, `extract_speed`, `extract_acceleration` |
| Stored activations | Shapes and dtype; clip order equals the split file; 16 seeded clips per dataset re-extracted live, bit-identical to the stored rows (48 / 48) | `extraction`: `verify` |

## Joined tables and storage

One table per variable (`artifacts/joined/<variable>.npz`) pairs each activation row with its label, value index,
split role, audit flags, tracked disk positions, and the motion fields; it pins the activation file's hash instead of
copying the array. `vjepa_physics.joined.load_joined` is the single loader for all later analyses.

| Checked | Result | Evidence |
|---|---|---|
| Joined tables | All clips, one entry each; role and flag counts as expected; tracked disk within 1 px of the metadata on every fully visible frame (matched by id, so a misaligned join would fail); rebuild byte-identical | `joined`: `build` |
| Storage | All 10 recorded artifacts match their hashes; the loader returns memory-mapped arrays aligned with their tables; activation files exactly their array size; 3.63 GiB on disk; loading the largest array fully into RAM peaks at 2.4 GB of 17.2 GB | `joined`: `storage` |

## Clean re-run

All six re-runnable checks were re-run from a clean, committed tree; every result was identical to the committed
one, except the free-disk and memory fields of `storage`, which describe the machine. The full extraction (about
1 h 45 min) was deliberately not repeated: its three results stayed untouched, and the live re-extraction of 48
seeded clips in `verify` (bit-identical) stands in for it. The split file was unchanged.

## Notes kept on record

- One seen direction angle has no test-seen clip (stratification is by octant, not by angle); pooled test metrics
  are unaffected, a per-angle test breakdown would have a gap (`splits`: `balance`).
- Direction exit clips: 8 in val-seen vs 20 in test-seen (about 15 expected each; within chance, as exit is not a
  stratification factor). Validation scores may look slightly better than test scores; the with/without-exit
  reporting shows the size of this (`splits`: `balance`).

## Open items and what each affects

| Item | Affects |
|---|---|
| The two balance notes above await the planning chat's view; the splits are kept unchanged meanwhile | Direction evaluation |
| Activations exist only on the local disk (git-ignored); the only way to recover them is a ~1 h 45 min re-extraction, and the recorded hashes prove a restored copy | Every later analysis |
| All-token statistics are nearly identical across datasets, most likely because the flat background fills most tokens (reasoning, untested) | Interpretation of pooled probes |
| Extraction slowed from ~0.9 to ~1.5 s per clip under sustained load (thermal throttling is a hypothesis, untested) | Time budget for steering runs only |
| Peak memory with a full array in RAM was ~0.6 GB above the array plus the process (memory-mapped pages counted as resident is a hypothesis) | Nothing; far within budget |