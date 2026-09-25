# Data audit — report

Date: 2026-09-25. Every claim points to a saved check result (`results/<subject>/checks.json`, key in backticks);
each entry carries its own provenance. All 21 audit keys were re-run from a clean, committed tree and reproduced the
committed results exactly (`results/evidence/checks.json`, key `rerun_identical`). Design decisions and their reasons
are in `docs/DECISIONS.md`.

**Verdict: data audit passed**, with three failures kept on record (all explained) and the open items listed at the end.

## Data

Three datasets: direction (1,500 clips), speed (1,536), acceleration (1,536); each clip 16 frames of 256×256 at 24 fps,
one orange disk on a flat background (29, 32, 29). `data/` is unchanged since the start and read-only.

## What was verified

| Area | Result | Evidence |
|---|---|---|
| Files and manifests | Every manifest line parses; counts match DATA.md; ids 0…N−1; every path resolves inside its dataset folder and exists; no orphan, duplicate or symlinked files; 9,147 files in total | `data_files`: `manifests` |
| Data unchanged | All 9,147 SHA-256 values match the fingerprint (Python and `shasum -c`); nothing under `data/` is writable | `data_files`: `fingerprint` |
| Metadata | Every file parses strictly and agrees with its manifest row; fps 24, 16 frames; speeds and accelerations ≥ 0; labels as DATA.md states; motion type consistent with the values; acceleration clips all start from rest; manifests sorted by label | `metadata`: `consistency`, `sorted_by_label` |
| Label grids and design | 64 values per dataset (speed 0.25–4.0 m/s and acceleration 0.25–10.0 m/s², both exactly linear; directions every 5.625°); 24 clips per value (direction 23 or 24); speed and acceleration clips share their angle by id; direction set has 12 motion groups, balanced across angles (13–20 clips per octant × group cell) | `design`: `value_grids`, `design_balance` |
| Start positions | Distinct, uniform, independent of every label (0 of 70 tests flagged at p < 1e-4) | `design`: `start_positions`, `label_independence` |
| Distance confound | Within the speed and acceleration sets the label fixes the distance travelled exactly (0.625 × speed, 0.1953 × acceleration); both sets cover 0.156–1.953 m (1,176 speed and 1,440 acceleration clips), matchable to within 0.6 px; in the direction set, motion type correlates with distance (r = −0.565) | `design`: `distance_confound`, `figure_design` |
| Video format | Every clip decodes to 16 frames of 256×256 with a constant 1/24 s step from t = 0; no black frames; no two clips decode to identical pixels | `videos`: `format`, `duplicates` |
| Two decoders | PyAV (the model's decoder) and OpenCV agree within the agreed tolerance on every clip (0 flagged, 0 failed) | `videos`: `decoders` |
| Disk tracking and pixel mapping | The disk is one object in every frame where visible; on every fully visible frame the tracked centre is within 0.783 px of the metadata (32 px/m, origin at pixel 128, y flipped, frame k at k/24 s — scale and origin fitted from the data); frame 0 shows the start position in every clip | `tracking`: `track`, `mapping`, `figure_tracking` |
| Clip flags | exit (113 direction clips), clipped (199 direction), sub-patch motion (150 / 240 / 360), frozen start (92 / 1 / 267); every exit clip is also clipped; tracked and predicted displacement agree within 1.27 px; table in `results/tracking/clip_flags.csv` | `tracking`: `flags` |

Figures: `results/videos/contact_sheet.png`, `results/design/design.png`, `results/tracking/tracking.png`.

## Failures kept on record

- `metadata` `documented_fields`: DATA.md lists `primary_label` for every clip; the direction set has none. Labels are
  taken from the dataset name, so nothing depends on it.
- `tracking` `documented_colour`: DATA.md says the disk is blue; it is orange in every clip (core ≈ 234, 114, 39).
- `videos` `format`, criterion `no_uniform_frames`: 185 frames in 52 direction clips are perfectly uniform. They are the
  frames after the disk has left the image: the background is one flat colour, they always end their clip, contain no
  disk pixels, and equal the background exactly (`videos` `uniform_frames`). That diagnostic's own pre-set rule missed
  3 frames by 0.39–0.55 px (disk predicted 10.0–10.2 px outside the image against a 10.56 px radius), within the known
  mapping error plus a half-pixel offset the rule ignored; it is also kept on record. Closed without re-scoring.

## Open items and what each affects

| Item | Affects |
|---|---|
| How to treat exit and clipped clips; 149 exit frames keep faint codec residue of the departed disk | Split design |
| Sub-patch motion and frozen start mark the low end of the label range, not data defects; excluding them would cut the label range | Split design |
| Direction angles 0°–151.875° have 24 clips, the rest 23 | Choice of held-out angles |
| Labels fix the distance travelled (speed and acceleration sets); motion type correlates with distance (direction set) | Interpretation of acceleration results; confound controls |
| Not re-checked: the number of clips showing ≤ 3 distinct positions; clip losses under the checkpoint's default resize and crop (they only motivate turning both off) | Nothing |
| Hypotheses, untested: exit-frame residue is codec residue; the 3-frame miss is the pixel-centre offset; 1,885 single frames shared between clips come from identical whole-pixel disk positions | Nothing |
| The decoders' red-channel difference sits exactly at the flag threshold in every clip | Only a future decoder or version change |