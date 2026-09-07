# AC-MOT Codex v10-style experiment log

This file tracks what each portable version is testing, what problem it fixed, and what result we are waiting for.

## v10_p1

Status: built and pushed to GitHub.

Purpose:

- Recreate the clean v10 / presentation logic in a separate Codex project.
- Compare:
  1. `Baseline_Default`
  2. `Baseline_TunedTracker`
  3. `ACMOT_V10STYLE_SCI`

Problem found during live run:

- FPS appeared extremely low when reading thousands of frames directly from Google Drive.
- This makes the timing number reflect Drive I/O, not only the actual detector/tracker pipeline.

Decision:

- Do not treat v10_p1 FPS as the final real-time number.
- Keep the method logic, but fix the timing protocol.

## v10_p2

Status: current version.

What changed:

- Dataset is first verified on Google Drive.
- Then the 17 sequence folders and 17 annotation files are copied to `/content`.
- Live benchmark reads frames from local `/content`, not directly from Drive.
- The run configuration records that Drive I/O is excluded from FPS and staging time is separate.

What did not change:

- Same YOLOv8n FP32 detector.
- Same ByteTrack baseline/tuned comparison.
- Same v10-style SCI.
- Same adaptive confidence / IoU / image-size ladder.
- No ReID in the production system.
- No detector-feedback controller.
- No adaptive birth / adaptive NMS / recovery controller in the production system.

Why this stays inside presentation scope:

- The research idea is still the simple v10/presentation story.
- The change only makes the real-time measurement fairer and cleaner.
- It does not add a new algorithmic claim.

What we are testing now:

- Whether `ACMOT_V10STYLE_SCI` improves MOTA/IDF1/HOTA/IDS against:
  - `Baseline_Default`
  - `Baseline_TunedTracker`
- Whether the local `/content` live timing is close enough to real-time for the thesis/journal story.

If v10_p2 result is good:

- Treat v10_p2 as the final candidate.
- Use TrackEval HOTA/CLEAR/Identity outputs as the official numbers.
- Use the saved run configuration and CSVs for presentation/paper reproducibility.

If v10_p2 result is not good:

- Create `v10_p3`.
- Only improve within the same v10 scope first:
  - tune SCI thresholds,
  - tune ByteTrack thresholds,
  - tune 640/736/832 switching thresholds,
  - keep ReID and heavy controller features as ablation-only unless they clearly prove improvement.

