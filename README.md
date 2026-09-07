# AC-MOT Codex v10-style Portable

This is a separate Codex-owned experiment folder. It does not replace the existing AC-MOT / ChatGPT project.

Current version: `v10_p2`

Purpose:

- Return to the simple presentation/v10 research story.
- Keep the production comparison clean:
  1. `Baseline_Default`
  2. `Baseline_TunedTracker`
  3. `ACMOT_V10STYLE_SCI`
- Use SCI + adaptive threshold + adaptive resolution only for the AC-MOT system.
- Keep ReID and heavier controller ideas out of the production system.
- Use real live timing on Google Colab T4.
- Stage the dataset from Google Drive to `/content` before timing, so FPS is not dominated by slow Drive reads.
- Export MOT-format predictions and run official TrackEval for HOTA/CLEAR/Identity where TrackEval is available.

Important notes:

- This notebook does not claim better results before running.
- Final benchmark mode requires a complete 17-sequence VisDrone test-dev dataset.
- If frames or GT are missing, the notebook stops before the expensive run.
- It saves every run configuration, sequence manifest, timing file, prediction files, and final tables to Google Drive.
- v10_p2 keeps the v10/presentation logic; it changes the timing protocol, not the tracking idea.

Main Colab notebook:

- `AC_MOT_Codex_v10style_Portable.ipynb`

After this folder is pushed to GitHub, open the notebook in Colab from:

`https://colab.research.google.com/github/AhmedCode110/ACMOT-Codex-V10Style-Portable/blob/main/AC_MOT_Codex_v10style_Portable.ipynb`

Default Colab paths:

- Dataset: `/content/drive/MyDrive/visdrone/VisDrone_Zips/VisDrone2019-MOT-test-dev/VisDrone2019-MOT-test-dev`
- Output: `/content/drive/MyDrive/VisDrone_Results/ACMOT_CODEX_V10STYLE`
- Work: `/content/acmot_codex_v10style`
- Local timing dataset: `/content/acmot_codex_v10style/dataset_local/VisDrone2019-MOT-test-dev`

How to run on any Colab account:

1. Open the Colab link above.
2. Runtime -> Change runtime type -> T4 GPU.
3. Make sure the Google Drive account has the VisDrone dataset at the default path, or edit `DATASET` in Cell 1.
4. Run cells from top to bottom.
5. Cell 2 checks Drive, then copies the verified dataset to `/content`.
6. If Cell 2 reports missing frames or GT, fix the dataset before running the live benchmark.

Version rule:

- If the next experiment changes only protocol/bugfixes, name it `v10_p3`, `v10_p4`, etc.
- If the method changes beyond the presentation/v10 logic, it should become a separate branch or ablation, not the main production claim.
