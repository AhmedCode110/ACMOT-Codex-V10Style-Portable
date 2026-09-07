# AC-MOT Colab notebooks

- `ACMOT_MASTER_SWEEP.ipynb`: cache-first master replay sweep notebook for tracker, SCI, calibrator, density-gate, combined trials, leaderboard, every-5 comparison, Pareto analysis, plots, and finalist selection.
- `ACMOT_FINAL_LIVE_BENCHMARK.ipynb`: gated final live benchmark notebook for the single best finalist selected by the master sweep. It measures real end-to-end FPS on a Tesla T4 only after `ALLOW_LIVE_RUN=True`.

Safety: the master sweep notebook does not run YOLO inference. The final live notebook is also blocked by default and runs expensive YOLO inference only after the manual safety flag is changed.
