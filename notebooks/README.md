# AC-MOT Colab notebooks

- `ACMOT_MASTER_SWEEP.ipynb`: cache-first master replay sweep notebook for tracker, SCI, calibrator, density-gate, combined trials, leaderboard, every-5 comparison, Pareto analysis, plots, and finalist selection.

Safety: the master sweep notebook does not run YOLO inference. It uses cached detections only and keeps live FP16 validation as a separate manual step.
