# AC-MOT Replay Versioning Rules

- Every optimization iteration is a separate Colab notebook: `ACMOT_Replay_Optimization_V1.ipynb`, `V2`, `V3`, ...
- Every version saves outputs under `MyDrive/VisDrone_Results/ACMOT_REPLAY_VERSIONS/V<number>/`.
- Reuse valid cached detections first. Do not rerun YOLO for replay development.
- V1-V4 each test one controlled experiment family.
- V5 automatically compares the best official TrackEval result from V1-V5.
- V10 compares V6-V10, and so on every 5 versions.
- Keep replay results labeled as `FP32_REPLAY_DEVELOPMENT` unless a later notebook explicitly uses another validated cache family.
- Do not treat replay throughput as live FPS.
- Only the strongest replay candidates should receive a later live FP16 benchmark.
- Never overwrite the frozen final run `codex_v10_p4_fp16_20260907_134125`.
