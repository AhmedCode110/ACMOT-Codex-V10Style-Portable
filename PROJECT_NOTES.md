# Project notes

This folder is intentionally separate from the main AC-MOT project.

## Why this version exists

Version 12 added stronger safety and reproducibility checks, but it also expanded the controller logic. This version returns to the simpler v10/presentation design so the research story stays clean:

- detection: YOLOv8n FP32
- tracking: ByteTrack
- scene intelligence: lightweight SCI
- adaptation: confidence / IoU / image-size ladder only
- production systems: three clear comparisons

## What was kept from v12

- strict dataset preflight
- no silent missing-GT or missing-frame skip
- real live timing on T4
- saved run configuration
- prediction export
- official TrackEval step for HOTA/CLEAR/Identity

## What was intentionally removed from production logic

- detector-feedback controller
- recovery-only mode
- adaptive birth threshold
- adaptive NMS
- cycling controller
- ReID remapping

Those ideas can still be studied as ablations later, but they should not be mixed into the main AC-MOT claim unless a separate validation run proves they help.

## Research claim this version can support if the run succeeds

A lightweight, training-free scene-complexity index can guide resolution and detector threshold adaptation for UAV multi-object tracking, improving the quality/speed trade-off over a default detector-tracker baseline.

The notebook must be run before making any numerical claim.

