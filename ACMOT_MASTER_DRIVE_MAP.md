# AC-MOT Master Drive Map

This file describes the intended logical organization. The live registry records actual accessible paths from the mounted Google Drive account.

## Preferred Shared Root

```text
/content/drive/MyDrive/ACMOT_MASTER
```

If the folder is shared through another Google account, add it as a shortcut into `My Drive` before running discovery.

Known accounts:

```text
a7medgouda1@gmail.com
a7medgoda1@gmail.com
```

## Important Existing Paths

Dataset:

```text
/content/drive/MyDrive/visdrone/VisDrone_Zips/VisDrone2019-MOT-test-dev/VisDrone2019-MOT-test-dev
```

Main v10 style results:

```text
/content/drive/MyDrive/VisDrone_Results/ACMOT_CODEX_V10STYLE
```

Legacy AC-MOT IDs results:

```text
/content/drive/MyDrive/VisDrone_Results/ACMOT_IDS
```

Persistent cache registry:

```text
/content/drive/MyDrive/VisDrone_Results/ACMOT_CACHE/registry/cache_registry.json
```

## Frozen Runs

```text
codex_v10_p4_fp16_20260907_134125
```

Status: `FROZEN_FINAL`

Expected path hint:

```text
/content/drive/MyDrive/VisDrone_Results/ACMOT_CODEX_V10STYLE/codex_v10_p4_fp16_20260907_134125
```

## Cross-Account Rule

Paths can change between Google accounts. The registry stores both Drive ID when available and path hints. If a shared folder is visible in Drive UI but not mounted in Colab, discovery should report it as not currently visible rather than declaring it missing.

## Physical Moves

Do not move old runs or large caches during discovery. Reference them in place unless a later migration is explicitly approved.
