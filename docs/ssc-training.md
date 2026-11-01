# Training with SSC

The default method is **SSC4**: four perpendicular neighbour-patch comparisons at
radius 1, averaged over 3×3 patches. SSC8 concatenates a separately normalized
radius-2 block. Both use the same fixed feature calculation and native nnU-Net
training machinery. There is no raw-intensity feature channel.

## Prepare CrossPan once

Download only the pinned source subset and prepare it with one command:

```sh
python scripts/prepare_crosspan.py --download
```

The helper downloads **369 T1-weighted MRI volumes and their segmentation masks**:
323 training and 46 validation volumes. Each volume has one 3D MRI NIfTI file and
one matching 3D mask NIfTI file, totaling 738 files. The training loader extracts
2D slices from these volumes; slices are not stored as separate NIfTI files.
The helper uses the exact catalog paths at the revision in
`configs/crosspan-t1/source.json`, with two Hugging Face download workers.
It never requests TEST, T2 or a full snapshot. Files default to `data/crosspan/`;
budget roughly **3.9 GB for images/masks**, plus staging copies and preprocessing
overhead. Existing source or staged files are checksum-verified and reused;
only missing paths are downloaded, and no network request occurs when all exist.
Corrupt present files stop preparation before network access; they are not silently repaired.

Alternatively, supply an existing local release with this layout:

```text
CrossPan/
  T1/train/images/   T1/train/labels/
  T1/val/images/     T1/val/labels/
```

Run `scripts/prepare_crosspan.py /path/to/CrossPan` in the installed project environment.
The small notebook sample does not contain enough training data. Downloads are
opt-in: without `--download`, the local source argument is required. Both routes
check catalog SHA256 values, stage the listed source cases and call nnU-Net's
preprocessor with one worker. There are no target/test cases in this step.
`--stage-only` downloads when requested and stages files/metadata without preprocessing.

The versioned reference lives in `configs/crosspan-t1/`: `cases.csv`, the official
`splits_final.json`, the package-derived `SSC2D.json` plan, the original
`dataset_fingerprint.json`, and `source.json` provenance.
The public dataset is named `Dataset801_CrossPanT1`; the plans are `SSC2D`. These
shorter identifiers preserve the pilot's numerical settings: patch 256×320,
batch 10, 1.25 mm in-plane spacing, one MRI input and binary pancreas labels.
SSC4/SSC8 share this plan, so changing features does not trigger replanning.

The release has 324 nominal TRAIN cases. `MCF_T1_0090` duplicates the image in VAL
`MCF_T1_0067`; we retain VAL and exclude that TRAIN entry, giving **323/46**.
Within-TRAIN aliases remain. Publisher IDs do not establish biological patient
independence. Original image/mask bytes, including known geometry warnings, are
preserved; the helper does not repair labels or silently discard further cases.
The original planning fingerprint included development VAL, so these are
development-validation results, not an untouched source test.

For another dataset, use the
[standard nnU-Net data format and preparation](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md)
and supply its plans/split to the training command. The SSC trainer expects a
single MRI input and a 2D configuration. CrossPan preparation is deliberately
separate from the reusable training adapter.

## Train and resume

```sh
python scripts/train_ssc.py 801 --device mps
python scripts/train_ssc.py 801 --device mps --channels 8
```

The first command uses SSC4. Defaults are 100 epochs, 250 updates/epoch, seed 1729
and fold 0. Omit `--device` to select CUDA, then MPS, then CPU automatically.
`--epochs` and `--seed` make intentional new studies explicit;
`--plans` selects another prepared plan. Inspect all options with `--help` or
print the resolved native command with `--dry-run`.

Use `--resume` with the same settings to continue an interrupted fit. Existing
fits are protected from accidental fresh starts, and resume requires an actual
checkpoint and matching recorded settings. If training finished but final validation
was interrupted, it resumes from the final checkpoint without extra training epochs.
Recorded complete runs are not repeated. Use a different `--workspace` for a
new seed/budget of the same method. SSC4 and SSC8 have separate trainer namespaces.

The launcher passes paths and settings to the native training entry point.
Native augmentation, Dice + cross-entropy loss, SGD, polynomial learning rate,
checkpointing and full-volume validation remain in nnU-Net. Checkpoints are saved
each epoch for interruption recovery. The recorded seed does not make asynchronous
augmentation or cross-device numerics bitwise deterministic.

Run one expensive job at a time on the M1 Mac.

## Files and inference

Default workspace:

```text
data/nnunet/
  raw/Dataset801_CrossPanT1/
  preprocessed/Dataset801_CrossPanT1/
  results/Dataset801_CrossPanT1/nnUNetTrainerSSC4__SSC2D__2d/fold_0/
```

The native fold folder contains checkpoints, logs, final `validation/summary.json`
and the launcher's `run.json` with resolved settings and input/code identities.
Validation patch Dice in the log differs from the final whole-volume result.
Neither validation scores nor T2 feedback automatically changes the training budget.

These are ordinary native nnU-Net checkpoints with a custom trainer. To use
`nnUNetv2_predict`, expose the same workspace paths and the trainer directory:

```sh
export nnUNet_raw="$PWD/data/nnunet/raw"
export nnUNet_preprocessed="$PWD/data/nnunet/preprocessed"
export nnUNet_results="$PWD/data/nnunet/results"
export nnUNet_extTrainer="$PWD/src/ml_biomed_project/ssc"
nnUNetv2_predict -i /path/to/images -o results/predictions -d 801 -c 2d -f 0 -tr nnUNetTrainerSSC4 -p SSC2D -device mps
```

This follows nnU-Net's documented
[external trainer mechanism](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how-to/share-models-with-custom-trainers.md).
The notebook's `load_model` helper loads the specifically pinned SSC4 demo checkpoint;
use the native prediction route above for newly trained models.
