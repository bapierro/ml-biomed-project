# Minimal segmentation pilot

**Version 1 — 24 October 2026.** Preparation is authorized and implemented; medical-data
downloads, model installation, and inference are **not authorized by this milestone**.
The prospective inference protocol below remains a draft until its resources are agreed.
Pancreas MRI is provisional pending supervisor feedback.

Use `ml-research-cycle` for this project's research work. Keep the protocol, amendments,
decisions, and handoff in this document rather than adding a tracking framework.

## Question and decision

Can a fixed pretrained MRI baseline produce a correctly aligned pancreas mask for two local
development volumes within our agreed storage, memory, and runtime limits?

This is an engineering feasibility question, not a comparison of segmentation methods.
The working hypothesis is that inference is feasible on the M1; plausible alternative
explanations for failure include invalid geometry, a wrong label mapping, software/device
incompatibility, and insufficient memory. A low Dice alone does not identify which explanation
is correct. No real-data observations or model runs exist yet.

The next decision is whether to authorize the listed files and a bounded baseline pilot.
The later scientific question about T1 versus T2 support examples is outside this milestone.

## Data roles and exact scan inventory

Source: [CrossPan](https://huggingface.co/datasets/linkai-peng/CrossPan), revision
`0da3821e8af2dfa4c22d56e282372614cb426227`. License metadata: **CC BY-NC 4.0**.
Sizes and SHA-256 values below were read from Hugging Face LFS metadata on 24 October 2026;
the actual files have not been downloaded or inspected.

| File within the dataset repository | Compressed bytes |
| --- | ---: |
| `T1/val/images/EMC014_T1_0000.nii.gz` | 594,820 |
| `T1/val/labels/EMC014_T1.nii.gz` | 1,828 |
| `T2/val/images/CAD_T2_110_0000.nii.gz` | 2,535,679 |
| `T2/val/labels/CAD_T2_110.nii.gz` | 5,516 |
| **Total** | **3,137,843 (2.99 MiB)** |

Expected SHA-256, in the same order:

```text
969229d4033a7ce17611161624c133422527947a5d45c80a20f5c1298e101e03
b60656e2d411288f9d3718e4d96b33850db9f78f4d020b6f3826ac794074f62f
5813e08252c120ca3af773b4034be0bce6cafafd777d63b543fa6bb42cb12dff
126efce387e2ae5faf976453f7a4c92650918eaae9c08efc1e6c112091642e31
```

Proposed local destination: `data/crosspan/`, preserving the listed paths. Select these two
cases before seeing their contents or results: they are the first lexicographic image names
in each sequence's validation folder. This convenience selection is not representative.

Explicitly assign both cases to **development and debugging**. Their annotations may be used
for inspection and evaluation, but not for inference crops, prompts, or tuning until the
protocol is amended. There is no training or support set in this pilot. Final-test files
remain untouched. Different filenames do not prove independent patients, and these T1/T2
cases are not a paired-subject experiment. Patient identity, cross-split duplicates, site
provenance, label conventions, and possible pretrained-model overlap remain unverified.

## Proposed baseline and complete weight inventory

Propose **TotalSegmentator 2.18.0**, without fine-tuning, using `total_mr` at its default
1.5 mm internal resolution. Code tag `v2.18.0` resolves to commit
`184fa4765456492b2ab81a981cc387465906b003`; the package has not been installed.

The [task configuration](https://github.com/wasserth/TotalSegmentator/blob/184fa4765456492b2ab81a981cc387465906b003/totalsegmentator/map_tasks_config.py#L64-L82)
requires **both** model IDs 850 and 851. The [inference code](https://github.com/wasserth/TotalSegmentator/blob/184fa4765456492b2ab81a981cc387465906b003/totalsegmentator/python_api.py#L317-L321)
downloads each required model. Neither cropping nor another model is requested in this
configuration: no `--fast`, `--fastest`, `--roi_subset`, or body/crop option.

| Required archive in release `v2.5.0-weights` | Asset ID | Compressed bytes |
| --- | ---: | ---: |
| [Dataset850_TotalSegMRI_part1_organs_1088subj.zip](https://github.com/wasserth/TotalSegmentator/releases/download/v2.5.0-weights/Dataset850_TotalSegMRI_part1_organs_1088subj.zip) | 220660581 | 231,681,650 |
| [Dataset851_TotalSegMRI_part2_muscles_1088subj.zip](https://github.com/wasserth/TotalSegmentator/releases/download/v2.5.0-weights/Dataset851_TotalSegMRI_part2_muscles_1088subj.zip) | 220660543 | 232,156,399 |
| **Weight archive total** | | **463,838,049 (442.35 MiB)** |

Sizes are verified against [GitHub release asset metadata](https://api.github.com/repos/wasserth/TotalSegmentator/releases/tags/v2.5.0-weights),
not against downloaded bytes. These assets publish no SHA-256 digest in that metadata;
record a local digest after any future approved retrieval. Asset IDs identify the observed
assets but are not content checksums. A release tag is not an immutable content guarantee.

Proposed weight destination: `models/totalsegmentator/`, using the upstream
`TOTALSEG_WEIGHTS_PATH` setting. The default normally writes outside the project; control
that location explicitly before running. Code is Apache-2.0; separately verify checkpoint
terms before retrieval rather than assuming the code license establishes them.

**Scans plus default weights: 466,975,892 bytes (445.34 MiB) compressed.** This excludes
the model package and its dependencies, unpacked archives, caches, predictions, and temporary
files. Those sizes and peak memory/runtime are **unknown**, not zero. No installed-space
estimate is presented as a verified measurement. Dependency resolution against the existing
PyTorch pin also remains to be checked before installation.

Smaller alternatives are not the chosen baseline: `--fast` uses model 852 at 3 mm
(126,399,352 compressed bytes), and `--fastest` uses model 853 at 6 mm (44,657,261 bytes).
Their different resolution changes the method. A pancreas-only ROI request also invokes a
cropping model, so it cannot be assumed to mean downloading just model 850.

## Prospective pilot protocol — not executed

- Run one case at a time on the M1 with explicit MPS selection. No training, tuning, support
  examples, or repeated searches for a favorable score. One attempt per case initially;
  retain any later retry and its reason alongside the original failure.
- Inspect image/label geometry and label meanings first. Use the complete image as model
  input. Extract the pancreas binary output without a reference-derived crop; the pinned
  [multilabel map](https://github.com/wasserth/TotalSegmentator/blob/184fa4765456492b2ab81a981cc387465906b003/totalsegmentator/map_to_binary.py#L232-L240)
  assigns pancreas label 7, but verify the actual output convention before conversion.
- Before scoring, require matching shapes, affines, spatial units, and verified binary masks.
  Do not silently resample or threshold to make a case pass. The notebook checks geometry;
  the shared `dice_score` function validates only arrays, not patient identity or geometry.
- Primary descriptive metric: one whole-volume Dice per case, using all voxels. Report cases
  separately with failures and undefined scores; no slice averaging, pooled two-case score,
  significance test, confidence interval, or T1-versus-T2 generalization claim.
- Dice is `2 * intersection / (prediction foreground + reference foreground)`. One empty
  mask against a nonempty mask scores 0; two empty masks return NaN and require explicit
  review. Verify any empty reference before using it for this pancreas pilot.
- Technical success means an inference attempt completes, output geometry and labels are
  valid, and the case can be inspected and evaluated within the approved budget. No minimum
  Dice establishes success or superiority here. A valid low score is a pilot observation,
  while a broken input/evaluator/device is an implementation failure.
- Preserve the code commit, `uv.lock`, command and options, software/checkpoint versions,
  verified file hashes, device, predictions, terminal status, elapsed time, peak memory, and
  disk growth. Keep local artifacts under `results/`; record the outcome and artifact links
  here. No run records or scores are fabricated during preparation.
- Before execution, agree on additional storage/download limits, maximum time per case and
  total time, and a memory stop condition. None is approved by the existence of this document.
  Stop affected work on invalid geometry/mapping, resource exhaustion, or a human stop.
  Do not automatically switch model settings or consume final-test cases after a failure.

The [CrossPan paper, sections 5.1 and 5.6](https://arxiv.org/html/2604.18797v1) motivates
the baseline, but its exact software/checkpoint recipe has not been established here. This
modern two-case feasibility check would not reproduce its reported cohort-level scores.

## Preparation evidence and handoff

- `notebooks/01-inspect-image-and-mask.ipynb` uses synthetic data by default and has no
  network calls. It checks alignment and plots the reference overlay. Its label-informed
  slice selection is for visualization only. Real-data mode requires explicitly provided
  local paths and does not invent a model prediction.
- `src/ml_biomed_project/metrics.py` contains the sole new shared function, `dice_score`.
  The evaluator's Git revision is its version; changes require rerunning affected checks
  and reviewing any future dependent experiment results.
- Verification: known-answer Dice tests, invalid-input checks, executable synthetic
  alignment-error examples, and visual review of the generated overlay. Only software and
  synthetic checks are performed; these do not establish medical segmentation feasibility.
- Current research status: **hypothesis**, with no real-data pilot observations. Preparation
  is ready for review once its checks pass. Review is local, with no independent reviewer.
- Next action: review this inventory and the missing resource limits together, then authorize
  a bounded download/install/inference step if the provisional direction still makes sense.

Preserve dated amendments in Git. Changes motivated by future observed results must be
identified as exploratory rather than presented as choices fixed before those results.
