# Research discussion reference

Status: provisional; final dataset, anatomy, and model choices require supervisor feedback.
The following is the supplied discussion summary, preserved as a reference on 24 October 2026.
Reported scores are literature results, not results produced by this project. No scans or model
weights were downloaded during environment setup. The current repository name deliberately
leaves room for a different research direction. Re-check dataset licenses and checkpoint
versions before the first experiment.

## Summary

**The leading project direction is now pancreas segmentation across T1- and T2-weighted MRI, using pretrained models on CrossPan.** This remains a proposed direction; we have not yet run the experiments.

- **Constraints:** a three-person student project, an Apple M1 from 2020, limited storage and compute, and a poster presentation. We therefore favoured inference with pretrained models over training from scratch.
- **How the topic evolved:** we considered Duke OCT, StructSeg Task 3, ACDC and M&Ms. Following your supervisor’s advice, the focus shifted toward a challenging organ and a documented failure of an established segmentation system.
- **Architectural takeaway:** nnU-Net remains an important baseline. Claims that newer architectures are better need careful scrutiny of training data, preprocessing, compute and evaluation. *nnU-Net Revisited* directly investigates these issues and innovation bias. [arxiv.org](https://arxiv.org/abs/2404.09556?utm_source=chatgpt.com)
- **Contribution:** a focused, reproducible study of a failure mode and a potential improvement could be worthwhile. The project does not need to introduce a new architecture or guarantee that UniverSeg beats TotalSegmentator.

## Main findings

### CrossPan gives us a concrete problem

CrossPan reports these mean pancreas Dice scores:

| Model and evaluation setting | T1 MRI | T2 MRI |
|---|---:|---:|
| nnU-Net **trained on T1** | 0.835 | 0.00027 |
| Pretrained **TotalSegmentator MRI**, without task-specific fine-tuning | 0.474 | 0.152 |
| Pretrained **MedSAM2**, without task-specific fine-tuning | 0.597 | 0.573 |

The nnU-Net row measures **T1 training → T2 testing**. The pretrained rows evaluate fixed checkpoints on different sequence cohorts. CrossPan already compares TotalSegmentator with **MedSAM2**, which is distinct from original MedSAM. [arxiv.org](https://arxiv.org/html/2604.18797v1?utm_source=chatgpt.com)

### Proposed research question

> **Can a few labeled examples from the target MRI sequence improve pancreas segmentation without retraining?**

The suggested core experiment is:

1. Establish **TotalSegmentator MRI** performance on T1 and T2.
2. Evaluate **UniverSeg on the same T2 patients**, first with T1 support examples and then with an equal-sized T2 support set.
3. Optionally add **MedSAM2**, explicitly reporting its patient-specific prompts.

UniverSeg supports this experiment because it conditions predictions on labeled examples supplied at inference, without updating its weights. [github.com](https://github.com/JJGO/UniverSeg?utm_source=chatgpt.com)

The potential contribution is the **controlled support-sequence comparison**. Repeating the published TotalSegmentator–MedSAM2 comparison would mainly be reproduction work. Novelty beyond that still needs verification.

### Feasibility

- CrossPan’s **T1 test folder is 972 MB** and **T2 test folder is 778 MB**: approximately **1.75 GB compressed combined**, plus weights and a small support/development subset. [huggingface.co](https://huggingface.co/datasets/linkai-peng/CrossPan/tree/main/T1/test?utm_source=chatgpt.com)
- **UniverSeg:** approximately 1.18 million parameters; official input workflow uses 128 × 128 images.
- **TotalSegmentator:** documents Apple M-series acceleration through `--device mps`.
- **MedSAM2:** its official installation is CUDA-oriented; practical M1 compatibility and runtime remain unverified. A small inference pilot is necessary. [arxiv.org](https://arxiv.org/html/2304.06131v1?utm_source=chatgpt.com)

### Qualifications we should preserve

- **Training overlap:** CrossPan extends earlier pancreas MRI data. A new dataset release does not automatically mean unseen patients for every pretrained checkpoint. [arxiv.org](https://arxiv.org/html/2604.18797v1?utm_source=chatgpt.com)
- **Sequence versus cohort:** different T1/T2 patients or institutions can confound the comparison. Keeping the T2 query patients fixed strengthens the support experiment.
- **Different information budgets:** automatic segmentation, labeled support examples and query-specific prompts are different workflows.
- **Evaluation:** use separate support/test patients, complete-volume patient-level metrics, repeated support selections and documented preprocessing.
- **Claims:** synthetic augmentation alone would not establish real-world OOD robustness; a Dice drop alone would not establish universal clinical unusability.
- **Baseline first:** reproduce TotalSegmentator’s behaviour with a pinned checkpoint before assuming its published failure persists.

## Sources and practical resources

### Core reading for the current direction

| Source | Why it matters | Links |
|---|---|---|
| **CrossPan — Peng et al., MIDL 2026** | Central benchmark and reported sequence-transfer failures. | [Paper](https://arxiv.org/abs/2604.18797) · [Project](https://crosspan.netlify.app/) · [Code](https://github.com/NUBagciLab/nnUNetDG) · [Dataset](https://huggingface.co/datasets/linkai-peng/CrossPan) |
| **TotalSegmentator MRI — Akinci D’Antonoli et al., Radiology 2025** | Automatic pretrained MRI baseline. | [Paper](https://pubs.rsna.org/doi/10.1148/radiol.241613) · [Code, weights and documentation](https://github.com/wasserth/TotalSegmentator) |
| **UniverSeg — Butoi et al., ICCV 2023** | Segmentation using labeled support examples without task-specific retraining. | [Paper](https://arxiv.org/abs/2304.06131) · [Code and weights](https://github.com/JJGO/UniverSeg) · [Project](https://universeg.csail.mit.edu/) |
| **MedSAM2 — Ma et al., 2025** | Optional prompted volumetric comparator. | [Paper](https://arxiv.org/abs/2504.03600) · [Code](https://github.com/bowang-lab/MedSAM2) · [Checkpoints](https://huggingface.co/wanglab/MedSAM2) |
| **nnU-Net Revisited — Isensee et al., MICCAI 2024** | Strong baselines, rigorous evaluation and innovation bias. | [Paper](https://arxiv.org/abs/2404.09556) · [nnU-Net framework](https://github.com/MIC-DKFZ/nnUNet) |

**Direct CrossPan downloads:** [T1 test images and labels](https://huggingface.co/datasets/linkai-peng/CrossPan/tree/main/T1/test) · [T2 test images and labels](https://huggingface.co/datasets/linkai-peng/CrossPan/tree/main/T2/test).

### Additional literature and alternatives explored

| Resource | Relevance | Links |
|---|---|---|
| **PanSegNet / PanSegData** | Pancreas-specific model, pretrained checkpoints, and earlier MRI data relevant to CrossPan’s provenance. | [Paper](https://arxiv.org/abs/2405.12367) · [Code and checkpoint links](https://github.com/NUBagciLab/PANSegNet) · [Data](https://osf.io/kysnj/) |
| **AbdoBench** | Comparison of abdominal MRI segmentation tools; provides evaluation code and prepared datasets. | [Paper](https://arxiv.org/abs/2507.17971) · [Repository and data links](https://github.com/deepakri201/AbdoBench) |
| **MultiverSeg — ICCV 2025** | Related approach combining labeled examples with interactive segmentation. | [Paper](https://arxiv.org/abs/2412.15058) · [Project](https://multiverseg.csail.mit.edu/) |
| **Primus** | Transformer-based 3D segmentation considered earlier; we have not established an appropriate pretrained checkpoint for the proposed task. | [Official documentation](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/primus.md) |
| **Original MedSAM** | Background for distinguishing the original method from MedSAM2. | [Paper](https://arxiv.org/abs/2304.12306) · [Code](https://github.com/bowang-lab/MedSAM) |

### Earlier dataset options

- **[ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html):** cardiac MRI; our initial manageable starting point.
- **[M&Ms](https://www.ub.edu/mnms/):** cardiac MRI across centres and scanner vendors; considered for studying acquisition-related generalization.
- **[StructSeg 2019](https://structseg2019.grand-challenge.org/Dataset/):** Task 3 covers organs at risk in chest CT.

**Recommended reading order:** CrossPan → UniverSeg → TotalSegmentator MRI → nnU-Net Revisited, followed by MedSAM2 if you include the prompted comparison.
