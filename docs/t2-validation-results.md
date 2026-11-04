# Current T2 validation results

4 November 2026. The SSC4-only, SSC4+MRI dropout and SSC4+MRI no-dropout
terminal100 comparison is complete and audited. All three fits use 323 T1 TRAIN /
46 development VAL cases, seed 1729 and 100 epochs. The two fusion models each
have raw-enabled and raw-muted inference policies: three fits, five policies.

Publisher T2/val: 73 scans; frozen allocation 56 scans / 53 filename-derived groups after 15 T1-development identifier overlaps and two unresolved aliases were excluded. Model-blind QC retained 56 scans / 53 groups. Primary T2 Dice averages scans within each group, then groups equally. Filename keys are not verified biological identities.

## Current SSC4 comparison

| Fit / inference policy | T1 validation case Dice, 46 cases | T2 validation group Dice, 53 groups | T2 validation scan Dice, 56 scans |
|---|---:|---:|---:|
| SSC4-only | 0.685556 | 0.497679 | 0.489867 |
| SSC4 + MRI, dropout p=0.5, raw enabled | 0.745106 | 0.058165 | 0.055814 |
| Same dropout weights, raw muted | 0.674708 | 0.488032 | 0.479569 |
| SSC4 + MRI, no dropout, raw enabled | 0.764545 | 0.005784 | 0.005475 |
| Same no-dropout weights, raw muted | 0.016356 | 0.010166 | 0.009621 |

Muting zeros raw feature 4 after unchanged SSC4. Each row is one policy on both
modalities; raw-enabled T1 and raw-muted T2 are not combined. No-dropout muting
changes T1 Dice by −0.748189 (unadjusted paired 95% CI [−0.777906, −0.714432])
and T2 by +0.004381 ([−0.004623, +0.015138]): it does not recover useful T2
performance. Dropout's useful muted mode remains slightly below the SSC4-only
means. One fit per recipe, exposed validation, unmatched resumed streams and
cached-versus-fresh source preprocessing limit the interpretation; subject/group
intervals do not measure training-seed robustness or establish equivalence.

Released weights and portable loaders:

- [SSC4-only](https://github.com/bapierro/ml-biomed-project/releases/tag/ssc4-demo-v1)
- [SSC4 + MRI with dropout](https://github.com/bapierro/ml-biomed-project/releases/tag/ssc4-raw-dropout-v1)
- [SSC4 + MRI without dropout](https://github.com/bapierro/ml-biomed-project/releases/tag/ssc4-raw-no-dropout-v1)

The [comparison notebook](../notebooks/ssc-domain-generalization.ipynb) includes
the five policies, four within-model muting intervals, matched 100-epoch source
histories and direct checkpoint links. Its new reporting cells use checked-in
scalar files; the saved-prediction browser and optional inference remain SSC4-only.
The original SSC4 release's demo scores concern an older test cohort, separately
from these validation results; the checkpoint itself is unchanged.

Local audited evidence:
`research/acquisition-shift/nnunet-ssc4-raw-no-drop-2026-11-03/terminal100/results/comparison.json`.
Portable reporting data: `notebooks/data/ssc4-terminal100-comparison.json` and
`notebooks/data/ssc4-terminal100-training.csv`.

## Earlier SSC8 ablations and context (2 November 2026)

The historical `DROP` and `STAGED` conditions below use SSC8 and are distinct
from the newly trained SSC4 fusion models above. Their retained values are
unchanged; no fit or inference was repeated for this documentation update.

| Existing model / condition | T1 validation Dice | T2 validation group Dice | T2 validation scan Dice |
|---|---:|---:|---:|
| SSC8-only | 0.689468 | 0.477886 | 0.466790 |
| SSC8 + MRI, STAGED | 0.737498 | 0.076602 | 0.072501 |
| SSC8 + MRI, legacy DROP | 0.726784 | 0.275031 | 0.266388 |
| SSC8 STAGED, raw muted | 0.615103 | 0.427641 | 0.414793 |
| SSC8 legacy DROP, raw muted | Not measured | 0.495315 | 0.482011 |

These replace earlier T2 test values in current development reporting. Historical test records remain unchanged. Per user clarification, earlier test evaluations were not used for tuning. Reserve further test evaluation for the final frozen comparison. The stopped consistency run remains incomplete and was not resumed.

No confidence intervals over training seeds are claimed. Raw-muted T1 was not
measured for the legacy SSC8 DROP model; it was measured for the current SSC4
dropout model above. SSC8 staged raw-muted T1 reuses its retained prior source
evaluation; no new T1 inference was run for this historical context.

Local evidence: research/acquisition-shift/t2-validation-2026-11-02/results/comparison.json
