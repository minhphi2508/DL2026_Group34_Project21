# Source Lock — Dataset v1

The source set is frozen to these four sources.

| Code | Source | Role | Train? | Paired GT? |
|---|---|---|---:|---:|
| HKM | NatLibFi/Finna-HKM-images | Core historical GT + natural damage holdout | Yes | Core becomes paired after our degradation |
| JOKA | NatLibFi/Finna-JOKA-images | Core historical GT + natural damage holdout | Yes | Core becomes paired after our degradation |
| SYNOLD | wushunshun/SynOld | Auxiliary scratch train + external scratch test | Optional train | Yes |
| MS-OLD | microsoft/Bringing-Old-Photos-Back-to-Life test images | Frozen real qualitative test | No | No |

## Fixed Finna core quota
- 400 HKM
- 400 JOKA
- source-stratified split:
  - train = 300 + 300
  - val = 50 + 50
  - test = 50 + 50

## Source policy
- No fifth source is added to `dataset_v1`.
- Any future source expansion creates `dataset_v2`.
- External benchmark images never enter Finna core train/val/test.
- Original-source image files are kept untouched until after the split is frozen.
