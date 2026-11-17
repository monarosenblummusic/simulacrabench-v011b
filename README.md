# SimulacraBench — Final submission code (v011b)

Code for team **moshegolderson**'s Final submission to SimulacraBench (NeurIPS 2026 competition, Codabench 17822), by Hersh Mittelman.

**Final result:** overall 0.48, rank #10 of 26 — UNICEF 0.34, World Bank 0.24, UNHCR 0.88.

## Method

Cross-fitted per-item ensembles with an anytime tabular in-context learner for survey completion. The pipeline (`main.py` + the `sbk` package) builds per-item predictions for each survey track, combining a TabICL v2 in-context learner with gradient-boosting, nearest-neighbour and MLP components, cross-fitted ensembling, and post-processing. The full method is described in the accompanying method report, "Cross-fitted per-item ensembles with an anytime tabular in-context learner for survey completion" (posted on OpenReview).

## Contents

Only the team's own code is published here: `main.py`, the `sbk/*.py` modules, and two attribution files under `sbk/weights/`.

## Dependencies (not bundled)

The pipeline depends on the public TabICL and TabPFN packages and their published model checkpoints (see `sbk/weights/SOURCE.txt` and `sbk/weights/TabICL_MODEL_CARD.md`). Third-party library code and model weights are **not** included in this repository; obtain them from their original sources.

## License

The code is the author's own work, released for non-commercial research reproduction, as specified in the method report.
