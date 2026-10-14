# Reference notebook review

The two user-supplied notebooks were statically inspected on 14 October 2026. Their code was not executed and their stored scores were not adopted as this project's results. Neither notebook replaces the six verified post-2022 journal references.

## xgb-fraud-with-magic-0-9600.ipynb
Cells 4, 17 and 21 describe transaction/identity joining and time-based validation. These reinforce the existing join and temporal holdout decisions. Cell 3 illustrates a curated anonymized-V subset and memory-conscious dtypes; the current 57-predictor ANN subset remains explicit in its own configuration.

Cells 9 and 11 factorize categories and build frequencies/group aggregates using concatenated training and competition-test covariates. This is transductive competition feature construction, not a training-only inductive inference pipeline. Reusing it unchanged in our holdout would allow future/held-out covariates to affect representation. There is no claim that covariate-only concatenation is label leakage by itself; it changes the evaluation assumptions and serving requirements.

Cells 29/31 infer a UID and create group features. These are useful future ideas, but a defensible online adaptation needs training-only fitted maps or causal past-only rolling state, unseen-entity fallbacks, and leakage tests. Cell 45 averages known training labels with test predictions by inferred UID. It is competition-specific label-informed post-processing and must not be copied into untouched held-out scoring.

Cell 22 invents a calendar origin for relative time. Our project keeps TransactionDT as relative elapsed seconds. Cells 23/39 GroupKFold by relative months include folds that train on later groups and validate earlier groups, so they are not the same forward temporal protocol as this project. Notebook title/public-leaderboard ROC-AUC claims are not comparable to our AP on labeled temporal holdouts.

## eda-and-models.ipynb
Cells 7, 30 and 31 motivate safe joining, missingness analysis, and relative-time distribution checks. Their useful EDA themes are incorporated into our notebook and dataset explorer.

Cell 36 computes transaction amount/identity/time-delta group statistics before the validation folds and separately on the whole competition-test table. An inductive adaptation should fit group statistics within training only and apply frozen maps; division by zero and unseen groups need explicit handling.

Cell 48 assigns TimeSeriesSplit and immediately overwrites it with KFold. Sorting rows first does not make ordinary KFold a forward temporal split. The executed object would therefore contradict the earlier time-based recommendation. Our split implementation/tests retain chronological boundaries and time ties.

## Incorporation
The project uses the references for design discussion, EDA priorities, typed category handling, resource-conscious feature selection, and future past-only/entity evaluation ideas. It does not copy tree-specific code, metrics, transductive statistics, calendar assertions or label-informed post-processing. The mandatory models remain baseline ANN, SMOTENC + ANN and class-weighted ANN with identical untouched holdouts.
