# Verified model research: stronger IEEE-CIS reference

Verified on 2026-10-14. This is a bounded research recommendation, not an executed training report.

## Recommendation

Add **CatBoostClassifier** as an optional, clearly named gradient-boosted-tree reference alongside the three mandatory ANN methods. Prioritize native categorical inputs, full available training observations within the existing temporal training partition, and validation-only early stopping. CatBoost is a defensible stronger candidate; whether it improves this project's actual average precision, recall and false positives must be measured.

The first-place team's own [Kaggle writeup](https://www.kaggle.com/competitions/ieee-fraud-detection/writeups/fraudsquad-1st-place-solution-part-2) identifies CatBoost, LightGBM and XGBoost as its final model families. It also describes ensemble/stacking and averaging predictions within reconstructed clients. The writeup supports investigating these families, not importing its leaderboard scores as project metrics. NVIDIA's [winning-solution account](https://developer.nvidia.com/blog/leveraging-machine-learning-to-detect-fraud-tips-to-developing-a-winning-kaggle-solution/) corroborates those families and the importance of validation and feature engineering. The page timed out on direct open; the indexed official record was available.

## Leakage and comparison protocol

1. Preserve the exact existing train/validation/test IDs and chronological boundaries. Sort training rows by TransactionDT. Describe TransactionDT as relative elapsed time, not a calendar timestamp. Final test data cannot appear in eval_set or any feature-fitting operation.
2. Exclude isFraud and TransactionID from model inputs. Preserve the canonical feature schema, deterministic transforms and fingerprint. Categorical-versus-numeric declarations must be explicit and saved; numeric card/address codes may be categorical by semantics, but anonymous V and D values should not be indiscriminately recast.
3. Do not concatenate training, validation and competition test rows to calculate frequency, group means, UID counts or category mappings. Any learned frequency/group transform must fit training observations only, then transform future rows using stored maps and documented unseen defaults. Avoid target-based UID smoothing and client prediction averaging: those need additional assumptions about entity reconstruction and batch inference and differ from independent transaction evaluation.
4. Explicitly persist counter_calc_method='SkipTest'. CatBoost documents that this excludes validation objects from Counter CTR counts; Full would include them. SkipTest is currently the documented default, but explicit configuration prevents accidental changes. [Official CTR settings](https://catboost.ai/docs/en/references/training-parameters/ctr#counter_calc_method).
5. Select architecture/tree settings, class weights, calibration and thresholds on validation only. Log all attempted configurations and their search budget. Threshold changes cannot change ROC-AUC or average precision because those are calculated from saved probabilities.
6. Report canonical unweighted scikit-learn metrics on native-prevalence held-out data, with the same observations as the ANN experiments. Separate stronger-model results from claims about ANN imbalance interventions. If repeatedly inspecting the existing test results has already guided tuning, label the comparison exploratory and plan a fresh future holdout.

## Verified CatBoost Python behavior

| Concern | Supported behavior and implementation decision | Primary documentation |
|---|---|---|
| Classifier interface | Constructor accepts iterations, depth, learning_rate, l2_leaf_reg, random_seed, task_type, thread_count, class_weights, eval_metric and cat_features. Named categorical columns require named DataFrame/Pool inputs. | [CatBoostClassifier](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier) |
| Validation and stopping | fit accepts eval_set as a Pool or (X,y), early_stopping_rounds, and use_best_model. Validation selects the best saved tree iteration. Use validation only, with both classes present. | [fit](https://catboost.ai/docs/en/concepts/python-reference_catboost_fit) |
| Categories | Floating categorical values and categorical NaN/None are unsuitable because their string hashes can be inconsistent across input formats. Use one stable sentinel and deterministic string representation for training, validation and inference. Native categories avoid ANN one-hot expansion. | [FAQ](https://catboost.ai/docs/en/concepts/faq#why-is-it-forbidden-to-use-float-values-and-nan-values-for-categorical-features) |
| Numeric missing values | Numeric NaN is supported. nan_mode='Min' considers missing values below ordinary values and considers a separating split. Preserve numeric NaNs rather than requiring median imputation. | [Missing values](https://catboost.ai/docs/en/concepts/algorithm-missing-values-processing) |
| Weighting | class_weights={0:1.0,1:w} scales class-object loss weights. Do not combine with auto_class_weights or scale_pos_weight. has_time=True preserves input order during categorical transformation/tree structure steps. | [Common parameters](https://catboost.ai/docs/en/references/training-parameters/common) |
| Evaluation metric | Binary PRAUC:type=Classic;use_weights=False is supported. It is a monitoring/selection metric, while Logloss is the objective. Recompute canonical average_precision_score separately: do not assume every library's PR-area computation equals sklearn AP. | [Classification metrics](https://catboost.ai/docs/en/concepts/loss-functions-classification#prauc) |
| CPU resources | thread_count limits CPU threads. used_ram_limit is an attempt that affects CTR calculations only and is not a hard cap on total process RAM. Limit one worker at a time, monitor actual memory, and persist threads/CTR complexity. | [Performance settings](https://catboost.ai/docs/en/references/training-parameters/performance) |
| Artifacts and progress | Use get_evals_result, get_best_iteration, predict_proba, save_model/load_model and get_all_params. Persist native model plus schema/preprocessor and canonical saved probabilities. | [Classifier methods](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier) |

## Small initial search

These are proposed settings based on practical judgment, not purported optimal parameters or copied winning parameters.

- Begin with depth 6 or 7, 1,200-1,500 iterations, learning_rate 0.05, l2_leaf_reg 5, seed 42, CPU, max_ctr_complexity=1 and 75-100 stopping patience. Choose an explicit thread count from available CPU/RAM, ordinarily 4-8 rather than oversubscribing the desktop.
- First use class weights {0:1,1:1}. Compare a modest {0:1,1:2} or {0:1,1:4} on validation if necessary; strong weighting is not guaranteed to improve ranking or probability calibration.
- Start with raw selected numeric/categorical features. Add deterministic per-row features (log1p nonnegative TransactionAmt, missing-value count, documented relative-time components) only as a separate feature configuration. Any group aggregates require training-only fitting and clear online availability.
- If resource and validation results justify it, test depth 8 or max_ctr_complexity=2. Record CPU time/memory. A broad search on a weak sample is less useful than a bounded search with representative temporal data.
- No SMOTE is required for the initial CatBoost reference. It is a separate tree baseline, not a substitute for the SMOTE+ANN intervention.

## API sketch

This sketch illustrates verified APIs. It has not been executed here and requires project-level type normalization and schema checks.

```python
from catboost import CatBoostClassifier, Pool

# X_train and X_valid preserve the saved feature names and chronological partitions.
# A shared transformer canonicalizes categorical values to strings/sentinel.
# Numeric features remain float values with NaN, never +/- infinity.
train_pool = Pool(X_train, label=y_train, cat_features=cat_columns)
valid_pool = Pool(X_valid, label=y_valid, cat_features=cat_columns)
model = CatBoostClassifier(
    loss_function="Logloss",
    eval_metric="PRAUC:type=Classic;use_weights=False",
    iterations=1500,
    depth=7,
    learning_rate=0.05,
    l2_leaf_reg=5.0,
    class_weights={0: 1.0, 1: 1.0},
    random_seed=42,
    task_type="CPU",
    thread_count=4,
    max_ctr_complexity=1,
    counter_calc_method="SkipTest",
    nan_mode="Min",
    has_time=True,
    allow_writing_files=False,
)
model.fit(train_pool, eval_set=valid_pool,
          early_stopping_rounds=100, use_best_model=True, verbose=100)
valid_probability = model.predict_proba(valid_pool)[:, 1]
# Select configuration and threshold using validation probabilities only.
# Predict the sealed test partition after selection; retain probabilities and IDs.
model.save_model(model_path)
```

For a secondary reference, XGBoost histogram trees with a training-fitted representation are also well motivated by the winning family. Use current constructor-level stopping configuration and validation eval_set; avoid outdated fit keyword assumptions. Official [XGBoost Python API](https://xgboost.readthedocs.io/en/stable/python/python_api.html) and [parameters](https://xgboost.readthedocs.io/en/stable/parameter.html) are authoritative. Its aucpr uses continuous interpolation, explicitly illustrating why external canonical average precision must remain separately labeled.

## Limits

No guarantee of “best available” or state-of-the-art follows from a leaderboard writeup. Competition data preparation, entity aggregation and ROC-AUC scoring differ from this project's temporal, native-prevalence average-precision-focused evaluation. No leaderboard value is reproduced or promised. No project source files were edited by this research task.

