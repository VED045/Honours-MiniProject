# Exploratory optimization

All scores below are measured on the existing temporal holdout. Rows are ordered by validation AP. Follow-up test reporting is exploratory; no leaderboard metric is imported.

| Model | Predictors | Validation AP | Test AP | ROC-AUC | Precision | Recall | F1 | Missed fraud |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Optimized CatBoost - all predictors depth 8 | 431 | 0.5978 | 0.5322 | 0.9015 | 0.5545 | 0.4895 | 0.5200 | 1576 |
| Optimized CatBoost - all predictors depth 7 | 431 | 0.5854 | 0.5227 | 0.8976 | 0.5451 | 0.4914 | 0.5169 | 1570 |
| CatBoost reference - expanded native predictors | 205 | 0.5720 | 0.5132 | 0.8940 | 0.5265 | 0.4820 | 0.5033 | 1599 |
| CatBoost reference - same 57 predictors | 57 | 0.5114 | 0.4600 | 0.8724 | 0.4857 | 0.4512 | 0.4678 | 1694 |
| Enhanced ANN - robust scaling | 136 | 0.4746 | 0.4552 | 0.8706 | 0.5308 | 0.4185 | 0.4680 | 1795 |
| Enhanced weighted ANN - weight 5 | 136 | 0.4669 | 0.4597 | 0.8715 | 0.4441 | 0.4555 | 0.4497 | 1681 |
