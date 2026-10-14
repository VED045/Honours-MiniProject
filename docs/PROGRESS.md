# Completion and optimization evidence

Project: `C:\Users\Ved\Desktop\Honours-MiniProject`  
Repository: https://github.com/VED045/Honours-MiniProject

## Score interpretation

The supplied XGB notebook's roughly 0.96 score refers to competition ROC-AUC. Average precision (AP) is a separate precision–recall ranking metric, not the arithmetic mean across models. Precision/recall/F1 are measured at a chosen threshold. The reference also uses different features, folds, train/test-concatenated aggregates and label-informed postprocessing. Those scores cannot be substituted for this project's measured results.

## Strongest saved validation result

Selected on validation AP, among completed full-data runs with the project's temporal evaluation. Model: **Optimized CatBoost - all predictors depth 8**, `8d5e7f11b8ac41bb99ff8dd3bbbbe23b`. 431 model input features; checkpoint 1800; validation AP 0.5978. The saved cutoff 0.24 was chosen on validation F1.

| Test metric | Measured value |
|---|---:|
| ROC-AUC | 0.9015 |
| Average precision | 0.5322 |
| Flag precision | 55.45% |
| Fraud recall | 48.95% |
| F1 | 0.5200 |
| Missed fraud | 1,576 |
| False alerts | 1,214 |
| Test transactions | 88,508 |

The earlier expanded CatBoost reference measured AP 0.5132 and ROC-AUC 0.8940. Changes in precision/recall reflect both model ranking and the validation-selected cutoff. At test prevalence 3.488%, high accuracy alone is insufficient. There is no claim that this model is the global best or production-ready.

Further tuning is **exploratory**: the held-out window had already been inspected before these follow-up requests. Parameters and final stage selection use validation, but a fresh later time window is needed for an untouched confirmatory evaluation. The nine full-data primary ANN runs remain separate; no exploratory run is mixed into their method means.

## Implemented and executed

- Full IEEE-CIS archive CRC verification, required CSV/key/target checks, one-to-one identity join, label-free duplicate removal and chronological 70/15/15 partitions.
- Baseline ANN, SMOTENC ANN and weighted-BCE ANN: nine full-data primary runs plus nine 30,000-row development runs across seeds 42/43/44. Enhanced ANN and CatBoost follow-ups are explicitly exploratory.
- Saved configurations, preprocessing, weights, split IDs, validation/test probabilities, canonical metrics, curves, timing and uncertainty. 25 completed artifact directories.
- Persistent FastAPI/SQLite queue and separate worker; validated prediction, integrated gradients/TreeSHAP, cancellation, configuration history and artifact exports.
- Eight responsive Next.js views with dropdowns/sliders, per-layer architecture controls, theme persistence, threshold comparisons and interactive model/technology explanations. Headline metrics come from one validation-selected model; AP, ROC-AUC and matching-holdout run means appear separately.
- Executed notebook and compiled seven-page paper, with verified post-2022 research. Only typography/layout was borrowed from the supplied formatting reference.

## Tested

- `python -m pytest -q --basetemp work/pytest-deployment-final`: 14 passed; one Starlette/anyio deprecation warning.
- `npm test`: 4 passed. `npm run typecheck`, `npm run lint`, `npm run build`: passed.
- `npm audit --omit=dev`: zero production dependency vulnerabilities. Development-only transitive audit findings require upstream fixes; no blind framework downgrade was performed.
- `python scripts/smoke_test.py`: registry, configurations, aggregate dataset summary, real registered-model prediction and explanation passed.
- Browser: production explanation tabs, weighted-loss slider, real saved-probability threshold exploration, feature search and dark theme verified. Earlier browser workflow submitted and completed an explicitly labeled synthetic test job.
- Artifact audit recomputes saved test metrics, checks disjoint partitions and common primary fingerprints/splits. Notebook executes top-to-bottom using the same artifacts.
- Private archive restoration passed: completed-only SQLite registry, actual native model inference/TreeSHAP, research references and aggregate dataset summary with raw CSVs absent. A regression test confirms the backup is checkpointed without modifying the live registry.

## Externally configured and publicly deployed

Source pushed to GitHub. GitHub Actions successfully runs Python tests, frontend validation, Docker build and a live container health/authentication smoke check. Installation uses separate package indexes for general dependencies and CPU PyTorch. See GitHub Actions for the final commit's run status.

Vercel frontend settings, Render persistent-disk blueprint, Docker image and private artifact restore workflow are supplied. No paid hosting subscription or public app deployment was created. Raw Kaggle data, secrets, environments and large generated model artifacts stay outside the public repository. The private completed-artifact archive enables the measured models on a persistent host.

Follow `docs/DEPLOYMENT.md`: deploy the persistent backend, restore trusted artifacts, set its private key and exact CORS origins, then import the GitHub repository into Vercel with root `frontend` and `NEXT_PUBLIC_API_BASE_URL` pointing to the backend.
