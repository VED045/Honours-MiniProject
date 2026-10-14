# Financial Fraud Detection Using Artificial Neural Networks with Imbalanced Learning

An educational research prototype with PyTorch ANNs, a FastAPI job API, a persistent worker, and a Next.js experiment laboratory. Synthetic runs validate the software only; IEEE-CIS claims require the licensed Kaggle files.

## Windows quick start
```powershell
Set-Location C:\Users\Ved\Desktop\Honours-MiniProject
# If Python is on PATH on a new machine:
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-torch.txt
.\.venv\Scripts\kaggle.exe auth login
# Accept the IEEE-CIS rules on Kaggle, then:
.\.venv\Scripts\python.exe scripts/download_dataset.py
.\.venv\Scripts\python.exe scripts/verify_dataset.py
# Three separate terminals from the project root:
.\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
.\.venv\Scripts\python.exe -m backend.worker
Set-Location frontend; npm ci; npm run dev
```
Open http://localhost:3000. The UI can submit real training jobs, monitor progress, compare saved predictions at new thresholds, export artifacts, and perform model inference. The laboratory recommends IEEE-CIS; acquire the licensed files before training. Synthetic configurations are clearly labeled software demonstrations. API documentation: http://localhost:8000/docs.

## Commands
```powershell
.\.venv\Scripts\python.exe scripts/run_experiments.py --dataset synthetic --seeds 42 43 44
.\.venv\Scripts\python.exe scripts/run_experiments.py --dataset ieee-cis --max-rows 30000
# Full labeled dataset: no development sample, resource guard still applies
.\.venv\Scripts\python.exe scripts/run_experiments.py --dataset ieee-cis --max-rows 0
.\.venv\Scripts\python.exe scripts/export_results.py
.\.venv\Scripts\python.exe -m pytest -q
Set-Location frontend; npm test; npm run lint; npm run typecheck; npm run build
```
The mandatory methods share a fixed split seed; training seed changes initialization without changing observations. Temporal splits keep equal TransactionDT values together. Relative time is not a calendar timestamp. Preprocessors fit exclusively on the training partition; SMOTENC operates before one-hot encoding. Final test probabilities are saved separately from validation probabilities. Threshold optimization uses validation only.

See docs/PROGRESS.md for executed evidence, docs/ARCHITECTURE.md for scientific details, docs/DEPLOYMENT.md for durable hosting, and docs/LITERATURE_MATRIX.md for verified references. Never load untrusted model/joblib artifacts. No production financial decisions should rely on this prototype.

## Deploy
Follow [the deployment guide](docs/DEPLOYMENT.md#deploy-this-github-repository-step-by-step) for Vercel + a persistent Python backend. Source, tests, notebook, paper and aggregate result tables are tracked. Licensed raw data, environment secrets, virtual environments, node_modules and generated training artifacts stay outside Git. Restore the private artifact archive to enable the existing trained models in a new deployment.

## Further optimization and score interpretation
`python scripts/run_improvements.py` reproduces the enhanced ANN and native CatBoost references. `python scripts/optimize_models.py` runs the bounded all-predictor CatBoost follow-up. These are exploratory and preserve the original controlled ANN study. Overview selects one model using validation metrics, displays its measured test precision/recall/F1, and places AP, ROC-AUC and mean run statistics to the side. AP is not a mean across runs. A Kaggle ROC-AUC of 0.96 does not imply 96% precision or recall.

Interactive Explainability includes model selectors, a weighted-loss demonstration, saved-prediction threshold exploration and the technology pipeline, alongside real integrated-gradients/TreeSHAP inference.
