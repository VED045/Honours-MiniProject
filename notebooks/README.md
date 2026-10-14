# Reproducible notebook
From the project root:
    .\.venv\Scripts\python.exe -m ipykernel install --prefix .venv --name fraud-lab --display-name "Financial Fraud Lab"
Open financial_fraud_detection.ipynb with the project kernel. The notebook finds the repository relative to its working directory. It verifies raw files, reads the canonical completed artifacts and produces EDA/evaluation plots. RUN_NEW_EXPERIMENTS defaults false to preserve recorded runs and avoid accidental lengthy training; set true to execute the three methods via the same tested pipeline. Artifact-reading cells execute top to bottom when saved runs exist. It does not silently fall back to synthetic empirical results.
