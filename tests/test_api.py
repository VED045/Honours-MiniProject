import importlib
import json
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from ml.config import ExperimentConfig
from ml.data import synthetic
from backend.worker import execute

def test_api_queue_prediction_and_errors(tmp_path,monkeypatch):
    monkeypatch.setenv("FRAUD_HOME",str(tmp_path))
    import backend.app as module
    module=importlib.reload(module)
    with TestClient(module.app) as client:
        assert client.get("/health").status_code==200
        assert client.get("/api/v1/models").json()==[]
        assert client.get("/api/v1/experiments/missing").status_code==404
        assert client.post("/api/v1/experiments",json={"epochs":0}).status_code==422
        missing=client.post("/api/v1/predict",json={"model_id":"a"*32,"features":{}})
        assert missing.status_code==404
        config=ExperimentConfig(epochs=1,bootstrap_samples=0)
        response=client.post("/api/v1/experiments",json=config.model_dump())
        assert response.status_code==202 and response.json()["status"]=="queued"
        job=module.store.claim()
        execute(module.store,job)
        uid=job["id"]
        assert client.get(f"/api/v1/experiments/{uid}").json()["status"]=="completed"
        metadata=client.get(f"/api/v1/models/{uid}/metadata").json()
        row=synthetic().iloc[1]
        values={f:None if pd.isna(row[f]) else row[f].item() if isinstance(row[f],np.generic) else row[f] for f in metadata["feature_schema"]}
        result=client.post("/api/v1/predict",json={"model_id":uid,"features":values,"explain":True})
        assert result.status_code==200 and 0<=result.json()["probability"]<=1
        assert client.post("/api/v1/predict",json={"model_id":uid,"features":{"unknown":4}}).status_code==422
        assert client.get(f"/api/v1/experiments/{uid}/metrics?threshold=0").json()["flagged"]==metadata["metrics"]["transactions"]
        assert client.get(f"/api/v1/metrics/comparison?ids={uid}").json()["comparable"]
        assert client.get(f"/api/v1/experiments/{uid}/export").headers["content-type"]=="application/zip"
        assert client.post("/api/v1/configurations",json=config.model_dump()).status_code==201
        assert len(client.get("/api/v1/configurations").json())==1
        assert client.get("/api/v1/dataset/summary?source=synthetic").json()["demonstration"]
        assert client.post("/api/v1/experiments",content="x"*70000).status_code==413

def test_private_api_auth(tmp_path,monkeypatch):
    monkeypatch.setenv("FRAUD_HOME",str(tmp_path))
    monkeypatch.setenv("API_KEY","private-test-key-at-least-24-characters")
    import backend.app as module
    module=importlib.reload(module)
    with TestClient(module.app) as client:
        assert client.get("/health").status_code==200
        denied=client.get("/api/v1/models",headers={"Origin":"http://localhost:3000"})
        assert denied.status_code==401 and denied.headers["access-control-allow-origin"]=="http://localhost:3000"
        assert client.get("/api/v1/models",headers={"Authorization":"Bearer private-test-key-at-least-24-characters"}).status_code==200
