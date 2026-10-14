from contextlib import asynccontextmanager
from functools import lru_cache
import hmac
import io
import json
import logging
import os
from pathlib import Path
import time
import zipfile
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ConfigDict, Field
from backend.store import Store
from ml.config import ExperimentConfig
from ml.data import summary
from ml.pipeline import infer
from ml.evaluation import metrics

HOME=Path(os.environ.get("FRAUD_HOME",Path(__file__).resolve().parents[1])).resolve()
store=Store(HOME)
API_KEY=os.environ.get("API_KEY","")
TRAINING_ENABLED=os.environ.get("TRAINING_ENABLED","true").lower()=="true"
logging.basicConfig(level=getattr(logging,os.environ.get("LOG_LEVEL","INFO").upper(),logging.INFO))

@asynccontextmanager
async def lifespan(app):
    if os.environ.get("REQUIRE_API_KEY","false").lower()=="true" and len(API_KEY)<24:
        raise RuntimeError("Non-local deployment requires a private API_KEY of at least 24 characters")
    yield

app=FastAPI(title="Financial Fraud Research API",version="1.0.0",lifespan=lifespan)
origins=os.environ.get("CORS_ORIGINS","http://localhost:3000,http://127.0.0.1:3000").split(",")


@app.middleware("http")
async def limits(request:Request,call_next):
    if request.method=="OPTIONS":
        return await call_next(request)
    if request.url.path not in ["/health"] and API_KEY:
        if not hmac.compare_digest(request.headers.get("Authorization",""),f"Bearer {API_KEY}"):
            return JSONResponse(status_code=401,content={"error":{"code":"unauthorized","message":"Private backend key required"}})
    if request.method=="POST":
        if len(await request.body())>65536:
            return JSONResponse(status_code=413,content={"error":{"code":"request_too_large","message":"Request limit is 64 KiB"}})
    started=time.perf_counter()
    result=await call_next(request)
    result.headers["X-Content-Type-Options"]="nosniff"
    result.headers["Cache-Control"]="no-store"
    logging.info("%s %s %s %.3fs",request.method,request.url.path,result.status_code,time.perf_counter()-started)
    return result

@app.exception_handler(HTTPException)
async def http_error(request,exc):
    return JSONResponse(status_code=exc.status_code,content={"error":{"code":str(exc.status_code),"message":str(exc.detail)}})

@app.exception_handler(RequestValidationError)
async def validation_error(request,exc):
    return JSONResponse(status_code=422,content={"error":{"code":"validation","message":"Invalid request","details":[{"location":list(e["loc"]),"message":e["msg"]} for e in exc.errors()]}})

@app.exception_handler(Exception)
async def unexpected_error(request,exc):
    logging.exception("Unhandled request failure")
    return JSONResponse(status_code=500,content={"error":{"code":"internal","message":"Unexpected backend error; consult server log"}})

def get_job(uid):
    try:
        return store.get(uid)
    except KeyError:
        raise HTTPException(404,"Experiment not found")

def result(uid):
    job=get_job(uid)
    if job["status"]!="completed":
        raise HTTPException(409,"Experiment has not completed")
    path=HOME/"artifacts"/job["id"]/"result.json"
    if not path.is_file():
        raise HTTPException(503,"Completed metadata exists but artifacts are unavailable")
    return json.loads(path.read_text())

@app.get("/health")
def health():
    return {"status":"ok","training_enabled":TRAINING_ENABLED,"storage":"sqlite_wal","worker_required":True}

@app.get("/api/v1/config/schema")
def configuration_schema():
    from scripts.run_experiments import FEATURES
    return {"defaults":ExperimentConfig(dataset="ieee-cis",features=FEATURES).model_dump(),"schema":ExperimentConfig.model_json_schema()}

@lru_cache(maxsize=4)
def cached_summary(source,version):
    snapshot=HOME/"data/processed"/f"summary-{source}.json"
    if snapshot.is_file():
        saved=json.loads(snapshot.read_text())
        if not version or saved.get("raw_signature")==[list(v) for v in version]:
            return {**saved,"raw_available":bool(version) if source=="ieee-cis" else True}
    return summary(HOME,source)

@app.get("/api/v1/dataset/summary")
def dataset_summary(source:str=Query("ieee-cis",pattern="^(ieee-cis|synthetic)$")):
    root=HOME/"data/raw/ieee-cis"
    version=tuple((p.name,p.stat().st_size,p.stat().st_mtime_ns) for p in root.glob("train*.csv"))
    try:
        return cached_summary(source,version)
    except (ValueError,FileNotFoundError) as exc:
        raise HTTPException(409,str(exc))

@app.get("/api/v1/experiments")
def experiments():
    rows=store.all()
    for row in rows:
        if row["status"]=="completed":
            path=HOME/"artifacts"/row["id"]/"result.json"
            row["artifact_available"]=path.is_file()
            if path.is_file():
                saved=json.loads(path.read_text())
                row.update(validation_metrics=saved["validation_metrics"],metrics=saved["metrics"],dataset=saved["dataset"],split_hash=saved["split_hash"],config_hash=saved["config_hash"])
    return rows

@app.post("/api/v1/config/validate")
def validate_configuration(config:ExperimentConfig):
    return {"configuration":config.model_dump(),"hash":config.config_hash()}

@app.post("/api/v1/experiments",status_code=202)
def submit(config:ExperimentConfig):
    if not TRAINING_ENABLED:
        raise HTTPException(503,"Training disabled on this host; browse saved results or run worker on a training host")
    if config.dataset=="ieee-cis" and not all((HOME/"data/raw/ieee-cis"/n).is_file() for n in ["train_transaction.csv","train_identity.csv"]):
        raise HTTPException(409,"IEEE-CIS files unavailable; acquire and verify dataset first")
    try:
        return store.create(config)
    except ValueError as exc:
        raise HTTPException(429,str(exc))

@app.get("/api/v1/experiments/{uid}")
def experiment(uid:str):
    return get_job(uid)

@app.post("/api/v1/experiments/{uid}/cancel")
def cancel(uid:str):
    get_job(uid)
    return store.cancel(uid)

@app.get("/api/v1/experiments/{uid}/artifacts")
def artifact(uid:str):
    return result(uid)

@app.get("/api/v1/experiments/{uid}/metrics")
def experiment_metrics(uid:str,threshold:float|None=Query(None,ge=0,le=1),partition:str=Query("test",pattern="^(validation|test)$")):
    saved=result(uid)
    if threshold is None and partition=="test":
        return saved["metrics"]
    with np.load(HOME/"artifacts"/uid/"predictions.npz") as data:
        return metrics(data[f"{partition}_y"],data[f"{partition}_probability"],saved["selected_threshold"] if threshold is None else threshold)

def selected(ids):
    values=[i for i in ids.split(",") if i]
    if not 1<=len(values)<=12:
        raise HTTPException(422,"Select between one and twelve completed experiments")
    return values

@app.get("/api/v1/metrics/comparison")
def comparison(ids:str,threshold:float|None=Query(None,ge=0,le=1)):
    records=[result(uid) for uid in selected(ids)]
    compatible=len({(r["dataset"]["fingerprint"],r["split_hash"]) for r in records})==1
    return {"comparable":compatible,"warning":None if compatible else "Different dataset/split: results are not a controlled comparison","experiments":[{"id":r["id"],"configuration":r["configuration"],"metrics":experiment_metrics(r["id"],threshold,"test"),"class_distribution":r["class_distribution"],"timing":r["timing"]} for r in records]}

@app.get("/api/v1/metrics/curves")
def metric_curves(ids:str):
    return [{"id":uid,"curves":result(uid)["curves"]} for uid in selected(ids)]

@app.get("/api/v1/models")
def models():
    return [{"id":j["id"],"name":j["config"]["name"],"method":j["config"]["method"],"dataset":j.get("dataset"),"metrics":j.get("metrics")} for j in experiments() if j["status"]=="completed" and j.get("artifact_available")]

@app.get("/api/v1/models/{uid}/metadata")
def metadata(uid:str):
    return result(uid)

class Prediction(BaseModel):
    model_config=ConfigDict(extra="forbid",allow_inf_nan=False)
    model_id:str=Field(pattern="^[a-f0-9]{32}$")
    features:dict[str,str|int|float|None]
    threshold:float|None=Field(default=None,ge=0,le=1)
    explain:bool=False

@app.post("/api/v1/predict")
def prediction(data:Prediction):
    result(data.model_id)
    try:
        return infer(str(HOME/"artifacts"/data.model_id),data.features,data.threshold,data.explain)
    except ValueError as exc:
        raise HTTPException(422,str(exc))

@app.get("/api/v1/configurations")
def configurations():
    return store.configs()

@app.post("/api/v1/configurations",status_code=201)
def save_configuration(config:ExperimentConfig):
    return store.save_config(config)

@app.get("/api/v1/experiments/{uid}/export")
def export(uid:str):
    result(uid)
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,"w",compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ["config.json","result.json","history.json","splits.json","predictions.npz","model.pt","model.cbm","preprocessor.joblib"]:
            path=HOME/"artifacts"/uid/name
            if path.is_file():
                archive.write(path,name)
    buffer.seek(0)
    return StreamingResponse(buffer,media_type="application/zip",headers={"Content-Disposition":f'attachment; filename="experiment-{uid}.zip"'})

@app.get("/api/v1/research")
def research():
    path=HOME/"docs/references.json"
    return json.loads(path.read_text()) if path.is_file() else []


app.add_middleware(CORSMiddleware,allow_origins=origins,allow_credentials=False,allow_methods=["GET","POST"],allow_headers=["Content-Type","Authorization"])
