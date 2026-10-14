import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from ml.config import ExperimentConfig
from ml.data import safe_join, synthetic, split_data, verify
from ml.preprocessing import Preprocessor
from ml.evaluation import metrics,choose_threshold
from ml.pipeline import run,infer
from backend.store import Store

def test_config_validation():
    for values in [{"dropout":[1,0]}, {"hidden_layers":[0,2]}, {"fraud_weight":0}, {"threshold":2}, {"features":["isFraud"]}, {"invented":1}]:
        with pytest.raises(ValidationError):
            ExperimentConfig(**values)
    assert ExperimentConfig().config_hash()!=ExperimentConfig(fraud_weight=10).config_hash()

def test_join_preserves_transaction_rows():
    a=pd.DataFrame({"TransactionID":[1,2],"isFraud":[0,1]})
    b=pd.DataFrame({"TransactionID":[2],"DeviceType":["mobile"]})
    assert len(safe_join(a,b))==2
    with pytest.raises(ValueError):
        safe_join(a,pd.concat([b,b]))

def test_temporal_ties_and_shared_observations():
    frame=synthetic()
    frame.TransactionDT=(frame.TransactionDT//1000)*1000
    parts,ids,hash1=split_data(frame,ExperimentConfig())
    for method in ["smote","cost_sensitive"]:
        _,other,hash2=split_data(frame,ExperimentConfig(method=method,seed=19))
        assert ids==other and hash1==hash2
    assert not set(ids["train"]) & set(ids["test"])
    assert frame.loc[parts[0],"TransactionDT"].max()<frame.loc[parts[1],"TransactionDT"].min()
    assert frame.loc[parts[1],"TransactionDT"].max()<frame.loc[parts[2],"TransactionDT"].min()

def test_preprocessing_fit_training_only_and_smote_training_only():
    frame=synthetic()
    config=ExperimentConfig(method="smote")
    parts,_,_=split_data(frame,config)
    train=frame.loc[parts[0]].copy()
    val=frame.loc[parts[1]].copy()
    test=frame.loc[parts[2]].copy()
    val.ProductCD="unseen"
    p=Preprocessor(config).fit(train)
    before=p.num_imputer.statistics_.copy()
    x,y,distribution=p.training(train,train.isFraud.to_numpy())
    assert len(x)>len(train) and distribution["strategy"]=="smotenc"
    assert len(p.transform(val))==len(val) and len(p.transform(test))==len(test)
    np.testing.assert_array_equal(before,p.num_imputer.statistics_)
    assert "unseen" not in p.ordinal.categories_[0]
    with pytest.raises(ValueError):
        Preprocessor(ExperimentConfig(method="smote",smote_strategy="smote")).fit(train).training(train,train.isFraud.to_numpy())

def test_metrics_and_threshold():
    y=np.array([0,0,1,1]); p=np.array([.1,.6,.4,.9])
    value=metrics(y,p,.5)
    assert value["confusion_matrix"]==[[1,1],[1,1]] and value["f1"]==.5
    assert metrics(y,p,.8)["flagged"]<=value["flagged"]
    with pytest.raises(ValueError):
        metrics(y,np.array([0,1,2,3]))
    config=ExperimentConfig(optimize_threshold=True)
    selected=choose_threshold(y,p,config)
    assert metrics(y,p,selected)["f1"]>=value["f1"]

@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    home=tmp_path_factory.mktemp("model")
    config=ExperimentConfig(epochs=2,bootstrap_samples=0)
    result=run(config,home,"a"*32)
    return home,result

def test_save_load_probability_and_explanation(trained):
    home,r=trained
    frame=synthetic().iloc[0]
    values={f:(None if pd.isna(frame[f]) else frame[f].item() if isinstance(frame[f],np.generic) else frame[f]) for f in r["feature_schema"]}
    output=infer(str(home/"artifacts"/r["id"]),values,explain=True)
    assert 0<=output["probability"]<=1 and output["explanation"]["contributions"]
    with np.load(home/"artifacts"/r["id"]/"predictions.npz") as data:
        assert data["test_y"].shape==data["test_probability"].shape
        assert (data["test_probability"]>=0).all() and (data["test_probability"]<=1).all()
    with pytest.raises(ValueError):
        infer(str(home/"artifacts"/r["id"]),{"isFraud":1})

def test_status_transitions_and_claim(tmp_path):
    s=Store(tmp_path)
    job=s.create(ExperimentConfig())
    assert job["status"]=="queued"
    assert s.claim()["status"]=="running" and s.claim() is None
    s.update(job["id"],"completed")
    with pytest.raises(ValueError):
        s.update(job["id"],"running")
    cancelled=s.create(ExperimentConfig())
    assert s.cancel(cancelled["id"])["status"]=="cancelled"

def test_dataset_integrity(tmp_path):
    pd.DataFrame({"TransactionID":[1,2],"TransactionDT":[0,1],"TransactionAmt":[2,3],"isFraud":[0,1]}).to_csv(tmp_path/"train_transaction.csv",index=False)
    pd.DataFrame({"TransactionID":[2],"DeviceType":["mobile"]}).to_csv(tmp_path/"train_identity.csv",index=False)
    assert verify(tmp_path)["train_transaction.csv"]["rows"]==2
    pd.DataFrame({"TransactionID":[2,2]}).to_csv(tmp_path/"train_identity.csv",index=False)
    with pytest.raises(ValueError):
        verify(tmp_path)

def test_atomic_claim_under_competing_workers(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    store=Store(tmp_path)
    jobs=[store.create(ExperimentConfig()) for _ in range(6)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        claimed=list(pool.map(lambda _:store.claim(),range(6)))
    assert len({j["id"] for j in claimed})==6
    assert {j["id"] for j in claimed}=={j["id"] for j in jobs}
    assert store.claim() is None

def test_training_cancellation(tmp_path):
    from ml.pipeline import Cancelled
    config=ExperimentConfig(epochs=2,bootstrap_samples=0)
    with pytest.raises(Cancelled):
        run(config,tmp_path,"b"*32,cancelled=lambda:True)
    assert not (tmp_path/"artifacts"/("b"*32)/"result.json").exists()

def test_native_catboost_roundtrip_and_leakage_guards(tmp_path):
    from ml.pipeline import load_model
    config=ExperimentConfig(method="catboost",tree_iterations=10,tree_threads=2,bootstrap_samples=0)
    artifact=run(config,tmp_path,"c"*32)
    model,processor,saved=load_model(str(tmp_path/"artifacts"/("c"*32)))
    params=saved["tree_parameters"]
    assert params["counter_calc_method"]=="SkipTest" and params["has_time"]
    assert saved["class_distribution"]["original"]==saved["class_distribution"]["resampled"]
    output=infer(str(tmp_path/"artifacts"/("c"*32)),{c:None for c in artifact["feature_schema"]},explain=True)
    assert 0<=output["probability"]<=1 and output["explanation"]["method"]=="TreeSHAP_logit"
    # Stable integer-coded categories agree across CSV float and JSON integer representations.
    assert processor.canonical(315.0)==processor.canonical(315)=="315"
