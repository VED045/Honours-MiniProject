from functools import lru_cache
import copy
import json
import platform
import time
from pathlib import Path
from importlib.metadata import version
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from ml.config import ExperimentConfig
from ml.data import load_dataset, split_data
from ml.preprocessing import Preprocessor
from ml.model import ANN, predict
from ml.evaluation import metrics, curves, choose_threshold, uncertainty

class Cancelled(Exception):
    pass

def save_json(path, value):
    path=Path(path)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding="utf-8")
    tmp.replace(path)

def run(config, home, run_id, progress=lambda value: None, cancelled=lambda: False):
    if config.method=="catboost":
        from ml.tree_pipeline import run_tree
        return run_tree(config,home,run_id,progress,cancelled)
    started=time.perf_counter()
    torch.set_num_threads(4)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.use_deterministic_algorithms(True)
    directory=Path(home)/"artifacts"/run_id
    directory.mkdir(parents=True,exist_ok=False)
    save_json(directory/"config.json",config.model_dump())
    frame, dataset=load_dataset(home,config)
    parts,ids,split_hash=split_data(frame,config)
    train,val,test=[frame.loc[p] for p in parts]
    processor=Preprocessor(config).fit(train)
    ytrain=train.isFraud.to_numpy().astype(int)
    xtrain,ytrain,distribution=processor.training(train,ytrain)
    xval,xtest=processor.transform(val),processor.transform(test)
    yval,ytest=val.isFraud.to_numpy().astype(int),test.isFraud.to_numpy().astype(int)
    if (xtrain.nbytes+xval.nbytes+xtest.nbytes)*3 > config.max_memory_mb*1024**2:
        raise MemoryError("Combined model matrices exceed configured memory budget")
    model=ANN(xtrain.shape[1],config)
    optimizer_class={"adam":torch.optim.Adam,"adamw":torch.optim.AdamW,"sgd":torch.optim.SGD}[config.optimizer]
    optimizer=optimizer_class(model.parameters(),lr=config.learning_rate,weight_decay=config.l2)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",patience=2) if config.scheduler=="plateau" else None
    loader=DataLoader(TensorDataset(torch.from_numpy(xtrain),torch.tensor(ytrain,dtype=torch.float32)),batch_size=config.batch_size,shuffle=True,generator=torch.Generator().manual_seed(config.seed))
    loss_fn=torch.nn.BCEWithLogitsLoss(reduction="none")
    w0,w1=(config.legitimate_weight,config.fraud_weight) if config.method=="cost_sensitive" else (1,1)
    best_score=-float("inf")
    best_state=None
    stale=0
    history=[]
    training_started=time.perf_counter()
    for epoch in range(config.epochs):
        if cancelled():
            raise Cancelled("Cancelled by user")
        model.train()
        total=0
        observed=0
        for xb,yb in loader:
            if cancelled():
                raise Cancelled("Cancelled by user")
            if config.batch_norm and len(xb)<2:
                model.eval()
            optimizer.zero_grad()
            logits=model(xb)
            weight=torch.where(yb>0.5,w1,w0)
            loss=(loss_fn(logits,yb)*weight).mean()
            if config.l1:
                loss=loss+config.l1*sum(p.abs().sum() for p in model.parameters())
            loss.backward()
            if config.gradient_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(),config.gradient_clip)
            optimizer.step()
            total+=loss.item()*len(xb)
            observed+=len(xb)
            model.train()
        pval=predict(model,xval)
        ptraining=predict(model,xtrain)
        model.eval()
        with torch.no_grad():
            vl=loss_fn(model(torch.from_numpy(xval)),torch.tensor(yval,dtype=torch.float32))
            val_loss=float((vl*torch.where(torch.tensor(yval)>0,w1,w0)).mean())
        vm=metrics(yval,pval,config.threshold)
        tm=metrics(ytrain,ptraining,config.threshold)
        score=-val_loss if config.checkpoint_metric=="val_loss" else vm[config.checkpoint_metric]
        improved=score>best_score+1e-7
        if improved:
            best_score=score
            best_state=copy.deepcopy(model.state_dict())
            best_epoch=epoch+1
            stale=0
        else:
            stale+=1
        row={"epoch":epoch+1,"train_loss":total/observed,"val_loss":val_loss,"val_average_precision":vm["average_precision"],"train_average_precision":tm["average_precision"],"val_roc_auc":vm["roc_auc"],"learning_rate":optimizer.param_groups[0]["lr"],"checkpoint":improved,"elapsed_seconds":time.perf_counter()-training_started}
        history.append(row)
        save_json(directory/"history.json",history)
        progress({"epoch":epoch+1,"epochs":config.epochs,"history":history,"message":f"Epoch {epoch+1}; validation AP {vm['average_precision']:.4f}"})
        if scheduler:
            scheduler.step(val_loss)
        if config.early_stopping and stale>=config.patience:
            break
    training_seconds=time.perf_counter()-training_started
    model.load_state_dict(best_state)
    pval=predict(model,xval)
    threshold=choose_threshold(yval,pval,config)
    inference_started=time.perf_counter()
    ptest=predict(model,xtest)
    inference_seconds=time.perf_counter()-inference_started
    result=metrics(ytest,ptest,threshold)
    result.update(training_seconds=training_seconds,inference_seconds=inference_seconds)
    np.savez_compressed(directory/"predictions.npz",validation_y=yval,validation_probability=pval,test_y=ytest,test_probability=ptest)
    torch.save(model.state_dict(),directory/"model.pt")
    joblib.dump(processor,directory/"preprocessor.joblib")
    save_json(directory/"splits.json",ids)
    feature_schema={name:{"type":"categorical" if name in processor.cat else "numeric", "nullable":True} for name in processor.features}
    artifact={"format_version":1,"id":run_id,"configuration":config.model_dump(),"config_hash":config.config_hash(),"dataset":dataset,"split_hash":split_hash,"feature_schema":feature_schema,"encoded_features":processor.feature_names,"input_width":xtrain.shape[1],"partition_counts":{"train":len(train),"validation":len(val),"test":len(test)},"class_distribution":distribution,"history":history,"metrics":result,"validation_metrics":metrics(yval,pval,threshold),"selected_threshold":threshold,"threshold_source":"validation" if config.optimize_threshold else "configuration","checkpoint_epoch":best_epoch,"early_stopped":len(history)<config.epochs,"curves":curves(ytest,ptest),"bootstrap_interval":uncertainty(ytest,ptest,config.bootstrap_samples,config.seed),"timing":{"training_seconds":training_seconds,"inference_seconds":inference_seconds,"total_seconds":time.perf_counter()-started},"software":{"python":platform.python_version(),**{p:version(p) for p in ["torch","pandas","numpy","scikit-learn","imbalanced-learn"]}},"paths":{"model":"model.pt","preprocessor":"preprocessor.joblib","predictions":"predictions.npz"}}
    save_json(directory/"result.json",artifact)
    return artifact

@lru_cache(maxsize=4)
def load_model(directory):
    directory=Path(directory)
    artifact=json.loads((directory/"result.json").read_text())
    config=ExperimentConfig.model_validate(artifact["configuration"])
    if artifact.get("model_family")=="catboost":
        from catboost import CatBoostClassifier
        model=CatBoostClassifier()
        model.load_model(str(directory/"model.cbm"))
        processor=joblib.load(directory/"preprocessor.joblib")
        return model,processor,artifact
    model=ANN(artifact["input_width"],config)
    model.load_state_dict(torch.load(directory/"model.pt",map_location="cpu",weights_only=True))
    model.eval()
    processor=joblib.load(directory/"preprocessor.joblib")
    return model,processor,artifact

def infer(directory, values, threshold=None, explain=False):
    model,processor,artifact=load_model(directory)
    required=set(artifact["feature_schema"])
    if set(values)!=required:
        raise ValueError(f"Expected exactly model features; missing {sorted(required-set(values))}; unknown {sorted(set(values)-required)}")
    for name, schema in artifact["feature_schema"].items():
        value=values[name]
        if value is not None and schema["type"]=="numeric" and (isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value)):
            raise ValueError(f"{name} must be a finite number or null")
        if value is not None and schema["type"]=="categorical" and not isinstance(value,(str,int,float)):
            raise ValueError(f"{name} must be a category or null")
    x=processor.transform(pd.DataFrame([values]))
    tree=artifact.get("model_family")=="catboost"
    probability=float(model.predict_proba(x)[0,1]) if tree else float(predict(model,x)[0])
    threshold=artifact["selected_threshold"] if threshold is None else threshold
    output={"probability":probability,"flagged":probability>=threshold,"threshold":threshold,"model_version":artifact["id"],"dataset":artifact["dataset"],"unknown_categories":[c for c in processor.cat if values[c] is not None and (processor.canonical(values[c]) not in processor.category_values[c] if tree else str(values[c]) not in set(processor.ordinal.categories_[processor.cat.index(c)]))]}
    if explain and tree:
        from catboost import Pool
        shap=model.get_feature_importance(Pool(x,cat_features=processor.cat),type="ShapValues")[0]
        output["explanation"]={"method":"TreeSHAP_logit","baseline":f"Expected logit {float(shap[-1]):.5f}","contributions":sorted([{"feature":f,"contribution":float(v)} for f,v in zip(processor.feature_names,shap[:-1])],key=lambda r:abs(r["contribution"]),reverse=True)[:20],"caution":"TreeSHAP contributions explain the model logit, not causal effects or calibrated probability changes."}
    elif explain:
        # Integrated gradients in encoded input space relative to an all-zero encoded baseline.
        input_tensor=torch.tensor(x,dtype=torch.float32)
        baseline=torch.zeros_like(input_tensor)
        gradients=[]
        for alpha in torch.linspace(0,1,64):
            point=(baseline+alpha*(input_tensor-baseline)).requires_grad_(True)
            gradients.append(torch.autograd.grad(model(point).sum(),point)[0])
        contributions=((input_tensor-baseline)*torch.stack(gradients).mean(0)).detach().numpy()[0]
        output["explanation"]={"method":"integrated_gradients_logit","baseline":"zero encoded vector; may be outside data manifold","contributions":sorted([{"feature":f,"contribution":float(v)} for f,v in zip(processor.feature_names,contributions)],key=lambda r:abs(r["contribution"]),reverse=True)[:20],"caution":"Model sensitivity relative to an encoded baseline; not causal evidence or a calibrated probability decomposition."}
    return output

