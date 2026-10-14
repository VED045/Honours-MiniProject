import json
import time
from pathlib import Path
from importlib.metadata import version
import platform
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier,Pool
from ml.data import load_dataset,split_data,KNOWN_CATEGORIES
from ml.evaluation import metrics,curves,choose_threshold,uncertainty

class NativePreprocessor:
    """No target/future-fitted encoding; CatBoost receives stable native categories."""
    @staticmethod
    def canonical(v):
        if pd.isna(v):return "__MISSING__"
        if isinstance(v,(int,float,np.number)) and float(v).is_integer():return str(int(v))
        return str(v)

    def __init__(self,config):
        self.config=config

    def fit(self,frame):
        self.features=self.config.features or [c for c in frame if c not in set(self.config.exclude_features)|{"isFraud","TransactionID"}]
        if not self.features or set(self.features)-set(frame) or set(self.features)&{"isFraud","TransactionID"}:
            raise ValueError("Invalid native tree predictors")
        self.cat=[c for c in self.features if c in KNOWN_CATEGORIES or c in self.config.categorical_features or not pd.api.types.is_numeric_dtype(frame[c])]
        self.num=[c for c in self.features if c not in self.cat]
        self.feature_names=self.features
        self.category_values={c:set(frame[c].map(self.canonical)) for c in self.cat}
        self.fill={c:(self.canonical(frame[c].mode().iloc[0]) if self.config.categorical_imputation=="most_frequent" and not frame[c].dropna().empty else "__MISSING__") for c in self.cat}
        return self

    def transform(self,frame):
        if set(self.features)-set(frame):raise ValueError("Required model fields missing")
        out=frame[self.features].copy()
        for c in self.cat:
            out[c]=out[c].map(lambda v:self.fill[c] if pd.isna(v) else self.canonical(v))
        for c in self.num:
            out[c]=pd.to_numeric(out[c],errors="raise").astype(float)
            if np.isinf(out[c]).any():raise ValueError(f"Infinite values in {c}")
        return out

def run_tree(config,home,run_id,progress,cancelled):
    from ml.pipeline import save_json,Cancelled
    started=time.perf_counter()
    directory=Path(home)/"artifacts"/run_id
    directory.mkdir(parents=True,exist_ok=False)
    save_json(directory/"config.json",config.model_dump())
    frame,dataset=load_dataset(home,config)
    parts,ids,split_hash=split_data(frame,config)
    train,val,test=[frame.loc[p] for p in parts]
    processor=NativePreprocessor(config).fit(train)
    xt,xv,xe=[processor.transform(f) for f in [train,val,test]]
    yt,yv,ye=[f.isFraud.to_numpy().astype(int) for f in [train,val,test]]
    weights={0:config.legitimate_weight,1:config.fraud_weight} if config.tree_class_weighted else {0:1,1:1}
    eval_metric={"average_precision":"PRAUC:type=Classic;use_weights=False","roc_auc":"AUC","val_loss":"Logloss"}[config.checkpoint_metric]
    model=CatBoostClassifier(iterations=config.tree_iterations,depth=config.tree_depth,learning_rate=config.tree_learning_rate,l2_leaf_reg=config.tree_l2,loss_function="Logloss",eval_metric=eval_metric,random_seed=config.seed,thread_count=config.tree_threads,task_type="CPU",class_weights=weights,has_time=True,max_ctr_complexity=1,counter_calc_method="SkipTest",nan_mode="Min",allow_writing_files=False)
    history=[]
    training_started=time.perf_counter()
    class Callback:
        def after_iteration(self,info):
            if cancelled():return False
            if info.iteration==1 or info.iteration%10==0:
                train_loss=float(info.metrics["learn"]["Logloss"][-1])
                val_loss=float(info.metrics["validation"]["Logloss"][-1])
                row={"epoch":info.iteration,"train_loss":train_loss,"val_loss":val_loss,"elapsed_seconds":time.perf_counter()-training_started,"checkpoint":False,"native_validation_score":float(info.metrics["validation"][eval_metric.split(";")[0]][-1]) if eval_metric.split(";")[0] in info.metrics["validation"] else None}
                history.append(row)
                save_json(directory/"history.json",history)
                progress({"epoch":info.iteration,"epochs":config.tree_iterations,"history":history,"message":f"Tree {info.iteration}; validation Logloss {val_loss:.4f}"})
            return True
    model.fit(Pool(xt,label=yt,cat_features=processor.cat),eval_set=Pool(xv,label=yv,cat_features=processor.cat),early_stopping_rounds=config.tree_patience if config.early_stopping else None,use_best_model=False,verbose=False,callbacks=[Callback()])
    if cancelled():raise Cancelled("Cancelled during tree training")
    # Native PR area is not canonical sklearn AP. Select exact AP on validation stages.
    best=-float("inf");best_tree=model.tree_count_;best_p=None
    for i,p in enumerate(model.staged_predict_proba(xv,eval_period=20)):
        count=min((i+1)*20,model.tree_count_)
        p=np.asarray(p)[:,1]
        m=metrics(yv,p,config.threshold)
        score=(-float(np.mean(-(yv*np.log(np.clip(p,1e-8,1))+(1-yv)*np.log(np.clip(1-p,1e-8,1)))))) if config.checkpoint_metric=="val_loss" else m[config.checkpoint_metric]
        if score>best:
            best,best_tree,best_p=score,count,p
    model.shrink(best_tree)
    pv=model.predict_proba(xv)[:,1]
    threshold=choose_threshold(yv,pv,config)
    train_seconds=time.perf_counter()-training_started
    inference_start=time.perf_counter()
    pe=model.predict_proba(xe)[:,1]
    inference_seconds=time.perf_counter()-inference_start
    m=metrics(ye,pe,threshold)
    m.update(training_seconds=train_seconds,inference_seconds=inference_seconds)
    np.savez_compressed(directory/"predictions.npz",validation_y=yv,validation_probability=pv,test_y=ye,test_probability=pe)
    model.save_model(str(directory/"model.cbm"))
    joblib.dump(processor,directory/"preprocessor.joblib")
    save_json(directory/"splits.json",ids)
    artifact={"format_version":2,"id":run_id,"model_family":"catboost","configuration":config.model_dump(),"config_hash":config.config_hash(),"dataset":dataset,"split_hash":split_hash,"feature_schema":{c:{"type":"categorical" if c in processor.cat else "numeric","nullable":True} for c in processor.features},"encoded_features":processor.features,"input_width":len(processor.features),"partition_counts":{"train":len(train),"validation":len(val),"test":len(test)},"class_distribution":{"original":np.bincount(yt).tolist(),"resampled":np.bincount(yt).tolist(),"strategy":"none"},"history":history,"metrics":m,"validation_metrics":metrics(yv,pv,threshold),"selected_threshold":threshold,"threshold_source":"validation" if config.optimize_threshold else "configuration","checkpoint_epoch":best_tree,"early_stopped":model.tree_count_<config.tree_iterations,"curves":curves(ye,pe),"bootstrap_interval":uncertainty(ye,pe,config.bootstrap_samples,config.seed),"timing":{"training_seconds":train_seconds,"inference_seconds":inference_seconds,"total_seconds":time.perf_counter()-started},"software":{"python":platform.python_version(),"catboost":version("catboost"),"numpy":version("numpy")},"paths":{"model":"model.cbm","preprocessor":"preprocessor.joblib","predictions":"predictions.npz"},"tree_parameters":model.get_all_params(),"preprocessing_notes":"Numeric NaN native Min; categorical training-fitted imputation/stable strings. ANN numerical imputation/scaling controls do not apply to this native-tree representation.","selection_notes":"Early stopping uses native PR area/AUC/Logloss. Final checkpoint selected by canonical validation metric at 20-tree stages. Follow-up model search is exploratory; no leaderboard score is imported."}
    save_json(directory/"result.json",artifact)
    return artifact
