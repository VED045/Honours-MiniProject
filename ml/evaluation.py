import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, balanced_accuracy_score, matthews_corrcoef, roc_curve, precision_recall_curve

def metrics(y, probability, threshold=0.5):
    y, p = np.asarray(y, dtype=int), np.asarray(probability, dtype=float)
    if y.shape != p.shape or p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be finite, one-dimensional and in [0,1]")
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be in [0,1]")
    pred = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    return {"accuracy": float(accuracy_score(y,pred)), "precision": float(precision_score(y,pred,zero_division=0)), "recall": float(recall_score(y,pred,zero_division=0)), "f1": float(f1_score(y,pred,zero_division=0)), "roc_auc": float(roc_auc_score(y,p)) if len(np.unique(y)) == 2 else None, "average_precision": float(average_precision_score(y,p)) if y.sum() else None, "specificity": float(tn/(tn+fp)) if tn+fp else 0, "false_positive_rate": float(fp/(tn+fp)) if tn+fp else 0, "false_negative_rate": float(fn/(tp+fn)) if tp+fn else 0, "balanced_accuracy": float(balanced_accuracy_score(y,pred)), "mcc": float(matthews_corrcoef(y,pred)), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "flagged": int(pred.sum()), "transactions": len(y), "prevalence": float(y.mean()), "threshold": float(threshold), "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]}

def curves(y,p):
    fpr,tpr,rt = roc_curve(y,p)
    precision,recall,pt = precision_recall_curve(y,p)
    def compress(a,b,limit=400):
        ids = np.unique(np.linspace(0,len(a)-1,min(len(a),limit)).astype(int))
        return [{"x":float(a[i]),"y":float(b[i])} for i in ids]
    return {"roc": compress(fpr,tpr), "pr": compress(recall,precision), "threshold": [{"threshold":float(t), **{k:v for k,v in metrics(y,p,t).items() if k in ["precision","recall","f1","fp","fn","flagged"]}} for t in np.linspace(0,1,51)]}

def choose_threshold(y,p,config):
    if not config.optimize_threshold:
        return config.threshold
    # Fixed grid prevents pathological O(n^2) searches on full IEEE-CIS.
    candidates = np.linspace(0.01,0.99,99)
    return float(max(candidates, key=lambda t: (metrics(y,p,t)[config.selection_metric],t)))

def uncertainty(y,p,n,seed):
    if not n:
        return {}
    rng = np.random.default_rng(seed)
    scores = {"average_precision":[], "roc_auc":[]}
    for _ in range(n):
        ids = rng.integers(0,len(y),len(y))
        if len(np.unique(y[ids])) < 2:
            continue
        for key,value in scores.items():
            value.append(metrics(y[ids],p[ids])[key])
    return {k:{"low":float(np.percentile(v,2.5)),"high":float(np.percentile(v,97.5)),"replicates":len(v)} for k,v in scores.items() if v}
