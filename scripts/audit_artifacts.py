import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ml.evaluation import metrics
from backend.store import Store
home=Path(__file__).resolve().parents[1]
store=Store(home)
checks=[]
primary=[]
for job in store.all():
    if job["status"]!="completed":continue
    directory=home/"artifacts"/job["id"]
    r=json.loads((directory/"result.json").read_text(encoding="utf-8"))
    split=json.loads((directory/"splits.json").read_text())
    assert not set(split["train"])&set(split["validation"])
    assert not set(split["train"])&set(split["test"])
    assert not set(split["validation"])&set(split["test"])
    with np.load(directory/"predictions.npz") as pred:
        assert len(pred["test_y"])==len(split["test"])
        m=metrics(pred["test_y"],pred["test_probability"],r["selected_threshold"])
        for key,val in m.items():
            if isinstance(val,(int,float)):assert abs(val-r["metrics"][key])<1e-10,(r["id"],key)
            else:assert val==r["metrics"][key]
    assert "isFraud" not in r["feature_schema"] and "TransactionID" not in r["feature_schema"]
    raw_hash=hashlib.sha256(json.dumps(r["configuration"],separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
    assert raw_hash==r["config_hash"],r["id"]
    if not r["dataset"]["demonstration"] and not r["dataset"]["development_sample"] and r["configuration"].get("run_role","primary")=="primary":
        primary.append(r)
    checks.append({"id":r["id"],"method":r["configuration"]["method"],"metrics_match":True,"partitions_disjoint":True,"config_hash_match":True})
assert len({r["split_hash"] for r in primary})==1
assert len({r["dataset"]["fingerprint"] for r in primary})==1
output={"completed_runs_verified":len(checks),"primary_full_runs":len(primary),"identical_primary_holdouts":True,"checks":checks}
(home/"docs/artifact_audit.json").write_text(json.dumps(output,indent=2))
print(json.dumps({k:v for k,v in output.items() if k!="checks"},indent=2))
