import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.store import Store
from backend.worker import execute
from ml.config import ExperimentConfig

FEATURES=["TransactionAmt","ProductCD","card4","card6","addr2","dist1","dist2"]+[f"C{i}" for i in range(1,15)]+["D1","D2","D3","D4","D10","D15"]+[f"V{i}" for i in range(1,31)]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset",choices=["ieee-cis","synthetic"],default="ieee-cis")
    parser.add_argument("--seeds",nargs="+",type=int,default=[42])
    parser.add_argument("--max-rows",type=int,default=30000)
    parser.add_argument("--epochs",type=int,default=15)
    parser.add_argument("--config",type=Path)
    args=parser.parse_args()
    home=Path(__file__).resolve().parents[1]
    store=Store(home)
    if args.config:
        configs=[ExperimentConfig.model_validate_json(args.config.read_text())]
    else:
        configs=[ExperimentConfig(run_role="primary",dataset=args.dataset,method=method,seed=seed,max_rows=args.max_rows,epochs=args.epochs,features=FEATURES if args.dataset=="ieee-cis" else [],name=f"{args.dataset} {method} seed {seed}") for seed in args.seeds for method in ["baseline","smote","cost_sensitive"]]
    failures=[]
    for config in configs:
        job=store.create(config)
        store.update(job["id"],"running")
        execute(store,store.get(job["id"]))
        final=store.get(job["id"])
        print(final["id"],config.method,final["status"],final["error"] or "",flush=True)
        if final["status"]!="completed":
            failures.append(final["id"])
    if failures:
        raise SystemExit(1)

if __name__=="__main__":
    main()
