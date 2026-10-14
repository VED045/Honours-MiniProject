"""Bounded follow-up search, validation-selected, never leaderboard score substitution."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.store import Store
from backend.worker import execute
from ml.config import ExperimentConfig
home=Path(__file__).resolve().parents[1]
store=Store(home)
for depth,lr,l2 in [(8,.04,7),(7,.035,3)]:
    config=ExperimentConfig(dataset="ieee-cis",method="catboost",run_role="exploratory",max_rows=0,features=[],name=f"Optimized CatBoost - all predictors depth {depth}",tree_iterations=1800,tree_depth=depth,tree_learning_rate=lr,tree_l2=l2,tree_patience=150)
    job=store.create(config)
    store.update(job["id"],"running")
    execute(store,store.get(job["id"]))
    final=store.get(job["id"])
    print(config.name,final["id"],final["status"],final["error"] or "",flush=True)
