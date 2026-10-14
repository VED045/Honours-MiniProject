import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.store import Store
from backend.worker import execute
from ml.config import ExperimentConfig
from scripts.run_experiments import FEATURES
home=Path(__file__).resolve().parents[1]
store=Store(home)
# Fixed, reference-motivated feature budget; no train/test-concatenated group statistics.
expanded=["TransactionAmt","ProductCD"]+[f"card{i}" for i in range(1,7)]+["addr1","addr2","dist1","dist2","P_emaildomain","R_emaildomain"]+[f"C{i}" for i in range(1,15)]+[f"D{i}" for i in range(1,16)]+[f"M{i}" for i in range(1,10)]+[f"V{i}" for i in range(1,340,3)]+[f"id_{i:02}" for i in range(1,39)]+["DeviceType","DeviceInfo"]
configs=[
ExperimentConfig(dataset="ieee-cis",method="catboost",run_role="exploratory",max_rows=0,features=FEATURES,name="CatBoost reference - same 57 predictors",tree_iterations=500,tree_depth=7),
ExperimentConfig(dataset="ieee-cis",method="catboost",run_role="exploratory",max_rows=0,features=expanded,name="CatBoost reference - expanded native predictors",tree_iterations=800,tree_depth=7),
ExperimentConfig(dataset="ieee-cis",method="baseline",run_role="exploratory",max_rows=0,features=FEATURES,name="Enhanced ANN - robust scaling",hidden_layers=[128,64,32],dropout=[.1,.1,.05],scaling="robust",learning_rate=.0005,batch_size=512,epochs=30,patience=7),
ExperimentConfig(dataset="ieee-cis",method="cost_sensitive",run_role="exploratory",max_rows=0,features=FEATURES,name="Enhanced weighted ANN - weight 5",hidden_layers=[128,64,32],dropout=[.1,.1,.05],scaling="robust",learning_rate=.0005,batch_size=512,epochs=30,patience=7,fraud_weight=5)
]
configs=configs[2:]+configs[:2]
for config in configs:
    job=store.create(config);store.update(job["id"],"running")
    execute(store,store.get(job["id"]))
    final=store.get(job["id"])
    print(config.name,final["id"],final["status"],final["error"] or "",flush=True)

