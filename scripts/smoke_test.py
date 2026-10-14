import argparse
import os
import httpx
parser=argparse.ArgumentParser()
parser.add_argument("--base-url",default="http://127.0.0.1:8000")
parser.add_argument("--private-key-env",default="API_KEY")
args=parser.parse_args()
key=os.environ.get(args.private_key_env,"")
with httpx.Client(base_url=args.base_url.rstrip("/"),timeout=240,headers={"Authorization":f"Bearer {key}"} if key else {}) as client:
    assert client.get("/health").json()["status"]=="ok"
    for path in ["/api/v1/models","/api/v1/configurations","/api/v1/dataset/summary"]:
        response=client.get(path);response.raise_for_status()
        print(path,response.status_code)
    models=client.get("/api/v1/models").json()
    if models:
        uid=models[0]["id"]
        metadata=client.get(f"/api/v1/models/{uid}/metadata").json()
        metric=client.get(f"/api/v1/experiments/{uid}/metrics").json()
        assert metric["average_precision"]==metadata["metrics"]["average_precision"]
        values={name:None for name in metadata["feature_schema"]}
        response=client.post("/api/v1/predict",json={"model_id":uid,"features":values,"explain":True})
        response.raise_for_status()
        output=response.json()
        assert 0<=output["probability"]<=1 and output["explanation"]["contributions"]
        print("registered model prediction and explanation passed",uid)
    else:
        print("No registered models; prediction smoke check skipped explicitly")
print("Smoke checks passed")
