import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ml.data import summary
import httpx
parser=argparse.ArgumentParser()
parser.add_argument("--base-url",default="")
args=parser.parse_args()
home=Path(__file__).resolve().parents[1]
out=home/"data/processed"
out.mkdir(parents=True,exist_ok=True)
for source in ["ieee-cis","synthetic"]:
    if args.base_url:
        response=httpx.get(args.base_url.rstrip("/")+"/api/v1/dataset/summary",params={"source":source},timeout=240)
        response.raise_for_status()
        data=response.json()
    else:
        data=summary(home,source)
    root=home/"data/raw/ieee-cis"
    data["raw_signature"]=[[p.name,p.stat().st_size,p.stat().st_mtime_ns] for p in root.glob("train*.csv")] if source=="ieee-cis" else []
    (out/f"summary-{source}.json").write_text(json.dumps(data,indent=2))
    print(source,data["rows"],data["columns"],data["fingerprint"])
