import json
import shutil
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ml.data import verify

if __name__=="__main__":
    home=Path(__file__).resolve().parents[1]
    root=home/"data/raw/ieee-cis"
    result=verify(root)
    result["free_disk_bytes"]=shutil.disk_usage(root).free
    (root/"manifest.json").write_text(json.dumps(result,indent=2))
    (home/"docs/dataset_verification.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
