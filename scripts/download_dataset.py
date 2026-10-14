"""Official Kaggle acquisition with safe reruns and optional automatic retry."""
import argparse
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ml.data import verify, ALL_FILES

def extract(archive,root):
    root=Path(root).resolve()
    root.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        bad=z.testzip()
        if bad:
            raise ValueError(f"Archive CRC failed: {bad}")
        infos={Path(i.filename).name:i for i in z.infolist() if Path(i.filename).name in ALL_FILES}
        if set(ALL_FILES)-set(infos):
            raise ValueError(f"Incomplete archive: missing {set(ALL_FILES)-set(infos)}")
        if shutil.disk_usage(root).free < sum(i.file_size for i in infos.values())*1.1:
            raise OSError("Insufficient free disk space")
        for name,info in infos.items():
            target=root/name
            if target.is_file() and target.stat().st_size==info.file_size:
                continue
            tmp=target.with_suffix(".partial")
            with z.open(info) as source,tmp.open("wb") as out:
                shutil.copyfileobj(source,out)
            tmp.replace(target)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--archive",type=Path)
    parser.add_argument("--watch-auth",action="store_true",help="Retry official acquisition every 60 seconds until authentication/rules are ready")
    parser.add_argument("--attempt-official",action="store_true",help="Attempt CLI even when existing training files are verified")
    args=parser.parse_args()
    home=Path(__file__).resolve().parents[1]
    root=home/"data/raw/ieee-cis"
    root.mkdir(parents=True,exist_ok=True)
    try:
        manifest=verify(root)
        if not args.attempt_official:
            print("Verified training files already exist; skipped download",manifest["fingerprint"])
            return
    except (FileNotFoundError,ValueError):
        pass
    archive=args.archive
    if archive is None:
        cli=Path(sys.executable).parent/("kaggle.exe" if sys.platform=="win32" else "kaggle")
        while True:
            # Capture logs but never print CLI output which could contain credential details.
            attempt=subprocess.run([str(cli),"competitions","download","-c","ieee-fraud-detection","-p",str(root)],capture_output=True,text=True,timeout=120)
            record={"mechanism":"official Kaggle CLI","attempted":True,"exit_code":attempt.returncode,"success":attempt.returncode==0,"credential_output_redacted":True}
            (home/"docs/download_attempt.json").write_text(json.dumps(record,indent=2))
            if attempt.returncode==0:
                break
            print("Official download failed. Run .\.venv\Scripts\kaggle.exe auth login; accept competition rules at https://www.kaggle.com/competitions/ieee-fraud-detection/rules. Do not share tokens in chat.")
            if not args.watch_auth:
                raise SystemExit(2)
            print("Waiting for authentication; retrying in 60 seconds. Ctrl+C stops.")
            time.sleep(60)
        archive=root/"ieee-fraud-detection.zip"
    extract(archive,root)
    manifest=verify(root)
    (root/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))

if __name__=="__main__":
    main()
