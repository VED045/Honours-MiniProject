import json
import sqlite3
import zipfile
from pathlib import Path
home=Path(__file__).resolve().parents[1]
work=home/"work/deployment"
work.mkdir(parents=True,exist_ok=True)
with sqlite3.connect(home/"artifacts/registry.sqlite3") as source,sqlite3.connect(work/"registry.sqlite3") as backup:
    source.backup(backup)
    backup.execute("DELETE FROM jobs WHERE status != 'completed'")
    backup.commit()
archive=work/"completed-artifacts.zip"
with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED) as z:
    z.write(work/"registry.sqlite3","artifacts/registry.sqlite3")
    for directory in (home/"artifacts").iterdir():
        if not directory.is_dir() or not (directory/"result.json").is_file():continue
        for path in directory.iterdir():
            if path.is_file() and path.suffix in [".json",".npz",".pt",".cbm",".joblib"]:
                z.write(path,str(path.relative_to(home)))
    for path in (home/"data/processed").glob("summary-*.json"):
        z.write(path,str(path.relative_to(home)))
    z.write(home/"docs/references.json","docs/references.json")
print(archive,archive.stat().st_size)
