"""Restore only an operator-supplied trusted research archive into an empty store."""
import argparse
from pathlib import Path
import zipfile
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("archive",type=Path)
parser.add_argument("--home",type=Path,required=True)
args=parser.parse_args()
target=args.home.resolve()
if (target/"artifacts/registry.sqlite3").exists():
    raise SystemExit("Registry already exists. Stop services and choose an empty storage directory; no overwrite performed.")
with zipfile.ZipFile(args.archive) as archive:
    for entry in archive.infolist():
        destination=(target/entry.filename).resolve()
        if not destination.is_relative_to(target) or entry.filename.startswith(("/","\\")):
            raise SystemExit("Unsafe archive path")
        if entry.filename.split("/")[0] not in {"artifacts","data","docs"}:
            raise SystemExit("Unexpected archive content")
        if (entry.external_attr>>16)&0o170000==0o120000:
            raise SystemExit("Symbolic link entries are forbidden")
    if archive.testzip() is not None:raise SystemExit("Archive integrity check failed")
    archive.extractall(target)
print("Trusted artifacts restored to",target)
