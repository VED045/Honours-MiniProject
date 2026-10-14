"""Create a consistent, private deployment bundle containing completed experiments."""
import argparse
from contextlib import closing
from pathlib import Path
import sqlite3
import zipfile

def package(home):
    home=Path(home).resolve()
    registry=home/'artifacts/registry.sqlite3'
    if not registry.is_file():raise FileNotFoundError('No experiment registry to export')
    work=home/'work/deployment'
    work.mkdir(parents=True,exist_ok=True)
    with closing(sqlite3.connect(registry)) as source,closing(sqlite3.connect(work/'registry.sqlite3')) as backup:
        source.backup(backup)
        backup.execute("DELETE FROM jobs WHERE status != 'completed'")
        backup.commit()
        backup.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        ids=[row[0] for row in backup.execute('SELECT id FROM jobs')]
    directories=[home/'artifacts'/uid for uid in ids]
    if any(not (d/'result.json').is_file() for d in directories):
        raise FileNotFoundError('A completed experiment is missing its result artifact')
    archive=work/'completed-artifacts.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        z.write(work/'registry.sqlite3','artifacts/registry.sqlite3')
        for directory in directories:
            for path in directory.iterdir():
                if path.is_file() and path.suffix in ['.json','.npz','.pt','.cbm','.joblib']:
                    z.write(path,str(path.relative_to(home)))
        for path in (home/'data/processed').glob('summary-*.json'):
            z.write(path,str(path.relative_to(home)))
        z.write(home/'docs/references.json','docs/references.json')
    return archive

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home',type=Path,default=Path(__file__).resolve().parents[1])
    archive=package(parser.parse_args().home)
    print(archive,archive.stat().st_size)
