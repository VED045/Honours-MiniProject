import json
import sqlite3
import zipfile
from backend.store import Store
from ml.config import ExperimentConfig
from scripts.package_artifacts import package

def test_archive_checkpoints_completed_registry_and_excludes_raw(tmp_path):
    store=Store(tmp_path)
    completed=store.create(ExperimentConfig())
    store.update(completed['id'],'running')
    store.update(completed['id'],'completed')
    store.create(ExperimentConfig())
    directory=tmp_path/'artifacts'/completed['id']
    directory.mkdir()
    (directory/'result.json').write_text(json.dumps({'test_fixture':True}))
    (tmp_path/'docs').mkdir()
    (tmp_path/'docs/references.json').write_text('{"records":[]}')
    (tmp_path/'data/raw').mkdir(parents=True)
    (tmp_path/'data/raw/private.csv').write_text('test fixture only')
    archive=package(tmp_path)
    with zipfile.ZipFile(archive) as z:
        assert not any(n.startswith('data/raw') for n in z.namelist())
        restored=tmp_path/'restored'
        z.extractall(restored)
    with sqlite3.connect(restored/'artifacts/registry.sqlite3') as db:
        assert db.execute('SELECT id,status FROM jobs').fetchall()==[(completed['id'],'completed')]
    assert len(store.all())==2 # Packaging did not alter the live registry.
