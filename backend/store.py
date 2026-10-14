import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

def now():
    return datetime.now(timezone.utc).isoformat()

class Store:
    def __init__(self, home):
        self.home=Path(home)
        (self.home/"artifacts").mkdir(parents=True,exist_ok=True)
        self.path=self.home/"artifacts/registry.sqlite3"
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY,status TEXT NOT NULL,config TEXT NOT NULL,created TEXT NOT NULL,updated TEXT NOT NULL,progress TEXT NOT NULL,error TEXT,cancel INTEGER DEFAULT 0)")
            db.execute("CREATE TABLE IF NOT EXISTS configs (id TEXT PRIMARY KEY,name TEXT,config TEXT,created TEXT)")

    def connect(self):
        db=sqlite3.connect(self.path,timeout=30)
        db.row_factory=sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        return db

    def create(self,config):
        uid=uuid.uuid4().hex
        stamp=now()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT count(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]>=20:
                raise ValueError("Queue is full; finish or cancel existing jobs")
            db.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?,?,0)",(uid,"queued",config.model_dump_json(),stamp,stamp,"{}",None))
        return self.get(uid)

    @staticmethod
    def decode(row):
        value=dict(row)
        value["config"]=json.loads(value["config"])
        value["progress"]=json.loads(value["progress"])
        return value

    def get(self,uid):
        with self.connect() as db:
            row=db.execute("SELECT * FROM jobs WHERE id=?",(uid,)).fetchone()
        if row is None:
            raise KeyError(uid)
        return self.decode(row)

    def all(self):
        with self.connect() as db:
            return [self.decode(r) for r in db.execute("SELECT * FROM jobs ORDER BY created DESC LIMIT 500")]

    def claim(self):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            # Heartbeat runs in its own thread; interrupted processes become explicit failures.
            db.execute("UPDATE jobs SET status='failed',error='Worker interrupted; duplicate configuration to retry',updated=? WHERE status='running' AND julianday(updated)<julianday('now','-10 minutes')",(now(),))
            row=db.execute("SELECT * FROM jobs WHERE status='queued' AND cancel=0 ORDER BY created LIMIT 1").fetchone()
            if row is None:
                return None
            db.execute("UPDATE jobs SET status='running',updated=? WHERE id=? AND status='queued'",(now(),row["id"]))
        return self.get(row["id"])

    def update(self,uid,status=None,progress=None,error=None):
        with self.connect() as db:
            row=db.execute("SELECT status FROM jobs WHERE id=?",(uid,)).fetchone()
            if row is None:
                raise KeyError(uid)
            old=row["status"]
            allowed={"queued":{"running","cancelled","failed"},"running":{"completed","failed","cancelled"},"completed":set(),"failed":set(),"cancelled":set()}
            if status and status != old and status not in allowed[old]:
                raise ValueError(f"Invalid status transition {old} -> {status}")
            db.execute("UPDATE jobs SET status=?,updated=?,progress=COALESCE(?,progress),error=COALESCE(?,error) WHERE id=?",(status or old,now(),json.dumps(progress) if progress is not None else None,error,uid))

    def cancel(self,uid):
        job=self.get(uid)
        with self.connect() as db:
            db.execute("UPDATE jobs SET cancel=1,status=CASE WHEN status='queued' THEN 'cancelled' ELSE status END,updated=? WHERE id=? AND status IN ('queued','running')",(now(),uid))
        return self.get(uid)

    def save_config(self,config):
        uid=uuid.uuid4().hex
        with self.connect() as db:
            db.execute("INSERT INTO configs VALUES (?,?,?,?)",(uid,config.name,config.model_dump_json(),now()))
        return {"id":uid,"config":config.model_dump(),"hash":config.config_hash()}

    def configs(self):
        with self.connect() as db:
            return [{"id":r["id"],"name":r["name"],"created":r["created"],"config":json.loads(r["config"])} for r in db.execute("SELECT * FROM configs ORDER BY created DESC")]
