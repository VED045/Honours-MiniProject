import argparse
import logging
import os
import threading
import time
from pathlib import Path
from backend.store import Store
from ml.config import ExperimentConfig
from ml.pipeline import run, Cancelled

HOME=Path(os.environ.get("FRAUD_HOME",Path(__file__).resolve().parents[1])).resolve()

def execute(store,job):
    uid=job["id"]
    stop=threading.Event()
    def heartbeat():
        while not stop.wait(30):
            store.update(uid)
    thread=threading.Thread(target=heartbeat,daemon=True)
    thread.start()
    try:
        config=ExperimentConfig.model_validate(job["config"])
        run(config,store.home,uid,progress=lambda value:store.update(uid,progress=value),cancelled=lambda:bool(store.get(uid)["cancel"]))
        store.update(uid,"cancelled" if store.get(uid)["cancel"] else "completed")
    except Cancelled:
        store.update(uid,"cancelled")
    except Exception as exc:
        logging.exception("Experiment %s failed",uid)
        # Only class and safe project-level errors; no environment or credentials.
        store.update(uid,"failed",error=f"{type(exc).__name__}: {str(exc)[:400]}")
    finally:
        stop.set()
        thread.join(timeout=2)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--once",action="store_true")
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
    store=Store(HOME)
    while True:
        job=store.claim()
        if job:
            execute(store,job)
        elif args.once:
            break
        else:
            time.sleep(2)
        if args.once:
            break

if __name__=="__main__":
    main()
