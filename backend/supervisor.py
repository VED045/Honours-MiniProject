"""Two independent processes sharing one persistent host; no HTTP-request training."""
import os
import signal
import subprocess
import sys
import time

def main():
    children=[]
    children.append(subprocess.Popen([sys.executable,"-m","uvicorn","backend.app:app","--host","0.0.0.0","--port",os.environ.get("PORT","8000")]))
    if os.environ.get("TRAINING_ENABLED","false").lower()=="true":
        children.append(subprocess.Popen([sys.executable,"-m","backend.worker"]))
    def stop(*args):
        for child in children:
            child.terminate()
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    try:
        while all(child.poll() is None for child in children):
            time.sleep(1)
    finally:
        stop()
        for child in children:
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:child.kill()
    raise SystemExit(next((c.returncode for c in children if c.returncode),0))

if __name__=="__main__":
    main()
