#!/usr/bin/env python3
"""
Start both the API and the Dashboard FastAPI apps by launching two uvicorn processes.

This script uses the same environment variables as the Dockerfile / project to
determine host/port for each app and will restart a server if it exits.
"""
import os
import sys
import time
import subprocess

PY = sys.executable or "python"

PROXY_HOST = os.getenv("PROXY_HOST", "0.0.0.0")
PROXY_PORT = int(os.getenv("PROXY_PORT", "8128"))
WEB_HOST = os.getenv("WEB_HOST", "0.0.0.0")
WEB_PORT = int(os.getenv("WEB_PORT", "8028"))

API_MODULE = "ollama_router:api_app"
WEB_MODULE = "ollama_router:web_app"

def start_uvicorn(module: str, host: str, port: int):
    return subprocess.Popen([
        PY, "-m", "uvicorn", module,
        "--host", host,
        "--port", str(port),
    ])

def main():
    api_proc = start_uvicorn(API_MODULE, PROXY_HOST, PROXY_PORT)
    web_proc = start_uvicorn(WEB_MODULE, WEB_HOST, WEB_PORT)

    try:
        while True:
            time.sleep(5)
            # Restart if any process exited
            if api_proc.poll() is not None:
                print("[start_servers] API server exited; restarting...")
                api_proc = start_uvicorn(API_MODULE, PROXY_HOST, PROXY_PORT)
            if web_proc.poll() is not None:
                print("[start_servers] Web server exited; restarting...")
                web_proc = start_uvicorn(WEB_MODULE, WEB_HOST, WEB_PORT)
    except KeyboardInterrupt:
        print("[start_servers] Shutting down child processes...")
        for p in (api_proc, web_proc):
            try:
                p.terminate()
            except Exception:
                pass

if __name__ == "__main__":
    main()
