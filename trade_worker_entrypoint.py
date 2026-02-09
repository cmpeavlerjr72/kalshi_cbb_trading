# trade_worker_entrypoint.py
import os
import time
import base64
from pathlib import Path
import subprocess

def bootstrap_kalshi_key_from_env():
    """
    Optional: store the private key as base64 in env var KALSHI_PRIVATE_KEY_B64.
    This writes it to a file and sets KALSHI_PRIVATE_KEY_PATH so existing code works.
    """
    b64 = (os.getenv("KALSHI_PRIVATE_KEY_B64") or "").strip()
    if not b64:
        return

    key_path = Path(os.getenv("KALSHI_PRIVATE_KEY_PATH", "/tmp/kalshi.key"))
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_bytes(base64.b64decode(b64))
    os.environ["KALSHI_PRIVATE_KEY_PATH"] = str(key_path)
    print(f"[bootstrap] Wrote Kalshi key to {key_path}")

def run_once() -> int:
    # Run the worker-friendly runner (this already handles R2 sync internally)
    p = subprocess.run(["python", "tonight_runner_cloud.py"], check=False)
    return int(p.returncode or 0)

def main():
    bootstrap_kalshi_key_from_env()

    # If you want the Render worker to run continuously (recommended):
    # set RUN_FOREVER=1 in Render env vars.
    run_forever = (os.getenv("RUN_FOREVER") or "0").strip() == "1"
    delay = int(os.getenv("RESTART_DELAY_SECS") or "30")

    if not run_forever:
        raise SystemExit(run_once())

    print(f"[entrypoint] RUN_FOREVER=1. Will restart after exit (delay={delay}s).")
    while True:
        rc = run_once()
        print(f"[entrypoint] runner exited rc={rc}. sleeping {delay}s...")
        time.sleep(max(5, delay))

if __name__ == "__main__":
    main()
