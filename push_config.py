#!/usr/bin/env python3
"""
Push strategy config overrides to R2 for live worker pickup.

Usage:
  python push_config.py                          # show current config
  python push_config.py config.json              # upload a file
  python push_config.py '{"mean_reversion":...}' # upload inline JSON
  python push_config.py --clear                  # reset to {} (code defaults)
"""
import os, sys, json
import boto3
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()

BUCKET = os.getenv("R2_BUCKET", "").strip()
KEY = "kalshi/config/strategy_config.json"


def get_client():
    return boto3.client(
        "s3",
        region_name="auto",
        endpoint_url=os.getenv("R2_ENDPOINT_URL", "").strip(),
        aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID", "").strip(),
        aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY", "").strip(),
        config=Config(signature_version="s3v4"),
    )


def read_config():
    try:
        resp = get_client().get_object(Bucket=BUCKET, Key=KEY)
        return json.loads(resp["Body"].read())
    except Exception as e:
        if "NoSuchKey" in str(e):
            return {}
        raise


def write_config(data: dict):
    get_client().put_object(
        Bucket=BUCKET,
        Key=KEY,
        Body=json.dumps(data, indent=2).encode("utf-8"),
        ContentType="application/json",
    )


def main():
    if len(sys.argv) < 2:
        current = read_config()
        print("Current config in R2:")
        print(json.dumps(current, indent=2) if current else "{} (empty — using code defaults)")
        return

    if sys.argv[1] == "--clear":
        write_config({})
        print("Cleared config to {} — worker will revert to code defaults in ~30s")
        return

    arg = sys.argv[1]
    if os.path.isfile(arg):
        data = json.loads(open(arg).read())
    else:
        data = json.loads(arg)

    if not isinstance(data, dict):
        print("Error: config must be a JSON object", file=sys.stderr)
        sys.exit(1)

    write_config(data)
    print(f"Pushed config to R2 — worker picks up in ~30-50s:")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
