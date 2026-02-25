#!/usr/bin/env python3
"""
Push tennis matches config to R2 for live worker pickup.

Usage:
  python push_matches.py                          # show current matches in R2
  python push_matches.py tennis_matches.json      # upload a file
  python push_matches.py '[{"label":"..."}]'      # upload inline JSON
  python push_matches.py --clear                  # reset to [] (no matches)
"""
import os, sys, json
import boto3
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()

BUCKET = os.getenv("R2_BUCKET", "").strip()
KEY = "kalshi/config/tennis_matches.json"


def get_client():
    return boto3.client(
        "s3",
        region_name="auto",
        endpoint_url=os.getenv("R2_ENDPOINT_URL", "").strip(),
        aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID", "").strip(),
        aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY", "").strip(),
        config=Config(signature_version="s3v4"),
    )


def read_matches():
    try:
        resp = get_client().get_object(Bucket=BUCKET, Key=KEY)
        return json.loads(resp["Body"].read())
    except Exception as e:
        if "NoSuchKey" in str(e):
            return []
        raise


def write_matches(data: list):
    get_client().put_object(
        Bucket=BUCKET,
        Key=KEY,
        Body=json.dumps(data, indent=2).encode("utf-8"),
        ContentType="application/json",
    )


def main():
    if len(sys.argv) < 2:
        current = read_matches()
        print("Current matches in R2:")
        if current:
            print(json.dumps(current, indent=2))
            print(f"\n({len(current)} matches)")
        else:
            print("[] (empty — no matches configured)")
        return

    if sys.argv[1] == "--clear":
        write_matches([])
        print("Cleared matches to [] — worker will have no matches until updated")
        return

    arg = sys.argv[1]
    if os.path.isfile(arg):
        data = json.loads(open(arg).read())
    else:
        data = json.loads(arg)

    if not isinstance(data, list):
        print("Error: matches must be a JSON array", file=sys.stderr)
        sys.exit(1)

    write_matches(data)
    print(f"Pushed {len(data)} matches to R2 — worker picks up in ~60s:")
    for m in data:
        print(f"  - {m.get('label', '?')} ({m.get('player_code', '?')} vs {m.get('opponent_code', '?')})")


if __name__ == "__main__":
    main()
