# test_key_load.py
import os
from dotenv import load_dotenv
from cryptography.hazmat.primitives import serialization

load_dotenv()

path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
print("KEY PATH:", path)

with open(path, "rb") as f:
    key = serialization.load_pem_private_key(f.read(), password=None)

print("✅ Private key loaded successfully:", type(key))
