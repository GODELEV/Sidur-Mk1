from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pyAesCrypt
from Crypto.Hash import SHA256
from Crypto.PublicKey import RSA
from Crypto.Signature import pkcs1_15

from .config import get_paths
from .logger import get_logger


LOGGER = get_logger(__name__)

# Example embedded RSA public key (PEM). Replace with your real key before release.
PUBLIC_KEY_PEM = b"""-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAw6Qm1fM1iQ8BqQF3oYgM
3zFA28VnN0A4mqzZ0zQz+qZx7gEw3Z4Uq9zE4r5o6P3mSgT8J7U6aS1z9CqV8AeC
c6ZpR9rS3YyHH8Uji9jdb6kqz8bbmV4GSkQ3XwZ2wq6O0m7cAef7iY+gGz4RZ5m5
1u2eQ9D+fx6V9Wn0nYx3D9Xl+I5tQYyA3o4q7o9m4s3f8r2B7jO0mKCs9cI+q8qA
b3C4T3ZhD7m0u3Z3b8pS7gqWw+fVQJdQ5bX0m1YF1fV5Jf3H6O0d9t8q6d6D7v8v
9q+f3x6z1s0jPzQ8s4VwKp2u8jX3oYzKQx9n8bQq1bE6b8j3QJmjuq6LZ2a5s8pX
wQIDAQAB
-----END PUBLIC KEY-----"""

# AES buffer size
AES_BUFFER_SIZE = 64 * 1024


@dataclass
class LicenseInfo:
    name: str
    email: str
    license_id: str
    expires_at: Optional[int]  # Unix timestamp or None

    @staticmethod
    def from_json(data: dict) -> "LicenseInfo":
        return LicenseInfo(
            name=data.get("name", ""),
            email=data.get("email", ""),
            license_id=data.get("license_id", ""),
            expires_at=data.get("expires_at"),
        )


class LicenseManager:
    def __init__(self) -> None:
        self.paths = get_paths()
        self.public_key = RSA.import_key(PUBLIC_KEY_PEM)
        self.license_file = self.paths.licenses_dir / "license.bin"

    def validate_signed_payload(self, payload: bytes, signature: bytes) -> bool:
        try:
            h = SHA256.new(payload)
            pkcs1_15.new(self.public_key).verify(h, signature)
            return True
        except Exception as exc:  # signature invalid
            LOGGER.error("License signature invalid: %s", exc)
            return False

    def save_encrypted_license(self, password: str, payload: bytes, signature: bytes) -> None:
        data = json.dumps({"payload": payload.decode("utf-8"), "signature": signature.hex()}).encode("utf-8")
        bio_in = io.BytesIO(data)
        bio_out = io.BytesIO()
        pyAesCrypt.encryptStream(bio_in, bio_out, password, AES_BUFFER_SIZE)
        self.license_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.license_file, "wb") as f:
            f.write(bio_out.getvalue())

    def load_and_validate(self, password: str) -> Optional[LicenseInfo]:
        if not self.license_file.exists():
            return None
        try:
            with open(self.license_file, "rb") as f:
                bio_in = io.BytesIO(f.read())
            bio_out = io.BytesIO()
            pyAesCrypt.decryptStream(bio_in, bio_out, password, AES_BUFFER_SIZE, len(bio_in.getvalue()))
            data = json.loads(bio_out.getvalue().decode("utf-8"))
            payload = data["payload"].encode("utf-8")
            signature = bytes.fromhex(data["signature"])  # hex
            if not self.validate_signed_payload(payload, signature):
                return None
            lic = LicenseInfo.from_json(json.loads(payload.decode("utf-8")))
            if lic.expires_at and int(time.time()) > int(lic.expires_at):
                LOGGER.error("License expired")
                return None
            return lic
        except Exception as exc:
            LOGGER.error("Failed to read license: %s", exc)
            return None
