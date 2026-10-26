from __future__ import annotations

import os
from pathlib import Path


def _load_env_file() -> None:
    """Load key-value pairs from root .env into os.environ if not already set."""
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if not env_file.is_file():
        return

    try:
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = val
    except OSError:
        pass


_load_env_file()
