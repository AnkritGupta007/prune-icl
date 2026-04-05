"""
Check whether a specific Hugging Face token can access
the gated repo meta-llama/Llama-3.1-8B.

Important:
- Replace TOKEN below with your real token locally on GAIVI.
- Do NOT commit this file to git with the real token inside.
- After testing, delete the token or replace it with a placeholder.
"""

from __future__ import annotations

import json
import requests

# Replace this locally on GAIVI with your real token.
TOKEN = "INSERT_YOUR_HF_TOKEN_HERE"

# Target gated repo and a simple file that should exist if access is allowed.
MODEL_REPO = "meta-llama/Llama-3.1-8B"
CONFIG_URL = f"https://huggingface.co/{MODEL_REPO}/resolve/main/config.json"
WHOAMI_URL = "https://huggingface.co/api/whoami-v2"


def main():
    # Basic validation to avoid confusing output.
    if TOKEN == "PASTE_YOUR_HF_TOKEN_HERE" or not TOKEN.strip():
        raise ValueError("Please edit this file and replace TOKEN with your real Hugging Face token.")

    headers = {"Authorization": f"Bearer {TOKEN}"}

    print("Checking token validity via whoami-v2...")
    whoami_resp = requests.get(WHOAMI_URL, headers=headers, timeout=30)
    print("whoami status_code:", whoami_resp.status_code)

    try:
        whoami_json = whoami_resp.json()
    except Exception:
        whoami_json = {"raw_text": whoami_resp.text}

    print("whoami response:")
    print(json.dumps(whoami_json, indent=2))

    print("\nChecking gated repo access for:", MODEL_REPO)
    repo_resp = requests.get(CONFIG_URL, headers=headers, timeout=30, allow_redirects=True)
    print("repo status_code:", repo_resp.status_code)

    # Print a short interpretation to make the result easy to understand.
    if repo_resp.status_code == 200:
        print("RESULT: Access granted. You can read the model repo.")
        try:
            print("config snippet:")
            text = repo_resp.text[:500]
            print(text)
        except Exception:
            pass
    elif repo_resp.status_code == 401:
        print("RESULT: Token is missing/invalid for this request.")
    elif repo_resp.status_code == 403:
        print("RESULT: Token is valid, but this account is NOT authorized for the gated repo.")
        print("response text:")
        print(repo_resp.text[:1000])
    else:
        print("RESULT: Unexpected status code.")
        print("response text:")
        print(repo_resp.text[:1000])


if __name__ == "__main__":
    main()
