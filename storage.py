import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

LOCAL_DIR = Path("data")
LOCAL_DIR.mkdir(exist_ok=True)
LOCAL_RESULTS = LOCAL_DIR / "participants.csv"
LOCAL_CHATS = LOCAL_DIR / "chat_logs.jsonl"


def new_participant_id():
    return "P" + uuid.uuid4().hex[:8].upper()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def storage_mode():
    return os.getenv("STORAGE_MODE", "LOCAL").upper()


def _post(payload: dict):
    url = os.getenv("GOOGLE_APPS_SCRIPT_URL", "").strip()
    if not url:
        raise RuntimeError("GOOGLE_APPS_SCRIPT_URL is missing. Configure the deployment secret before collecting real responses.")
    # Apps Script ContentService web apps commonly return a 302 redirect
    # after doPost() has executed. The redirected googleusercontent.com URL
    # is a one-time response URL and expects GET, not another POST. If we
    # repeat the POST at that URL, Google returns HTTP 405 Method Not Allowed.
    # Therefore: POST to the /exec URL once, then GET the redirect target.
    response = requests.post(url, json=payload, timeout=30, allow_redirects=False)
    if response.status_code in (301, 302, 303, 307, 308):
        redirect_url = response.headers.get("Location")
        if not redirect_url:
            raise RuntimeError("Google Apps Script returned a redirect without a Location header.")
        response = requests.get(redirect_url, timeout=30, allow_redirects=True)

    response.raise_for_status()
    try:
        body = response.json()
    except Exception as exc:
        raise RuntimeError(
            f"Remote storage returned a non-JSON response (HTTP {response.status_code}). "
            "Check the Apps Script deployment and its execution logs."
        ) from exc
    if body.get("ok") is False:
        raise RuntimeError(body.get("error", "Remote storage rejected the response."))
    if body.get("ok") is not True:
        raise RuntimeError("Remote storage did not confirm that the response was saved.")
    return body


def append_result(record: dict):
    if storage_mode() == "APPS_SCRIPT":
        _post({"type": "response", "record": record})
        return
    df = pd.DataFrame([record])
    if LOCAL_RESULTS.exists():
        df.to_csv(LOCAL_RESULTS, mode="a", header=False, index=False)
    else:
        df.to_csv(LOCAL_RESULTS, index=False)


def append_chat(participant_id, messages):
    payload = {
        "participant_id": participant_id,
        "timestamp_utc": utc_now(),
        "messages": messages,
    }
    if storage_mode() == "APPS_SCRIPT":
        _post({"type": "chat", "record": payload})
        return
    with open(LOCAL_CHATS, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
