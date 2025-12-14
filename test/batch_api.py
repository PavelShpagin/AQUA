#!/usr/bin/env python3
"""
Minimal OpenAI Batch API prototype to prove end-to-end functionality.

- Creates a JSONL with 3 chat completions
- Uploads file
- Creates batch
- Polls until completion
- Downloads and prints parsed outputs

Env requirements:
  OPENAI_API_KEY

Optional:
  FORCE_OPENAI_DIRECT_BATCH (on/off) defaults to direct OpenAI
"""

import os
import time
import json
import tempfile
import requests
from dotenv import load_dotenv


OPENAI_BASE = os.getenv("OPENAI_BASE", "https://api.openai.com/v1").rstrip("/")

# Load environment variables from .env if present
load_dotenv()


def _auth_headers():
    key = (os.getenv("OPENAI_API_KEY", "").strip() or os.getenv("API_TOKEN", "").strip())
    if not key:
        raise RuntimeError("OPENAI_API_KEY or API_TOKEN not set")
    headers = {"Authorization": f"Bearer {key}"}
    # Optional proxy headers (e.g., Transparent API / Sparta)
    calling = os.getenv("LLM_PROXY_CALLING_SERVICE", "").strip()
    if calling:
        headers["LLM-Proxy-Calling-Service"] = calling
    req_id = os.getenv("X_REQUEST_ID", "").strip()
    if req_id:
        headers["X-Request-Id"] = req_id
    return headers


def create_jsonl(prompts, model="gpt-4.1-nano"):
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    for i, p in enumerate(prompts):
        rec = {
            "custom_id": f"p-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": p},
                ],
                "max_tokens": 128,
                "temperature": 0.0,
                "response_format": {"type": "json_object"}
            }
        }
        tmp.write(json.dumps(rec) + "\n")
    tmp.close()
    return tmp.name


def upload_file(path):
    url = f"{OPENAI_BASE}/files"
    with open(path, "rb") as f:
        resp = requests.post(url, headers=_auth_headers(), files={"file": f}, data={"purpose": "batch"}, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"upload failed: {resp.status_code} {resp.text[:200]}")
    return resp.json()["id"]


def create_batch(file_id):
    url = f"{OPENAI_BASE}/batches"
    payload = {"input_file_id": file_id, "endpoint": "/v1/chat/completions", "completion_window": "24h"}
    resp = requests.post(url, headers={**_auth_headers(), "Content-Type": "application/json"}, json=payload, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"create batch failed: {resp.status_code} {resp.text[:200]}")
    return resp.json()["id"]


def poll_batch(batch_id, timeout_sec=900):
    url = f"{OPENAI_BASE}/batches/{batch_id}"
    start = time.time()
    last_status = None
    last_print_ts = 0.0

    def fmt_dur(seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    while time.time() - start < timeout_sec:
        try:
            resp = requests.get(url, headers=_auth_headers(), timeout=45)
        except Exception as e:
            # transient network; show heartbeat
            now = time.time()
            if now - last_print_ts >= 15.0:
                print(f"[poll] {fmt_dur(now - start)} network error: {e}")
                last_print_ts = now
            time.sleep(2.0)
            continue

        if resp.status_code == 200:
            data = resp.json()
            st = str(data.get("status", "")).lower() or "unknown"
            counts = data.get("request_counts") or {}
            total = counts.get("total")
            completed = counts.get("completed")
            failed = counts.get("failed")
            now = time.time()

            # periodic progress or on status change
            if st != last_status or (now - last_print_ts) >= 15.0:
                extra = []
                if total is not None:
                    extra.append(f"done {completed or 0}/{total}")
                if failed:
                    extra.append(f"failed {failed}")
                print(f"[poll] {fmt_dur(now - start)} status={st}" + (" | " + ", ".join(extra) if extra else ""))
                last_status = st
                last_print_ts = now

            if st in {"completed", "succeeded", "success"}:
                ofid = data.get("output_file_id") or (data.get("output_file_ids", [None]) or [None])[0]
                return ofid
            if st in {"failed", "canceled", "expired"}:
                raise RuntimeError(f"batch {batch_id} failed: {data}")

        else:
            now = time.time()
            if now - last_print_ts >= 15.0:
                print(f"[poll] {fmt_dur(now - start)} http {resp.status_code}: {resp.text[:120]}")
                last_print_ts = now

        time.sleep(2.0)

    return None


def download_file(file_id):
    url = f"{OPENAI_BASE}/files/{file_id}/content"
    for _ in range(4):
        resp = requests.get(url, headers=_auth_headers(), timeout=120)
        if resp.status_code == 200 and resp.text.strip():
            return resp.text
        time.sleep(2.0)
    raise RuntimeError(f"download failed: {resp.status_code} {resp.text[:200]}")


def main():
    prompts = [
        "Return JSON: {\"message\": \"hello\"}",
        "Return JSON with key 'sum' for 2+3",
        "Return JSON with key 'lang' detecting 'Hola mundo'",
    ]
    jsonl = create_jsonl(prompts)
    print(f"JSONL: {jsonl}")
    fid = upload_file(jsonl)
    print(f"file_id: {fid}")
    bid = create_batch(fid)
    print(f"batch_id: {bid}")
    # Default long wait (2h) but can be overridden via BATCH_MAX_WAIT_SECS
    ofid = poll_batch(bid, timeout_sec=int(os.getenv("BATCH_MAX_WAIT_SECS", "7200")))
    if not ofid:
        raise SystemExit("Timed out waiting for batch")
    print(f"output_file_id: {ofid}")
    text = download_file(ofid)
    print("--- raw output head ---")
    print("\n".join(text.splitlines()[:5]))
    print("--- parsed ---")
    parsed = []
    for ln in text.splitlines():
        if not ln.strip():
            continue
        rec = json.loads(ln)
        body = rec.get("response", {}).get("body")
        if isinstance(body, str):
            try:
                body = json.loads(body)
            except Exception:
                pass
        content = ""
        if isinstance(body, dict):
            ch = (body.get("choices") or [{}])[0]
            msg = ch.get("message", {})
            raw = msg.get("content")
            if isinstance(raw, str):
                content = raw
            elif isinstance(raw, list):
                # look for json parts
                for part in raw:
                    if isinstance(part, dict) and part.get("type") in ("output_json", "json"):
                        content = json.dumps(part.get("json"), ensure_ascii=False)
                        break
                if not content:
                    texts = [str(p.get("text", "")) for p in raw if isinstance(p, dict)]
                    content = "\n".join([t for t in texts if t])
        parsed.append({"custom_id": rec.get("custom_id"), "content": content})
    for r in parsed:
        print(r)


if __name__ == "__main__":
    main()

    #
    # Sharded demo mode: submit multiple small batch jobs and show per-shard progress
    # Controlled via env vars: BATCH_POC_SHARDS, BATCH_POC_PER_SHARD, MODEL
    #
    shards = int(os.getenv("BATCH_POC_SHARDS", "4"))
    per_shard = int(os.getenv("BATCH_POC_PER_SHARD", "5"))
    model = os.getenv("MODEL", "gpt-4.1-nano").strip()

    def make_prompts(n):
        base = [
            "Return JSON: {\"message\": \"hello\"}",
            "Return JSON with key 'sum' for 2+3",
            "Return JSON with key 'lang' detecting 'Hola mundo'",
        ]
        out = []
        for i in range(n):
            out.append(base[i % len(base)] + f" #[{i}]")
        return out

    jobs = []
    print(f"Submitting {shards} shards, {per_shard} requests each (model={model})")
    for si in range(shards):
        prompts = make_prompts(per_shard)
        jsonl = create_jsonl(prompts, model=model)
        fid = upload_file(jsonl)
        bid = create_batch(fid)
        jobs.append({
            "shard": si,
            "jsonl": jsonl,
            "file_id": fid,
            "batch_id": bid,
            "status": "submitted",
            "total": None,
            "done": 0,
            "failed": 0,
            "finalized": False,
        })
        print(f"shard {si}: file_id={fid} batch_id={bid}")

    def get_status(batch_id):
        url = f"{OPENAI_BASE}/batches/{batch_id}"
        r = requests.get(url, headers=_auth_headers(), timeout=45)
        if r.status_code != 200:
            return {"status": f"http_{r.status_code}", "raw": r.text}
        return r.json()

    def poll_many(all_jobs, timeout_sec=int(os.getenv("BATCH_MAX_WAIT_SECS", "7200"))):
        start = time.time()
        last_print = 0.0
        def fmt(seconds: float) -> str:
            m, s = divmod(int(seconds), 60)
            h, m = divmod(m, 60)
            return f"{h:02d}:{m:02d}:{s:02d}"
        done_states = {"completed", "succeeded", "success"}
        fail_states = {"failed", "canceled", "expired"}
        while time.time() - start < timeout_sec:
            total_req = 0
            total_done = 0
            total_failed = 0
            shards_done = 0
            for j in all_jobs:
                if j.get("finalized"):
                    shards_done += 1
                    continue
                data = get_status(j["batch_id"]) or {}
                st = str(data.get("status", "unknown")).lower()
                j["status"] = st
                counts = data.get("request_counts") or {}
                # Stable per-shard totals and cumulative counts
                seen_total = int(counts.get("total") or 0)
                seen_done = int(counts.get("completed") or 0)
                seen_failed = int(counts.get("failed") or 0)
                if j["total"] is None and seen_total:
                    j["total"] = seen_total
                j["done"] = max(j["done"], seen_done)
                j["failed"] = max(j["failed"], seen_failed)
                total_req += int(j["total"] or seen_total)
                total_done += j["done"]
                total_failed += j["failed"]
                if st in done_states:
                    j["finalized"] = True
                    shards_done += 1
                if st in fail_states:
                    j["finalized"] = True
                    shards_done += 1

            now = time.time()
            if now - last_print >= 15.0:
                print(f"[poll] {fmt(now - start)} shards done {shards_done}/{len(all_jobs)} | requests done {total_done}/{total_req} | failed {total_failed}")
                last_print = now

            if shards_done == len(all_jobs):
                return True
            time.sleep(2.0)
        return False

    def download_completed(all_jobs):
        """Download and show a head for each completed shard."""
        for j in all_jobs:
            if not j.get("finalized"):
                continue
            data = get_status(j["batch_id"]) or {}
            if str(data.get("status", "")).lower() not in {"completed", "succeeded", "success"}:
                continue
            ofid = data.get("output_file_id") or (data.get("output_file_ids", [None]) or [None])[0]
            if not ofid:
                print(f"shard {j['shard']} completed without output_file_id")
                continue
            try:
                text = download_file(ofid)
            except Exception as e:
                print(f"shard {j['shard']} download error: {e}")
                continue
            head = "\n".join(text.splitlines()[:2])
            print(f"--- shard {j['shard']} head ---\n{head}\n------------------------------")

    all_done = poll_many(jobs)
    print(f"All shards done: {all_done}")
    download_completed(jobs)

