#!/usr/bin/env python3
import os, json, time, traceback, requests, redis
from functools import lru_cache
from flask import Flask, jsonify, render_template
import pandas as pd
import praw

# ---------- Config ----------
FRESH_TTL = int(os.getenv("FRESH_TTL_SECONDS", "90"))     # snapshot is "fresh" for 90s
STALE_TTL = int(os.getenv("STALE_TTL_SECONDS", "3600"))   # keep & serve stale up to 1h
POST_LIMIT = int(os.getenv("POST_LIMIT", "20"))

HF_ENDPOINT_URL = os.getenv("HF_ENDPOINT_URL")
HF_TOKEN        = os.getenv("HF_TOKEN")

REDIS_URL = os.getenv("REDIS_URL")
rds = redis.Redis.from_url(REDIS_URL) if REDIS_URL else None

def cache_get(key):
    if not rds: return None
    val = rds.get(key)
    return json.loads(val) if val else None

def cache_set(key, value, ttl):
    if not rds: return
    rds.set(key, json.dumps(value), ex=ttl)

# ---------- Flask ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
    static_url_path="/static",
)

# ---------- Reddit API ----------
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
USER_AGENT = os.getenv("USER_AGENT") or "reddit_rtsent (vercel)"

try:
    reddit = praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        user_agent=USER_AGENT,
    )
    reddit.read_only = True
except Exception as e:
    print("Reddit init failed:", e)
    reddit = None

# ---------- Hugging Face Inference Endpoint (batched) ----------
def hf_infer_batch(texts, max_length=128, timeout=20):
    if not texts:
        return []
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {"inputs": texts, "parameters": {"truncation": True, "max_length": max_length}}

    # simple retry on 429/5xx
    delay = 0.25
    for _ in range(3):
        resp = requests.post(HF_ENDPOINT_URL, headers=headers, data=json.dumps(payload), timeout=timeout)
        if resp.status_code in (429, 500, 502, 503, 504):
            time.sleep(delay); delay *= 2; continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError(f"HF inference failed: {resp.status_code} {resp.text[:200]}")

def to_sym(label, score):
    # map to [-1,1] like your old logic
    p_pos = score if label in ("LABEL_1","POSITIVE") else 1.0 - score
    return 2.0*p_pos - 1.0

def predict_sentiments(texts):
    outs = hf_infer_batch(texts, max_length=128)
    return [to_sym(o["label"], float(o["score"])) for o in outs]

# ---------- Cached data fetch ----------
def get_data(subreddit: str, post_limit: int = POST_LIMIT):
    """
    Returns (avg, med, data[{id,title,sentiment}]) using:
      - page snapshot cache: sub:<sr>:v1  (fresh 90s, stale 1h)
      - per-post score cache: sub:<sr>:scores (1h)
    """
    if reddit is None:
        raise RuntimeError("Reddit client not initialized (check CLIENT_ID/SECRET/USER_AGENT).")

    page_key = f"sub:{subreddit}:v1"
    snap = cache_get(page_key)
    now = time.time()

    # Serve fresh snapshot immediately
    if snap and (now - snap.get("ts", 0) <= FRESH_TTL):
        rows = snap["headlines"]
        return float(snap["average"]), float(snap["median"]), rows

    # Fetch latest posts
    posts = list(reddit.subreddit(subreddit).new(limit=post_limit))
    ids    = [p.id for p in posts]
    titles = [p.title for p in posts]

    # Per-post score cache
    score_key = f"sub:{subreddit}:scores"
    score_map = cache_get(score_key) or {}  # {post_id: sentiment}

    missing_idx = [i for i, pid in enumerate(ids) if pid not in score_map]
    if missing_idx:
        new_titles = [titles[i] for i in missing_idx]
        new_scores = predict_sentiments(new_titles)      # batched call to HF Endpoint
        for i, s in zip(missing_idx, new_scores):
            score_map[ids[i]] = s
        cache_set(score_key, score_map, ttl=3600)

    sentiments = [score_map[pid] for pid in ids if pid in score_map]
    if len(sentiments) != len(ids):
        # backfill if something went missing unexpectedly
        all_scores = predict_sentiments(titles)
        score_map = {pid: s for pid, s in zip(ids, all_scores)}
        cache_set(score_key, score_map, ttl=3600)
        sentiments = all_scores

    rows = [{"id": pid, "title": t, "sentiment": s} for pid, t, s in zip(ids, titles, sentiments)]
    df = pd.DataFrame(rows)
    avg = float(df["sentiment"].mean()) if not df.empty else 0.0
    med = float(df["sentiment"].median()) if not df.empty else 0.0

    # write new snapshot; keep up to STALE_TTL so stale can still be served next time
    snapshot = {"average": avg, "median": med, "headlines": rows, "ts": now}
    cache_set(page_key, snapshot, ttl=STALE_TTL)

    return avg, med, rows

# ---------- Routes ----------
@app.route("/health")
def health():
    return jsonify({
        "env": {
            "CLIENT_ID": bool(CLIENT_ID),
            "CLIENT_SECRET": bool(CLIENT_SECRET),
            "USER_AGENT": bool(USER_AGENT),
            "HF_ENDPOINT_URL": bool(HF_ENDPOINT_URL),
            "REDIS": bool(rds),
        }
    })

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/modelnotes")
def modelnotes_page():
    return render_template("modelnotes.html")
# script.js calls this to fetch data
@app.route("/api/sub/<subreddit>")
def api_sub(subreddit):
    try:
        avg, med, headlines = get_data(subreddit, POST_LIMIT)
        return jsonify(average=avg, median=med, headlines=headlines, ts=time.time())
    except Exception as e:
        print("api_sub ERROR:", e, "\n", traceback.format_exc())
        return jsonify(error=str(e)), 500
# this is used for the template html site
@app.route("/<subreddit>")
def subreddit_page(subreddit):
    return render_template("subreddit.html", subreddit=subreddit)

# tiny warm route to preheat the endpoint without blocking users
@app.route("/warm")
def warm():
    try:
        requests.post(
            HF_ENDPOINT_URL,
            headers={"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"},
            data=json.dumps({"inputs": ["ok"], "parameters": {"truncation": True, "max_length": 8}}),
            timeout=8
        )
    except Exception:
        pass
    return jsonify(ok=True)

# Vercel handler
app = app

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
