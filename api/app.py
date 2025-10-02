import os
import time
import traceback
from functools import lru_cache

import requests
from flask import Flask, jsonify, render_template
import pandas as pd
import praw

# ---------- Flask: point to templates/static outside /api ----------
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

# ---------- Hugging Face Inference API (PUBLIC model: no token) ----------
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "alissawu/realtime-reddit-distilbert")
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"

def _hf_call(payload, timeout=25):
    """
    Call HF once without auth header (public model). Retry once on 503 cold start.
    Raise with details on non-200.
    """
    r = requests.post(HF_API_URL, json=payload, timeout=timeout)
    if r.status_code == 503:  # model cold start
        time.sleep(1.0)
        r = requests.post(HF_API_URL, json=payload, timeout=timeout)
    if r.status_code != 200:
        # bubble up useful error text into Vercel logs
        raise RuntimeError(f"HF API error {r.status_code}: {r.text}")
    return r.json()

# ---- single / batch inference helpers ----
@lru_cache(maxsize=2048)
def _classify_one(text: str):
    data = _hf_call({"inputs": text})
    # normalize to list[dict]
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data
    if isinstance(data, list) and data and isinstance(data[0], list):
        return data[0]
    raise RuntimeError(f"Unexpected HF response: {data}")

def predict_sentiment(text: str) -> float:
    out = _classify_one(text)[0]
    label = out["label"]; score = float(out["score"])
    p_pos = score if label == "LABEL_1" else 1.0 - score
    return 2.0 * p_pos - 1.0

def predict_sentiments(texts):
    """Batch classify many texts in a single HF call."""
    if not texts:
        return []
    data = _hf_call({"inputs": texts})
    # expected: list-of-list-of-dicts
    if not (isinstance(data, list) and data and isinstance(data[0], list)):
        data = [data]  # handle single-item fallback
    res = []
    for item in data:
        out = item[0]
        label = out["label"]; score = float(out["score"])
        p_pos = score if label == "LABEL_1" else 1.0 - score
        res.append(2.0 * p_pos - 1.0)
    return res

def get_data(subreddit: str, post_limit: int = 20):
    if reddit is None:
        raise RuntimeError("Reddit client not initialized (check CLIENT_ID/SECRET/USER_AGENT).")
    posts = list(reddit.subreddit(subreddit).new(limit=post_limit))
    titles = [p.title for p in posts]
    sentiments = predict_sentiments(titles)

    data = [{"title": t, "sentiment": s} for t, s in zip(titles, sentiments)]
    df = pd.DataFrame(data)
    avg = float(df["sentiment"].mean()) if not df.empty else 0.0
    med = float(df["sentiment"].median()) if not df.empty else 0.0
    headlines = data  # already <= 20
    return avg, med, headlines

# ---------- Health & routes ----------
@app.route("/health")
def health():
    return jsonify({
        "env": {
            "CLIENT_ID": bool(CLIENT_ID),
            "CLIENT_SECRET": bool(CLIENT_SECRET),
            "USER_AGENT": bool(USER_AGENT),
            "HF_MODEL_ID": HF_MODEL_ID,
            "HF_TOKEN_present": False,   # always false in this token-free build
        }
    })

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/modelnotes")
def modelnotes_page():
    return render_template("modelnotes.html")

# --- API routes BEFORE catch-all ---
@app.route("/fetch_sentiment/<subreddit>")
def fetch_sentiment(subreddit):
    try:
        avg, med, _ = get_data(subreddit)
        return jsonify(average=avg, median=med)
    except Exception as e:
        print("fetch_sentiment ERROR:", e, "\n", traceback.format_exc())
        return jsonify(error=str(e)), 500

@app.route("/fetch_headlines/<subreddit>")
def fetch_headlines(subreddit):
    try:
        _, _, headlines = get_data(subreddit, 20)
        return jsonify(headlines=headlines)
    except Exception as e:
        print("fetch_headlines ERROR:", e, "\n", traceback.format_exc())
        return jsonify(error=str(e)), 500

# --- Catch-all page route LAST ---
@app.route("/<subreddit>")
def subreddit_page(subreddit):
    return render_template("subreddit.html", subreddit=subreddit)

# Vercel handler
app = app

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
