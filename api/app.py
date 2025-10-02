#!/usr/bin/env python3
import os, traceback
from functools import lru_cache

# Cache Hugging Face downloads in /tmp for faster cold starts
os.environ.setdefault("HF_HOME", "/tmp/hf")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf")

from flask import Flask, jsonify, render_template
import pandas as pd
import praw
from transformers import pipeline

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

# ---------- Hugging Face pipeline ----------
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "alissawu/realtime-reddit-distilbert")

@lru_cache(maxsize=1)
def get_pipeline():
    """Lazy-load pipeline once per cold start."""
    return pipeline(
        "text-classification",
        model=HF_MODEL_ID,
        tokenizer=HF_MODEL_ID,
        truncation=True,
    )

def predict_sentiments(texts):
    if not texts:
        return []
    clf = get_pipeline()
    outputs = clf(texts, truncation=True)
    res = []
    for out in outputs:
        label = out["label"]
        score = float(out["score"])
        if label in ("LABEL_1", "POSITIVE"):
            p_pos = score
        else:
            p_pos = 1.0 - score
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
    return avg, med, data

# ---------- Routes ----------
@app.route("/health")
def health():
    return jsonify({
        "env": {
            "CLIENT_ID": bool(CLIENT_ID),
            "CLIENT_SECRET": bool(CLIENT_SECRET),
            "USER_AGENT": bool(USER_AGENT),
            "HF_MODEL_ID": HF_MODEL_ID,
        }
    })

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/modelnotes")
def modelnotes_page():
    return render_template("modelnotes.html")

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

@app.route("/<subreddit>")
def subreddit_page(subreddit):
    return render_template("subreddit.html", subreddit=subreddit)

# Vercel handler
app = app

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
