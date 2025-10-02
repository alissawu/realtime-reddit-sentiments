#!/usr/bin/env python3
"""
Flask app serving a fine-tuned DistilBERT sentiment model via Hugging Face.
- Keeps original routes and JSON shape.
- Maps POSITIVE probability p ∈ [0,1] to polarity ∈ [-1,1] as (2p-1).
"""

import os
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, jsonify, render_template, request
import pandas as pd
import praw

# ==== Flask ====
app = Flask(__name__)

# ==== Reddit API (unchanged) ====
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
USER_AGENT = os.getenv("USER_AGENT")

reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT
)

# ==== Hugging Face pipeline ====
from transformers import pipeline

# Directory created by train_final.py (override with env if you want)
HF_MODEL_DIR = os.environ.get("HF_MODEL_DIR", "model_distilbert")

# Lazy init to avoid import cost during module import on some hosts
def _load_pipeline():
    clf = pipeline(
        task="text-classification",
        model=HF_MODEL_DIR,
        tokenizer=HF_MODEL_DIR,
        return_all_scores=False,
        truncation=True
    )
    return clf

clf = _load_pipeline()

def predict_sentiment(text: str) -> float:
    """
    Returns polarity in [-1, 1].
    Model uses LABEL_0 (negative) and LABEL_1 (positive).
    """
    out = clf(text)[0]  # {'label': 'LABEL_0' or 'LABEL_1', 'score': p}
    label = out["label"]
    score = float(out["score"])
    
    # LABEL_1 = positive, LABEL_0 = negative
    if label == "LABEL_1":
        p_pos = score
    else:  # LABEL_0
        p_pos = 1.0 - score
    
    return 2.0 * p_pos - 1.0

def get_data(subreddit: str, post_limit: int = 50):
    posts = reddit.subreddit(subreddit).new(limit=post_limit)
    data = []
    for post in posts:
        score = predict_sentiment(post.title)
        data.append({"title": post.title, "sentiment": score})
    df = pd.DataFrame(data)
    average_sentiment = float(df["sentiment"].mean()) if not df.empty else 0.0
    median_sentiment = float(df["sentiment"].median()) if not df.empty else 0.0
    headlines = df.tail(50).to_dict(orient="records")
    return average_sentiment, median_sentiment, headlines

# ==== Routes (unchanged) ====
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/modelnotes")
def modelnotes_page():
    return render_template("modelnotes.html")

@app.route("/<subreddit>")
def subreddit_page(subreddit):
    return render_template("subreddit.html", subreddit=subreddit)

@app.route("/fetch_sentiment/<subreddit>")
def fetch_sentiment(subreddit):
    avg, med, _ = get_data(subreddit)
    return jsonify(average=avg, median=med)

@app.route("/fetch_headlines/<subreddit>")
def fetch_headlines(subreddit):
    _, _, headlines = get_data(subreddit, 50)
    return jsonify(headlines=headlines)

# Vercel adapter
app = app

if __name__ == "__main__":
    app.run(debug=True)
