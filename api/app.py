import os, json, traceback
import requests
from flask import Flask, jsonify, render_template, request
import pandas as pd
import praw

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"),
            static_folder=os.path.join(BASE_DIR, "static"), static_url_path="/static")

# ---- Reddit
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
USER_AGENT = os.getenv("USER_AGENT") or "reddit_rtsent (vercel)"

try:
    reddit = praw.Reddit(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, user_agent=USER_AGENT)
except Exception as e:
    reddit = None

# ---- HF Inference API
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "alissawu/realtime-reddit-distilbert")
HF_TOKEN = os.getenv("HF_TOKEN")
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

def hf_classify(text: str):
    try:
        r = requests.post(HF_API_URL, headers=HF_HEADERS, json={"inputs": text}, timeout=20)
        if r.status_code == 401:
            raise RuntimeError("HF 401 Unauthorized (private model or bad token)")
        if r.status_code == 503:
            # model cold start on HF; try again once after a brief pause
            return [{"label": "LABEL_1", "score": 0.5}]  # fallback neutral
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return data
        if isinstance(data, list) and data and isinstance(data[0], list):
            return data[0]
        raise RuntimeError(f"Unexpected HF response: {data}")
    except Exception as e:
        print("HF ERROR:", e, "\n", traceback.format_exc())
        # raise to let route return 500, or return neutral:
        raise

def predict_sentiment(text: str) -> float:
    out = hf_classify(text)[0]
    label = out["label"]; score = float(out["score"])
    p_pos = score if label == "LABEL_1" else 1.0 - score
    return 2.0 * p_pos - 1.0

def get_data(subreddit: str, post_limit: int = 50):
    if reddit is None:
        raise RuntimeError("Reddit client not initialized (check CLIENT_ID/SECRET/USER_AGENT).")
    try:
        posts = reddit.subreddit(subreddit).new(limit=post_limit)
        data = []
        for post in posts:
            score = predict_sentiment(post.title)
            data.append({"title": post.title, "sentiment": score})
        df = pd.DataFrame(data)
        avg = float(df["sentiment"].mean()) if not df.empty else 0.0
        med = float(df["sentiment"].median()) if not df.empty else 0.0
        headlines = df.tail(50).to_dict(orient="records")
        return avg, med, headlines
    except Exception as e:
        print("REDDIT/HF PIPELINE ERROR:", e, "\n", traceback.format_exc())
        raise

# health test
@app.route("/health")
def health():
    return jsonify({
        "env": {
            "CLIENT_ID": bool(CLIENT_ID),
            "CLIENT_SECRET": bool(CLIENT_SECRET),
            "USER_AGENT": bool(USER_AGENT),
            "HF_MODEL_ID": HF_MODEL_ID,
            "HF_TOKEN_present": bool(HF_TOKEN),
        }
    })

# routes 
@app.route("/")
def index(): return render_template("index.html")

@app.route("/modelnotes")
def modelnotes_page(): return render_template("modelnotes.html")

@app.route("/<subreddit>")
def subreddit_page(subreddit): return render_template("subreddit.html", subreddit=subreddit)

@app.route("/fetch_sentiment/<subreddit>")
def fetch_sentiment(subreddit):
    avg, med, _ = get_data(subreddit)
    return jsonify(average=avg, median=med)

@app.route("/fetch_headlines/<subreddit>")
def fetch_headlines(subreddit):
    _, _, headlines = get_data(subreddit, 50)
    return jsonify(headlines=headlines)

app = app
