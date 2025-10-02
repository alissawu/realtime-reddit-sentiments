import os
import requests
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, jsonify, render_template, request
import pandas as pd
import praw

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
    static_url_path="/static"
)
# ----- Reddit API -----
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
USER_AGENT = os.getenv("USER_AGENT")

reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT
)

# ----- Hugging Face Inference API -----
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "alissawu/realtime-reddit-distilbert")
HF_TOKEN = os.getenv("HF_TOKEN")  # create on hf.co/settings/tokens (Read is enough)

HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

def hf_classify(text: str):
    # returns [{'label': 'LABEL_1', 'score': 0.97}] or similar
    resp = requests.post(HF_API_URL, headers=HF_HEADERS, json={"inputs": text}, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    # Some models return a list-of-list; normalize:
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data  # already normalized
    if isinstance(data, list) and data and isinstance(data[0], list):
        return data[0]
    raise RuntimeError(f"Unexpected HF response: {data}")

def predict_sentiment(text: str) -> float:
    """
    Convert HF label/score into polarity [-1, 1].
    LABEL_1 = positive, LABEL_0 = negative
    """
    out = hf_classify(text)[0]
    label = out["label"]
    score = float(out["score"])
    p_pos = score if label == "LABEL_1" else 1.0 - score
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
