#export FLASK_APP=rtsent.py
# .env variables
from dotenv import load_dotenv
load_dotenv()

import os
client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")
user_agent = os.getenv("USER_AGENT")

from flask import Flask, jsonify, render_template, request
app = Flask(__name__)  # __name__ is built-in Python var, indicates current module

from textblob import TextBlob
import pandas as pd

# Reddit API Set-up
import praw
reddit = praw.Reddit(client_id=client_id,
                     client_secret=client_secret,
                     user_agent=user_agent)

def get_data(sub, post_limit=50):
    """Fetch a specified amount of posts from a subreddit and perform sentiment analysis."""
    posts = reddit.subreddit(sub).new(limit=post_limit)
    data = []  # Sentiments and headlines
    for post in posts:
        analysis = TextBlob(post.title)
        data.append({
            'title': post.title,
            'sentiment': analysis.sentiment.polarity
        })
    df = pd.DataFrame(data)
    average_sentiment = df['sentiment'].mean()
    median_sentiment = df['sentiment'].median()
    headlines = df.tail(50).to_dict(orient='records')
    return average_sentiment, median_sentiment, headlines

@app.route('/')
def index():
    """Serve the index page that can link to different subreddit analyses."""
    return render_template('index.html')

@app.route('/<subreddit>')
def subreddit_page(subreddit):
    """Render a page for each subreddit dynamically."""
    return render_template('subreddit.html', subreddit=subreddit)

@app.route('/fetch_sentiment/<subreddit>')
def fetch_sentiment(subreddit):
    """Fetch and return the average and median sentiment for a specific subreddit."""
    average_sentiment, median_sentiment, _ = get_data(subreddit)
    return jsonify(average=average_sentiment, median=median_sentiment)

@app.route('/fetch_headlines/<subreddit>')
def fetch_headlines(subreddit):
    """Fetch and return the latest 50 headlines from a specific subreddit."""
    _, _, headlines = get_data(subreddit, 50)
    return jsonify(headlines=headlines)

if __name__ == '__main__':
    # run the app
    app.run(debug=True)
