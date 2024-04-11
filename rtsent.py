#export FLASK_APP=rtsent.py
from dotenv import load_dotenv
load_dotenv()  

import os
client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")
user_agent = os.getenv("USER_AGENT")


from flask import Flask, jsonify, render_template
app = Flask(__name__) #__name__ is built-in Python var, indicates current module

from textblob import TextBlob
import pandas as pd

# Reddit API Set-up
import praw
reddit = praw.Reddit(client_id=client_id, 
                     client_secret=client_secret, 
                     user_agent=user_agent)


def get_sentiments(sub, post_limit=50):
    # fetch x amt posts from subreddit
    posts = reddit.subreddit(sub).new(limit=post_limit)

    # simple sentiment analytic 
    sentiments = []
    for post in posts:
        analysis = TextBlob(post.title)
        sentiment = analysis.sentiment.polarity
        sentiments.append(sentiment)

    # create dataframe 
    df = pd.DataFrame(sentiments, columns=['Sentiment'])

    # calculate average and median sentiment
    average_sentiment = df['Sentiment'].mean()
    median_sentiment = df['Sentiment'].median()

    return average_sentiment, median_sentiment

@app.route('/fetch_sentiment')
def fetch_sentiment():
    average_sentiment, median_sentiment = get_sentiments('politics')
    return jsonify(average=average_sentiment, median=median_sentiment)
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    # run the app
    app.run(debug=True)