#export FLASK_APP=rtsent.py
#.env variables
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


def get_data(sub, post_limit=50):
    # fetch x amt posts from subreddit
    posts = reddit.subreddit(sub).new(limit=post_limit)

    # simple sentiment analytic 
    data = [] #sentiments and headlines
    for post in posts:
        analysis = TextBlob(post.title)
        data.append({
            'title': post.title,
            'sentiment': analysis.sentiment.polarity
        })

    # create dataframe 
    df = pd.DataFrame(data)
    df = df.sort_values(by='timestamp', ascending=False)

    # calculate average and median sentiment
    average_sentiment = df['sentiment'].mean()
    median_sentiment = df['sentiment'].median()
    # last 50 headlines to dict
    headlines = df.tail(50).to_dict(orient = 'records') 

    return average_sentiment, median_sentiment, headlines

# ABOUT PAGE
@app.route('/')
def index():
    return render_template('index.html')
# POLITICS / SENTIMENT PAGE
@app.route('/politics')
def politics():
    return render_template('politics.html')

@app.route('/fetch_sentiment')
def fetch_sentiment():
    average_sentiment, median_sentiment, _ = get_data('politics')
    return jsonify(average=average_sentiment, median=median_sentiment)

@app.route('/fetch_headlines')
def fetch_headlines():
    _, _, headlines = get_data('politics', 50)
    return jsonify(headlines=headlines)


if __name__ == '__main__':
    # run the app
    app.run(debug=True)