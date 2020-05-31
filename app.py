import GetOldTweets3 as got
from flask import Flask 
import pandas as pd
import pickle
import json

app = Flask(__name__)

def getTweets(username):
    count = 200
    # Creation of query object
    tweetCriteria = got.manager.TweetCriteria().setUsername(username).setMaxTweets(count)
    
    # Creation of list that contains all tweets
    tweets = got.manager.TweetManager.getTweets(tweetCriteria)
    # Creating list of chosen tweet data
    user_tweets = [tweet.text for tweet in tweets]
    return(user_tweets)

def evalAccount(username):
    with open('./models/classifier.pkl', 'rb') as f:
        clf = pickle.load(f)
    with open('./models/victorizer.pkl', 'rb') as f:
        victorizer = pickle.load(f)

    data_from_twitter = getTweets(username)

    if not data_from_twitter:
        mssg = "I am not able to access @{} account".format(username)
        rezpons = {'status': 500, 'msg': mssg}
        return json.dumps(rezpons, ensure_ascii = False)


    new_test = victorizer.transform(data_from_twitter)
    
    
    result = clf.predict(new_test)
        
    neg_count = 0
    pos_count = 0

    if len(pd.value_counts(result)) <= 1:
        if result[0] == 4':
            pos_count = len(result)
        else:
            neg_count = len(result)
    else:
        neg_count = pd.value_counts(result)[0]
        pos_count = pd.value_counts(result)[4]
        
        neg_perc = pd.value_counts(result)[0]/200 * 100
        pos_perc = pd.value_counts(result)[4]/200 * 100

    if neg_count > pos_count:
        msg = "I found out that the percentage of the negative tweets is {}% compared to the positive ones.".format(neg_perc)
        rezpons = {'status': 200, 'msg': msg}
        return json.dumps(rezpons, ensure_ascii = False)
    else:
        
        msg =  "I found out that the percentage of the positive tweets is {}% compared to the negative ones.".format(pos_perc)
        rezpons = {'status': 200, 'msg': msg}
        return json.dumps(rezpons, ensure_ascii = False)


@app.route("/")
def index():
    return 'You home'

@app.route("/predict/<username>")
def home(username):
    return evalAccount(username)

if __name__ == '__main__':
    app.run()
