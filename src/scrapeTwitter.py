import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import snscrape.modules.twitter as sntwitter

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from itertools import product

nltk.download('wordnet')
nltk.download('stopwords')

my_stopwords = nltk.corpus.stopwords.words('english')
word_rooter = nltk.stem.snowball.PorterStemmer(ignore_stopwords=False).stem
my_punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@'

def remove_links(tweet):
    '''Takes a string and removes web links from it'''
    tweet = re.sub(r'http\S+', '', tweet) # remove http links
    tweet = re.sub(r'bit.ly/\S+', '', tweet) # rempve bitly links
    tweet = tweet.strip('[link]') # remove [links]
    return tweet


def remove_users(tweet):
    '''Takes a string and removes retweet and @user information'''
    tweet = re.sub('(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet) # remove retweet
    tweet = re.sub('(@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet) # remove tweeted at
    return tweet


# cleaning master function
def clean_tweet(tweet, bigrams=False):
    tweet = remove_users(tweet)
    tweet = remove_links(tweet)
    tweet = tweet.lower() # lower case
    tweet = re.sub('['+my_punctuation + ']+', ' ', tweet) # strip punctuation
    tweet = re.sub('\s+', ' ', tweet) #remove double spacing
    tweet = re.sub('([0-9]+)', '', tweet) # remove numbers
    tweet_token_list = [word for word in tweet.split(' ')
                            if word not in my_stopwords] # remove stopwords

    tweet_token_list = [word_rooter(word) if '#' not in word else word
                        for word in tweet_token_list] # apply word rooter
    if bigrams:
        tweet_token_list = tweet_token_list+[tweet_token_list[i]+'_'+tweet_token_list[i+1]
                                            for i in range(len(tweet_token_list)-1)]
    tweet = ' '.join(tweet_token_list)
    return tweet


def display_topics(model, feature_names, no_top_words):
    topic_dict = {}
    for topic_idx, topic in enumerate(model.components_):
        topic_dict["Topic %d words" % (topic_idx)]= ['{}'.format(feature_names[i])
                        for i in topic.argsort()[:-no_top_words - 1:-1]]
        topic_dict["Topic %d weights" % (topic_idx)]= ['{:.1f}'.format(topic[i])
                        for i in topic.argsort()[:-no_top_words - 1:-1]]
    return pd.DataFrame(topic_dict)


def scrapeTwitter(KEY_PHRASES, START_DATE, END_DATE, MAX_TWEETS_PER_PHRASE,
                  MIN_FOLLOWERS, MAX_TWEETS_PER_USER, NUMBER_OF_TOPICS,
                  NO_TOP_WORDS):

    key_words = [item for words in KEY_PHRASES for item in words.split(' ')]
    key_word_sets = [item for word in key_words for item in wordnet.synsets(word)]

    # Creating list to append tweet data to
    tweet_list = []

    # Using TwitterSearchScraper to scrape data and append tweets to list
    num_phrases = len(KEY_PHRASES)
    for (j, key_phrase) in enumerate(KEY_PHRASES):
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper(
                            key_phrase + ' since:' + START_DATE + ' until:' + END_DATE).get_items()):
            if i > MAX_TWEETS_PER_PHRASE:
                print("Scraped " + str(j) + " of " + str(num_phrases) + " key phrases.")
                break
            tweet_list.append([tweet.user, tweet.user.username, tweet.user.followersCount, tweet.user.verified])
    # Creating a dataframe from the tweets list above
    tweet_df = pd.DataFrame(tweet_list, columns=['User Object', 'Username', 'FollowerCount', 'Verified'])
    tweet_df = tweet_df.loc[tweet_df['FollowerCount'] >= MIN_FOLLOWERS]
    tweet_df = tweet_df.drop_duplicates(subset=['Username'])

    candidate_usernames = tweet_df['Username'].values
    num_candidates = candidate_usernames.shape[0]
    print("There are " + str(num_candidates) + " candidate users.")

    user_max_scores = []
    user_avg_scores = []
    user_LDA_model = []

    for (index, twitter_user) in enumerate(candidate_usernames):

        if index % 20 == 1:
            print(str(index) + " users have been processed.")
        # Creating list to append tweet data to
        user_tweets = []

        # Using TwitterSearchScraper to scrape data and append tweets to list
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper('from:' + twitter_user).get_items()):
            if i > MAX_TWEETS_PER_USER:
                break
            user_tweets.append([tweet.date, tweet.id, tweet.content, tweet.user.username])

        # Creating a dataframe from the tweets list above
        user_tweets_df = pd.DataFrame(user_tweets, columns=['Datetime', 'Tweet Id', 'Text', 'Username'])

        documents = list(user_tweets_df['Text'].values)
        cleaned_document = [clean_tweet(tweet) for tweet in documents]
        cleaned_document = [cleaned_tweet for cleaned_tweet in cleaned_document if cleaned_tweet != ' ']

        # the vectorizer object will be used to transform text to vector form
        vectorizer = CountVectorizer(token_pattern='\w+|\$[\d\.]+|\S+')

        # apply transformation
        tf = vectorizer.fit_transform(cleaned_document).toarray()

        # tf_feature_names tells us what word each column in the matric represents
        tf_feature_names = vectorizer.get_feature_names()

        model = LatentDirichletAllocation(n_components=NUMBER_OF_TOPICS, random_state=0)
        model.fit(tf)

        topic_df = display_topics(model, tf_feature_names, NO_TOP_WORDS)

        user_words = []
        for topic_number in range(NUMBER_OF_TOPICS):
            topic_words = topic_df["Topic " + str(topic_number) + " words"].values
            user_words += [word for word in topic_words]

        user_word_sets = [item for word in user_words for item in wordnet.synsets(word)]

        max_val = 0
        avg_val = 0
        count = 0
        #max_sets = ()
        for set_1, set_2 in product(key_word_sets, user_word_sets):
            similarity = set_1.wup_similarity(set_2)
            avg_val += similarity
            count += 1
            if similarity > max_val:
                max_val = similarity
                #max_sets = (set_1, set_2)
        if count > 0:
            avg_val /= count

        user_max_scores.append(max_val)
        user_avg_scores.append(avg_val)
        user_LDA_model.append(model)

    tweet_df['Max Scores'] = user_max_scores
    tweet_df['Avg Scores'] = user_avg_scores
    tweet_df['LDA Model'] = user_LDA_model

    return tweet_df
