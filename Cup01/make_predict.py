#!/usr/bin/env python
# coding: utf-8

debug = False
MIN = 1 # n-gram
MAX = 1 # n-gram
MAX_DF = 0.6
HASH_POWER = 10 # hash to 2**HASH_POWER features

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd
import re
import nltk
import os
import xgboost as xgb
import datetime

from xgboost import XGBClassifier
from tqdm import tqdm
from bs4 import BeautifulSoup
from summa import keywords
from summa.summarizer import summarize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from time import strptime
from cup01 import *

def preprocessor(text):
    text = BeautifulSoup(text, 'html.parser').get_text()
    r = '(?::|;|=|X)(?:-)?(?:\)|\(|D|P)'
    emoticons = re.findall(r, text)
    text = re.sub(r, '', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' ' + ' '.join(emoticons).replace('-','')
    return text

nltk.download('stopwords')
stop = stopwords.words('english')
stop = stop + extra_stopwords()

def tokenizer_stem_nostop(text):
    porter = PorterStemmer()
    return [porter.stem(w) for w in re.split('\s+', text.strip()) if w not in stop and re.match('[a-zA-Z]+', w)]

df = pd.read_csv('./input/test.csv')
if debug:
    df = df.iloc[:100] # debug

df_train_contents = df['Page content'].values.tolist()
days, pub_days, channels, img_counts, topics, authors, titles, social_media_counts, contents, num_hrefs, num_self_hrefs = get_all_datas(df_train_contents)
rate_positive_words, rate_negative_words, avg_positive_polarity, min_positive_polarity, max_positive_polarity, avg_negative_polarity, min_negative_polarity, max_negative_polarity = get_word_sentiment_features(contents)
n_tokens_titles, n_tokens_contents, n_unique_tokens, n_non_stop_words, n_non_stop_unique_tokens = get_some_n_features(titles, contents)
global_sentiment_polarity, global_subjectivity, title_subjectivity_list, title_sentiment_polarity_list, abs_title_subjectivity, abs_title_sentiment_polarity = get_sentiment_features(titles, contents)

df = pd.DataFrame({'Page content':df_train_contents,
                   'Id':df.Id[:],
                   'topic':topics,
                   'channel':channels,
                   'weekday':days,
                   'pub_date' : pub_days,
                   'author':authors,
                   'img count':img_counts,
                   'title':titles,
                   'content':contents,
                   'media count': social_media_counts,
                   'n_tokens_title' : n_tokens_titles,
                   'n_tokens_content': n_tokens_contents,
                   'n_unique_tokens' : n_unique_tokens,
                   'n_non_stop_words': n_non_stop_words,
                   'n_non_stop_unique_tokens': n_non_stop_unique_tokens,
                   'num_hrefs' : num_hrefs,
                   'num_self_hrefs' : num_self_hrefs,
                   'global_sentiment_polarity' : global_sentiment_polarity,
                   'global_subjectivity' : global_subjectivity,
                   'title_subjectivity' : title_subjectivity_list,
                   'title_sentiment_polarity' : title_sentiment_polarity_list,
                   'abs_title_subjectivity' : abs_title_subjectivity,
                   'abs_title_sentiment_polarity' : abs_title_sentiment_polarity,
                   'rate_positive_words' : rate_positive_words,
                   'rate_negative_words' : rate_negative_words,
                   'avg_positive_polarity' : avg_positive_polarity,
                   'min_positive_polarity' : min_positive_polarity,
                   'max_positive_polarity' : max_positive_polarity,
                   'avg_negative_polarity' : avg_negative_polarity,
                   'min_negative_polarity' : min_negative_polarity,
                   'max_negative_polarity' : max_negative_polarity
                  })

#df['day_of_month'] = df['pub_date'].apply(lambda x: int(x.split()[1]))
#df['month'] = df['pub_date'].apply(lambda x: strptime(x.split()[2], '%b').tm_mon)
#df['hour'] = df['pub_date'].apply(lambda x: strptime(x.split()[4], '%X')[3])

df['day_of_month'] = df['pub_date'].apply(lambda x: int(x.split()[1]) if x != ' noneday' else 0)
df['month'] = df['pub_date'].apply(lambda x: strptime(x.split()[2], '%b').tm_mon if x != ' noneday' else 0)
df['hour'] = df['pub_date'].apply(lambda x: strptime(x.split()[4], '%X')[3] if x  != ' noneday' else 0)

df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x ==' Sat' or x == ' Sun' else 0)
del df_train_contents, df['pub_date']
tqdm.pandas()
df['Page content'] = df['Page content'].progress_apply(preprocessor) # 此步驟約要花五分鐘
tqdm.pandas()
df['keywords'] = df['Page content'].progress_apply(tokenizer_stem_nostop)
df['keywords'] = df['keywords'].progress_apply(lambda x: ' '.join(x))
tqdm.pandas()
df['keywords'] = df['keywords'].progress_apply(lambda x: keywords.keywords(x).replace('\n', ' '))

#######
df_train = pd.read_csv('./input/input_feature.csv')
col = 'author'
df2 = df_train.groupby(f'{col}').mean().reset_index().sort_values(by='Popularity', ascending=False)[[f'{col}', 'Popularity']]
df2.columns=[f'{col}', 'avg_popularity']

author_avg_score = {}
for i, row in df2.iterrows():
    author_name = row['author']
    score = row['avg_popularity']
    author_avg_score[author_name] = score

df['author_popularity'] = df['author'].apply(lambda x: author_avg_score[x] if x in author_avg_score else 0.0)
del df2, df_train
######

doc = df['keywords']
hashvec = HashingVectorizer(n_features=2**HASH_POWER)
doc_hash = hashvec.transform(doc).toarray()
doc = df['title']
hashvec = HashingVectorizer(n_features=2**HASH_POWER)
doc_hash_title = hashvec.transform(doc).toarray()
# channel
channel_ohe = OneHotEncoder(handle_unknown='ignore')
channel_str = channel_ohe.fit_transform(df['channel'].values.reshape(-1,1)).toarray()
print(channel_str.shape)
# weekday
weekday_ohe = OneHotEncoder(handle_unknown='ignore')
weekday_str = weekday_ohe.fit_transform(df['weekday'].values.reshape(-1,1)).toarray()
print(weekday_str.shape)
# ohe author
author_ohe = OneHotEncoder(handle_unknown='ignore')
author_str = author_ohe.fit_transform(df['author'].values.reshape(-1,1)).toarray()
print(author_str.shape)

img_count = df['img count'].values.reshape(-1,1)
media_count = df['media count'].values.reshape(-1,1)

def flatten(t):
    return [item for sublist in t for item in sublist]

df_X_train = []
for i in tqdm(range(len(channel_str))):
    temp = []
    temp.append(img_count[i])
    temp.append(media_count[i])
    temp.append(channel_str[i])
    temp.append(weekday_str[i])
    temp.append(author_str[i])
    temp.append(doc_hash[i])
    temp.append(doc_hash_title[i])
    temp = flatten(temp)
    temp.append(df['day_of_month'][i])
    temp.append(df['month'][i])
    temp.append(df['hour'][i])
    temp.append(df['n_tokens_title'][i])
    temp.append(df['n_tokens_content'][i])
    temp.append(df['n_unique_tokens'][i])
    temp.append(df['n_non_stop_words'][i])
    temp.append(df['n_non_stop_unique_tokens'][i])
    temp.append(df['num_hrefs'][i])
    temp.append(df['num_self_hrefs'][i])
    temp.append(df['global_sentiment_polarity'][i])
    temp.append(df['global_subjectivity'][i])
    temp.append(df['title_subjectivity'][i])
    temp.append(df['title_sentiment_polarity'][i])
    temp.append(df['abs_title_subjectivity'][i])
    temp.append(df['abs_title_sentiment_polarity'][i])
    temp.append(df['rate_positive_words'][i])
    temp.append(df['rate_negative_words'][i])
    temp.append(df['avg_positive_polarity'][i])
    temp.append(df['min_positive_polarity'][i])
    temp.append(df['max_positive_polarity'][i])
    temp.append(df['avg_negative_polarity'][i])
    temp.append(df['min_negative_polarity'][i])
    temp.append(df['max_negative_polarity'][i])
    temp.append(df['is_weekend'][i])
    temp.append(df['author_popularity'][i])
    df_X_train.append(temp)
    del temp

del df, doc_hash, img_count, media_count, channel_str, weekday_str, author_str

pd.DataFrame(df_X_train).to_csv("./input/X_test.csv", index=False, header=False)
