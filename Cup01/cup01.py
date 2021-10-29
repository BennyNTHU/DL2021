import numpy as np
import scipy as sp
import pandas as pd
import nltk
import re
import string
from bs4 import BeautifulSoup
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def fetch_datetime(soup):
    if soup.time.has_attr('datetime'):
        date = soup.time.attrs['datetime']
        day = ' '+ date[0:3]
    else:
        day  = ' noneday'
    return day

def fetch_pubday(soup):
    if soup.time.has_attr('datetime'):
        date = soup.time.attrs['datetime']
        pub_day = '' + date[:]
    else:
        pub_day =  ' noneday'
    return pub_day

def fetch_channel(soup):
    channel = soup.article['data-channel']
    return channel

def fetch_img_count(soup):
    c = 0
    find_all_images = soup.find_all('img')
    for i in find_all_images:
        c = c+1
    return c

def fetch_topics(soup):
    footer = soup.footer
    ta = footer.find_all('a')
    topic = []
    for t in ta:
        topic.append(t.get_text())
    topic_text = ' '.join(topic)
    return topic_text

def fetch_authors(soup):
    footer = soup.span
    if footer != None:
        ta = footer.findAll('a')
        authors = []
        for t in ta:
            authors.append(t.get_text())
        if len(authors) == 0:
            authors_text = 'NaN'
        else:
            authors_text = ''.join(authors)
    else:
        authors_text   = 'NaN'
    return authors_text

def fetch_titles(soup):
    footer = soup.h1
    if footer != None:
        titles = footer.get_text()
    else:
        titles = 'NaN'
    return titles

def fetch_social_media_count(soup):
    c = 0
    for frame in soup("iframe"):
        if frame.get('src').find("youtube") != None:
            c = c+1
        elif frame.get('src').find("instagram") != None:
            c = c+1
        elif frame.get('src').find("vine") != None:
            c = c+1
    return c

def fetch_href(soup):
    all_a_tags = soup.find_all('a', href=True)
    num_href = len(all_a_tags)
    num_self_href = 0
    for tag in all_a_tags:
        href = tag['href']
        if 'mashable' in href:
            num_self_href += 1
    return num_href, num_self_href

def get_some_n_features(titles, contents):
    n_tokens_titles = []
    n_tokens_contents = []
    n_unique_tokens = []
    n_non_stop_words = []
    n_non_stop_unique_tokens = []

    for title, content in zip(titles, contents):
        title_tokens = process(title)
        n_tokens_titles.append(len(title_tokens))

        content_tokens = process(content)
        n_tokens_contents.append(len(content_tokens))
        
        len_content_tokens = len(content_tokens)
        if len(content_tokens) == 0:
            print(content)
            len_content_tokens = 1
            
        set_content_token = set(content_tokens)
        unique_token_rate = len(set_content_token) / len_content_tokens
        n_unique_tokens.append(unique_token_rate)
        stop = stopwords.words('english')
        non_stop_words = [w for w in content_tokens if w not in stop]

        non_stop_word_rate = len(non_stop_words) / len_content_tokens
        n_non_stop_words.append(non_stop_word_rate)

        set_non_stop_words = set(non_stop_words)
        n_non_stop_unique_tokens_rate = len(set_non_stop_words) / len_content_tokens
        n_non_stop_unique_tokens.append(n_non_stop_unique_tokens_rate)
        
    return n_tokens_titles, n_tokens_contents, n_unique_tokens, n_non_stop_words, n_non_stop_unique_tokens

def get_sentiment_features(titles, contents):
    global_sentiment_polarity = []
    global_subjectivity = []
    title_subjectivity_list = []
    title_sentiment_polarity_list = []
    abs_title_subjectivity = []
    abs_title_sentiment_polarity = []
    
    for title, content in zip(titles, contents):
        title_blob = TextBlob(title)
        title_polarity = title_blob.sentiment.polarity
        title_subjectivity = title_blob.sentiment.subjectivity
       
        title_sentiment_polarity_list.append(title_polarity)
        title_subjectivity_list.append(title_subjectivity)
        abs_title_subjectivity.append(abs(title_subjectivity))
        abs_title_sentiment_polarity.append(abs(title_polarity))
              
        content_blob = TextBlob(content)
        content_polarity = content_blob.sentiment.polarity
        content_subjectivity = content_blob.sentiment.subjectivity
        global_sentiment_polarity.append(content_polarity)
        global_subjectivity.append(content_subjectivity)
        
    return global_sentiment_polarity, global_subjectivity, title_subjectivity_list, title_sentiment_polarity_list, \
abs_title_subjectivity, abs_title_sentiment_polarity

def get_word_sentiment_features(contents):
    rate_positive_words = []
    rate_negative_words = []
    avg_positive_polarity = []
    min_positive_polarity = []
    max_positive_polarity = []
    avg_negative_polarity = []
    min_negative_polarity = []
    max_negative_polarity = []
    
    for content in contents:
        content_tokens = process(content)
        
        pos_count = 0
        pos_score = 0.0
        min_pos_polarity = 1.0
        max_pos_polarity = 0.0
        
        neg_count = 0
        neg_score = 0.0
        min_neg_polarity = 0.0
        max_neg_polarity = -1.0
        
        for token in content_tokens:
            blob = TextBlob(token)
            sentiment_score = blob.sentiment.polarity

            if sentiment_score > 0.0: # positive
                pos_count += 1
                pos_score += sentiment_score
                if sentiment_score < min_pos_polarity:
                    min_pos_polarity = sentiment_score
                if sentiment_score > max_pos_polarity:
                    max_pos_polarity = sentiment_score
            elif sentiment_score < 0.0: # negative
                neg_count += 1
                neg_score += sentiment_score
                if sentiment_score < min_neg_polarity:
                    min_neg_polarity = sentiment_score
                if sentiment_score > max_neg_polarity:
                    max_neg_polarity = sentiment_score
                    
        if len(content_tokens) == 0:
            rate_positive_words.append(0.0)
            rate_negative_words.append(0.0)
            avg_positive_polarity.append(0.0)
            min_positive_polarity.append(0.0)
            max_positive_polarity.append(0.0)
            avg_negative_polarity.append(0.0)
            min_negative_polarity.append(0.0)
            max_negative_polarity.append(0.0)
            continue
            
        if pos_count == 0:
            max_pos_polarity = 0.0
            min_pos_polarity = 0.0
        else:
            pos_score /= pos_count
            
        if neg_count == 0:
            min_neg_polarity = 0.0
            max_neg_polarity = 0.0
        else:
            neg_score /= neg_count
  
        pos_rate = pos_count / len(content_tokens)
        neg_rate = neg_count / len(content_tokens)
        rate_positive_words.append(pos_rate)
        rate_negative_words.append(neg_rate)
        
        avg_positive_polarity.append(pos_score)
        min_positive_polarity.append(min_pos_polarity)
        max_positive_polarity.append(max_pos_polarity)
        
        avg_negative_polarity.append(neg_score)
        min_negative_polarity.append(min_neg_polarity)
        max_negative_polarity.append(max_neg_polarity)

    return rate_positive_words, rate_negative_words, avg_positive_polarity, min_positive_polarity, max_positive_polarity,\
avg_negative_polarity, min_negative_polarity, max_negative_polarity

def get_all_datas(texts):
    days = []
    channels = []
    img_counts = []
    topics = []
    authors = []
    titles = []
    social_media_counts = []
    contents = []
    num_hrefs = []
    num_self_hrefs = []
    pub_days = []
    
    for text in texts:
        soup = BeautifulSoup(text, "lxml")
        contents.append(soup.find('article').get_text())
        topics.append(fetch_topics(soup))
        channels.append(fetch_channel(soup))
        days.append(fetch_datetime(soup))
        authors.append(fetch_authors(soup))
        img_counts.append(fetch_img_count(soup))
        titles.append(fetch_titles(soup))
        social_media_counts.append(fetch_social_media_count(soup))

        # input()
        num_href, num_self_href = fetch_href(soup)
        num_hrefs.append(num_href)
        num_self_hrefs.append(num_self_href)
        pub_days.append(fetch_pubday(soup))
        
    return days, pub_days, channels, img_counts, topics, authors, titles, social_media_counts, contents, num_hrefs, num_self_hrefs

def extra_stopwords():
    extra_stopwords = ["ain't", "amn't", "aren't", "can't", "could've", "couldn't",
                    "daresn't", "didn't", "doesn't", "don't", "gonna", "gotta", 
                    "hadn't", "hasn't", "haven't", "he'd", "he'll", "he's", "how'd",
                    "how'll", "how's", "I'd", "I'll", "I'm", "I've", "isn't", "it'd",
                    "it'll", "it's", "let's", "mayn't", "may've", "mightn't", 
                    "might've", "mustn't", "must've", "needn't", "o'clock", "ol'",
                    "oughtn't", "shan't", "she'd", "she'll", "she's", "should've",
                    "shouldn't", "somebody's", "someone's", "something's", "that'll",
                    "that're", "that's", "that'd", "there'd", "there're", "there's", 
                    "these're", "they'd", "they'll", "they're", "they've", "this's",
                    "those're", "tis", "twas", "twasn't", "wasn't", "we'd", "we'd've",
                    "we'll", "we're", "we've", "weren't", "what'd", "what'll", 
                    "what're", "what's", "what've", "when's", "where'd", "where're",
                    "where's", "where've", "which's", "who'd", "who'd've", "who'll",
                    "who're", "who's", "who've", "why'd", "why're", "why's", "won't",
                    "would've", "wouldn't", "y'all", "you'd", "you'll", "you're", 
                    "you've", "'s", "'d", "'m", "abov", "afterward", "ai", "alon", "alreadi", "alway", "ani", 
                     "anoth", "anyon", "anyth", "anywher", "becam", "becaus", "becom", "befor", 
                     "besid", "ca", "cri", "dare", "describ", "did", "doe", "dure", "els", 
                     "elsewher", "empti", "everi", "everyon", "everyth", "everywher", "fifti", 
                     "forti", "gon", "got", "henc", "hereaft", "herebi", "howev", "hundr", "inde", 
                     "let", "ll", "mani", "meanwhil", "moreov", "n't", "na", "need", "nobodi", "noon", 
                     "noth", "nowher", "ol", "onc", "onli", "otherwis", "ought", "ourselv", "perhap", 
                     "pleas", "sever", "sha", "sinc", "sincer", "sixti", "somebodi", "someon", "someth", 
                     "sometim", "somewher", "ta", "themselv", "thenc", "thereaft", "therebi", "therefor", 
                     "togeth", "twelv", "twenti", "ve", "veri", "whatev", "whenc", "whenev", 
                    "wherea", "whereaft", "wherebi", "wherev", "whi", "wo", "anywh", "el", "elsewh", "everywh", 
                    "ind", "otherwi", "plea", "somewh", "yourselv"]
    return extra_stopwords

def process(text):
    porter = PorterStemmer()
    stop = stopwords.words('english')
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    tokens  = nltk.word_tokenize(text)
    tokens = [porter.stem(w) for w in tokens]
    # Join the words back into one string separated by space, and return the result.
    return tokens
