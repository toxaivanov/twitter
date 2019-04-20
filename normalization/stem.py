import csv      
import pandas as pd
import re
import numpy as np 
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer


#train  = pd.read_csv('')
train = pd.read_csv('./input.csv', sep=';', usecols=[0,8,9] , names=['id', 'tweet_id', 'tweet_text'] )

print(train['tweet_text'].head(5))

def remove_pattern(input_txt):
    input_txt = re.sub("@[\w]*", " ", str(input_txt) )
    return input_txt  
    
train['tweet_text'] = np.vectorize(remove_pattern)(train['tweet_text'])
train['tweet_text'] = train['tweet_text'].str.replace("[^a-zA-Z#]", " ")
train['tweet_text'] = train['tweet_text'].apply(lambda x:' '.join(w for w in x.split() if len(w)>3))

tokenized_tweet = train['tweet_text'].apply(lambda x: word_tokenize(x))

print(tokenized_tweet.head(10))


stemmer = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming


for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])


tokenized_tweet.to_csv('./output.csv')


