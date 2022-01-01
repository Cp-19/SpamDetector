# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 14:57:42 2022

@author: DELL
"""

import numpy as np 
import pandas as pd

df=pd.read_csv("spam.csv")
print(df.columns)
print(df['v2'].head())

drop_col=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']
df.drop(drop_col,axis=1,inplace=True)

from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()
df['v1']=label.fit_transform(df['v1'])
print(df['v1'].head())

#ham-0 and spam-1

X=df.iloc[:,1]
Y=df.iloc[:,0]

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer

from textblob import TextBlob

tokenizer=RegexpTokenizer('\w+')
sw=set(stopwords.words('english'))
ps=PorterStemmer()

def preprocess(X):
    
    X=X.lower()
    
    #Correct spelling
    X=str(TextBlob(X).correct())
    
    word_list=tokenizer.tokenize(X)
    words_left=[w for w in word_list if w not in sw]
    word_left=[ps.stem(w) for w in words_left]
    clean_word=' '.join(word_left)
    return clean_word

def getdoc(document):
    d=[]
    for word in document:
        d.append(preprocess(word))
        
    return d

stemmed_doc=getdoc(X)

print("Count_Vectorizer")
#next day