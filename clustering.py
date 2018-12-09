from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
# clean the descriptions
import nltk
from nltk.corpus import stopwords # import stopwords 
from string import punctuation # import punctuations
from nltk.corpus import wordnet # to check if a word is english
from nltk.stem import PorterStemmer #Porter stemming algorithm to remove and replace well-known suffixes of English words
# Create the vocabularies
from collections import defaultdict
import math
import json
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from scipy import sparse
import matplotlib.pyplot as plt
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

def clean_locali(s):
    s=int(re.findall(r'[\d]',s)[0])
    return(s)
def clean_superficie(s):
    s=int(s)
    return(s)
def clean_piano(s):
    try:
        s=int(s)
    except:
        if s=='>10' or s=='Ultimo':
            s=20
        if s=='Piano rialzato' or s=='Ammezzato':
            s=0.5
        if s=='Piano terra':
            s=0
        if s=='Seminterrato':
            s=-0.5
        if s=='Interrato':
            s=-1
    return(s)
def clean_Price(s):
    s=int(str(s).replace('.',''))
    return(s)
def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3
def Union(lst1, lst2): 
    final_list = list(set(lst1) | set(lst2)) 
    return final_list

def jaccard_similarity(lista1,lista2):
    inter=len(intersection(lista1,lista2))
    union=len(Union(lista1, lista2))
    res=round(inter/union,2)
    return res

with open('vocabulary_tfidf.json') as json_data:
    vocabulary_tfidf= json.load(json_data)
def get_id_word(cluster_i, vocabulary_tfidf=vocabulary_tfidf):
    h=[]
    try:
        for doc in cluster_i:
            for ID in vocabulary_tfidf[str(doc)]:
                v=ID[0]
                h.append(v)
    except:
        pass
    return(h)

with open('map_term_ID.json') as json_data:
    map_term_ID= json.load(json_data)

def fromID_toSTR(lista, map_term_ID=map_term_ID):
    l=[]
    try:
        for j in lista:
            l.append(map_term_ID[str(j)])
        s=' '.join(l)
    except:
        pass
    return(s)

def transform_format(val):
    if val == 0:
        return 255
    else:
        return val


