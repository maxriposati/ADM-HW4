#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from bs4 import BeautifulSoup
import requests
import re
import pandas 
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

def read_announcement(soup):
    df_results_1 = pandas.DataFrame(columns=["Title", "Price", "Locali", "Superficie", "Bagni", "Piano", "Description URL"])
    for ul in soup.find_all('ul'):
        title, price, locali, superficie, bagni, piano = [None]*6
        if ul.get("class") == ["listing-features", "list-piped"]:

            # Title of the Announcement
            title = ul.find_previous_sibling("p").a.text

            for li in ul.find_all("li"):
                # Price
                if (li.get("class")) == ['lif__item','lif__pricing']:
                    s=li.text
                    if re.search(r'[\d]+.[\d]{3}.[\d]{3}',s):
                        price=re.findall(r'[\d]+.[\d]{3}.[\d]{3}',s)[0]
                        price=int(price.replace('.',''))
                    elif re.search(r'[\d]+.[\d]{3}',s):
                        price=re.findall(r'[\d]+.[\d]{3}',s)[0]
                        price=int(price.replace('.',''))

            # Locali
            for div in ul.find_all('div' , text="locali"): # there is only one div with text = "locali" in a single "ul"
                if div.get("class") == ["lif__text", "lif--muted"]:
                    s=div.previous_sibling.span.text
                    s=s.replace('+','')
                    l=s.split('-')
                    locali=l[len(l)-1]

            # Superficie
            for div in ul.find_all('div' , text="superficie"): # there is only one div with text = "superficie" in a single "ul"
                if div.get("class") == ["lif__text", "lif--muted"]:
                    superficie = div.previous_sibling.span.text

            # Bagni
            for div in ul.find_all('div' , text="bagni"): # there is only one div with text = "bagni" in a single "ul"
                if div.get("class") == ["lif__text", "lif--muted"]:
                    bagni = div.previous_sibling.span.text
                    bagni=int(bagni.replace('+',''))

            # Piano
            for div in ul.find_all('div' , text="piano"): # there is only one div with text = "piano" in a single "ul"
                if div.get("class") == ["lif__text", "lif--muted"]:
                    piano = div.previous_sibling.abbr.get("title")

            # Append all the features in the dataframe using a temporary df
            list_to_append = [[title, price, locali, superficie, bagni, piano]]

            # It's possible that not all the announcements will have all the fields mentioned above, 
            # if it's the case don't take it into account.
            if None in list_to_append[0]:
                continue
                
            # If the announcement has all the features we save also the url to the the description
            url = ul.find_previous_sibling("p").a.get("href")
            list_to_append[0].append(url)

            df_temp = pandas.DataFrame(list_to_append, columns=["Title", "Price", "Locali", "Superficie", "Bagni", "Piano", "Description URL"] )       
            df_results_1 = pandas.concat([df_results_1, df_temp], ignore_index=True)

    return df_results_1

def cleaning_function(unclnd_string):
    stemmer = PorterStemmer() # instantiate the PorterStemmer class
    clnd_string = ""  # create an empty string 
    
    for single_word in str(unclnd_string).split(): # read every feature word by word
        
        ##### PUNCTUATION CHECK 
        # the variable "wrong_char" is a set of the punctuation in "single_word"
        wrong_char = set(single_word).intersection(punctuation)
        # for loop for every wrong charachter contained in "single_word"
        for wrg in wrong_char:
            single_word = single_word.replace( wrg, "" ) # remove the punctuation 
        ##### Italian OR NUMBER CHECK    
        if (not wordnet.synsets(single_word, lang='ita')) and not (single_word.isdigit()):
            # Not an italian Word
            continue # Skip the word
        ##### STEMMING       
        single_word = stemmer.stem(single_word) # remove affixes from a word
        ##### STOPWORD 
        if (single_word not in stop_words) and (single_word != ''): # remove the word if it is a stopword 
            clnd_string = (clnd_string + " " + single_word.lower())  # append words in the string created earlier 
            
    clnd_string = clnd_string[1:] # remove the final space
        
    return(clnd_string)

def tf_idf ( term, document , D ):
    # the document is passed array like: a list of all the words
    # D is the number of document that contain the term
    
    # find the frequency
    freq = document.count(term) / len(document)
    
    # tf_idf computation
    tf_idf = math.log(10000/D) * freq
    
    return tf_idf

