
#importing libraries
import numpy as np
import pandas as pd
from nltk.corpus import PlaintextCorpusReader
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer, SnowballStemmer
nltk.download('wordnet')
nltk.download('punkt')
import string
import matplotlib.pyplot as plt

#unzipping the dataset.
!unzip /content/data_text_classify.zip

#creating Negative corpus by giving the path and reading it.
corpus_root1 = '/content/txt_sentoken/neg'
filelists = PlaintextCorpusReader(corpus_root1, '.*')

#read all the text files in negative folder
a=filelists.fileids()

Neg_Corpus=[]

for file in a:
  g=open("{}/{}".format(corpus_root1,file),'r',encoding='latin-1')
  x=g.read()
  Neg_Corpus.append(x)
print(Neg_Corpus[0])

len(Neg_Corpus)

print(Neg_Corpus)

#creating dataframe

negative_reviews = pd.DataFrame(
    {'review':Neg_Corpus,'label': 'Neg'}
)

negative_reviews

#creating positive corpus by giving the path and reading it.
corpus_root2 = '/content/txt_sentoken/pos'
filelists = PlaintextCorpusReader(corpus_root2, '.*')

#read all the text files in positive folder
b=filelists.fileids()

Pos_Corpus=[]

for file in b:
  f=open("{}/{}".format(corpus_root2,file),'r',encoding='latin-1')
  y=f.read()
  Pos_Corpus.append(y)
print(Pos_Corpus[0])

len(Pos_Corpus)

#creating dataframe

positive_reviews = pd.DataFrame(
    {'review':Pos_Corpus,'label': 'POS'}
)

positive_reviews
