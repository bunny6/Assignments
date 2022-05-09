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

#for reading each text file from all folders and doing lemmatization,stopword removal.
def for_reading_file(path):    
  filelists = PlaintextCorpusReader(path, '.*')
  a=filelists.fileids()
#list Containg all the text files of Sport Folder
  Corpus_document=[]
  new=[]
  for file in a:
    f = open('{}/{}'.format(path,file), 'r', encoding="latin-1")
    #Got an utf-8 error so used encoding while reading the text in the file
    text_data=f.read().split('\n')
    text_data = list(filter(None, text_data))
    new.append(text_data)

  new1=[]
  for i in new:
    a=' '.join(i)
    new1.append(a)
 
#Joined each word in the file to form a string in  Array
  for i in new1:
     Corpus_document.append(i.split())
   
  # done tokenization for each file in  Array
  Scorpus=[]
  for item in  Corpus_document:
    new=[]
    for item2 in item:
        item2= re.sub('[^a-zA-Z]','',item2) #Replacing the punctuation marks into empty charcter using sub function.
        item2=item2.lower()  #converted to lower case
        item2=nltk.stem.WordNetLemmatizer().lemmatize(item2) #Applying Lemmatization on  Each word
        if item2 not in set(stopwords.words('english')) and len(item2)>2:
            new.append(item2)
    Scorpus.append(new)   

  # Array after removing stopwords ,converting to lower case words

  Scorpus1=[]
  for i in Scorpus:
    new=[]
    for j in i:
        if j!="":
            if j in new:
                break
            else:
                new.append(j)
      
    Scorpus1.append(new)    
       
   # array after removing the empty elements and finding the unique words
  Scorpus2=[]
  for i in Scorpus1:
    Scorpus2.append(" ".join(i))

  return Scorpus1,Scorpus2 #array after making string of words from each file 


# business corpus"
corpus_root1 = 'document_classification/bbc_fulltext/bbc/business'
#business folder path
c1,c2=for_reading_file(path=corpus_root1)
df1=pd.DataFrame({'page':c2,"Text":c1,"category":"Business"})
#data frame for business class


# Entertainment
corpus_root1 = 'document_classification/bbc_fulltext/bbc/entertainment'
#path of Entertainment Folder
c1,c2=for_reading_file(path=corpus_root1)
df2=pd.DataFrame({'page':c2,"Text":c1,"category":"Entertainment"})
#data frame for Entertainment class


# politics
corpus_root1 = 'document_classification/bbc_fulltext/bbc/politics'
c1,c2=for_reading_file(path=corpus_root1)
df3=pd.DataFrame({'page':c2,"Text":c1,"category":"Politics"})
#data frame for politcs class


# Sport
corpus_root1 = 'document_classification/bbc_fulltext/bbc/sport'
#sport folder Path
c1,c2=for_reading_file(path=corpus_root1)
df4=pd.DataFrame({'page':c2,"Text":c1,"category":"Sports"})
#data frame for sports class

# Tech
corpus_root1 = 'document_classification/bbc_fulltext/bbc/tech'
#Tech Folder path
c1,c2=for_reading_file(path=corpus_root1)
df5=pd.DataFrame({'page':c2,"Text":c1,"category":"Tech"})
#data frame for tech class
print(df5)

#Final Data Frame
DF=pd.concat((df1,df2,df3,df4,df5))
#dataframe after concatenating all the dataframes tech,sport,entertainment,politics and business
print(DF)

# tfidf vectorizor
#applied tfidf vectorizor
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(input='content', analyzer = 'word', lowercase=True, stop_words='english',\
                                   ngram_range=(1, 3), min_df=40, max_df=0.20,\
                                  norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True)
text_vector = vectorizer.fit_transform(DF.page)
dtm = text_vector.toarray()
features = vectorizer.get_feature_names()

# Label Encoding
from sklearn.preprocessing import LabelEncoder
label_enc = LabelEncoder()
DF['label'] = label_enc.fit_transform(DF['category'])


h = pd.DataFrame(data = text_vector.todense(), columns = vectorizer.get_feature_names())
X = text_vector
y = DF.label.values



# splitting the data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

print(X_train)
print(y_train)

# Model Training
# Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
svc1 = RandomForestClassifier(random_state = 0)
svc1.fit(X_train, y_train)
svc1_pred = svc1.predict(X_test)
#print(f"Train Accuracy: {svc1.score(X_train, y_train)*100:.3f}%")
print(f"Test Accuracy: {svc1.score(X_test, y_test)*100:.3f}%")

# K Nearest Neighbours
from sklearn.neighbors import KNeighborsClassifier
svc4 = KNeighborsClassifier()
#pprint(svc4.get_params())
svc4.fit(X_train, y_train)
svc4_pred = svc4.predict(X_test)
#print(f"Train Accuracy: {svc4.score(X_train, y_train)*100:.3f}%")
print(f"Test Accuracy: {svc4.score(X_test, y_test)*100:.3f}%")

