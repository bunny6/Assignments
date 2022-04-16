#importing the Libraries
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

#creating business corpus by giving the path and reading it.
corpus_root1 = '/home/neosoft/Downloads/NLP/document_classification/bbc_fulltext_document_classification/bbc/business'
filelists = PlaintextCorpusReader(corpus_root1, '.*')

#read all the text files in business folder
a=filelists.fileids()

wordslist = filelists.words('510.txt')

print(wordslist)

print(a)

businessCorpus=[]

for file in a:
  wordslist = filelists.words(file) #Read all the words in the each text file iterating through the loop
  businessCorpus.append(wordslist)

print(businessCorpus)

Bcorpus=[]
for item in businessCorpus:
    new=[]
    for item2 in item:
        item2= re.sub('[^a-zA-Z]','',item2) #Replacing the punctuation marks into empty charcter using sub function.
        item2=item2.lower()  #Converting  words to lower case 
        item2=nltk.stem.WordNetLemmatizer().lemmatize(item2)
        if item2 not in set(stopwords.words('english')) and len(item2)>2: 
            #separating the words that are not stopwords and length of the words greater than 2
            new.append(item2)
    Bcorpus.append(new)   

#business corpus array after removing stopwords and cnverting to lower case and applying lemmatization
print(Bcorpus) 

Bcorpus1=[]
for i in Bcorpus:
    new=[]
    for j in i:
        if j!="":
            if j in new:
                break
            else:
                new.append(j)
      
    Bcorpus1.append(new)    
      
print(Bcorpus1)

#Business array after removing the empty values

print(Bcorpus1[0])
      
Bcorpus2=[]
for i in Bcorpus1:
    Bcorpus2.append(" ".join(i))

#Business list after converting the words after doing limatization and finding unique words into string in each document
print(Bcorpus2) 

#Business Class DataFrame
df1=pd.DataFrame({'page':Bcorpus2,'class':"Business"})

#added new column in the business Dataframe which contains the list of Bag of words created
df1["Text"]=Bcorpus1
print(df1)
      
#Now doing same process for Entertainment
corpus_root1 = 'document_classification/bbc_fulltext_document_classification/bbc/entertainment'

#path of Entertainment Folder
filelists = PlaintextCorpusReader(corpus_root1, '.*')

a=filelists.fileids()
      
#list containing all the text files from Entertainment Folder
print(a)

entertainmentCorpus=[]

for file in a:
    wordslist = filelists.words(file)
    #Read all the words in each file of Entertainment Folder
    entertainmentCorpus.append(wordslist)

print(entertainmentCorpus)

print(entertainmentCorpus)

Ecorpus=[]
for item in entertainmentCorpus:
    new=[]
    for item2 in item:
        item2= re.sub('[^a-zA-Z]','',item2) #Replacing the punctuation marks into empty charcter using sub function.
        item2=item2.lower()  #converted each word to lower case 
        item2=nltk.stem.WordNetLemmatizer().lemmatize(item2) #applied lemmatization on each word
        if item2 not in set(stopwords.words('english')) and len(item2)>2:
            new.append(item2)
    Ecorpus.append(new)   

print(Ecorpus) 
#Entertainment Array after applying lemmatization,changing tom lower case,replacing punctuations.

Ecorpus1=[]
for i in Ecorpus:
    new=[]
    for j in i:
        if j!="":
            if j in new:
                break
            else:
                new.append(j)

    Ecorpus1.append(new)    

print(Ecorpus1) #Entertainment Array after removing null elements and unique words

print(Ecorpus1[0])

Ecorpus2=[]
for i in Ecorpus1:
    Ecorpus2.append(" ".join(i))

# Entertainment Array after making all words in each text file to string.
print(Ecorpus2) 
      
#Entertainment Data Frame
      
df2=pd.DataFrame({'page':Ecorpus2,'class':"Entertainment"})

#Added new column which has the Bagwords as rows

df2["Text"]=Ecorpus1
print(df2)

#politics

#Politics Folder Path
corpus_root1 = 'document_classification/bbc_fulltext_document_classification/bbc/politics'

filelists = PlaintextCorpusReader(corpus_root1, '.*')

#Read all the text files in the entertainment Folder
a=filelists.fileids()

print(a)

politicsCorpus=[]

for file in a:
    wordslist = filelists.words(file)
    #Read all the words in each text file of Politics folder text files
    politicsCorpus.append(wordslist)

print(politicsCorpus)

Pcorpus=[]
for item in politicsCorpus:
    new=[]
    for item2 in item:
        item2= re.sub('[^a-zA-Z]','',item2) #Replacing the punctuation marks into empty charcter using sub function.
        item2=item2.lower() #changing case of the letter to lower case
        item2=nltk.stem.WordNetLemmatizer().lemmatize(item2) #Applied Lemmatization on each word of Politics text files
        if item2 not in set(stopwords.words('english')) and len(item2)>2: #Words which are not in stopwords and words length greater than 2 are found
            new.append(item2)
    Pcorpus.append(new)   

#Politics Array after applying Lemmatization and removing stopwords
print(Pcorpus) 

Pcorpus1=[]
for i in Pcorpus:
    new=[]
    for j in i:
        if j!="":
            if j in new:
                break
            else:
                new.append(j)
      
    Pcorpus1.append(new)    
      
#Politics Array after removing empty elements and finding the unique words
print(Pcorpus1) 

print(Pcorpus1[0])

Pcorpus2=[]
for i in Pcorpus1:
    Pcorpus2.append(" ".join(i))

#Politics Array after joining the words in each text file to form a string
      
print(Pcorpus2)

#Data frame for Politics      
df3=pd.DataFrame({'page':Pcorpus2,'class':"Politics"})

#added new column to Politics DataFrame which has bag of words of each text file
df3["Text"]=Pcorpus1

print(df3)

#Sport

#sport folder Path
corpus_root1 = 'document_classification/bbc_fulltext_document_classification/bbc/sport'

filelists = PlaintextCorpusReader(corpus_root1, '.*')

#list Containg all the text files of Sport Folder
a=filelists.fileids()

print(a)

sportsCorpus=[]

import codecs
new=[]
for file in a:
    f = open('document_classification/bbc_fulltext_document_classification/bbc/sport/{}'.format(file), 'r', encoding="latin-1")
    #Got an utf-8 error so used encoding while reading the text in the file
    text_data=f.read().split('\n')
  
    text_data = list(filter(None, text_data))
    
    new.append(text_data)
    
print(new)

new1=[]
for i in new:
    a=' '.join(i)
   
    new1.append(a)
    
#Joined each word in the file to form a string in Sports Array
print(new1)
      
# done tonenization for each file in Sports Array
for i in new1:
    sportsCorpus.append(i.split())
    
print(sportsCorpus)

Scorpus=[]
for item in sportsCorpus:
    new=[]
    for item2 in item:
        item2= re.sub('[^a-zA-Z]','',item2) #Replacing the punctuation marks into empty charcter using sub function.
        item2=item2.lower()  #converted to lower case
        item2=nltk.stem.WordNetLemmatizer().lemmatize(item2) #Applying Lemmatization on  Each word
        if item2 not in set(stopwords.words('english')) and len(item2)>2:
            new.append(item2)
    Scorpus.append(new)   

print(Scorpus) #sports Array after removing stopwords ,converting to lower case words

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
       
print(Scorpus1) #sports array after removing the empty elements and finding the unique words

Scorpus2=[]
for i in Scorpus1:
    Scorpus2.append(" ".join(i))

print(Scorpus2) #sports array after making string of words from each file 

#data frame for sports class
df4=pd.DataFrame({'page':Scorpus2,'class':"Sport"})

#added new column containing the list of words of each text file
df4["Text"]=Scorpus1

print(df4)

#Tech

#Tech Folder path
corpus_root1 = 'document_classification/bbc_fulltext_document_classification/bbc/tech'

filelists = PlaintextCorpusReader(corpus_root1, '.*')

#list containing all the files of tech folder
a=filelists.fileids()

print(a)

techCorpus=[]

for file in a:
    wordslist = filelists.words(file) #read all the words from each file of tech folder
    techCorpus.append(wordslist)

print(techCorpus)

print(techCorpus)

Tcorpus=[]
for item in techCorpus:
    new=[]
    for item2 in item:
        item2= re.sub('[^a-zA-Z]','',item2) #Replacing the punctuation marks into empty charcter using sub function.
        item2=item2.lower() #converted to lower case  
        item2=nltk.stem.WordNetLemmatizer().lemmatize(item2) #applied lemmatization 
        if item2 not in set(stopwords.words('english')) and len(item2)>2: #removed stopwords and words less than size 3
            new.append(item2)
    Tcorpus.append(new)   

print(Tcorpus) #Tech array after removing stopwords and doing lemmatization 

Tcorpus1=[]
for i in Tcorpus:
    new=[]
    for j in i:
        if j!="":
            if j in new:
                break
            else:
                new.append(j)

    Tcorpus1.append(new)    
      
#Tech array after removing empty elements and duplicates
print(Tcorpus1) 

print(Tcorpus1[0])

Tcorpus2=[]
for i in Tcorpus1:
    Tcorpus2.append(" ".join(i))

# tech array after joing the list elements of each file in tech folder
print(Tcorpus2) 

#Dataframe for Tech 
df5=pd.DataFrame({'page':Tcorpus2,'class':"Tech"})

#added new column in Tech dataframe 
df5["Text"]=Tcorpus1

print(df5)

#dataframe after concatenating all the dataframes tech,sport,entertainment,politics and business
DF=pd.concat((df1,df2,df3,df4,df5))

print(DF)
      
#renamed column in dataframe
DF = DF.rename(columns={'page': 'page', 'class': 'category'})

print(DF)

#applied tfidf vectorizor
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(input='content', analyzer = 'word', lowercase=True, stop_words='english'ngram_range=(1, 3), min_df=40, max_df=0.20,norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True)
text_vector = vectorizer.fit_transform(DF.page)
dtm = text_vector.toarray()
features = vectorizer.get_feature_names()

#Label Encoding
from sklearn.preprocessing import LabelEncoder
label_enc = LabelEncoder()
DF['label'] = label_enc.fit_transform(DF['category'])
DF.tail()

DF[DF['label']==3]

print(text_vector)

h = pd.DataFrame(data = text_vector.todense(), columns = vectorizer.get_feature_names())
h.iloc[:,:]

dictionary = Dictionary(DF['Text'])
print('Nr. of unique words in initital documents:', len(dictionary))

# Filter out words that occur less than 10 documents, or more than 20% of the documents
dictionary.filter_extremes(no_below=10, no_above=0.2)
print('Nr. of unique words after removing rare and common words:', len(dictionary))

print(dictionary)

X = text_vector
y = DF.label.values

DF[DF["label"]==0]

#splitting the data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

print(X_train)
print(y_train)

#Model Training

#Random Forest
from sklearn.ensemble import RandomForestClassifier

svc1 = RandomForestClassifier(random_state = 0)
svc1.fit(X_train, y_train)
svc1_pred = svc1.predict(X_test)
#print(f"Train Accuracy: {svc1.score(X_train, y_train)*100:.3f}%")
print(f"Test Accuracy: {svc1.score(X_test, y_test)*100:.3f}%")

#K Neighbours
from sklearn.neighbors import KNeighborsClassifier
svc4 = KNeighborsClassifier()

#pprint(svc4.get_params())
svc4.fit(X_train, y_train)
svc4_pred = svc4.predict(X_test)
#print(f"Train Accuracy: {svc4.score(X_train, y_train)*100:.3f}%")
print(f"Test Accuracy: {svc4.score(X_test, y_test)*100:.3f}%")

