import pandas as pd
import numpy as np
from nltk.corpus import PlaintextCorpusReader
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('wordnet')
nltk.download('punkt')
import string
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.corpora import Dictionary
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt


titles=[]
all_data=[]
def reading_file(path):    
  filelists = PlaintextCorpusReader(path, '.*')
  a=filelists.fileids()
#list Containg all the text files of Sport Folder
  
  for file in a:
    f = open('{}/{}'.format(path,file), 'r', encoding="latin-1")
    #Got an utf-8 error so used encoding while reading the text in the file
    text_data=f.read().split('\n')
    text_data = list(filter(None, text_data))
    
    titles.append(text_data[0])
    all_data.append(( text_data[0], text_data[1:]))
             
  return all_data #sports array after making string of words from each file 

# business corpus
titles=[]
all_data=[]
#business folder path
all_data=reading_file('document_classification/bbc_fulltext/bbc/business')
df1 = pd.DataFrame(all_data, columns=[ 'title', 'text'])
df1['category']="Business"


# Entertainment
titles=[]
all_data=[]
#path of Entertainment Folder
all_data=reading_file('document_classification/bbc_fulltext/bbc/entertainment')
df2 = pd.DataFrame(all_data, columns=[ 'title', 'text'])
df2['category']="Entertainment"

# politics
titles=[]
all_data=[]
corpus_root3 =('document_classification/bbc_fulltext/bbc/politics')
all_data=reading_file(path=corpus_root3)
df3 = pd.DataFrame(all_data, columns=[ 'title', 'text'])
df3['category']="Politics"
print(df3)

# Sport
titles=[]
all_data=[]
all_data=reading_file('document_classification/bbc_fulltext/bbc/sport')

df4 = pd.DataFrame(all_data, columns=[ 'title', 'text'])
df4['category']="Sports"
print(df4)

# Tech
titles=[]
all_data=[]
all_data=reading_file('document_classification/bbc_fulltext/bbc/tech')

df5 = pd.DataFrame(all_data, columns=[ 'title', 'text'])
df5['category']="Tech"
print(df5)

#Final Data Frame

df=pd.concat((df1,df2,df3,df4,df5))
#dataframe after concatenating all the dataframes tech,sport,entertainment,politics and business

print(df)


from sklearn.preprocessing import LabelEncoder
label_enc = LabelEncoder()
df['label'] = label_enc.fit_transform(df['category'])



# An array of words
df['text']=df['text'].apply(str)
df_txt=np.array(df['text'])
print(df_txt)

stopwords = nltk.corpus.stopwords.words('english')

def docs_preprocessor(docs):
    # Remain only letters
    tokenizer = RegexpTokenizer('[A-Za-z]\w+')
    
    for i in range(len(docs)):
         # Convert to lowercase
        print(i) 
        print(docs[i][0]) 
        
        docs[i] = docs[i].lower() 
        docs[i] = tokenizer.tokenize(docs[i]) 
        print(docs[i])
        # Split into words
         
    
    # Lemmatize all words with len>2 in documents 
    lemmatizer = WordNetLemmatizer()
    docs = [[nltk.stem.WordNetLemmatizer().lemmatize(token) for token in doc if len(token) > 2 and token not in stopwords] for doc in docs]
    
    return docs


df_txt = docs_preprocessor(df_txt) #df_txt = np.array(df['text'])

print(df_txt)

# Create a dictionary representation of the documents
dictionary = Dictionary(df_txt)
print('Nr. of unique words in initital documents:', len(dictionary))

# Filter out words that occur less than 10 documents, or more than 20% of the documents
dictionary.filter_extremes(no_below=10, no_above=0.2)
print('Nr. of unique words after removing rare and common words:', len(dictionary))

df['text2'] = df_txt

df['text3'] = [' '.join(map(str, j)) for j in df['text2']]

df.iloc[1475:1480,:]

# Word Vectors

# Classification Model
#vectorizer = TfidfVectorizer(stop_words = 'english', lowercase=True)
vectorizer = TfidfVectorizer(input='content', analyzer = 'word', lowercase=True, stop_words='english',\
                                   ngram_range=(1, 3), min_df=40, max_df=0.20,\
                                  norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True)
text_vector = vectorizer.fit_transform(df.text3)
dtm = text_vector.toarray()
features = vectorizer.get_feature_names()

h = pd.DataFrame(data = text_vector.todense(), columns = vectorizer.get_feature_names())
h.iloc[990:1000,280:300]

corpus = [dictionary.doc2bow(txt) for txt in df_txt]

print(f'Number of unique tokens: {len(dictionary)}')
print(f'Number of documents: {len(corpus)}')



# Classification Model

X = text_vector
y = df.label.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

svc1 = RandomForestClassifier(random_state = 42)
svc1.fit(X_train, y_train)
svc1_pred = svc1.predict(X_test)
#print(f"Train Accuracy: {svc1.score(X_train, y_train)*100:.3f}%")
print(f"Test Accuracy: {svc1.score(X_test, y_test)*100:.3f}%")

svc1 = RandomForestClassifier(random_state = 42)
svc1.fit(X_train, y_train)
svc1_pred = svc1.predict(X_test)
#print(f"Train Accuracy: {svc1.score(X_train, y_train)*100:.3f}%")
print(f"Test Accuracy: {svc1.score(X_test, y_test)*100:.3f}%")

svc3 = SGDClassifier(random_state = 42)
svc3.fit(X_train, y_train)
svc3_pred = svc3.predict(X_test)
#print(f"Train Accuracy: {svc3.score(X_train, y_train)*100:.3f}%")
print(f"Test Accuracy: {svc3.score(X_test, y_test)*100:.3f}%")

svc4 = KNeighborsClassifier()
#pprint(svc4.get_params())
svc4.fit(X_train, y_train)
svc4_pred = svc4.predict(X_test)
#print(f"Train Accuracy: {svc4.score(X_train, y_train)*100:.3f}%")
print(f"Test Accuracy: {svc4.score(X_test, y_test)*100:.3f}%")



y_pred1 = vectorizer.transform(['Hour ago, I contemplated retirement for a lot of reasons. I felt like people were not sensitive enough to my injuries. I felt like a lot of people were backed, why not me? I have done no less. I have won a lot of games for the team, and I am not feeling backed, said Ashwin'])
yy = svc1.predict(y_pred1)
print(yy)
result = ""
if yy == [0]:
  result = "Business News"
elif yy == [1]:
  result = "Entertainment news"
elif yy == [2]:
  result = "Politics News"
elif yy == [3]:
  result = "Sports News"
elif yy == [4]:
  result = "Tech News"
print(result)

y_pred1=vectorizer.transform(["Since these raw materials can be processed into paper, the opportunity to provide information to a larger population had also become available during this period in history"])
yy = svc1.predict(y_pred1)
print(yy)
result = ""
if yy == [0]:
  result = "Business News"
elif yy == [1]:
  result = "Entertainment news"
elif yy == [2]:
  result = "Politics News"
elif yy == [3]:
  result = "Sports News"
elif yy == [4]:
  result = "Tech News"
print(result)

y_pred1 = vectorizer.transform(["Politics is the set of activities that are associated with making decisions in groups, or other forms of power relations among individuals, such as the distribution of resources or status. The branch of social science that studies politics and government is referred to as political science"])
yy = svc1.predict(y_pred1)
print(yy)
result = ""
if yy == [0]:
  result = "Business News"
elif yy == [1]:
  result = "Entertainment news"
elif yy == [2]:
  result = "Politics News"
elif yy == [3]:
  result = "Sports News"
elif yy == [4]:
  result = "Tech News"
print(result)


