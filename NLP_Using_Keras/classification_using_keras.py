#importing libraries.
import tensorflow as tf
import numpy as np
import pandas as pd
from nltk.corpus import PlaintextCorpusReader
import re
import nltk
nltk.download('omw-1.4')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer, SnowballStemmer
nltk.download('wordnet')
nltk.download('punkt')
import string
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix, roc_curve, auc, 
                             mean_squared_error, log_loss, precision_recall_curve, classification_report, 
                             precision_recall_fscore_support)
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from zipfile import ZipFile

#importing dataset.
#unzip /content/data_text_classify.zip
import zipfile
with zipfile.ZipFile("data_text_classify.zip","r") as zip_ref:
    zip_ref.extractall("~/Documents/NLP_using_keras")

#creating corpus by giving the path and reading it.
corpus_root1 = '~/Documents/NLP_using_keras/txt_sentoken/neg'
filelists = PlaintextCorpusReader(corpus_root1, '.*')

#read all the text files in negative folder
a=filelists.fileids()

Neg_Corpus=[]

for file in a:
  g=open("{}/{}".format(corpus_root1,file),'r',encoding='latin-1')
  x=g.read()
  Neg_Corpus.append(x)
print(Neg_Corpus[0])

negative_reviews = pd.DataFrame(
    {'review':Neg_Corpus,'label': 'Neg'}
)

#creating  corpus by giving the path and reading it.
corpus_root2 = '~/Documents/NLP_using_keras/txt_sentoken/pos'
filelists = PlaintextCorpusReader(corpus_root2, '.*')

#read all the text files
b=filelists.fileids()

Pos_Corpus=[]

for file in b:
  f=open("{}/{}".format(corpus_root2,file),'r',encoding='latin-1')
  y=f.read()
  Pos_Corpus.append(y)
print(Pos_Corpus[0])

positive_reviews = pd.DataFrame(
    {'review':Pos_Corpus,'label': 'POS'}
)

#applying lower to positive and negative reviews.
positive_reviews.review = positive_reviews.review.apply(lambda x:x.lower())
negative_reviews.review = negative_reviews.review.apply(lambda x:x.lower())

#punctuation
punctuations = list(string.punctuation)
punctuations

#applying punctuation
positive_reviews.review = positive_reviews.review.apply(lambda x: " ".join(x for x in x.split() if x not in punctuations))
negative_reviews.review = negative_reviews.review.apply(lambda x: " ".join(x for x in x.split() if x not in punctuations))

#downloading stopwords
nltk.download('stopwords')

stop = stopwords.words('english')

#applying stopwords
positive_reviews.review = positive_reviews.review.apply(lambda x: " ".join(x for x in x.split() if x not in stop))
negative_reviews.review = negative_reviews.review.apply(lambda x: " ".join(x for x in x.split() if x not in stop))

positive_reviews.review[2]

nltk.download('punkt')

#applying tokenizer
positive_reviews['review_tokenized'] = positive_reviews.review.apply(lambda x: word_tokenize(x))
negative_reviews['review_tokenized'] = negative_reviews.review.apply(lambda x: word_tokenize(x))

#imporing lemmatizer
lemmatizer = WordNetLemmatizer()

#applying lemmatization
positive_reviews['review_lemmatized'] = positive_reviews.review_tokenized.apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
negative_reviews['review_lemmatized'] = negative_reviews.review_tokenized.apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

positive_review_list = positive_reviews['review_lemmatized'].tolist()
negative_review_list = negative_reviews['review_lemmatized'].tolist()

positive_review_list = [item for sublist in positive_review_list for item in sublist]
negative_review_list = [item for sublist in negative_review_list for item in sublist]

#printing total number of words.
all_words = (positive_review_list + negative_review_list)

print('Number of total words in corpus',len(all_words))

word_counter = Counter(all_words)

most_common_words = word_counter.most_common()[:10]
most_common_words = pd.DataFrame(most_common_words)
most_common_words.columns = ['word', 'freq']
most_common_words

most_common_words.word.tolist()[:3]

#removing most common words.
remove = most_common_words.word.tolist()[:3]
remove


negative_reviews['review_lemmatized'] = negative_reviews['review_lemmatized'].apply(lambda x: [y for y in x if y not in remove])
positive_reviews['review_lemmatized'] = positive_reviews['review_lemmatized'].apply(lambda x: [y for y in x if y not in remove])

#joining the sentences
positive_reviews['review_lemmatized_train'] = positive_reviews.review_lemmatized.apply(lambda x: ' '.join(x))
negative_reviews['review_lemmatized_train'] = negative_reviews.review_lemmatized.apply(lambda x: ' '.join(x))

#spliting the data into X and Y
x = (positive_reviews['review_lemmatized_train'].append(negative_reviews['review_lemmatized_train']))
y = (positive_reviews['label'].append(negative_reviews['label']))

#encoding the target column
le = preprocessing.LabelEncoder()
le.fit(y)

list(le.classes_)

y = le.transform(y)

print('Labels for \'{}\' are \'{}\' respectively.'.format(le.inverse_transform(np.unique(y)),np.unique(y)))

#creating dataframe of X and Y.
df = pd.DataFrame(
    {'review':x,'label': y}
)

#shuffling the dataset
df = df.sample(frac = 1)

df.head(15)

#splitting the data into 80-20.
train_size=int(df.shape[0] * 0.8)
X_train = df.review[:train_size]
Y_train = df.label[:train_size]

X_test = df.review[train_size: ]
Y_test = df.label[train_size: ]

corpus1=[]
for text in df['review']:
  words=[word.lower() for word in word_tokenize(text)]
  corpus1.append(words)

train_size1=int(df.shape[0] * 0.8)
x_train = df.review[:train_size]
y_train1 = df.label[:train_size]

x_test = df.review[train_size: ]
y_test = df.label[train_size: ]

num_words=len(corpus1)
print(num_words)

#applying tokenization
tokenizer = Tokenizer(num_words)
tokenizer.fit_on_texts(x_train)
x_train=tokenizer.texts_to_sequences(x_train)
x_train=pad_sequences(x_train, maxlen=128,truncating='post',padding='post')

x_test=tokenizer.texts_to_sequences(x_test)
x_test=pad_sequences(x_test, maxlen=128,truncating='post',padding='post')

#importing model.
model = Sequential()
model.add(Embedding(input_dim=num_words, output_dim=100, input_length=128, trainable=True))
model.add(LSTM(100, dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()

#training the model.
history= model.fit(x_train,y_train1, epochs=10,batch_size=64,validation_data=(x_test,y_test))

#plotting training loss and validation loss.
plt.figure(figsize=(16,5))
epochs= range(1, len(history.history['accuracy'])+1)
plt.plot(epochs, history.history['loss'],'b',label='Training Loss',color='red')
plt.plot(epochs, history.history['val_loss'],'b',label='Validation Loss')
plt.legend()
plt.show()

#plotting training accuracy and validation accuracy.
plt.figure(figsize=(16,5))
epochs= range(1, len(history.history['accuracy'])+1)
plt.plot(epochs, history.history['accuracy'],'b',label='Training Accuracy',color='red')
plt.plot(epochs, history.history['val_accuracy'],'b',label='Validation Accuracy')
plt.legend()
plt.show()

#predicting on new/unseen sentences.
validation_sentence=['It had some good parts like storyline although the actors performed really well and that is why overall I enjoyed.']
validation_sentence_tokenized=tokenizer.texts_to_sequences(validation_sentence)
validation_sentence_padded=pad_sequences(validation_sentence_tokenized, maxlen=128, truncating='post', padding='post')
print(validation_sentence[0])
print('Probability of positive: {}'.format(model.predict(validation_sentence_padded)[0]))

print(validation_sentence_padded)

az=model.predict(validation_sentence_padded)[0]

print(az)

def prediction(validation_sentence):
  validation_sentence_tokenized=tokenizer.texts_to_sequences(validation_sentence)
  validation_sentence_padded=pad_sequences(validation_sentence_tokenized, maxlen=128, truncating='post', padding='post')
  print(validation_sentence)
  az=model.predict(validation_sentence_padded)[0]
  print('Probability of positive: {}'.format(model.predict(validation_sentence_padded)[0][0]))
  if az[0] > 0.5:
    print('Class of the document is 1')
  else:
    print('Class of the document is 0')
validation_sentence= ['No sense of humour, pathetic acting, and boring story']
prediction(validation_sentence)

#sentences which are tested.
# I watched the movie today, cinemetography was fine but the actors performed well
# I watched the movie today, cinemetography was not much good
# Overall a great comedy movie
# Great sense of humour, great acting, and outstanding story
# No sense of humour, pathetic acting, and boring story

y_test=y_test.to_numpy()

y_pred = model.predict(x_test)

for i in y_pred:
  i[0]=i[0].round(2)



print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
