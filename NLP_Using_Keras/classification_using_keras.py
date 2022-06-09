#import libraries.
import tensorflow as tf
import numpy as np
import pandas as pd
from nltk.corpus import PlaintextCorpusReader
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
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

tf.__version__

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout

#unziping the dataset.
!unzip /content/data_text_classify.zip

#creating business corpus by giving the path and reading it.
corpus_root1 = '/content/txt_sentoken/neg'
filelists = PlaintextCorpusReader(corpus_root1, '.*')

#read all the text files in business folder
a=filelists.fileids()

Neg_Corpus=[]

for file in a:
  g=open("{}/{}".format(corpus_root1,file),'r',encoding='latin-1')
  x=g.read()
  Neg_Corpus.append(x)
print(Neg_Corpus[0])

len(Neg_Corpus)

print(Neg_Corpus)

negative_reviews = pd.DataFrame(
    {'review':Neg_Corpus,'label': 'Neg'}
)

#creating business corpus by giving the path and reading it.
corpus_root2 = '/content/txt_sentoken/pos'
filelists = PlaintextCorpusReader(corpus_root2, '.*')

#read all the text files in business folder
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

positive_reviews.review = positive_reviews.review.apply(lambda x:x.lower())

negative_reviews.review = negative_reviews.review.apply(lambda x:x.lower())

punctuations = list(string.punctuation)
punctuations

positive_reviews.review = positive_reviews.review.apply(lambda x: " ".join(x for x in x.split() if x not in punctuations))
negative_reviews.review = negative_reviews.review.apply(lambda x: " ".join(x for x in x.split() if x not in punctuations))

nltk.download('stopwords')

stop = stopwords.words('english')

print('Total stop words:',len(stop))

positive_reviews.review = positive_reviews.review.apply(lambda x: " ".join(x for x in x.split() if x not in stop))
negative_reviews.review = negative_reviews.review.apply(lambda x: " ".join(x for x in x.split() if x not in stop))

positive_reviews.review[2]

nltk.download('punkt')

positive_reviews['review_tokenized'] = positive_reviews.review.apply(lambda x: word_tokenize(x))
negative_reviews['review_tokenized'] = negative_reviews.review.apply(lambda x: word_tokenize(x))

lemmatizer = WordNetLemmatizer()
 
print(lemmatizer.lemmatize('increases'))

positive_reviews['review_lemmatized'] = positive_reviews.review_tokenized.apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
negative_reviews['review_lemmatized'] = negative_reviews.review_tokenized.apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

positive_review_list = positive_reviews['review_lemmatized'].tolist()
negative_review_list = negative_reviews['review_lemmatized'].tolist()

positive_review_list = [item for sublist in positive_review_list for item in sublist]
negative_review_list = [item for sublist in negative_review_list for item in sublist]

print('Number of positive words',len(positive_review_list))

print('Number of negative words',len(negative_review_list))

all_words = (positive_review_list + negative_review_list)

print('Number of total words in corpus',len(all_words))

word_counter = Counter(all_words)

most_common_words = word_counter.most_common()[:10]
most_common_words = pd.DataFrame(most_common_words)
most_common_words.columns = ['word', 'freq']
most_common_words

most_common_words.sort_values(by='freq',ascending=True).plot(x='word', kind='barh')

import seaborn as sns
sns.distplot(positive_reviews['review'].apply(lambda y: len(y)), label='positive reviews',hist=False)
sns.distplot(negative_reviews['review'].apply(lambda y: len(y)), label='negative reviews',hist=False)
plt.legend()
plt.show()

most_common_words.word.tolist()[:3]

remove = most_common_words.word.tolist()[:3]
remove

negative_reviews['review_lemmatized'] = negative_reviews['review_lemmatized'].apply(lambda x: [y for y in x if y not in remove])
positive_reviews['review_lemmatized'] = positive_reviews['review_lemmatized'].apply(lambda x: [y for y in x if y not in remove])

print(positive_reviews)

positive_reviews['review_lemmatized_train'] = positive_reviews.review_lemmatized.apply(lambda x: ' '.join(x))
negative_reviews['review_lemmatized_train'] = negative_reviews.review_lemmatized.apply(lambda x: ' '.join(x))

print(positive_reviews)

x = (positive_reviews['review_lemmatized_train'].append(negative_reviews['review_lemmatized_train']))
y = (positive_reviews['label'].append(negative_reviews['label']))

le = preprocessing.LabelEncoder()
le.fit(y)

list(le.classes_)

y = le.transform(y)

print('Labels for \'{}\' are \'{}\' respectively.'.format(le.inverse_transform(np.unique(y)),np.unique(y)))

y.shape

print(x)

df = pd.DataFrame(
    {'review':x,'label': y}
)

df

df = df.sample(frac = 1)

df.head(15)

df.label.value_counts()

df.tail()

df.shape

train_size=int(df.shape[0] * 0.8)
X_train = df.review[:train_size]
Y_train = df.label[:train_size]

X_test = df.review[train_size: ]
Y_test = df.label[train_size: ]

print(X_train)

text= df['review'][0]
print(text)

corpus1=[]
for text in df['review']:
  words=[word.lower() for word in word_tokenize(text)]
  corpus1.append(words)

num_words=len(corpus1)
print(num_words)

train_size1=int(df.shape[0] * 0.8)
x_train = df.review[:train_size]
y_train1 = df.label[:train_size]

x_test = df.review[train_size: ]
y_test = df.label[train_size: ]

tokenizer = Tokenizer(num_words)
tokenizer.fit_on_texts(x_train)
x_train=tokenizer.texts_to_sequences(x_train)
x_train=pad_sequences(x_train, maxlen=128,truncating='post',padding='post')

x_train[0], len(x_train[0])

x_test=tokenizer.texts_to_sequences(x_test)
x_test=pad_sequences(x_test, maxlen=128,truncating='post',padding='post')

x_test[0], len(x_test[0])

print(x_train.shape, y_train1.shape)
print(x_test.shape, y_test.shape)

model = Sequential()
model.add(Embedding(input_dim=num_words, output_dim=100, input_length=128, trainable=True))
model.add(LSTM(100,dropout=0.1, return_sequences=True))
model.add(LSTM(100, dropout=0.1))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()

history= model.fit(x_train,y_train1, epochs=20,batch_size=64,validation_data=(x_test,y_test))

plt.figure(figsize=(16,5))
epochs= range(1, len(history.history['accuracy'])+1)
plt.plot(epochs, history.history['loss'],'b',label='Training Loss',color='red')
plt.plot(epochs, history.history['val_loss'],'b',label='Validation Loss')
plt.legend()
plt.show()

plt.figure(figsize=(16,5))
epochs= range(1, len(history.history['accuracy'])+1)
plt.plot(epochs, history.history['accuracy'],'b',label='Training Accuracy',color='red')
plt.plot(epochs, history.history['val_accuracy'],'b',label='Validation Accuracy')
plt.legend()
plt.show()

validation_sentence=['This movie was not good at all. It had some good parts like acting was pretty good but the story was not impressing at all.']
validation_sentence_tokenized=tokenizer.texts_to_sequences(validation_sentence)
validation_sentence_padded=pad_sequences(validation_sentence_tokenized, maxlen=128, truncating='post', padding='post')
print(validation_sentence[0])
print('Probability of positive: {}'.format(model.predict(validation_sentence_padded)[0]))

validation_sentence=['It had some good parts like storyline although the actors performed really well and that is why overall I enjooyed it.']
validation_sentence_tokenized=tokenizer.texts_to_sequences(validation_sentence)
validation_sentence_padded=pad_sequences(validation_sentence_tokenized, maxlen=128, truncating='post', padding='post')
print(validation_sentence[0])
print('Probability of positive: {}'.format(model.predict(validation_sentence_padded)[0]))

validation_sentence=['i can watch this movie forever just because of the beauty in its cinematography.']
validation_sentence_tokenized=tokenizer.texts_to_sequences(validation_sentence)
validation_sentence_padded=pad_sequences(validation_sentence_tokenized, maxlen=128, truncating='post', padding='post')
print(validation_sentence[0])
print('Probability of positive: {}'.format(model.predict(validation_sentence_padded)[0]))

validation_sentence=['today i watched stranger things. Overall the movie was good.']
validation_sentence_tokenized=tokenizer.texts_to_sequences(validation_sentence)
validation_sentence_padded=pad_sequences(validation_sentence_tokenized, maxlen=128, truncating='post', padding='post')
print(validation_sentence[0])
print('Probability of positive: {}'.format(model.predict(validation_sentence_padded)[0]))

