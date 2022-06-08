
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

#applying lower casing.
positive_reviews.review = positive_reviews.review.apply(lambda x:x.lower())

print(positive_reviews)

negative_reviews.review = negative_reviews.review.apply(lambda x:x.lower())

print(negative_reviews)

positive_reviews.review[2]

negative_reviews.review[2]

#Applying punctuation

punctuations = list(string.punctuation)
punctuations

positive_reviews.review = positive_reviews.review.apply(lambda x: " ".join(x for x in x.split() if x not in punctuations))
negative_reviews.review = negative_reviews.review.apply(lambda x: " ".join(x for x in x.split() if x not in punctuations))

positive_reviews.review[2]

negative_reviews.review[2]

#Removing Stopwords

nltk.download('stopwords')

stop = stopwords.words('english')

print('Total stop words:',len(stop))

positive_reviews.review = positive_reviews.review.apply(lambda x: " ".join(x for x in x.split() if x not in stop))
negative_reviews.review = negative_reviews.review.apply(lambda x: " ".join(x for x in x.split() if x not in stop))

positive_reviews.review[2]

negative_reviews.review[2]

nltk.download('punkt')

#Applying Tokenization and adding a column in the dataframe

positive_reviews['review_tokenized'] = positive_reviews.review.apply(lambda x: word_tokenize(x))
negative_reviews['review_tokenized'] = negative_reviews.review.apply(lambda x: word_tokenize(x))

print(positive_reviews)

#Applying lemmatizer and adding it to the dataframe

lemmatizer = WordNetLemmatizer()
 
print(lemmatizer.lemmatize('increases'))

positive_reviews['review_lemmatized'] = positive_reviews.review_tokenized.apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
negative_reviews['review_lemmatized'] = negative_reviews.review_tokenized.apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

print(positive_reviews)

#Plotting frequencies of words

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

#Removing unimportants words

most_common_words.word.tolist()[:3]

remove = most_common_words.word.tolist()[:3]
remove

negative_reviews['review_lemmatized'] = negative_reviews['review_lemmatized'].apply(lambda x: [y for y in x if y not in remove])
positive_reviews['review_lemmatized'] = positive_reviews['review_lemmatized'].apply(lambda x: [y for y in x if y not in remove])

print(positive_reviews)

positive_reviews['review_lemmatized_train'] = positive_reviews.review_lemmatized.apply(lambda x: ' '.join(x))
negative_reviews['review_lemmatized_train'] = negative_reviews.review_lemmatized.apply(lambda x: ' '.join(x))

print(positive_reviews)

#Defining target and feature variables

x = (positive_reviews['review_lemmatized_train'].append(negative_reviews['review_lemmatized_train']))
y = (positive_reviews['label'].append(negative_reviews['label']))

#Label encoding target column

le = preprocessing.LabelEncoder()
le.fit(y)

list(le.classes_)

y = le.transform(y)

print('Labels for \'{}\' are \'{}\' respectively.'.format(le.inverse_transform(np.unique(y)),np.unique(y)))

#Creating Bag of words

bow = CountVectorizer(max_features=25000, lowercase=True,analyzer = "word")
train_bow_neg = bow.fit_transform(x)
train_bow_neg

train_bow_neg.toarray().shape

vocab = bow.get_feature_names()

# Sum up the counts of each vocabulary word
dist = np.sum(train_bow_neg.toarray(), axis=0)

word_freq = pd.DataFrame({'word':vocab,'freq':dist})

word_freq.sort_values(by='freq',ascending=False)[:10]

#Splitting into train and test dataset.

from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve
xtrain, xtest, ytrain, ytest = train_test_split \
                (train_bow_neg.toarray(), y,test_size=0.3, \
                random_state=1000)

print ("No. of True Cases in training data set for" , ytrain.sum())
print ("No. of True Cases in testing data set for",ytest.sum())

print ("Ratio of True Cases in training data set: " , round(ytrain.sum()/len(ytrain),2))
print ("Ratio of True Cases in testing data set: ", round(ytest.sum()/len(ytest),2))

xtrain.shape

train_bow_neg

#Training a multinomial Naive Bayes Classifier

from sklearn.naive_bayes import MultinomialNB
print( "Training the multinomial Naive Bayes Classifier")

# Initialize a Random Forest classifier with 100 trees
NB = MultinomialNB() 
NB_clf = NB.fit( xtrain, ytrain )

prob_test  = NB.predict_proba(xtest)
prob_train = NB.predict_proba(xtrain)

#Probability Threshold = 0.5 (default) 
pred_test  = NB.predict(xtest)
pred_train = NB.predict(xtrain)

train_acc = accuracy_score(ytrain, pred_train)
test_acc  = accuracy_score(ytest, pred_test)
print ("Train Accuracy :: ", train_acc)
print ("Test Accuracy :: ", test_acc)

pip install scikit-plot

import scikitplot as skplt

print ("\n Confusion matrix: \n")
skplt.metrics.plot_confusion_matrix(ytest, pred_test, title="Confusion Matrix",
                text_fontsize='large')
plt.show()

