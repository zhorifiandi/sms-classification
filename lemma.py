############################################################################################################
# 
# Author :
# 1. Ari Pratama Zhorifiandi
# 2. Anwar Ramadha
#
############################################################################################################

# Data Initialization

import pandas as pd
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer 

# For first time use, run these lines for importing data
# import nltk
# nltk.download('punkt')
# nltk.download('wordnet')

class LemmaTokenizer(object):
	def __init__(self):
		self.wnl = WordNetLemmatizer()
	def __call__(self, doc):
		return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

############################################################################################################
# Preprocessing Data

data = pd.read_csv("spam.csv", encoding = 'latin-1')

# Drop column and change name

data = data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1)

data = data.rename(columns={'v1':'label', 'v2':'text'})


# convert label to a numerical variable

data['label_num'] = data.label.map({'ham':0, 'spam':1})


############################################################################################################
# Splitting data into training data and test data randomly

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size = 0.45, random_state = 10)


############################################################################################################
# Using stopword removal
from sklearn.feature_extraction.text import CountVectorizer

prestopwords = pd.read_csv("stopwords.csv")
stopwords = [a[0] for a in prestopwords.values.tolist()]


############################################################################################################
# Converting text into matrix of token counts
#
# Without Lemmatizing 
print('\nWithout Using Lemmatizing ')
vect = CountVectorizer(stop_words=stopwords)
#
# With Lemmatizing
# print('\nUsing Lemmatizing ')
# vect = CountVectorizer(stop_words=stopwords, tokenizer=LemmaTokenizer())

vect.fit(X_train)

X_train_df = vect.transform(X_train)
X_test_df = vect.transform(X_test)


# Using TF-IDF for Normalizing data
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_df)
X_test_tfidf = tfidf_transformer.fit_transform(X_test_df)

############################################################################################################
# Build Model using Multinomial Naive Bayes

prediction = dict()
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()

# Without TF-IDF Option
# print('Without Using TF-IDF : ')
# model.fit(X_train_df,y_train)
# prediction["Multinomial"] = model.predict(X_test_df)


# # With TF-IDF Option
print('Using TF-IDF : ')
model.fit(X_train_tfidf,y_train)
prediction["Multinomial"] = model.predict(X_test_tfidf)

############################################################################################################
# Counting Accuracy 
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

print(accuracy_score(y_test,prediction["Multinomial"]))
print(classification_report(y_test, prediction['Multinomial'], target_names = ["Ham", "Spam"]))

############################################################################################################

