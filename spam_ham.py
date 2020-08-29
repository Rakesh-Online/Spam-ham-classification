import nltk
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

"""
#Using stemming 
word_stemmer = PorterStemmer()
print(word_stemmer.stem('history'))

#using Lematization
lematizer = WordNetLemmatizer()
print(lematizer.lemmatize("history"))

print(string.punctuation)

print(stopwords.words('english'))
"""
df = pd.read_csv(r'C:\Users\rakes\Documents\Raq files\Data Science\Data sets\NB.csv', encoding = 'ISO-8859-1')

#print(df.describe())
#print(df.groupby('type').describe())


def message_text_process(mess):
    no_punc = [char for char in mess if char not in string.punctuation]
    no_punc = ''.join(no_punc)
    return [word for word in no_punc.split() if word.lower() not in stopwords.words('english')]

df['text'].head().apply(message_text_process)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
bag_words = CountVectorizer(analyzer = message_text_process).fit(df['text'])
print(len(bag_words.vocabulary_))

message_bagwords = bag_words.transform(df['text'])
print(message_bagwords)

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(message_bagwords)

message_tfidf = tfidf_transformer.transform(message_bagwords)
# model building
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(message_tfidf, df['type'], test_size=0.2)
from sklearn.naive_bayes import MultinomialNB
spam_detect = MultinomialNB().fit(X_train,y_train)
predicted = spam_detect.predict(X_test)
expected = y_test

from sklearn import metrics
from sklearn.metrics import accuracy_score
print(metrics.classification_report(expected,predicted))
print(metrics.confusion_matrix(expected,predicted))
print(accuracy_score(expected,predicted))

import pickle

pickle.dump(spam_detect,open("spam_model.pkl","wb"))
pickle.dump(cv, open('cv.pkl', "wb"))
