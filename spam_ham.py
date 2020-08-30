import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

df = pd.read_csv('spam.csv',encoding='latin-1')
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

df['label'] = df['class'].map({'ham':0, 'spam':1})
X = df['message']
y = df['label']

cv = CountVectorizer()
X = cv.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X,y, test_size=0.2)

clf = MultinomialNB()
clf.fit(X_train,y_train)
print(clf.score(X_test, y_test))

joblib.dump(clf, 'spam_model.pkl')
joblib.dump(cv, 'cv.pkl')


