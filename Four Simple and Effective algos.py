# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 22:07:59 2023

@author: swank
"""

#Guessing the number: linear regression
#Using more variables

from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
boston = load_boston()
X = scale(boston.data)
y = boston.target

################################################################
#Had previous problems with the Boston dataset
from sklearn.linear_model import LinearRegression
regression = LinearRegression(normalize=True)
regression.fit(X, y)
print(regression.score(X, y))
print([a + ':' + str(round(b, 2)) for a, b in zip(
    boston.feature_names, regression.coef_,)])


#Understanding limitations and potential problems
#Moving to Logistic Regression
#Applying logistic regression

from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data[:-1,:]
y = iris.target[:-1]

from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression()
logistic.fit(X, y)
single_row_pred = logistic.predict(
    iris.data[-1, :].reshape(1, -1))
single_row_pred_proba = logistic.predict_proba(
    iris.data[-1, :].reshape(1, -1))
print ('Predicted class %s, real class %s' 
       % (single_row_pred, iris.target[-1]))
print ('Probabilities for each class from 0 to 2: %s' 
       % single_row_pred_proba)

#Considering when classes are more than two
from sklearn.datasets import load_digits
digits = load_digits()
train = range(0, 1700)
test = range(1700, len(digits.data))
X = digits.data[train]
y = digits.target[train]
tX = digits.data[test]
ty = digits.target[test]
from sklearn.multiclass import OneVsRestClassifier 
from sklearn.multiclass import OneVsOneClassifier
OVR = OneVsRestClassifier(logistic).fit(X, y)
OVO = OneVsOneClassifier(logistic).fit(X, y)
print('One vs rest accuracy: %.3f' % OVR.score(tX, ty))
print('One vs one accuracy: %.3f' % OVO.score(tX, ty))

#Making Things as Simple as Naïve Bayes
#Predicting text classifications
from sklearn.datasets import fetch_20newsgroups
newsgroups_train = fetch_20newsgroups(
    subset='train', remove=('headers', 'footers', 'quotes'))
newsgroups_test = fetch_20newsgroups(
    subset='test', remove=('headers', 'footers', 'quotes'))

from sklearn.naive_bayes import BernoulliNB, MultinomialNB
Bernoulli = BernoulliNB(alpha=0.01)
Multinomial = MultinomialNB(alpha=0.01)
import sklearn.feature_extraction.text as txt
multinomial = txt.HashingVectorizer(stop_words='english', 
                                binary=False, norm=None)
binary = txt.HashingVectorizer(stop_words='english',
                           binary=True, norm=None)

import numpy as np
target = newsgroups_train.target
target_test = newsgroups_test.target
multi_X = np.abs(
    multinomial.transform(newsgroups_train.data))
multi_Xt = np.abs(
    multinomial.transform(newsgroups_test.data))
bin_X = binary.transform(newsgroups_train.data)
bin_Xt = binary.transform(newsgroups_test.data)
Multinomial.fit(multi_X, target)
Bernoulli.fit(bin_X, target)
​
from sklearn.metrics import accuracy_score
for name, model, data in [('BernoulliNB', Bernoulli, bin_Xt), 
                      ('MultinomialNB', Multinomial, multi_Xt)]:
    accuracy = accuracy_score(y_true=target_test, 
                              y_pred=model.predict(data))
    print ('Accuracy for %s: %.3f' % (name, accuracy))
print('number of posts in training: %i' 
      % len(newsgroups_train.data))
D={word:True for post in newsgroups_train.data 
   for word in post.split(' ')}
print('number of distinct words in training: %i' 
      % len(D))
print('number of posts in test: %i' 
      % len(newsgroups_test.data))

#Exploring Lazy Learning with K-nearest Neighbors
#Predicting after observing neighbors

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

digits = load_digits()
train = range(0, 1700)
test = range(1700, len(digits.data))
pca = PCA(n_components = 25)
pca.fit(digits.data[train])
X = pca.transform(digits.data[train]) 
y = digits.target[train]
tX = pca.transform(digits.data[test]) 
ty = digits.target[test]

from sklearn.neighbors import KNeighborsClassifier
kNN = KNeighborsClassifier(n_neighbors=5, p=2)
kNN.fit(X, y)
print('Accuracy: %.3f' % kNN.score(tX,ty) )
print('Prediction: %s Actual: %s' 
      % (kNN.predict(tX[-15:,:]),ty[-15:]))

#Choosing wisely your k parameter
for k in [1, 5, 10, 50, 100, 200]:
    kNN = KNeighborsClassifier(n_neighbors=k).fit(X, y)
    print('for k = %3i accuracy is %.3f' 
          % (k, kNN.score(tX, ty)))