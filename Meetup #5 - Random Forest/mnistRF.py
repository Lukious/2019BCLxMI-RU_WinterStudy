# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 10:14:36 2020

@author: 이충섭
"""
from tensorflow.keras.datasets import mnist
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import matplotlib
from sklearn import metrics

(train_x, train_y), (test_x, test_y) = mnist.load_data()

X = train_x
X = X.reshape([-1, 28*28])
Z = test_x
Z = Z.reshape([-1, 28*28])
y = train_y

rnd_clf = RandomForestClassifier(n_estimators=1, criterion="entropy", n_jobs=-1, max_features="auto", random_state=42)
rnd_clf.fit(X, y)

pred = rnd_clf.predict(Z)

print('정확도:', metrics.accuracy_score(test_y, pred))

def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = matplotlib.cm.hot,
              interpolation="nearest")
    plt.axis("off")

plot_digit(rnd_clf.feature_importances_)

cbar = plt.colorbar(ticks=[rnd_clf.feature_importances_.min(), rnd_clf.feature_importances_.max()])
cbar.ax.set_yticklabels(['Not useful', 'Important'])

plt.show()

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

log_clf = LogisticRegression(random_state=42)
rnd_clf = RandomForestClassifier(random_state=42)
svm_clf = SVC(random_state=42)

voting_clf = VotingClassifier(
   estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
   voting='hard')
voting_clf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
   clf.fit(X_train, y_train)
   y_pred = clf.predict(X_test)
   print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

