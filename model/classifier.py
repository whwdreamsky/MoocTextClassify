from sklearn.tree import DecisionTreeClassifier as dtc
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score
import pandas


def loadData(featurefile):
	

def DecisionTree():

w
scoring = ['precision_macro', 'recall_macro','f1_macro']
clf = dtc()
socres = cross_validate(clf,x,y,scoring=scoring,cv=10,return_train_score=False)
socres
clf.predict(x)