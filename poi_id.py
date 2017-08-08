#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import numpy as np
from time import time
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

### Task 1: Select what features you'll use.

features_list = ['poi', 'salary', 'deferral_payments', 'total_payments',
'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 
'total_stock_value', 'expenses', 'exercised_stock_options', 'director_fees'] 


print 'Number of Features:', len(features_list)

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

#Checking the number of Values
print 'Number of Data Values:', len(data_dict)

### Task 2: Remove outliers
data_dict.pop('TOTAL',)

### Task 3: Create new feature(s)

def bonus_ratio(numerator, denominator):
	if (numerator == 'NAN') or (denominator == 'NAN') or (denominator == 0):
		fraction = 0
	else:
		fraction = float(numerator)/float(denominator)
	return fraction

def total_to_bonus_ratio(dict):
	for key in dict:
		total = dict[key]['total_payments']
		bonus = dict[key]['bonus']
		total_to_bonus = bonus_ratio(total, bonus)
		dict[key]['total_to_bonus_ratio'] = total_to_bonus
data_dict_new = total_to_bonus_ratio(data_dict)

print 'Number of Data Values:', len(data_dict)

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
#See classifiers_only.py for my starter classifiers

#Pipeline with scaling, splitting, decision tree classifier, and gridsearchcv
mmscaler = MinMaxScaler()
skbest = SelectKBest()
dt = DecisionTreeClassifier()

sss = StratifiedShuffleSplit(10, random_state = 42)

param = {"skbest__k": range(1,10),
         "dt__criterion": ["gini", "entropy"],
         "dt__min_samples_split": [2, 10, 20],
         "dt__max_depth": [None, 2, 5, 10],
         "dt__min_samples_leaf": [1, 5, 10],
         "dt__max_leaf_nodes": [None, 5, 10, 20]
         }
pipe = Pipeline(steps=[('mmscaler', mmscaler), ('skbest', skbest), ('dt', dt)])
newclf = GridSearchCV(pipe, param, scoring = 'f1', cv = sss, verbose = 5, n_jobs = 10)	
newclf.fit(features, labels)
clf = newclf.best_estimator_

clf.fit(features, labels)
pred = clf.predict(features)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. 

from tester import test_classifier
test_classifier(clf, my_dataset, features_list)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. 

dump_classifier_and_data(clf, my_dataset, features_list)