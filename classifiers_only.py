import sys
import pickle
sys.path.append("../tools/")
import numpy as np
from time import time
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
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
#I could not figure out how to add the new feature to the feature list
def bonus_ratio(numerator, denominator):
	if (numerator == 'NaN') or (denominator == 'NaN') or (denominator == 0):
		fraction = 0
	else:
		fraction = float(numerator)/float(denominator)
	return fraction

def total_to_bonus_ratio(features_list):
	for key in features_list:
		total = features_list[key]['total_payments']
		bonus = features_list[key]['bonus']
		total_to_bonus = bonus_ratio(total, bonus)
		features_list[key]['total_to_bonus_ratio'] = total_to_bonus
total_to_bonus_ratio(data_dict)
features_list.insert(1, 'total_to_bonus_ratio')

print 'Number of Features:', len(features_list)

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#Use SelectKBest to get the optimal number of features
K_Best = SelectKBest(f_classif, k=8)
features_KBest = K_Best.fit_transform(features, labels)
print 'Shape of features after applying SelectKBest:', features_KBest.shape
features_scores = ['%.2f' % elem for elem in K_Best.scores_ ]
features_scores_pvalues = ['%.3f' % elem for elem in  K_Best.pvalues_ ]
features_selected_tuple=[(features_list[i+1], features_scores[i], 
	features_scores_pvalues[i]) for i in K_Best.get_support(indices = True)]

#Printing out the scores for the best features
print 'Feature Scores:', features_scores
print 'Feature scores P:', features_scores_pvalues
print 'Features Selected Tuple:', features_selected_tuple


#Splitting the features and labels into training and testing
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state = 42)

### Task 4: Try a varity of classifiers

#Gaussian Naive Bayes Classifier
#clf = GaussianNB()
#clf.fit(features_train, labels_train)
#pred = clf.predict(features_test)

#Decision Tree Classifier by itself
clf = DecisionTreeClassifier(criterion = 'gini', random_state = 42)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
t0 = time()
print "prediction time:", round(time()-t0, 3), "s"

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. 

#recall = recall_score (labels_test, pred)
#precision = precision_score(labels_test, pred)
#f1 = f1_score(labels_test, pred)
#print 'Recall Score:', recall
#print 'Precision Score:', precision
#print 'F1 Score:', f1

from tester import test_classifier
test_classifier(clf, my_dataset, features_list)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. 

dump_classifier_and_data(clf, my_dataset, features_list)