from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split #enables validation and bagging
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics
 # Feature Importance
from sklearn import datasets
from sklearn.ensemble import ExtraTreesClassifier


#Load the dataset

AH_data = pd.read_csv("tree_addhealth.csv")
data_clean = AH_data.dropna() #drops rows with ANY NA values

print data_clean.dtypes
print data_clean.describe()

#Split into training and testing sets
predictor_list = ['BIO_SEX','HISPANIC','WHITE','BLACK','NAMERICAN','ASIAN','age',
'ALCEVR1','ALCPROBS1','marever1','TREG1','inhever1','cigavail','DEP1','ESTEEM1','VIOL1',
'PASSIST','DEVIANT1','SCHCONN1','GPA1','EXPEL1','FAMCONCT','PARACTV','PARPRES']

predictors = data_clean[predictor_list]

targets = data_clean.cocever1 #Target variable here is cocever1 : Whether the individual has ever used cocaine

pred_train, pred_test, tar_train, tar_test  = train_test_split(predictors, targets, test_size=.4) #for cross validation

print pred_train.shape
print pred_test.shape
print tar_train.shape
print tar_test.shape

#Build model on training data
from sklearn.ensemble import RandomForestClassifier

classifier=RandomForestClassifier(n_estimators=25) #25 trees
classifier=classifier.fit(pred_train,tar_train)

predictions=classifier.predict(pred_test)

print sklearn.metrics.confusion_matrix(tar_test,predictions)
print sklearn.metrics.accuracy_score(tar_test, predictions)


# fit an Extra Trees model to the data
#a classifier similar to RandomForestClassifer , with few differences in sampling and splitting method
model = ExtraTreesClassifier()
model.fit(pred_train,tar_train)

print predictor_list
print model.feature_importances_ #prints importances of the predictor variables
plt.figure(figsize=(20,8))
plt.cla()
x = range(0, 10*len(predictor_list),10)
plt.xticks(x, predictor_list,fontsize=9)
plt.bar(x,model.feature_importances_,3.5,align='center')
plt.savefig('feature_importances_bar.png')
plt.clf()

"""
Running a different number of trees and see the effect
 of that on the accuracy of the prediction
"""

trees=range(25)
accuracy=np.zeros(25)

for idx in range(len(trees)):
   classifier=RandomForestClassifier(n_estimators=idx + 1)
   classifier=classifier.fit(pred_train,tar_train)
   predictions=classifier.predict(pred_test)
   accuracy[idx]=sklearn.metrics.accuracy_score(tar_test, predictions)

plt.cla()
plt.plot(trees, accuracy)
plt.savefig('treecount_vs_accuracy_1.png')

