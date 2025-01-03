# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 11:44:00 2020

@author: SUPPU SMILEY
"""
# Import libraries
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score
import matplotlib.pyplot as pl
# Read student data
student_data = pd.read_csv("studentdata.csv")
print ("Student data read successfully!")
#Calculate number of students
n_students = student_data.shape[0]
#Calculate number of features
n_features = student_data.shape[1]
#Calculate passing students
n_passed = sum(student_data['passed'].str.lower()=='yes')
#Calculate failing students
n_failed = sum(student_data['passed'].str.lower()=='no')
#Calculate graduation rate
grad_rate = n_passed/n_students
# Print the results
print ("Total number of students: {}".format(n_students))
print ("Number of features: {}".format(n_features))
print ("Number of students who passed: {}".format(n_passed))
print ("Number of students who failed: {}".format(n_failed))
print ("Graduation rate of the class: {:.2f}%".format(grad_rate))
# Extract feature columns
feature_cols = list(student_data.columns[:-1])
# Extract target column 'passed'
target_col = student_data.columns[-1] 
# Show the list of columns
print ("Feature columns:\n{}".format(feature_cols))
print ("\nTarget column: {}".format(target_col))
# Separate the data into feature data and target data (X_all and y_all, respectively)
X_all = student_data[feature_cols]
y_all = student_data[target_col]
# Show the feature information by printing the first five rows
print ("\nFeature values:")
print (X_all.head())
def preprocess_features(X):
    ''' Preprocesses the student data and converts non-numeric binary variables into
        binary (0/1) variables. Converts categorical variables into dummy variables. '''
    
    # Initialize new output DataFrame
    output = pd.DataFrame(index = X.index)

    # Investigate each feature column for the data
    for col, col_data in X.iteritems():
        
        # If data type is non-numeric, replace all yes/no values with 1/0
        # To better distinguish between the binary variables and the categorical varuables 
        if col_data.dtype == object :
            col_data = col_data.replace(['yes', 'no'], [1, 0])

        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object :
            # Example: 'school' => 'school_GP' and 'school_MS'
            col_data = pd.get_dummies(col_data, prefix = col)  
        
        # Collect the revised columns
        output = output.join(col_data)
    
    return output



X_all = preprocess_features(X_all)
print ("Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns)))
# Import any additional functionality you may need here
import random
#Set the number of training points
num_train = 300
# Set the number of testing points
num_test = X_all.shape[0] - num_train
#Shuffle and split the dataset into the number of training and testing points above
random.seed(312)
shuffled_index=list(range(0,X_all.shape[0])) #cannot do a=random.shuffle(a)
random.shuffle(shuffled_index)
X_train = X_all.iloc[shuffled_index[:num_train],]
X_test =  X_all.iloc[shuffled_index[num_train:],]
y_train = y_all.iloc[shuffled_index[:num_train]]
y_test = y_all.iloc[shuffled_index[num_train:]]
# Show the results of the split
print ("Training set has {} samples.".format(X_train.shape[0]))
print ("Testing set has {} samples.".format(X_test.shape[0]))
def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''
    
    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    
    # Print the results
    print ("Trained model in {:.4f} seconds".format(end - start))

    
def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''
    
    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    end = time()
    
    # Print and return results
    print ("Made predictions in {:.4f} seconds.".format(end - start))
    return f1_score(target.values, y_pred, pos_label='yes')


def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''
    
    # Indicate the classifier and the training set size
    print ("Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train)))
    
    # Train the classifier
    train_classifier(clf, X_train, y_train)
    
    # Print the results of prediction for both training and testing
    print ("F1 score for training set: {:.4f}.".format(predict_labels(clf, X_train, y_train)))
    print ("F1 score for test set: {:.4f}.".format(predict_labels(clf, X_test, y_test)))
    #Import the three supervised learning models from sklearn
# from sklearn import model_A
from sklearn.ensemble import RandomForestClassifier

# from sklearn import model_B
from sklearn.linear_model import SGDClassifier

# from skearln import model_C
from sklearn.svm import SVC





#Initialize the three models
clf_A = RandomForestClassifier(random_state=312)
clf_B = SGDClassifier(random_state=312)
clf_C = SVC(random_state=312)



# Set up the training set sizes
X_train_100 = X_train[:100]
y_train_100 = y_train[:100]

X_train_200 = X_train[:200]
y_train_200 = y_train[:200]

X_train_300 = X_train[:300]
y_train_300 = y_train[:300]       
        

# Execute the 'train_predict' function for each classifier and each training set size
# train_predict(clf, X_train, y_train, X_test, y_test)
#clfs=[clf_A, clf_B, clf_C]
#for i in range(0, len(clfs)):
train_predict(clf_A,X_train_100, y_train_100, X_test, y_test)
train_predict(clf_A,X_train_200, y_train_200, X_test, y_test)
train_predict(clf_A,X_train_300, y_train_300, X_test, y_test)
print('\n')


train_predict(clf_B,X_train_100, y_train_100, X_test, y_test)
train_predict(clf_B,X_train_200, y_train_200, X_test, y_test)
train_predict(clf_B,X_train_300, y_train_300, X_test, y_test)
print('\n')

train_predict(clf_C,X_train_100, y_train_100, X_test, y_test)
train_predict(clf_C,X_train_200, y_train_200, X_test, y_test)
train_predict(clf_C,X_train_300, y_train_300, X_test, y_test)
print('\n')

# use plots to visualiza the results
def model_fit(clf, X_train, y_train, X_test, y_test):
    ''' Train the model using different size of the training samples. '''
    score_train=[]
    score_test=[]
    for i in range(1,4):
        X_train_temp=X_train[:i*100]
        y_train_temp=y_train[:i*100]
        train_predict(clf,X_train_temp, y_train_temp, X_test, y_test)
        score_train.append(predict_labels(clf, X_train_temp, y_train_temp))
        score_test.append(predict_labels(clf, X_test, y_test))
    
    
    a,=pl.plot([100,200,300], score_train, 'r-', label='training set')
    b,= pl.plot([100,200,300], score_test, 'b-', label='testng set')  
    pl.ylim([0.5,1.1])
    pl.title("F1 Score of {}".format((clf.__class__.__name__)))
    pl.legend(handles=[a, b])
    pl.show()
    
model_fit(clf_A, X_train, y_train, X_test, y_test)
model_fit(clf_B, X_train, y_train, X_test, y_test)
model_fit(clf_C, X_train, y_train, X_test, y_test)

clf_rf=RandomForestClassifier(max_depth=5, random_state=312)
train_predict(clf_rf,X_train_100, y_train_100, X_test, y_test)
train_predict(clf_rf,X_train_200, y_train_200, X_test, y_test)
train_predict(clf_rf,X_train_300, y_train_300, X_test, y_test)
print('\n')

# Import 'GridSearchCV' and 'make_scorer'
from sklearn.grid_search import GridSearchCV 
from sklearn.metrics import make_scorer

# Create the parameters list you wish to tune
parameters = {'loss' : ['hinge', 'log', 'modified_huber'],
              'alpha' : [0.0001, 0.001, 0.005, 0.01],
              'n_iter' : [5, 20, 50, 100]
             }

# Initialize the classifier
clf = SGDClassifier(random_state=312)

# Make an f1 scoring function using 'make_scorer' 
f1_scorer = make_scorer(f1_score, pos_label='yes')

# Perform grid search on the classifier using the f1_scorer as the scoring method
grid_obj = GridSearchCV(clf, param_grid=parameters, scoring=f1_scorer)

# Fit the grid search object to the training data and find the optimal parameters
grid_obj.fit(X_train, y_train)

# Get the estimator
clf = grid_obj.best_estimator_
print(clf)

# Report the final F1 score for training and testing after parameter tuning
print ("Tuned model has a training F1 score of {:.4f}.".format(predict_labels(clf, X_train, y_train)))
print ("Tuned model has a testing F1 score of {:.4f}.".format(predict_labels(clf, X_test, y_test)))