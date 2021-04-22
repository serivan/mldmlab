#!/usr/bin/env python
# coding: utf-8

# Here you will learn to submit your model to a machine learning competition in Python.  It's fun, and it will give you a way to see your progress as your skills keep improving.*
# 
# # Introduction
# Machine learning competitions are a great way to improve your skills and measure your progress as a data scientist. If you are using data from a competition on Kaggle, you can easily submit it from your notebook.  Here's how you do it.
# 
# # Example
# We're doing very minimal data set up here so we can focus on how to submit modeling results to competitions. Other tutorials will teach you how build great models. So the model in this example will be fairly simple. We'll start with the code to read data, select predictors, and fit a model.

# In[1]:


import pip
import sys
#if not 'sklearn' in sys.modules.keys():
#    pip.main(['install', 'sklearn'])
#if not 'kaggle' in sys.modules.keys():
#    pip.main(['install', 'kaggle'])
import random

print("Random number with seed 2020")
# first call
random.seed(2020)


# In[2]:


import numpy as np
import pandas as pd
import graphviz

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt

# Read the data
train = pd.read_csv('https://raw.githubusercontent.com/serivan/mldmlab/master/Datasets/Kaggle2020/train.csv')


# In[3]:


train["Quality"] = np.where(train["Quality"].str.contains("Good"), 1, 0)

train.dtypes


# In[4]:


train


# ## Feature engineering phase

# In[5]:


# pull data into target (y) and predictors (X)
train_y = train.Quality
predictor_cols = ['fixed.acidity','volatile.acidity','citric.acid','residual.sugar','chlorides','free.sulfur.dioxide','total.sulfur.dioxide','density','pH','sulphates','alcohol']

# Create training predictors data
train_X = train[predictor_cols]

# Create our imputer to replace missing values with the mean e.g.
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp = imp.fit(train_X)


# Impute our data, then train
train_X_imp = imp.transform(train_X)


# Here we define a function for preperaing a submission file

# In[6]:


def generateSubmission(myModel, submissionFile, description):
    # Read the test data
    test = pd.read_csv('https://raw.githubusercontent.com/serivan/mldmlab/master/Datasets/Kaggle2020/test.csv')


    # Treat the test data in the same way as training data. In this case, pull same columns.
    test_X = test[predictor_cols]

    # Impute each test item, then predict
    test_X_imp = imp.transform(test_X)
    
    # Use the model to make predictions
    predicted_q = myModel.predict(test_X_imp)
    # We will look at the predicted Qualities to ensure we have something sensible.
    print(predicted_q)
    
    #submission file
    my_submission = pd.DataFrame({'Id': test.Id, 'Quality': predicted_q})
    # you could use any filename. We choose submission here
    my_submission.to_csv(submissionFile, index=False)
    
    #Submit authomatically; kaggle API authentication needed
    #!kaggle competitions submit -c mldm-classification-competition-2020 -f {submissionFile} -m '{description}'


# ## Training

# ### You can train directly on the training set

# In[7]:


my_model = DecisionTreeClassifier(random_state=1)
my_model.fit(train_X_imp, train_y)
my_model.score(train_X_imp, train_y)


# In[8]:


# The snippet below will retrieve the feature importances from the model and make them into a DataFrame.
feature_importances = pd.DataFrame(my_model.feature_importances_,
                                   index = train_X.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)
feature_importances


# In[9]:


y_pred = my_model.predict(train_X_imp)
print("Confusion Matrix:")
print(confusion_matrix(train_y, y_pred))

print("Classification Report:")
print(classification_report(train_y, y_pred))


# In[10]:


# generate a submission file
generateSubmission(my_model,'firstDecisionTree.csv',"Default Decision tree")


# ### You can consider different models splitting in training and testing

# In[11]:


xTrain, xTest, yTrain, yTest = train_test_split(train_X_imp, train_y, train_size = 0.8, random_state = 0)


# In[12]:


classification_tree_wine = DecisionTreeClassifier(max_depth = 6)
classification_tree_wine.fit(xTrain, yTrain)
classification_tree_wine.score(xTrain, yTrain)


# In[13]:


classification_tree_wine.score(xTest,yTest)


# In[14]:


dor_data=export_graphviz(classification_tree_wine, 
                out_file = "carseat_tree.dot", 
                feature_names = predictor_cols)


with open("carseat_tree.dot") as f:
    dot_graph = f.read()
    
display(graphviz.Source(dot_graph))


# In[15]:


# The snippet below will retrieve the feature importances from the model and make them into a DataFrame.
feature_importances = pd.DataFrame(classification_tree_wine.feature_importances_,
                                   index = predictor_cols,
                                   columns=['importance']).sort_values('importance', ascending=False)
feature_importances


# In[16]:


yPred = classification_tree_wine.predict(xTest)
print("Confusion Matrix:")
print(confusion_matrix(yTest, yPred))

print("Classification Report:")
print(classification_report(yTest, yPred))


# In[17]:


# generate a submission file
generateSubmission(my_model,'secondDecisionTree.csv', "User defined decision tree evaluated with test set")


# ### Or you can consider different models using cross validation

# In[18]:


from sklearn.model_selection import cross_val_score
import seaborn as sns


# In[19]:


dtc = DecisionTreeClassifier()
cv_scores = cross_val_score(dtc, train_X_imp, train_y, cv=10, scoring='accuracy',verbose=1)
sns.distplot(cv_scores)
plt.title('Average score: {}'.format(np.mean(cv_scores)))


# In[20]:


# generate a submission file
generateSubmission(my_model,'DecisionTree.csv', "User defined decision tree evaluated with cross validation")


# ### Parameter Tuning

# In every classification technique, there are some parameters that can be tuned to optimize the classification. Some parameters that can be tuned in the decision tree is max depth (the depth of the tree), max feature (the feature used to classify), criterion, and splitter.
# 
# To search to tune parameter is to use Grid Search. Basically, it explores a range of parameters and finds the best combination of parameters. Then repeat the process several times until the best parameters are discovered. We will also use Stratified k-fold cross-validation that will prevent a certain class only split them to the same subset.

# In[21]:


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold


# In[22]:


dtc = DecisionTreeClassifier()
parameter_grid = {'criterion': ['gini', 'entropy'],
                  'splitter': ['best', 'random'],
                  'max_depth': [1, 2, 3],
                  'max_features': [1, 2, 3]}

cross_validation = StratifiedKFold(n_splits=10)
cross_validation.get_n_splits(train_X_imp, train_y)
#Create the scoring dictionary
SCORING = {'accuracy': 'accuracy',
'balanced_accuracy': 'balanced_accuracy',
'precision': 'precision_macro',
'recall': 'recall_macro',
'f1': 'f1_macro'}

grid_search = GridSearchCV(dtc, param_grid=parameter_grid, cv=cross_validation, )
#grid_search = GridSearchCV(dtc, param_grid=parameter_grid, cv=cross_validation, scoring=SCORING,return_train_score=True, refit='accuracy')

grid_search.fit(train_X_imp, train_y)


# In[23]:


grid_search.cv_results_


# In[24]:


print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))

best_dtc = grid_search.best_estimator_
best_dtc


# In[25]:


my_model=best_dtc
my_model.fit(train_X_imp, train_y)
my_model.score(train_X_imp, train_y)


# In[26]:


y_pred = my_model.predict(train_X_imp)
print("Confusion Matrix:")
print(confusion_matrix(train_y, y_pred))

print("Classification Report:")
print(classification_report(train_y, y_pred))


# In[27]:


# The snippet below will retrieve the feature importances from the model and make them into a DataFrame.
feature_importances = pd.DataFrame(my_model.feature_importances_,
                                   index = train_X.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)
feature_importances


# ### Submit the final model

# In[28]:


# generate a submission file
#generateSubmission(my_model,'DecisionTree.csv', "User defined decision tree evaluated with grid search")


# Step by step commands...
# 
# In addition to your training data, there will be test data. This is frequently stored in a file with the title `test.csv`. This data won't include a column with your target (y), because that is what we'll have to predict and submit.  Here is sample code to do that. 

# In[29]:


# Read the test data
test = pd.read_csv('https://raw.githubusercontent.com/serivan/mldmlab/master/Datasets/Kaggle2020/test.csv')


# Treat the test data in the same way as training data. In this case, pull same columns.
test_X = test[predictor_cols]

# Impute each test item, then predict
test_X_imp = imp.transform(test_X)

# Use the model to make predictions
predicted_q = my_model.predict(test_X_imp)
# We will look at the predicted Qualities to ensure we have something sensible.
print(predicted_q)


# # Prepare Submission File
# We make submissions in CSV files.  Your submissions usually have two columns: an ID column and a prediction column.  The ID field comes from the test data (keeping whatever name the ID field had in that data, which for the housing data is the string 'Id'). The prediction column will use the name of the target field.
# 
# We will create a DataFrame with this data, and then use the dataframe's `to_csv` method to write our submission file.  Explicitly include the argument `index=False` to prevent pandas from adding another column in our csv file.

# In[30]:


my_submission = pd.DataFrame({'Id': test.Id, 'Quality': predicted_q})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)


# # Make Submission
# Hit the blue **Publish** button at the top of your notebook screen.  It will take some time for your kernel to run.  When it has finished your navigation bar at the top of the screen will have a tab for **Output**.  This only shows up if you have written an output file (like we did in the **Prepare Submission File** step).  
# 
# Otherwise, if you have a kaggle API token (https://www.kaggle.com/docs/api#getting-started-installation-&-authentication), you can use the following command (remove the '#'):

# In[31]:


# !kaggle competitions submit -c mldm-classification-competition-2020 -f submission.csv -m "Please describe the technique used"


# 
# # Last Steps 
# Click on the Output button.  This will bring you to a screen with an option to **Submit to Competition**.  Hit that and you will see how your model performed.
# 
# If you want to go back to improve your model, click the Edit button, which re-opens the kernel.  You'll need to re-run all the cells when you re-open the kernel.
# 
# # Conclusion
# You've completed Level 1 of Machine Learning.  Congrats.  
# 
# If you are ready to keep improving your model (and your skills), start level 2 of [Learn Machine Learning](https://www.kaggle.com/learn/machine-learning). 
# 
# Level 2 covers more powerful models, techniques to include non-numeric data, and more.  You can make more submissions to the competition and climb up the leaderboard as you go through the course.
# 
# 
# 
