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
if not 'sklearn' in sys.modules.keys():
    pip.main(['install', 'sklearn'])
#if not 'kaggle' in sys.modules.keys():
#    pip.main(['install', 'kaggle'])


# In[2]:


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# Read the data
train = pd.read_csv('https://raw.githubusercontent.com/serivan/mldmlab/master/Datasets/Kaggle-Wine-train.csv')

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

my_model = RandomForestClassifier(n_estimators=100)
my_model.fit(train_X_imp, train_y)


# In[3]:


# The snippet below will retrieve the feature importances from the model and make them into a DataFrame.
feature_importances = pd.DataFrame(my_model.feature_importances_,
                                   index = train_X.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)
feature_importances


# In addition to your training data, there will be test data. This is frequently stored in a file with the title `test.csv`. This data won't include a column with your target (y), because that is what we'll have to predict and submit.  Here is sample code to do that. 

# In[4]:


# Read the test data
test = pd.read_csv('https://raw.githubusercontent.com/serivan/mldmlab/master/Datasets/Kaggle-Wine-test.csv')
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

# In[5]:


my_submission = pd.DataFrame({'Id': test.Id, 'Quality': predicted_q})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)


# # Make Submission
# Hit the blue **Publish** button at the top of your notebook screen.  It will take some time for your kernel to run.  When it has finished your navigation bar at the top of the screen will have a tab for **Output**.  This only shows up if you have written an output file (like we did in the **Prepare Submission File** step).  
# 
# Otherwise, if you have a kaggle API token, you can use the following command (remove the '#'):

# In[6]:


# !kaggle competitions submit -c unibs-mldm-classification -f submission.csv -m "Please descrive the technique used"


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
