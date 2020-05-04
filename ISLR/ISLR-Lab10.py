
# coding: utf-8

# This lab on Ridge Regression and the Lasso is a Python adaptation of p. 251-255 of "Introduction to Statistical Learning with Applications in R" by Gareth James, Daniela Witten, Trevor Hastie and Robert Tibshirani. Adapted by R. Jordan Crouser at Smith College for SDS293: Machine Learning (Spring 2016).
# 
# # 6.6: Ridge Regression and the Lasso

# In[ ]:

get_ipython().magic('matplotlib inline')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.metrics import mean_squared_error


# We will use the `sklearn` package in order to perform ridge regression and
# the lasso. The main functions in this package that we care about are `Ridge()`, which can be used
# to fit ridge regression models, and `Lasso()` which will fit lasso models. They also have cross-validated counterparts: `RidgeCV()` and `LassoCV()`. We'll use these a bit later.
# 
# Before proceeding, let's first ensure that the missing values have
# been removed from the data, as described in the previous lab.

# In[ ]:

df = pd.read_csv('Hitters.csv').dropna().drop('Player', axis = 1)
df.info()
dummies = pd.get_dummies(df[['League', 'Division', 'NewLeague']])


# We will now perform ridge regression and the lasso in order to predict `Salary` on
# the `Hitters` data. Let's set up our data:

# In[ ]:

y = df.Salary

# Drop the column with the independent variable (Salary), and columns for which we created dummy variables
X_ = df.drop(['Salary', 'League', 'Division', 'NewLeague'], axis = 1).astype('float64')

# Define the feature set X.
X = pd.concat([X_, dummies[['League_N', 'Division_W', 'NewLeague_N']]], axis = 1)

X.info()


# # 6.6.1 Ridge Regression
# The `Ridge()` function has an alpha argument ($\lambda$, but with a different name!) that is used to tune the model. We'll generate an array of alpha values ranging from very big to very small, essentially
# covering the full range of scenarios from the null model containing
# only the intercept, to the least squares fit:

# In[ ]:

alphas = 10**np.linspace(10,-2,100)*0.5
alphas


# Associated with each alpha value is a vector of ridge regression coefficients, which we'll
# store in a matrix `coefs`. In this case, it is a $19 \times 100$
# matrix, with 19 rows (one for each predictor) and 100
# columns (one for each value of alpha). Remember that we'll want to standardize the
# variables so that they are on the same scale. To do this, we can use the
# `normalize = True` parameter:

# In[ ]:

ridge = Ridge(normalize = True)
coefs = []

for a in alphas:
    ridge.set_params(alpha = a)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)
    
np.shape(coefs)


# We expect the coefficient estimates to be much smaller, in terms of $l_2$ norm,
# when a large value of alpha is used, as compared to when a small value of alpha is
# used. Let's plot and find out:

# In[ ]:

ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')


# We now split the samples into a training set and a test set in order
# to estimate the test error of ridge regression and the lasso:

# In[ ]:

# Split data into training and test sets
X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)


# Next we fit a ridge regression model on the training set, and evaluate
# its MSE on the test set, using $\lambda = 4$:

# In[ ]:

ridge2 = Ridge(alpha = 4, normalize = True)
ridge2.fit(X_train, y_train)             # Fit a ridge regression on the training data
pred2 = ridge2.predict(X_test)           # Use this model to predict the test data
print(pd.Series(ridge2.coef_, index = X.columns)) # Print coefficients
print(mean_squared_error(y_test, pred2))          # Calculate the test MSE


# The test MSE when alpha = 4 is 106216. Now let's see what happens if we use a huge value of alpha, say $10^{10}$:

# In[ ]:

ridge3 = Ridge(alpha = 10**10, normalize = True)
ridge3.fit(X_train, y_train)             # Fit a ridge regression on the training data
pred3 = ridge3.predict(X_test)           # Use this model to predict the test data
print(pd.Series(ridge3.coef_, index = X.columns)) # Print coefficients
print(mean_squared_error(y_test, pred3))          # Calculate the test MSE


# This big penalty shrinks the coefficients to a very large degree, essentially reducing to a model containing just the intercept. This over-shrinking makes the model more biased, resulting in a higher MSE.

# Okay, so fitting a ridge regression model with alpha = 4 leads to a much lower test
# MSE than fitting a model with just an intercept. We now check whether
# there is any benefit to performing ridge regression with alpha = 4 instead of
# just performing least squares regression. Recall that least squares is simply
# ridge regression with alpha = 0.

# In[ ]:

ridge2 = Ridge(alpha = 0, normalize = True)
ridge2.fit(X_train, y_train)             # Fit a ridge regression on the training data
pred = ridge2.predict(X_test)            # Use this model to predict the test data
print(pd.Series(ridge2.coef_, index = X.columns)) # Print coefficients
print(mean_squared_error(y_test, pred))           # Calculate the test MSE


# It looks like we are indeed improving over regular least-squares!
# 
# Instead of arbitrarily choosing alpha $ = 4$, it would be better to
# use cross-validation to choose the tuning parameter alpha. We can do this using
# the cross-validated ridge regression function, `RidgeCV()`. By default, the function
# performs generalized cross-validation (an efficient form of LOOCV), though this can be changed using the
# argument `cv`.

# In[ ]:

ridgecv = RidgeCV(alphas = alphas, scoring = 'neg_mean_squared_error', normalize = True)
ridgecv.fit(X_train, y_train)
ridgecv.alpha_


# Therefore, we see that the value of alpha that results in the smallest cross-validation
# error is 0.57. What is the test MSE associated with this value of
# alpha?

# In[ ]:

ridge4 = Ridge(alpha = ridgecv.alpha_, normalize = True)
ridge4.fit(X_train, y_train)
mean_squared_error(y_test, ridge4.predict(X_test))


# This represents a further improvement over the test MSE that we got using
# alpha $ = 4$. Finally, we refit our ridge regression model on the full data set,
# using the value of alpha chosen by cross-validation, and examine the coefficient
# estimates.

# In[ ]:

ridge4.fit(X, y)
pd.Series(ridge4.coef_, index = X.columns)


# As expected, none of the coefficients are exactly zero - ridge regression does not
# perform variable selection!
# 
# # 6.6.2 The Lasso
# We saw that ridge regression with a wise choice of alpha can outperform least
# squares as well as the null model on the Hitters data set. We now ask
# whether the lasso can yield either a more accurate or a more interpretable
# model than ridge regression. In order to fit a lasso model, we'll
# use the `Lasso()` function; however, this time we'll need to include the argument `max_iter = 10000`.
# Other than that change, we proceed just as we did in fitting a ridge model:

# In[ ]:

lasso = Lasso(max_iter = 10000, normalize = True)
coefs = []

for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(scale(X_train), y_train)
    coefs.append(lasso.coef_)
    
ax = plt.gca()
ax.plot(alphas*2, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')


# Notice that in the coefficient plot that depending on the choice of tuning
# parameter, some of the coefficients are exactly equal to zero. We now
# perform 10-fold cross-validation to choose the best alpha, refit the model, and compute the associated test error:

# In[ ]:

lassocv = LassoCV(alphas = None, cv = 10, max_iter = 100000, normalize = True)
lassocv.fit(X_train, y_train)

lasso.set_params(alpha=lassocv.alpha_)
lasso.fit(X_train, y_train)
mean_squared_error(y_test, lasso.predict(X_test))


# This is substantially lower than the test set MSE of the null model and of
# least squares, and only a little worse than the test MSE of ridge regression with alpha
# chosen by cross-validation.
# 
# However, the lasso has a substantial advantage over ridge regression in
# that the resulting coefficient estimates are sparse. Here we see that 13 of
# the 19 coefficient estimates are exactly zero:

# In[ ]:

# Some of the coefficients are now reduced to exactly zero.
pd.Series(lasso.coef_, index=X.columns)


# # Your turn!
# Now it's time to test out these approaches (ridge regression and the lasso) and evaluation methods (validation set, cross validation) on other datasets. You may want to work with a team on this portion of the lab.
# You may use any of the datasets included in ISLR, or choose one from the UCI machine learning repository (http://archive.ics.uci.edu/ml/datasets.html). Download a dataset, and try to determine the optimal set of parameters to use to model it! You are free to use the same dataset you used in Lab 9, or you can choose a new one.

# In[ ]:

# Your code here


# To get credit for this lab, post your responses to the following questions:
#  - Which dataset did you choose?
#  - What was your response variable (i.e. what were you trying to model)?
#  - Did you expect ridge regression to outperform the lasso, or vice versa?
#  - Which predictors turned out to be important in the final model(s)?
#  
# to Moodle: https://moodle.smith.edu/mod/quiz/view.php?id=259464
