
# coding: utf-8

# This lab on Decision Trees is a Python adaptation of p. 324-331 of "Introduction to Statistical Learning with
# Applications in R" by Gareth James, Daniela Witten, Trevor Hastie and Robert Tibshirani. Original adaptation by J. Warmenhoven, updated by R. Jordan Crouser at Smith
# College for SDS293: Machine Learning (Spring 2016).

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import graphviz


# # 8.3.1 Fitting Classification Trees
# 
# The `sklearn` library has a lot of useful tools for constructing classification and regression trees:

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import confusion_matrix, mean_squared_error


# We'll start by using **classification trees** to analyze the `Carseats` data set. In these
# data, `Sales` is a continuous variable, and so we begin by converting it to a
# binary variable. We use the `ifelse()` function to create a variable, called
# `High`, which takes on a value of `Yes` if the `Sales` variable exceeds 8, and
# takes on a value of `No` otherwise. We'll append this onto our dataFrame using the `.map()` function, and then do a little data cleaning to tidy things up:

df3 = pd.read_csv('Carseats.csv').drop('Unnamed: 0', axis=1)
df3['High'] = df3.Sales.map(lambda x: 1 if x>8 else 0)
df3.ShelveLoc = pd.factorize(df3.ShelveLoc)[0]
df3.Urban = df3.Urban.map({'No':0, 'Yes':1})
df3.US = df3.US.map({'No':0, 'Yes':1})
df3.info()


# In order to properly evaluate the performance of a classification tree on
# the data, we must estimate the test error rather than simply computing
# the training error. We first split the observations into a training set and a test
# set:

X = df3.drop(['Sales', 'High'], axis = 1)
y = df3.High

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state = 0)


# We now use the `DecisionTreeClassifier()` function to fit a classification tree in order to predict
# `High`. Unfortunately, manual pruning is not implemented in `sklearn`: http://scikit-learn.org/stable/modules/tree.html
# 
# However, we can limit the depth of a tree using the `max_depth` parameter:

classification_tree_carseats = DecisionTreeClassifier(max_depth = 6)
classification_tree_carseats.fit(X_train, y_train)
classification_tree_carseats.score(X_train, y_train)


# We see that the training accuracy is 92.2%.
# 
# One of the most attractive properties of trees is that they can be
# graphically displayed. Unfortunately, this is a bit of a roundabout process in `sklearn`. We use the `export_graphviz()` function to export the tree structure to a temporary `.dot` file,
# and the `graphviz.Source()` function to display the image:

export_graphviz(classification_tree_carseats, 
                out_file = "carseat_tree.dot", 
                feature_names = X_train.columns)

with open("carseat_tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)


# The most important indicator of `High` sales appears to be `Price`.

# Finally, let's evaluate the tree's performance on
# the test data. The `predict()` function can be used for this purpose. We can then build a confusion matrix, which shows that we are making correct predictions for
# around 72.5% of the test data set:

pred = classification_tree_carseats.predict(X_test)
cm = pd.DataFrame(confusion_matrix(y_test, pred).T, 
                  index = ['No', 'Yes'], 
                  columns = ['No', 'Yes'])
print(cm)
# (37+21)/80 = 0.745


# # 8.3.2 Fitting Regression Trees
# 
# Now let's try fitting a **regression tree** to the `Boston` data set from the `MASS` library. First, we create a
# training set, and fit the tree to the training data using `medv` (median home value) as our response:

boston_df = pd.read_csv('Boston.csv')
X = boston_df.drop('medv', axis = 1)
y = boston_df.medv
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state = 0)

# Pruning not supported. Choosing max depth 2)
regr_tree_boston = DecisionTreeRegressor(max_depth = 2)
regr_tree_boston.fit(X_train, y_train)


# Let's take a look at the tree:

export_graphviz(regr_tree_boston, 
                out_file = "boston_tree.dot", 
                feature_names = X_train.columns)

with open("boston_tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)


# The variable `lstat` measures the percentage of individuals with lower
# socioeconomic status. The tree indicates that lower values of `lstat` correspond
# to more expensive houses. The tree predicts a median house price
# of \$45,766 for larger homes (`rm>=7.4351`) in suburbs in which residents have high socioeconomic
# status (`lstat<7.81`).
# 
# Now let's see how it does on the test data:

pred = regr_tree_boston.predict(X_test)

plt.scatter(pred, 
            y_test, 
            label = 'medv')

plt.plot([0, 1], 
         [0, 1], 
         '--k', 
         transform = plt.gca().transAxes)

plt.xlabel('pred')
plt.ylabel('y_test')

mean_squared_error(y_test, pred)


# The test set MSE associated with the regression tree is
# 35.4. The square root of the MSE is therefore around 5.95, indicating
# that this model leads to test predictions that are within around \$5,950 of
# the true median home value for the suburb.
#     
# # 8.3.3 Bagging and Random Forests
# 
# Let's see if we can improve on this result using **bagging** and **random forests**. The exact results obtained in this section may
# depend on the version of `python` and the version of the `RandomForestRegressor` package
# installed on your computer, so don't stress out if you don't match up exactly with the book. Recall that **bagging** is simply a special case of
# a **random forest** with $m = p$. Therefore, the `RandomForestRegressor()` function can
# be used to perform both random forests and bagging. Let's start with bagging:

# Bagging: using all features
bagged_boston = RandomForestRegressor(max_features = 13, random_state = 1)
bagged_boston.fit(X_train, y_train)


# The argument `max_features = 13` indicates that all 13 predictors should be considered
# for each split of the tree -- in other words, that bagging should be done. How
# well does this bagged model perform on the test set?

pred = bagged_boston.predict(X_test)

plt.scatter(pred, 
            y_test, 
            label = 'medv')

plt.plot([0, 1], 
         [0, 1], 
         '--k', 
         transform = plt.gca().transAxes)

plt.xlabel('pred')
plt.ylabel('y_test')

mean_squared_error(y_test, pred)


# The test set MSE associated with the bagged regression tree is significantly lower than our single tree!

# We can grow a random forest in exactly the same way, except that
# we'll use a smaller value of the `max_features` argument. Here we'll
# use `max_features = 6`:

# Random forests: using 6 features
random_forest_boston = RandomForestRegressor(max_features = 6, random_state = 1)

random_forest_boston.fit(X_train, y_train)

pred = random_forest_boston.predict(X_test)
mean_squared_error(y_test, pred)


# The test set MSE is even lower; this indicates that random forests yielded an
# improvement over bagging in this case.
# 
# Using the `feature_importances_` attribute of the `RandomForestRegressor`, we can view the importance of each
# variable:

Importance = pd.DataFrame({'Importance':random_forest_boston.feature_importances_*100}, 
                          index = X.columns)

Importance.sort_values(by = 'Importance', 
                       axis = 0, 
                       ascending = True).plot(kind = 'barh', 
                                              color = 'r', )

plt.xlabel('Variable Importance')
plt.gca().legend_ = None


# The results indicate that across all of the trees considered in the random
# forest, the wealth level of the community (`lstat`) and the house size (`rm`)
# are by far the two most important variables.

# # 8.3.4 Boosting
# 
# Now we'll use the `GradientBoostingRegressor` package to fit **boosted
# regression trees** to the `Boston` data set. The
# argument `n_estimators = 500` indicates that we want 500 trees, and the option
# `interaction.depth = 4` limits the depth of each tree:

boosted_boston = GradientBoostingRegressor(n_estimators = 500, 
                                           learning_rate = 0.01, 
                                           max_depth = 4, 
                                           random_state = 1)

boosted_boston.fit(X_train, y_train)


# Let's check out the feature importances again:

feature_importance = boosted_boston.feature_importances_*100

rel_imp = pd.Series(feature_importance, 
                    index = X.columns).sort_values(inplace = False)

rel_imp.T.plot(kind = 'barh', 
               color = 'r', )

plt.xlabel('Variable Importance')

plt.gca().legend_ = None


# We see that `lstat` and `rm` are again the most important variables by far. Now let's use the boosted model to predict `medv` on the test set:

mean_squared_error(y_test, boosted_boston.predict(X_test))


# The test MSE obtained is similar to the test MSE for random forests
# and superior to that for bagging. If we want to, we can perform boosting
# with a different value of the shrinkage parameter $\lambda$. Here we take $\lambda = 0.2$:

boosted_boston2 = GradientBoostingRegressor(n_estimators = 500, 
                                            learning_rate = 0.2, 
                                            max_depth = 4, 
                                            random_state = 1)
boosted_boston2.fit(X_train, y_train)

mean_squared_error(y_test, boosted_boston2.predict(X_test))


# In this case, using $\lambda = 0.2$ leads to a slightly lower test MSE than $\lambda = 0.01$.
# 
# To get credit for this lab, post your responses to the following questions:
#  - What's one real-world scenario where you might try using Bagging?
#  - What's one real-world scenario where you might try using Random Forests?
#  - What's one real-world scenario where you might try using Boosting?
#  
# to Moodle: https://moodle.smith.edu/mod/quiz/view.php?id=264671
