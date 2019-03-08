
# coding: utf-8

# This lab on Support Vector Machines is a Python adaptation of p. 359-366 of "Introduction to Statistical
# Learning with Applications in R" by Gareth James, Daniela Witten, Trevor Hastie and Robert Tibshirani. Original
# adaptation by J. Warmenhoven, updated by R. Jordan Crouser at Smith College for SDS293: Machine
# Learning (Spring 2016).




import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# We'll define a function to draw a nice plot of an SVM
def plot_svc(svc, X, y, h=0.02, pad=0.25):
    x_min, x_max = X[:, 0].min()-pad, X[:, 0].max()+pad
    y_min, y_max = X[:, 1].min()-pad, X[:, 1].max()+pad
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.2)

    plt.scatter(X[:,0], X[:,1], s=70, c=y, cmap=mpl.cm.Paired)
    # Support vectors indicated in plot by vertical lines
    sv = svc.support_vectors_
    plt.scatter(sv[:,0], sv[:,1], c='k', marker='x', s=100, linewidths='1')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
    print('Number of support vectors: ', svc.support_.size)


# # 9.6 Lab: Support Vector Machines
# 
# In this lab, we'll use the ${\tt SVC}$ module from the ${\tt sklearn.svm}$ package to demonstrate the support vector classifier
# and the SVM:




from sklearn.svm import SVC


# # 9.6.1 Support Vector Classifier
# 
# The ${\tt SVC()}$ function can be used to fit a
# support vector classifier when the argument ${\tt kernel="linear"}$ is used. This
# function uses a slightly different formulation of the equations we saw in lecture to build the
# support vector classifier. The ${\tt c}$ argument allows us to specify the cost of
# a violation to the margin. When the ${\tt c}$ argument is **small**, then the margins
# will be wide and many support vectors will be on the margin or will
# violate the margin. When the ${\tt c}$ argument is large, then the margins will
# be narrow and there will be few support vectors on the margin or violating
# the margin.
# 
# We can use the ${\tt SVC()}$ function to fit the support vector classifier for a
# given value of the ${\tt cost}$ parameter. Here we demonstrate the use of this
# function on a two-dimensional example so that we can plot the resulting
# decision boundary. Let's start by generating a set of observations, which belong
# to two classes:




# Generating random data: 20 observations of 2 features and divide into two classes.
np.random.seed(5)
X = np.random.randn(20,2)
y = np.repeat([1,-1], 10)

X[y == -1] = X[y == -1]+1


# Let's plot the data to see whether the classes are linearly separable:




plt.scatter(X[:,0], X[:,1], s=70, c=y, cmap=mpl.cm.Paired)
plt.xlabel('X1')
plt.ylabel('X2')


# Nope; not linear. Next, we fit the support vector classifier:




svc = SVC(C=1, kernel='linear')
svc.fit(X, y)


# We can now plot the support vector classifier by calling the ${\tt plot\_svc()}$ function on the output of the call to ${\tt SVC()}$, as well as the data used in the call to ${\tt SVC()}$:




plot_svc(svc, X, y)


# The region of feature space that will be assigned to the −1 class is shown in
# light blue, and the region that will be assigned to the +1 class is shown in
# brown. The decision boundary between the two classes is linear (because we
# used the argument ${\tt kernel="linear"}$).
# 
# The support vectors are plotted with crosses
# and the remaining observations are plotted as circles; we see here that there
# are 13 support vectors. We can determine their identities as follows:




svc.support_


# What if we instead used a smaller value of the ${\tt cost}$ parameter?




svc2 = SVC(C=0.1, kernel='linear')
svc2.fit(X, y)
plot_svc(svc2, X, y)


# Now that a smaller value of the ${\tt c}$ parameter is being used, we obtain a
# larger number of support vectors, because the margin is now **wider**.
# 
# The ${\tt sklearn.grid\_search}$ module includes a a function ${\tt GridSearchCV()}$ to perform cross-validation. In order to use this function, we pass in relevant information about the set of models that are under consideration. The
# following command indicates that we want perform 10-fold cross-validation to compare SVMs with a linear
# kernel, using a range of values of the cost parameter:




from sklearn.model_selection import GridSearchCV

# Select the optimal C parameter by cross-validation
tuned_parameters = [{'C': [0.001, 0.01, 0.1, 1, 5, 10, 100]}]
clf = GridSearchCV(SVC(kernel='linear'), tuned_parameters, cv=10, scoring='accuracy')
clf.fit(X, y)


# We can easily access the cross-validation errors for each of these models:




clf.cv_results_


# The ${\tt GridSearchCV()}$ function stores the best parameters obtained, which can be accessed as
# follows:




clf.best_params_


# c=0.001 is best according to ${\tt GridSearchCV}$. 
# 
# As usual, the ${\tt predict()}$ function can be used to predict the class label on a set of
# test observations, at any given value of the cost parameter. Let's
# generate a test data set:




np.random.seed(1)
X_test = np.random.randn(20,2)
y_test = np.random.choice([-1,1], 20)
X_test[y_test == 1] = X_test[y_test == 1]-1


# Now we predict the class labels of these test observations. Here we use the
# best model obtained through cross-validation in order to make predictions:




svc2 = SVC(C=0.001, kernel='linear')
svc2.fit(X, y)
y_pred = svc2.predict(X_test)
pd.DataFrame(confusion_matrix(y_test, y_pred), index=svc2.classes_, columns=svc2.classes_)


# With this value of ${\tt c}$, 14 of the test observations are correctly
# classified.

# Now consider a situation in which the two classes are linearly separable.
# Then we can find a separating hyperplane using the ${\tt svm()}$ function. First we'll give our simulated data a little nudge so that they are linearly separable:




X_test[y_test == 1] = X_test[y_test == 1] -1
plt.scatter(X_test[:,0], X_test[:,1], s=70, c=y_test, cmap=mpl.cm.Paired)
plt.xlabel('X1')
plt.ylabel('X2')


# Now the observations are **just barely linearly** separable. We fit the support
# vector classifier and plot the resulting hyperplane, using a very large value
# of ${\tt cost}$ so that no observations are misclassified.




svc3 = SVC(C=1e5, kernel='linear')
svc3.fit(X_test, y_test)
plot_svc(svc3, X_test, y_test)


# No training errors were made and only three support vectors were used.
# However, we can see from the figure that the margin is very narrow (because
# the observations that are **not** support vectors, indicated as circles, are very close to the decision boundary). It seems likely that this model will perform
# poorly on test data. Let's try a smaller value of ${\tt cost}$:




svc4 = SVC(C=1, kernel='linear')
svc4.fit(X_test, y_test)
plot_svc(svc4, X_test, y_test)


# Using ${\tt cost=1}$, we misclassify a training observation, but we also obtain
# a much wider margin and make use of five support vectors. It seems
# likely that this model will perform better on test data than the model with
# ${\tt cost=1e5}$.
# 
# # 9.6.2 Support Vector Machine
# 
# In order to fit an SVM using a **non-linear kernel**, we once again use the ${\tt SVC()}$
# function. However, now we use a different value of the parameter kernel.
# To fit an SVM with a polynomial kernel we use ${\tt kernel="poly"}$, and
# to fit an SVM with a radial kernel we use ${\tt kernel="rbf"}$. In the former
# case we also use the ${\tt degree}$ argument to specify a degree for the polynomial
# kernel, and in the latter case we use ${\tt gamma}$ to specify a
# value of $\gamma$ for the radial basis kernel.
# 
# Let's generate some data with a non-linear class boundary:




from sklearn.model_selection import train_test_split

np.random.seed(8)
X = np.random.randn(200,2)
X[:100] = X[:100] +2
X[101:150] = X[101:150] -2
y = np.concatenate([np.repeat(-1, 150), np.repeat(1,50)])

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=2)

plt.scatter(X[:,0], X[:,1], s=70, c=y, cmap=mpl.cm.Paired)
plt.xlabel('X1')
plt.ylabel('X2')


# See how one class is kind of stuck in the middle of another class? This suggests that we might want to use a **radial kernel** in our SVM. Now let's fit
# the training data using the ${\tt SVC()}$ function with a radial kernel and $\gamma = 1$:




svm = SVC(C=1.0, kernel='rbf', gamma=1)
svm.fit(X_train, y_train)
plot_svc(svm, X_test, y_test)


# Not too shabby! The plot shows that the resulting SVM has a decidedly non-linear
# boundary. We can see from the figure that there are a fair number of training errors
# in this SVM fit. If we increase the value of cost, we can reduce the number
# of training errors:




# Increasing C parameter, allowing more flexibility
svm2 = SVC(C=100, kernel='rbf', gamma=1.0)
svm2.fit(X_train, y_train)
plot_svc(svm2, X_test, y_test)


# However, this comes at the price of a more irregular decision boundary that seems to be at risk of overfitting the data. We can perform cross-validation using ${\tt GridSearchCV()}$ to select the best choice of
# $\gamma$ and cost for an SVM with a radial kernel:




tuned_parameters = [{'C': [0.01, 0.1, 1, 10, 100],
                     'gamma': [0.5, 1,2,3,4]}]
clf = GridSearchCV(SVC(kernel='rbf'), tuned_parameters, cv=10, scoring='accuracy')
clf.fit(X_train, y_train)
clf.best_params_


# Therefore, the best choice of parameters involves ${\tt cost=1}$ and ${\tt gamma=0.5}$. We
# can plot the resulting fit using the ${\tt plot\_svc()}$ function, and view the test set predictions for this model by applying the ${\tt predict()}$
# function to the test data:




plot_svc(clf.best_estimator_, X_test, y_test)
print(confusion_matrix(y_test, clf.best_estimator_.predict(X_test)))
print(clf.best_estimator_.score(X_test, y_test))


# 85% of test observations are correctly classified by this SVM. Not bad!
# 
# # 9.6.3 ROC Curves
# 
# The ${\tt auc()}$ function from the ${\tt sklearn.metrics}$ package can be used to produce ROC curves such as those we saw in lecture:




from sklearn.metrics import auc
from sklearn.metrics import roc_curve


# Let's start by fitting two models, one more flexible than the other:




# More constrained model
svm3 = SVC(C=1, kernel='rbf', gamma=1)
svm3.fit(X_train, y_train)





# More flexible model
svm4 = SVC(C=1, kernel='rbf', gamma=50)
svm4.fit(X_train, y_train)


# SVMs and support vector classifiers output class labels for each observation.
# However, it is also possible to obtain fitted values for each observation,
# which are the numerical scores used to obtain the class labels. For instance,
# in the case of a support vector classifier, the fitted value for an observation
# $X = (X_1,X_2, . . .,X_p)^T$ takes the form $\hat\beta_0 + \hat\beta_1X_1 + \hat\beta_2X_2 + . . . + \hat\beta_pX_p$.
# 
# For an SVM with a non-linear kernel, the equation that yields the fitted
# value is given in (9.23) on p. 352 of the ISLR book. In essence, the sign of the fitted value determines
# on which side of the decision boundary the observation lies. Therefore, the
# relationship between the fitted value and the class prediction for a given
# observation is simple: if the fitted value exceeds zero then the observation
# is assigned to one class, and if it is less than zero than it is assigned to the
# other.
# 
# In order to obtain the fitted values for a given SVM model fit, we
# use the ${\tt .decision\_function()}$ method of the SVC:




y_train_score3 = svm3.decision_function(X_train)
y_train_score4 = svm4.decision_function(X_train)


# Now we can produce the ROC plot to see how the models perform on both the training and the test data:




y_train_score3 = svm3.decision_function(X_train)
y_train_score4 = svm4.decision_function(X_train)

false_pos_rate3, true_pos_rate3, _ = roc_curve(y_train, y_train_score3)
roc_auc3 = auc(false_pos_rate3, true_pos_rate3)

false_pos_rate4, true_pos_rate4, _ = roc_curve(y_train, y_train_score4)
roc_auc4 = auc(false_pos_rate4, true_pos_rate4)

fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(14,6))
ax1.plot(false_pos_rate3, true_pos_rate3, label='SVM $\gamma = 1$ ROC curve (area = %0.2f)' % roc_auc3, color='b')
ax1.plot(false_pos_rate4, true_pos_rate4, label='SVM $\gamma = 50$ ROC curve (area = %0.2f)' % roc_auc4, color='r')
ax1.set_title('Training Data')

y_test_score3 = svm3.decision_function(X_test)
y_test_score4 = svm4.decision_function(X_test)

false_pos_rate3, true_pos_rate3, _ = roc_curve(y_test, y_test_score3)
roc_auc3 = auc(false_pos_rate3, true_pos_rate3)

false_pos_rate4, true_pos_rate4, _ = roc_curve(y_test, y_test_score4)
roc_auc4 = auc(false_pos_rate4, true_pos_rate4)

ax2.plot(false_pos_rate3, true_pos_rate3, label='SVM $\gamma = 1$ ROC curve (area = %0.2f)' % roc_auc3, color='b')
ax2.plot(false_pos_rate4, true_pos_rate4, label='SVM $\gamma = 50$ ROC curve (area = %0.2f)' % roc_auc4, color='r')
ax2.set_title('Test Data')

for ax in fig.axes:
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([-0.05, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")


# # 9.6.4 SVM with Multiple Classes
# 
# If the response is a factor containing more than two levels, then the ${\tt svm()}$
# function will perform multi-class classification using the one-versus-one approach.
# We explore that setting here by generating a third class of observations:




np.random.seed(8)
XX = np.vstack([X, np.random.randn(50,2)])
yy = np.hstack([y, np.repeat(0,50)])
XX[yy ==0] = XX[yy == 0] +4

plt.scatter(XX[:,0], XX[:,1], s=70, c=yy, cmap=plt.cm.prism)
plt.xlabel('XX1')
plt.ylabel('XX2')


# Fitting an SVM to multiclass data uses identical syntax to fitting a simple two-class model:




svm5 = SVC(C=1, kernel='rbf')
svm5.fit(XX, yy)
plot_svc(svm5, XX, yy)


# # Application to Handwritten Letter Data
# 
# We now examine [`Optical Recognition of Handwritten Digits Data Set`](http://archive.ics.uci.edu/ml/datasets/optical+recognition+of+handwritten+digits), which contains 5,620 samples of handwritten digits 0..9. You can use these links to download the [training data](http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra) and [test data](http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes), and then we'll load them into python:




X_train = pd.read_csv('optdigits.tra', header=None)
y_train = X_train[64]
X_train = X_train.drop(X_train.columns[64], axis=1)

X_test = pd.read_csv('optdigits.tes', header=None)
y_test = X_test[64]
X_test = X_test.drop(X_test.columns[64], axis=1)


# Let's take a look at the dimensions of this dataset:




print(X_train.shape)
print(X_test.shape)


# This data set consists of preprocessed images of handwriting samples gathered from 43 different people. Each image was converted into an 8x8 matrix (64 pixels), which was then flattened into a vector of 64 numeric values. The final column contains the class label for each digit.
# 
# The training and test sets consist of 3,823 and 1,797 observations respectively. Let's see what one of these digits looks like:




plt.imshow(X_train.values[1].reshape(8,8), cmap="gray") 
plt.show()


# That's a pretty messy digit. Let's peek at the true class:




y_train[0]


# Phew, looks like our SVM has its work cut out for it! Let's start with a linear kernel to see how we do:




svc = SVC(kernel='linear')
svc.fit(X_train, y_train)

# Print a nice confusion matrix
cm = confusion_matrix(y_train, svc.predict(X_train))
cm_df = pd.DataFrame(cm.T, index=svc.classes_, columns=svc.classes_)
print(cm_df)


# We see that there are **no training errors**. In fact, this is not surprising,
# because the large number of variables relative to the number of observations
# implies that it is easy to find hyperplanes that fully separate the classes. We
# are most interested not in the support vector classifier’s performance on the
# training observations, but rather its performance on the test observations:




cm = confusion_matrix(y_test, svc.predict(X_test))
print(pd.DataFrame(cm.T, index=svc.classes_, columns=svc.classes_))


# We see that using `cost = 10` yields just 71 test set errors on this data. Now try using the ${\tt GridSearchCV()}$ function to select an optimal value for ${\tt c}$. Consider values in the range 0.01 to 100:




# Your code here


# To get credit for this lab, report your optimal values and comment on your final model's performance on Moodle: https://moodle.smith.edu/mod/quiz/view.php?id=266457
