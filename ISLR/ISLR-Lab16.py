
# coding: utf-8

# This lab on K-Means and Hierarchical Clustering in R is an adaptation of p. 404-407, 410-413 of "Introduction to
# Statistical Learning with Applications in R" by Gareth James, Daniela Witten, Trevor Hastie and Robert
# Tibshirani. Initial python implementation by [Hyun Bong Lee](https://github.com/hyunblee), adapted by R. Jordan Crouser at Smith College for SDS293: Machine Learning (Fall 2017).
# 
# 
# # 10.5.1 K-Means Clustering
# The `sklearn` function `Kmeans()` performs K-means clustering in R. We begin with
# a simple simulated example in which there truly are two clusters in the
# data: the first 25 observations have a mean shift relative to the next 25
# observations.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)
X = np.random.randn(50,2)
X[0:25, 0] = X[0:25, 0] + 3
X[0:25, 1] = X[0:25, 1] - 4

f, ax = plt.subplots(figsize=(6, 5))
ax.scatter(X[:,0], X[:,1], s=50) 
ax.set_xlabel('X0')
ax.set_ylabel('X1')


# We now perform K-means clustering with `K = 2`:




from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 2, random_state = 123).fit(X)


# The cluster assignments of the 50 observations are contained in
# `kmeans.labels_`:




print(kmeans.labels_)


# The K-means clustering perfectly separated the observations into two clusters
# even though we did not supply any group information to `Kmeans()`. We
# can plot the data, with each observation colored according to its cluster
# assignment:




plt.figure(figsize=(6,5))
plt.scatter(X[:,0], X[:,1], s = 50, c = kmeans.labels_, cmap = plt.cm.bwr) 
plt.scatter(kmeans.cluster_centers_[:, 0], 
            kmeans.cluster_centers_[:, 1], 
            marker = '*', 
            s = 150,
            color = 'cyan', 
            label = 'Centers')
plt.legend(loc = 'best')
plt.xlabel('X0')
plt.ylabel('X1')


# Here the observations can be easily plotted because they are two-dimensional.
# If there were more than two variables then we could instead perform PCA
# and plot the first two principal components score vectors.
# 
# In this example, we knew that there really were two clusters because
# we generated the data. However, for real data, in general we do not know
# the true number of clusters. We could instead have performed K-means
# clustering on this example with `K  =  3`. If we do this, K-means clustering will split up the two "real" clusters, since it has no information about them:




kmeans_3_clusters = KMeans(n_clusters = 3, random_state = 123)
kmeans_3_clusters.fit(X)

plt.figure(figsize=(6,5))
plt.scatter(X[:,0], X[:,1], s=50, c=kmeans_3_clusters.labels_, cmap=plt.cm.prism) 
plt.scatter(kmeans_3_clusters.cluster_centers_[:, 0], kmeans_3_clusters.cluster_centers_[:, 1], marker='*', s=150,
            color='blue', label='Centers')
plt.legend(loc='best')
plt.xlabel('X0')
plt.ylabel('X1')


# To run the `Kmeans()` function in python with multiple initial cluster assignments,
# we use the `n_init` argument (default: 10). If a value of `n_init` greater than one
# is used, then K-means clustering will be performed using multiple random
# assignments, and the `Kmeans()` function will
# report only the best results. Here we compare using `n_init = 1`:




km_out_single_run = KMeans(n_clusters = 3, n_init = 1, random_state = 123).fit(X)
km_out_single_run.inertia_


# to `n_init = 20`:




km_out_single_run = KMeans(n_clusters = 3, n_init = 20, random_state = 123).fit(X)
km_out_single_run.inertia_


# Note that `.inertia_` is the total within-cluster sum of squares,
# which we seek to minimize by performing K-means clustering.
# 
# It is generally recommended to always run K-means clustering with a large
# value of `n_init`, such as 20 or 50 to avoid getting stuck in an undesirable local
# optimum.
# 
# When performing K-means clustering, in addition to using multiple initial
# cluster assignments, it is also important to set a random seed using the
# `random_state` parameter. This way, the initial cluster assignments can
# be replicated, and the K-means output will be fully reproducible.
# 
# # 10.5.2 Hierarchical Clustering
# The `linkage()` function from `scipy` implements several clustering functions in python. In the following example we use the data from the previous section to plot the hierarchical
# clustering dendrogram using complete, single, and average linkage clustering,
# with Euclidean distance as the dissimilarity measure. We begin by
# clustering observations using complete linkage:




from scipy.cluster.hierarchy import linkage

hc_complete = linkage(X, "complete")


# We could just as easily perform hierarchical clustering with average or single linkage instead:




hc_average = linkage(X, "average")
hc_single = linkage(X, "single")


# We can now plot the dendrograms obtained using the usual `dendrogram()` function.
# The numbers at the bottom of the plot identify each observation:




from scipy.cluster.hierarchy import dendrogram

# calculate full dendrogram
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    hc_complete,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()


# To determine the cluster labels for each observation associated with a
# given cut of the dendrogram, we can use the `cut_tree()` function:




from scipy.cluster.hierarchy import cut_tree
print(cut_tree(hc_complete, n_clusters = 2).T) # Printing transpose just for space


# For this data, complete and average linkage generally separates the observations
# into their correct groups.

# # 10.6: NCI60 Data Example
# 
# Unsupervised techniques are often used in the analysis of genomic data. In this portion of the lab, we'll see how hierarchical and K-means clustering compare on the `NCI60` cancer cell line microarray data, which
# consists of 6,830 gene expression measurements on 64 cancer cell lines:




# The NCI60 data
nci_labs = pd.read_csv("NCI60_labs.csv", index_col = 0)
nci_data = pd.read_csv("NCI60_data.csv", index_col = 0)


# Each cell line is labeled with a cancer type. We'll ignore the
# cancer types in performing clustering, as these are unsupervised
# techniques. After performing clustering, we'll use this column to see the extent to which these cancer types agree with the results of these
# unsupervised techniques.
# 
# The data has 64 rows and 6,830 columns.




nci_data.shape


# Let's take a look at the cancer types for the cell lines:




nci_labs.x.value_counts(sort=True)


# # 10.6.2 Clustering the Observations of the NCI60 Data
# We now proceed to hierarchically cluster the cell lines in the `NCI60` data,
# with the goal of finding out whether or not the observations cluster into
# distinct types of cancer.

# We now perform hierarchical clustering of the observations using complete,
# single, and average linkage. We'll use standard Euclidean distance as the dissimilarity
# measure:




nci_data.index = nci_labs.x

fig, ax = plt.subplots(3,1, figsize=(15,22))
fig.subplots_adjust(hspace=0.5)

linkages = ['complete', 'single', 'average']
for link, axis in zip(linkages, fig.axes):
    hc = linkage(y = nci_data, method=link, metric='euclidean') 
    axis.set_title("Linkage=%s" % link, size=15)
    axis.set_xlabel('Sample Type', size=15)
    axis.set_ylabel('Distance', size=15)
    dendrogram(hc, ax=axis, labels=nci_data.index, leaf_rotation=90, leaf_font_size=10)


# We see that the choice of linkage
# certainly does affect the results obtained. Typically, single linkage will tend
# to yield trailing clusters: very large clusters onto which individual observations
# attach one-by-one. On the other hand, complete and average linkage
# tend to yield more balanced, attractive clusters. For this reason, complete
# and average linkage are generally preferred to single linkage. Clearly cell
# lines within a single cancer type do tend to cluster together, although the
# clustering is not perfect. 
# 
# Let's use our complete linkage hierarchical clustering
# for the analysis. We can cut the dendrogram at the height that will yield a particular
# number of clusters, say 4:




nci_hc_complete = linkage(y = nci_data, method="complete", metric='euclidean') 

nci_hc_complete_4_clusters = cut_tree(nci_hc_complete, n_clusters = 4) # Printing transpose just for space

pd.crosstab(index = nci_data.index, 
            columns = nci_hc_complete_4_clusters.T[0], 
            rownames = ['Cancer Type'], 
            colnames = ['Cluster'])


# There are some clear patterns. All the leukemia cell lines fall in cluster 2,
# while the breast cancer cell lines are spread out over three different clusters.
# We can plot the cut on the dendrogram that produces these four clusters by adding an `axhline()`, which draws a horizontal line on top of our plot:




fig, ax = plt.subplots(1,1, figsize = (15,8))
dendrogram(nci_hc_complete, 
           labels = nci_data.index, 
           leaf_font_size = 14, 
           show_leaf_counts = True)  

plt.axhline(y=110, c='k', ls='dashed')
plt.show()


# We claimed earlier that K-means clustering and hierarchical
# clustering with the dendrogram cut to obtain the same number
# of clusters can yield **very** different results. How do these `NCI60` hierarchical
# clustering results compare to what we get if we perform K-means clustering
# with `K = 4`?




kmean_4 = KMeans(n_clusters = 4, random_state = 123, n_init = 150)    
kmean_4.fit(nci_data)
kmean_4.labels_


# We can use a confusion matrix to compare the differences in how the two methods assigned observations to clusters:




pd.crosstab(index = kmean_4.labels_, 
            columns = nci_hc_complete_4_clusters.T[0], 
            rownames = ['K-Means'], 
            colnames = ['Hierarchical'])


# We see that the four clusters obtained using hierarchical clustering and Kmeans
# clustering are somewhat different. Cluster 0 in K-means clustering is almost
# identical to cluster 2 in hierarchical clustering. However, the other clusters
# differ: for instance, cluster 2 in K-means clustering contains a portion of
# the observations assigned to cluster 0 by hierarchical clustering, as well as
# all of the observations assigned to cluster 1 by hierarchical clustering.
# 
# To get credit for this lab, use a similar analysis to compare the results of your K-means clustering to the results of your hierarchical clustering with single and average linkage. What differences do you notice? Post your response to Moodle: https://moodle.smith.edu/mod/quiz/view.php?id=267171
