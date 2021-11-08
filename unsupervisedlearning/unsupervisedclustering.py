import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score, f1_score, silhouette_score, adjusted_mutual_info_score
from sklearn import metrics
import mlrose_hiive
from supervised_learning.data_load import  load_women_diabetes_data, load_heart_stroke_data
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FastICA, FactorAnalysis
from sklearn import random_projection
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve

from scipy.stats import kurtosis

import numpy as np
import time
import logging

import matplotlib.pyplot as plt




X1, y1 = load_women_diabetes_data('../data/diabetes_in_women/data.csv')
X2, y2 = load_heart_stroke_data('../data/heart-stroke/data.csv')

X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.25, random_state=0)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.25, random_state=0)


# Print information about the datasets
print ("<------------------------ Women Diabetes Dataset ------------------------>")
print ("Total number of examples: ", X1.shape[0])
print ("Fraction of positive examples: %.2f%%" % (y1[y1 == 1].shape[0]/y1.shape[0]*100.0))
print ("Number of features: ", X1.shape[1])
print()
print ("<------------------------ Heart Stroke Dataset ------------------------>")
print ("Total number of examples: ", X2.shape[0])
print ("Fraction of positive examples: %.2f%%" % (y2[y2 == 1].shape[0]/y2.shape[0]*100.0))
print ("Number of features: ", X2.shape[1])
print()


random_seed = 7641
np.random.seed(random_seed)

# Standardize data
X1 = scale(X1)
X2 = scale(X2)


##Clustering

# Find the optimal k
k_grid = np.arange(2, 31)

# k_grid = np.arange(2, 31, 2)
loss1 = np.zeros(k_grid.size)
loss2 = np.zeros(k_grid.size)
score1 = np.zeros(k_grid.size)
score2 = np.zeros(k_grid.size)
for idx, k in enumerate(k_grid):
    print ("k = ", k)
    kmeans = KMeans(n_clusters=k, random_state=random_seed,init='k-means++',max_iter=1000,tol=.00001)
    kmeans.fit(X1)
    loss1[idx] = kmeans.inertia_
    score1[idx] = silhouette_score(X1, kmeans.labels_)
    kmeans.fit(X2)
    loss2[idx] = kmeans.inertia_
    score2[idx] = silhouette_score(X2, kmeans.labels_)

# Plot loss vs k to find best k
plt.figure()
plt.plot(k_grid, loss1)
# plt.xticks(k_grid)
plt.xlabel('k')
plt.ylabel('Loss')
plt.title('Dataset 1')
plt.grid()
plt.savefig('../plots/kmeans_loss_1.png')

plt.figure()
plt.plot(k_grid, loss2)
# plt.xticks(k_grid)
plt.xlabel('k')
plt.ylabel('Loss')
plt.title('Dataset 2')
plt.grid()
plt.savefig('../plots/kmeans_loss_2.png')

plt.figure()
plt.plot(k_grid, score1)
# plt.xticks(k_grid)
plt.xlabel('k')
plt.ylabel('Loss')
plt.title('Dataset 1')
plt.grid()
plt.savefig('../plots/kmeans_sc_1.png')

plt.figure()
plt.plot(k_grid, score2)
# plt.xticks(k_grid)
plt.xlabel('k')
plt.ylabel('Loss')
plt.title('Dataset 2')
plt.grid()
plt.savefig('../plots/kmeans_sc_2.png')

k1 = 10
k2 = 10

kmeans1 =  KMeans(n_clusters=k1, random_state=random_seed,init='k-means++',max_iter=1000,tol=.00001)
kmeans2 =  KMeans(n_clusters=k2, random_state=random_seed,init='k-means++',max_iter=1000,tol=.00001)
kmeans1.fit(X1)
kmeans2.fit(X2)


print ("<------------------------ Dataset 1 ------------------------>")
print('Inertia: ', kmeans1.inertia_)
score1 = silhouette_score(X1, kmeans1.labels_)
print('Silhouette score: ', score1)
ami1 = adjusted_mutual_info_score(y1, kmeans1.labels_)
print('Adjusted Mutual Information (AMI) score: ', ami1)
print()
print ("<------------------------ Dataset 2 ------------------------>")
print('Inertia: ', kmeans2.inertia_)
score2 = silhouette_score(X2, kmeans2.labels_)
print('Silhouette score: ', score2)
ami2 = adjusted_mutual_info_score(y2, kmeans2.labels_)
print('Adjusted Mutual Information (AMI) score: ', ami2)
print()



# Dataset 1
plt.figure()
plt.hist(kmeans1.labels_, bins=np.arange(0, k1 + 1) - 0.5, rwidth=0.5, zorder=2)
plt.xticks(np.arange(0, k1))
plt.xlabel('Cluster label')
plt.ylabel('Number of samples')
plt.title('Dataset 1')
plt.grid()
plt.savefig('../plots/kmeans_hist_1.png')

# Dataset 2
plt.figure()
plt.hist(kmeans2.labels_, bins=np.arange(0, k2 + 1) - 0.5, rwidth=0.5, zorder=2)
plt.xticks(np.arange(0, k2))
plt.xlabel('Cluster label')
plt.ylabel('Number of samples')
plt.title('Dataset 2')
plt.grid()
plt.savefig('../plots/kmeans_hist_2.png')


# Dataset 1
df1 = pd.read_csv('../data/diabetes_in_women/data.csv', nrows=1)
to_drop = ['Outcome']
df1 = df1.drop(to_drop, axis=1)
cols1 = df1.columns
df1 = pd.DataFrame(X1, columns = cols1)
df1['Outcome'] = y1
df1['class'] = kmeans1.labels_
print(df1.head(5))

# Parallel coordinates plot
rand_idx1 = np.random.randint(0, df1.shape[1] - 2, 5)   # randomly pick 5 features for visualization
idx_viz1 = np.append(rand_idx1, [df1.shape[1] - 2, df1.shape[1] - 1])    # Add ground truth label and cluster label

# plt.figure(figsize=(10, 7.5), dpi=250)
plt.figure(figsize=(8, 6), dpi=200)
pd.plotting.parallel_coordinates(df1.iloc[:, idx_viz1], 'class', colormap='Set1')
plt.xticks(rotation=30)
plt.xlabel('Features')
plt.ylabel('Value (standardized)')
plt.title('Dataset 1: k-means visualization')
plt.tight_layout()
plt.savefig('../plots/kmeans_viz_1.png')

# Dataset 2
df2 = pd.read_csv('../data/heart-stroke/data.csv',  nrows=1)
to_drop = ['id', 'stroke']
df2 = df2.drop(to_drop, axis=1)
cols2 = df2.columns
df2 = pd.DataFrame(X2, columns = cols2)
df2['stroke'] = y2
df2['class'] = kmeans2.labels_
print(df2.head(5))

# Parallel coordinates plot
rand_idx2 = np.random.randint(0, df2.shape[1] - 2, 5)   # randomly pick 5 features for visualization
idx_viz2 = np.append(rand_idx2, [df2.shape[1] - 2, df2.shape[1] - 1])    # Add ground truth label and cluster label

# plt.figure(figsize=(10, 7.5), dpi=250)
plt.figure(figsize=(8, 6), dpi=200)
pd.plotting.parallel_coordinates(df2.iloc[:, idx_viz2], 'class', colormap='Set1')
plt.xticks(rotation=30)
plt.xlabel('Features')
plt.ylabel('Value (standardized)')
plt.title('Dataset 2: k-means visualization')
plt.tight_layout()
plt.savefig('../plots/kmeans_viz_2.png')



# Find the optimal number of components using BIC

n_components_grid1 = np.arange(2, 31)
n_components_grid2 = np.arange(2, 51)
bic1 = np.zeros(n_components_grid1.size)
bic2 = np.zeros(n_components_grid2.size)

print ("<------------------------ Dataset 1 ------------------------>")
for idx, n_components in enumerate(n_components_grid1):
    print ("Number of components = ", n_components)
    gmm1 = GaussianMixture(n_components=n_components, random_state=random_seed, max_iter=1000, covariance_type = 'spherical', \
        tol = .0001, init_params = 'random')
    gmm1.fit(X1)
    bic1[idx] = gmm1.bic(X1)
print()
print ("<------------------------ Dataset 2 ------------------------>")
for idx, n_components in enumerate(n_components_grid2):
    print ("Number of components = ", n_components)
    gmm2 = GaussianMixture(n_components=n_components, random_state=random_seed, max_iter=1000, covariance_type = 'spherical', \
        tol = .0001, init_params = 'random')
    gmm2.fit(X2)
    bic2[idx] = gmm2.bic(X2)



# Plot BIC vs number of components
plt.figure()
plt.plot(n_components_grid1, bic1)
# plt.xticks(k_grid)
plt.xlabel('Number of components')
plt.ylabel('BIC')
plt.title('Dataset 1')
plt.grid()
plt.savefig('../plots/gmm_bic_1.png')

plt.figure()
plt.plot(n_components_grid2, bic2)
# plt.xticks(k_grid)
plt.xlabel('Number of components')
plt.ylabel('BIC')
plt.title('Dataset 2')
plt.grid()
plt.savefig('../plots/gmm_bic_2.png')




n_best_1 = n_components_grid1[np.argmin(bic1)]
print('Optimal number of components for dataset 1: ', n_best_1)
n_best_2 = n_components_grid2[np.argmin(bic2)]
print('Optimal number of components for dataset 2: ', n_best_2)




# Run GMM-EM again with the optimal number of components
gmm1 = GaussianMixture(n_components=n_best_1, random_state=random_seed,max_iter=1000, covariance_type = 'spherical', \
        tol = .0001, init_params = 'random')
gmm2 = GaussianMixture(n_components=n_best_2, random_state=random_seed,max_iter=1000, covariance_type = 'spherical', \
        tol = .0001, init_params = 'random')
gmm1.fit(X1)
gmm_labels1 = gmm1.predict(X1)
gmm2.fit(X2)
gmm_labels2 = gmm2.predict(X2)




print ("<------------------------ Dataset 1 ------------------------>")
print('BIC: ', gmm1.bic(X1))
score1_gmm = silhouette_score(X1, gmm_labels1)
print('Silhouette score: ', score1_gmm)
ami1_gmm = adjusted_mutual_info_score(y1, gmm_labels1)
print('Adjusted Mutual Information (AMI) score: ', ami1_gmm)
print()
print ("<------------------------ Dataset 2 ------------------------>")
print('BIC: ', gmm2.bic(X2))
score2_gmm = silhouette_score(X2, gmm_labels2)
print('Silhouette score: ', score2_gmm)
ami2_gmm = adjusted_mutual_info_score(y2, gmm_labels2)
print('Adjusted Mutual Information (AMI) score: ', ami2_gmm)
print()




# Dataset 1
plt.figure()
plt.hist(gmm_labels1, bins=np.arange(0, n_best_1 + 1) - 0.5, rwidth=0.5, zorder=2)
plt.xticks(np.arange(0, n_best_1))
plt.xlabel('Cluster label')
plt.ylabel('Number of samples')
plt.title('Dataset 1')
plt.grid()
plt.savefig('../plots/gmm_hist_1.png')

# Dataset 2
plt.figure()
plt.hist(gmm_labels2, bins=np.arange(0, n_best_2 + 1) - 0.5, rwidth=0.5, zorder=2)
plt.xticks(np.arange(0, n_best_2))
plt.xlabel('Cluster label')
plt.ylabel('Number of samples')
plt.title('Dataset 2')
plt.grid()
plt.savefig('../plots/gmm_hist_2.png')



# Dataset 1
df1_gmm = pd.read_csv('../data/diabetes_in_women/data.csv', nrows=1)
to_drop = ['Outcome']
df1_gmm = df1_gmm.drop(to_drop, axis=1)
cols1 = df1_gmm.columns
df1_gmm = pd.DataFrame(X1, columns = cols1)
df1_gmm['Outcome'] = y1
df1_gmm['class'] = gmm_labels1
print(df1_gmm.head(5))





rand_idx1 = np.random.randint(0, df1_gmm.shape[1] - 2, 5)   # randomly pick 5 features for visualization
idx_viz1 = np.append(rand_idx1, [df1_gmm.shape[1] - 2, df1_gmm.shape[1] - 1])    # Add ground truth label and cluster label


print(df1_gmm.shape[1])

# Parallel coordinates plot
idx_viz1_gmm = idx_viz1    # pick the same 5 features used in kmeans for visualization

# plt.figure(figsize=(10, 7.5), dpi=250)
plt.figure(figsize=(8, 6), dpi=200)
pd.plotting.parallel_coordinates(df1_gmm.iloc[:, idx_viz1_gmm], 'class', colormap='Set1')
plt.xticks(rotation=30)
plt.xlabel('Features')
plt.ylabel('Value (standardized)')
plt.title('Dataset 1: GMM-EM visualization')
plt.tight_layout()
plt.savefig('../plots/gmm_viz_1.png')

# Dataset 2
df2_gmm = pd.read_csv('../data/heart-stroke/data.csv',  nrows=1)
to_drop = ['id', 'stroke']
df2_gmm = df2_gmm.drop(to_drop, axis=1)
cols2 = df2_gmm.columns
df2_gmm = pd.DataFrame(X2, columns = cols2)
df2_gmm['stroke'] = y2
df2_gmm['class'] = gmm_labels2
print(df2_gmm.head(5))

rand_idx2 = np.random.randint(0, df2_gmm.shape[1] - 2, 5)   # randomly pick 5 features for visualization
idx_viz2 = np.append(rand_idx2, [df2_gmm.shape[1] - 2, df2_gmm.shape[1] - 1])    # Add ground truth label and cluster label

print(df2_gmm.shape[1])

# Parallel coordinates plot
idx_viz2_gmm = idx_viz2    # pick the same 5 features used in kmeans for visualization

# plt.figure(figsize=(10, 7.5), dpi=250)
plt.figure(figsize=(8, 6), dpi=200)
pd.plotting.parallel_coordinates(df2_gmm.iloc[:, idx_viz2_gmm], 'class', colormap='Set1')
plt.xticks(rotation=30)
plt.xlabel('Features')
plt.ylabel('Value (standardized)')
plt.title('Dataset 2: GMM-EM visualization')
plt.tight_layout()
plt.savefig('../plots/gmm_viz_2.png')



plt.figure()
plt.barh(np.arange(n_best_1), gmm1.weights_, zorder=2)
plt.gca().invert_yaxis()  # labels read top-to-bottom
plt.title('Dataset 1')
plt.xlabel('Weight')
plt.grid()
plt.savefig('../plots/gmm_wts_1.png')

plt.figure()
plt.barh(np.arange(n_best_2), gmm2.weights_, zorder=2)
plt.gca().set_yticks(np.arange(n_best_2))
plt.gca().invert_yaxis()  # labels read top-to-bottom
plt.title('Dataset 2')
plt.xlabel('Weight')
plt.grid()
plt.savefig('../plots/gmm_wts_2.png')

probs1 = np.amax(gmm1.predict_proba(X1), axis=1)
plt.figure()
plt.barh(np.arange(X1.shape[0]), probs1, zorder=2)
plt.gca().invert_yaxis()  # labels read top-to-bottom
plt.title('Dataset 1')
plt.xlabel('Posterior probabilities')
plt.grid()
plt.savefig('../plots/gmm_probs_1.png')

probs2 = np.amax(gmm2.predict_proba(X2), axis=1)
plt.figure()
plt.barh(np.arange(X2.shape[0]), probs2, zorder=2)
plt.gca().invert_yaxis()  # labels read top-to-bottom
plt.title('Dataset 2')
plt.xlabel('Posterior probabilities')
plt.grid()
plt.savefig('../plots/gmm_probs_2.png')



##dimensionality reduction algorithms

# 1. PCA


pca1 = PCA(random_state=3)
pca1.fit(X1)

pca2 = PCA(random_state=3)
pca2.fit(X2)

variance1 = np.cumsum(pca1.explained_variance_)
print(variance1)
# Plot variance explained by each component to find the best number of components
plt.figure()
plt.bar(np.arange(1, pca1.explained_variance_ratio_.size + 1), pca1.explained_variance_ratio_*100)
#plt.xticks(np.arange(1, pca1.explained_variance_ratio_.size + 1))
plt.xlabel('Component')
plt.ylabel('Eigenvalues')
plt.title('Dataset 1')
plt.grid()
plt.savefig('../plots/pca_var_1.png')

plt.figure()
plt.bar(np.arange(1, pca1.explained_variance_ratio_.size + 1), pca1.explained_variance_ratio_*100)
plt.xticks(np.arange(1, pca2.explained_variance_ratio_.size + 1))
plt.xlabel('Component')
plt.ylabel('Eigenvalues')
plt.title('Dataset 2')
plt.grid()
plt.savefig('../plots/pca_var_2.png')

plt.figure()
plt.plot(np.arange(1, pca1.explained_variance_ratio_.size + 1), np.cumsum(pca1.explained_variance_ratio_))
plt.xticks(np.arange(1, pca1.explained_variance_ratio_.size + 1))
plt.xlabel('Component')
plt.ylabel('Variance (cumulative)')
plt.title('Dataset 1')
plt.grid()
plt.savefig('../plots/pca_var_cum_1.png')

plt.figure()
plt.plot(np.arange(1, pca2.explained_variance_ratio_.size + 1), np.cumsum(pca2.explained_variance_ratio_))
plt.xticks(np.arange(1, pca2.explained_variance_ratio_.size + 1))
plt.xlabel('Component')
plt.ylabel('Variance (cumulative)')
plt.title('Dataset 2')
plt.grid()
plt.savefig('../plots/pca_var_cum_2.png')

# Choose the number of components that capture 85% of the variance
n1 = 6
n2 = 7

# Transform the data
pca1 = PCA(n_components=n1,random_state=3)
X1_transform = pca1.fit_transform(X1)
pca2 = PCA(n_components=n2,random_state=3)
X2_transform = pca2.fit_transform(X2)


X1_inverse = pca1.inverse_transform(X1_transform)
mse1 = np.sum(np.square(X1 - X1_inverse))/X1_inverse.size
print('MSE for dataset 1: ', mse1)
X2_inverse = pca2.inverse_transform(X2_transform)
mse2 = np.sum(np.square(X2 - X2_inverse))/X2_inverse.size
print('MSE for dataset 2: ', mse2)




# 2. ICA


nComponents = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,30,40,100]
ica1 = FastICA(random_state = 3)
kurtosisList1 = []
for item in nComponents:
    ica1.set_params(n_components=item)
    fit1 = ica1.fit_transform(X1)
    kurtosisList1.append(np.mean(kurtosis(fit1)))

plt.figure()
plt.plot(nComponents,kurtosisList1,label = 'Kurtosis')
plt.ylabel('Kurtosis')
plt.xlabel('Components')
plt.legend(loc=0)
plt.title('ICA Kurtosis')
plt.savefig('../plots/var1-Kurtosis-ICA.png')

ica2 = FastICA(random_state = 3)
kurtosisList2 = []
for item in nComponents:
    ica2.set_params(n_components=item)
    fit2 = ica2.fit_transform(X2)
    kurtosisList2.append(np.mean(kurtosis(fit2)))

plt.figure()
plt.plot(nComponents,kurtosisList2,label = 'Kurtosis')
plt.ylabel('Kurtosis')
plt.xlabel('Components')
plt.legend(loc=0)
plt.title('ICA Kurtosis')
plt.savefig('../plots/var2-Kurtosis-ICA.png')


# 3.Randomized Projections

nComponents = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,23,30]#,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,30,40
reconError1 = []
meanError1 = []
stdError1 = []
for item in nComponents:
	rp1 = random_projection.GaussianRandomProjection(n_components = item,random_state=3)
	X1_transformed = rp1.fit_transform(X1)
	X1_projected = (X1_transformed.dot(rp1.components_)) - X1
	reconError1.append(((X1 - X1_projected) ** 2).mean())
	meanError1.append(np.mean(((X1 - X1_projected) ** 2).mean()))
	stdError1.append(np.std(((X1 - X1_projected) ** 2).mean()))
print(meanError1)
print(stdError1)
plt.figure()
plt.plot(nComponents,meanError1, label = 'Mean')
plt.plot(nComponents,reconError1, label = 'recon')
plt.plot(nComponents,stdError1, label = 'STD')
plt.ylabel('Error')
plt.xlabel('Components')
plt.legend(loc=0)
plt.title('Random Projection Error Dataset 1')
plt.savefig('../plots/var1-Error-RP.png')


nComponents = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,23,30]#,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,30,40
reconError2 = []
meanError2 = []
stdError2 = []
for item in nComponents:
	rp2 = random_projection.GaussianRandomProjection(n_components = item,random_state=3)
	X2_transformed = rp2.fit_transform(X2)
	X2_projected = (X2_transformed.dot(rp2.components_)) - X2
	reconError2.append(((X2 - X2_projected) ** 2).mean())
	meanError2.append(np.mean(((X2 - X2_projected) ** 2).mean()))
	stdError2.append(np.std(((X2 - X2_projected) ** 2).mean()))
plt.figure()
plt.plot(nComponents,meanError2, label = 'Mean')
plt.plot(nComponents,reconError2, label = 'recon')
plt.plot(nComponents,stdError2, label = 'STD')
plt.ylabel('Error')
plt.xlabel('Components')
plt.legend(loc=0)
plt.title('Random Projection Error Dataset 2')
plt.savefig('../plots/var2-Error-RP.png')



#4. Factor Analysis

nComponents = [2,3,4,5,6,7,8,9,10,11,12,13,14,15]#,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,30,40
reconError1 = []
meanError1 = []
stdError1 = []
for item in nComponents:
	fa1 = FactorAnalysis(n_components = item)
	X_transformed1 = fa1.fit_transform(X1)
	X_projected1 = (X_transformed1.dot(fa1.components_)) - X1
	reconError1.append(((X1 - X_projected1) ** 2).mean())
	meanError1.append(np.mean(((X1 - X_projected1) ** 2).mean()))
	stdError1.append(np.std(((X1 - X_projected1) ** 2).mean()))

plt.figure()
plt.plot(nComponents,meanError1, label = 'Mean')
plt.plot(nComponents,stdError1, label = 'STD')
plt.ylabel('Error')
plt.xlabel('Components')
plt.legend(loc=0)
plt.title('Factor Analysis Error Dataset 1')
plt.savefig('../plots/var1-Error-FA.png')



nComponents = [2,3,4,5,6,7,8,9,10,11,12,13,14,15]#,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,30,40
reconError2 = []
meanError2 = []
stdError2 = []
for item in nComponents:
	fa2 = FactorAnalysis(n_components = item)
	X_transformed2 = fa2.fit_transform(X2)
	X_projected2 = (X_transformed2.dot(fa2.components_)) - X2
	reconError2.append(((X2 - X_projected2) ** 2).mean())
	meanError2.append(np.mean(((X2 - X_projected2) ** 2).mean()))
	stdError2.append(np.std(((X2 - X_projected2) ** 2).mean()))

print(meanError2)
print(stdError2)
plt.figure()
plt.plot(nComponents,meanError2, label = 'Mean')
plt.plot(nComponents,stdError2, label = 'STD')
plt.ylabel('Error')
plt.xlabel('Components')
plt.legend(loc=0)
plt.title('Factor Analysis Error Dataset 2')
plt.savefig('../plots/var2-Error-FA.png')




# Clustering with dimensionality reduction


def kmeansPartDiabetes(dataTraining, dataTesting, dataTraining_Class, dataTesting_Class, expName):

	nComponents = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,30,100]
	rpData = random_projection.GaussianRandomProjection(random_state=3, n_components=6) \
		.fit_transform(dataTraining)
	icaData = FastICA(random_state=3, n_components=6).fit_transform(dataTraining)
	pcaData = PCA(random_state=3, n_components=6).fit_transform(dataTraining)
	faData = FactorAnalysis(random_state=3, n_components=6).fit_transform(dataTraining)
	homoScoresKMrpData = []
	homoScoresKMicaData = []
	homoScoresKMpcaData = []
	homoScoresKMfaData = []

	compScoresKMrpData = []
	compScoresKMicaData = []
	compScoresKMpcaData = []
	compScoresKMfaData = []

	silScoresKMrpData = []
	silScoresKMicaData = []
	silScoresKMpcaData = []
	silScoresKMfaData = []

	scoreKmrpData = []
	scoreKmicaData = []
	scoreKmpcaData = []
	scoreKmfaData = []
	for item in nComponents:
		km = KMeans(n_clusters=item, init='k-means++', max_iter=1000, tol=.00001, random_state=3, \
					algorithm='auto')
		km.fit(rpData)
		predictions = km.predict(rpData)
		homoScoresKMrpData.append(metrics.homogeneity_score(dataTraining_Class, km.labels_))
		compScoresKMrpData.append(metrics.completeness_score(dataTraining_Class, km.labels_))
		scoreKmrpData.append(abs(km.score(rpData)))
		silScoresKMrpData.append(metrics.silhouette_score(rpData, km.labels_))
	for item in nComponents:
		km = KMeans(n_clusters=item, init='k-means++', max_iter=1000, tol=.00001, random_state=3, \
					algorithm='auto')
		km.fit(icaData)
		predictions = km.predict(icaData)
		homoScoresKMicaData.append(metrics.homogeneity_score(dataTraining_Class, km.labels_))
		compScoresKMicaData.append(metrics.completeness_score(dataTraining_Class, km.labels_))
		scoreKmicaData.append(abs(km.score(icaData)))
		silScoresKMicaData.append(metrics.silhouette_score(icaData, km.labels_))
	for item in nComponents:
		km = KMeans(n_clusters=item, init='k-means++', max_iter=1000, tol=.00001, random_state=3, \
					algorithm='auto')
		km.fit(pcaData)
		predictions = km.predict(pcaData)
		homoScoresKMpcaData.append(metrics.homogeneity_score(dataTraining_Class, km.labels_))
		compScoresKMpcaData.append(metrics.completeness_score(dataTraining_Class, km.labels_))
		scoreKmpcaData.append(abs(km.score(pcaData)))
		silScoresKMpcaData.append(metrics.silhouette_score(pcaData, km.labels_))
	for item in nComponents:
		km = KMeans(n_clusters=item, init='k-means++', max_iter=1000, tol=.00001, random_state=3, \
					algorithm='auto')
		km.fit(faData)
		predictions = km.predict(faData)
		homoScoresKMfaData.append(metrics.homogeneity_score(dataTraining_Class, km.labels_))
		compScoresKMfaData.append(metrics.completeness_score(dataTraining_Class, km.labels_))
		scoreKmfaData.append(abs(km.score(faData)))
		silScoresKMfaData.append(metrics.silhouette_score(faData, km.labels_))

	plt.figure()
	plt.plot(nComponents, homoScoresKMpcaData, label='PCA')
	plt.plot(nComponents, homoScoresKMfaData, label='FA')
	plt.plot(nComponents, homoScoresKMicaData, label='ICA')
	plt.plot(nComponents, homoScoresKMrpData, label='RP')
	plt.ylabel('Score')
	plt.xlabel('Components')
	plt.legend(loc=0)
	plt.title('Kmeans With DR - Homo')
	plt.savefig(f'3-{expName}-Kmeans-DR-Homo.png')

	plt.figure(15)
	plt.plot(nComponents, compScoresKMfaData, label='FA')
	plt.plot(nComponents, compScoresKMpcaData, label='PCA')
	plt.plot(nComponents, compScoresKMicaData, label='ICA')
	plt.plot(nComponents, compScoresKMrpData, label='RP')
	plt.ylabel('Score')
	plt.xlabel('Components')
	plt.legend(loc=0)
	plt.title('Kmeans With DR - Comp')
	plt.savefig(f'3-{expName}-Kmeans-DR-CO.png')

	plt.figure(16)
	plt.plot(nComponents, silScoresKMfaData, label='FA')
	plt.plot(nComponents, silScoresKMpcaData, label='PCA')
	plt.plot(nComponents, silScoresKMicaData, label='ICA')
	plt.plot(nComponents, silScoresKMrpData, label='RP')
	plt.ylabel('Score')
	plt.xlabel('Components')
	plt.legend(loc=0)
	plt.title('Kmeans With DR - Sil')
	plt.savefig(f'3-{expName}-Kmeans-DR-Sil.png')


def kmeansPartStroke(dataTraining, dataTesting, dataTraining_Class, dataTesting_Class, expName):

	nComponents = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,30,100]
	rpData = random_projection.GaussianRandomProjection(random_state=3, n_components=7) \
		.fit_transform(dataTraining)
	icaData = FastICA(random_state=3, n_components=7).fit_transform(dataTraining)
	pcaData = PCA(random_state=3, n_components=7).fit_transform(dataTraining)
	faData = FactorAnalysis(random_state=3, n_components=7).fit_transform(dataTraining)
	homoScoresKMrpData = []
	homoScoresKMicaData = []
	homoScoresKMpcaData = []
	homoScoresKMfaData = []

	compScoresKMrpData = []
	compScoresKMicaData = []
	compScoresKMpcaData = []
	compScoresKMfaData = []

	silScoresKMrpData = []
	silScoresKMicaData = []
	silScoresKMpcaData = []
	silScoresKMfaData = []

	scoreKmrpData = []
	scoreKmicaData = []
	scoreKmpcaData = []
	scoreKmfaData = []
	for item in nComponents:
		km = KMeans(n_clusters=item, init='k-means++', max_iter=1000, tol=.00001, random_state=3, \
					algorithm='auto')
		km.fit(rpData)
		predictions = km.predict(rpData)
		homoScoresKMrpData.append(metrics.homogeneity_score(dataTraining_Class, km.labels_))
		compScoresKMrpData.append(metrics.completeness_score(dataTraining_Class, km.labels_))
		scoreKmrpData.append(abs(km.score(rpData)))
		silScoresKMrpData.append(metrics.silhouette_score(rpData, km.labels_))
	for item in nComponents:
		km = KMeans(n_clusters=item, init='k-means++', max_iter=1000, tol=.00001, random_state=3, \
					algorithm='auto')
		km.fit(icaData)
		predictions = km.predict(icaData)
		homoScoresKMicaData.append(metrics.homogeneity_score(dataTraining_Class, km.labels_))
		compScoresKMicaData.append(metrics.completeness_score(dataTraining_Class, km.labels_))
		scoreKmicaData.append(abs(km.score(icaData)))
		silScoresKMicaData.append(metrics.silhouette_score(icaData, km.labels_))
	for item in nComponents:
		km = KMeans(n_clusters=item, init='k-means++', max_iter=1000, tol=.00001, random_state=3, \
					algorithm='auto')
		km.fit(pcaData)
		predictions = km.predict(pcaData)
		homoScoresKMpcaData.append(metrics.homogeneity_score(dataTraining_Class, km.labels_))
		compScoresKMpcaData.append(metrics.completeness_score(dataTraining_Class, km.labels_))
		scoreKmpcaData.append(abs(km.score(pcaData)))
		silScoresKMpcaData.append(metrics.silhouette_score(pcaData, km.labels_))
	for item in nComponents:
		km = KMeans(n_clusters=item, init='k-means++', max_iter=1000, tol=.00001, random_state=3, \
					algorithm='auto')
		km.fit(faData)
		predictions = km.predict(faData)
		homoScoresKMfaData.append(metrics.homogeneity_score(dataTraining_Class, km.labels_))
		compScoresKMfaData.append(metrics.completeness_score(dataTraining_Class, km.labels_))
		scoreKmfaData.append(abs(km.score(faData)))
		silScoresKMfaData.append(metrics.silhouette_score(faData, km.labels_))

	plt.figure()
	plt.plot(nComponents, homoScoresKMpcaData, label='PCA')
	plt.plot(nComponents, homoScoresKMfaData, label='FA')
	plt.plot(nComponents, homoScoresKMicaData, label='ICA')
	plt.plot(nComponents, homoScoresKMrpData, label='RP')
	plt.ylabel('Score')
	plt.xlabel('Components')
	plt.legend(loc=0)
	plt.title('Kmeans With DR - Homo')
	plt.savefig(f'3-{expName}-Kmeans-DR-Homo.png')

	plt.figure()
	plt.plot(nComponents, compScoresKMfaData, label='FA')
	plt.plot(nComponents, compScoresKMpcaData, label='PCA')
	plt.plot(nComponents, compScoresKMicaData, label='ICA')
	plt.plot(nComponents, compScoresKMrpData, label='RP')
	plt.ylabel('Score')
	plt.xlabel('Components')
	plt.legend(loc=0)
	plt.title('Kmeans With DR - Comp')
	plt.savefig(f'3-{expName}-Kmeans-DR-CO.png')

	plt.figure(16)
	plt.plot(nComponents, silScoresKMfaData, label='FA')
	plt.plot(nComponents, silScoresKMpcaData, label='PCA')
	plt.plot(nComponents, silScoresKMicaData, label='ICA')
	plt.plot(nComponents, silScoresKMrpData, label='RP')
	plt.ylabel('Score')
	plt.xlabel('Components')
	plt.legend(loc=0)
	plt.title('Kmeans With DR - Sil')
	plt.savefig(f'3-{expName}-Kmeans-DR-Sil.png')


def expectedMaxPartDiabetes(dataTraining, dataTesting, dataTraining_Class, dataTesting_Class, expName):

	nComponents = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,30,100]
	rpData = random_projection.GaussianRandomProjection(random_state=3, n_components=6) \
		.fit_transform(dataTraining)
	icaData = FastICA(random_state=3, n_components=6).fit_transform(dataTraining)
	pcaData = PCA(random_state=3, n_components=6).fit_transform(dataTraining)
	faData = FactorAnalysis(random_state=3, n_components=6).fit_transform(dataTraining)
	homoScoresEMrpData = []
	homoScoresEMicaData = []
	homoScoresEMpcaData = []
	homoScoresEMfaData = []

	compScoresEMrpData = []
	compScoresEMicaData = []
	compScoresEMpcaData = []
	compScoresEMfaData = []

	silScoresEMrpData = []
	silScoresEMicaData = []
	silScoresEMpcaData = []
	silScoresEMfaData = []

	aicScoresEMrpData = []
	aicScoresEMicaData = []
	aicScoresEMpcaData = []
	aicScoresEMfaData = []

	bicScoresEMrpData = []
	bicScoresEMicaData = []
	bicScoresEMpcaData = []
	bicScoresEMfaData = []
	for item in nComponents:
		em = GaussianMixture(n_components=item, max_iter=1000, covariance_type='spherical', \
							 tol=.0001, init_params='random', random_state=3, warm_start=False)
		em.fit(rpData)
		predictions = em.predict(rpData)
		aicScoresEMrpData.append(em.aic(rpData))
		bicScoresEMrpData.append(em.bic(rpData))
		homoScoresEMrpData.append(metrics.homogeneity_score(dataTraining_Class, predictions))
		compScoresEMrpData.append(metrics.completeness_score(dataTraining_Class, predictions))
		silScoresEMrpData.append(metrics.silhouette_score(rpData, predictions, metric='euclidean'))
	for item in nComponents:
		em = GaussianMixture(n_components=item, max_iter=1000, covariance_type='spherical', \
							 tol=.0001, init_params='random', random_state=3, warm_start=False)
		em.fit(icaData)
		predictions = em.predict(icaData)
		aicScoresEMicaData.append(em.aic(icaData))
		bicScoresEMicaData.append(em.bic(icaData))
		homoScoresEMicaData.append(metrics.homogeneity_score(dataTraining_Class, predictions))
		compScoresEMicaData.append(metrics.completeness_score(dataTraining_Class, predictions))
		silScoresEMicaData.append(metrics.silhouette_score(icaData, predictions, metric='euclidean'))
	for item in nComponents:
		em = GaussianMixture(n_components=item, max_iter=1000, covariance_type='spherical', \
							 tol=.0001, init_params='random', random_state=3, warm_start=False)
		em.fit(pcaData)
		predictions = em.predict(pcaData)
		aicScoresEMpcaData.append(em.aic(pcaData))
		bicScoresEMpcaData.append(em.bic(pcaData))
		homoScoresEMpcaData.append(metrics.homogeneity_score(dataTraining_Class, predictions))
		compScoresEMpcaData.append(metrics.completeness_score(dataTraining_Class, predictions))
		silScoresEMpcaData.append(metrics.silhouette_score(pcaData, predictions, metric='euclidean'))
	for item in nComponents:
		em = GaussianMixture(n_components=item, max_iter=1000, covariance_type='spherical', \
							 tol=.0001, init_params='random', random_state=3, warm_start=False)
		em.fit(faData)
		predictions = em.predict(faData)
		aicScoresEMfaData.append(em.aic(faData))
		bicScoresEMfaData.append(em.bic(faData))
		homoScoresEMfaData.append(metrics.homogeneity_score(dataTraining_Class, predictions))
		compScoresEMfaData.append(metrics.completeness_score(dataTraining_Class, predictions))
		silScoresEMfaData.append(metrics.silhouette_score(faData, predictions, metric='euclidean'))
	plt.figure()
	plt.plot(nComponents, homoScoresEMpcaData, label='PCA')
	plt.plot(nComponents, homoScoresEMfaData, label='FA')
	plt.plot(nComponents, homoScoresEMicaData, label='ICA')
	plt.plot(nComponents, homoScoresEMrpData, label='RP')
	plt.ylabel('Score')
	plt.xlabel('Components')
	plt.legend(loc=0)
	plt.title('EM With DR - Homo')
	plt.savefig(f'{expName}-EM-DR-Homo.png')

	plt.figure()
	plt.plot(nComponents, compScoresEMfaData, label='FA')
	plt.plot(nComponents, compScoresEMpcaData, label='PCA')
	plt.plot(nComponents, compScoresEMicaData, label='ICA')
	plt.plot(nComponents, compScoresEMrpData, label='RP')
	plt.ylabel('Score')
	plt.xlabel('Components')
	plt.legend(loc=0)
	plt.title('EM With DR - Comp')
	plt.savefig(f'{expName}-EM-DR-CO.png')

	plt.figure()
	plt.plot(nComponents, silScoresEMfaData, label='FA')
	plt.plot(nComponents, silScoresEMpcaData, label='PCA')
	plt.plot(nComponents, silScoresEMicaData, label='ICA')
	plt.plot(nComponents, silScoresEMrpData, label='RP')
	plt.ylabel('Score')
	plt.xlabel('Components')
	plt.legend(loc=0)
	plt.title('EM With DR - Sil')
	plt.savefig(f'{expName}-EM-DR-Sil.png')

	plt.figure()
	plt.plot(nComponents, aicScoresEMfaData, label='FA')
	plt.plot(nComponents, aicScoresEMpcaData, label='PCA')
	plt.plot(nComponents, aicScoresEMicaData, label='ICA')
	plt.plot(nComponents, aicScoresEMrpData, label='RP')
	plt.ylabel('Score')
	plt.xlabel('Components')
	plt.legend(loc=0)
	plt.title('EM With DR - AIC')
	plt.savefig(f'{expName}-EM-DR-AIC.png')

	plt.figure()
	plt.plot(nComponents, bicScoresEMfaData, label='FA')
	plt.plot(nComponents, bicScoresEMpcaData, label='PCA')
	plt.plot(nComponents, bicScoresEMicaData, label='ICA')
	plt.plot(nComponents, bicScoresEMrpData, label='RP')
	plt.ylabel('Score')
	plt.xlabel('Components')
	plt.legend(loc=0)
	plt.title('EM With DR - BIC')
	plt.savefig(f'{expName}-EM-DR-BIC.png')


def expectedMaxPartStroke(dataTraining, dataTraining_Class, dataTesting, dataTesting_Class, expName):
	
	nComponents = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,30,100]
	rpData = random_projection.GaussianRandomProjection(random_state=3, n_components=7).fit_transform(dataTraining)
	icaData = FastICA(random_state=3, n_components=7).fit_transform(dataTraining)
	pcaData = PCA(random_state=3, n_components=7).fit_transform(dataTraining)
	faData = FactorAnalysis(random_state=3, n_components=7).fit_transform(dataTraining)
	homoScoresEMrpData = []
	homoScoresEMicaData = []
	homoScoresEMpcaData = []
	homoScoresEMfaData = []

	compScoresEMrpData = []
	compScoresEMicaData = []
	compScoresEMpcaData = []
	compScoresEMfaData = []

	silScoresEMrpData = []
	silScoresEMicaData = []
	silScoresEMpcaData = []
	silScoresEMfaData = []

	aicScoresEMrpData = []
	aicScoresEMicaData = []
	aicScoresEMpcaData = []
	aicScoresEMfaData = []

	bicScoresEMrpData = []
	bicScoresEMicaData = []
	bicScoresEMpcaData = []
	bicScoresEMfaData = []
	for item in nComponents:
		em = GaussianMixture(n_components=item, max_iter=1000, covariance_type='spherical', \
							 tol=.0001, init_params='random', random_state=3, warm_start=False)
		em.fit(rpData)
		predictions = em.predict(rpData)
		aicScoresEMrpData.append(em.aic(rpData))
		bicScoresEMrpData.append(em.bic(rpData))
		homoScoresEMrpData.append(metrics.homogeneity_score(dataTraining_Class, predictions))
		compScoresEMrpData.append(metrics.completeness_score(dataTraining_Class, predictions))
		silScoresEMrpData.append(metrics.silhouette_score(rpData, predictions, metric='euclidean'))
	for item in nComponents:
		em = GaussianMixture(n_components=item, max_iter=1000, covariance_type='spherical', \
							 tol=.0001, init_params='random', random_state=3, warm_start=False)
		em.fit(icaData)
		predictions = em.predict(icaData)
		aicScoresEMicaData.append(em.aic(icaData))
		bicScoresEMicaData.append(em.bic(icaData))
		homoScoresEMicaData.append(metrics.homogeneity_score(dataTraining_Class, predictions))
		compScoresEMicaData.append(metrics.completeness_score(dataTraining_Class, predictions))
		silScoresEMicaData.append(metrics.silhouette_score(icaData, predictions, metric='euclidean'))
	for item in nComponents:
		em = GaussianMixture(n_components=item, max_iter=1000, covariance_type='spherical', \
							 tol=.0001, init_params='random', random_state=3, warm_start=False)
		em.fit(pcaData)
		predictions = em.predict(pcaData)
		aicScoresEMpcaData.append(em.aic(pcaData))
		bicScoresEMpcaData.append(em.bic(pcaData))
		homoScoresEMpcaData.append(metrics.homogeneity_score(dataTraining_Class, predictions))
		compScoresEMpcaData.append(metrics.completeness_score(dataTraining_Class, predictions))
		silScoresEMpcaData.append(metrics.silhouette_score(pcaData, predictions, metric='euclidean'))
	for item in nComponents:
		em = GaussianMixture(n_components=item, max_iter=1000, covariance_type='spherical', \
							 tol=.0001, init_params='random', random_state=3, warm_start=False)
		em.fit(faData)
		predictions = em.predict(faData)
		aicScoresEMfaData.append(em.aic(faData))
		bicScoresEMfaData.append(em.bic(faData))
		homoScoresEMfaData.append(metrics.homogeneity_score(dataTraining_Class, predictions))
		compScoresEMfaData.append(metrics.completeness_score(dataTraining_Class, predictions))
		silScoresEMfaData.append(metrics.silhouette_score(faData, predictions, metric='euclidean'))
	plt.figure()
	plt.plot(nComponents, homoScoresEMpcaData, label='PCA')
	plt.plot(nComponents, homoScoresEMfaData, label='FA')
	plt.plot(nComponents, homoScoresEMicaData, label='ICA')
	plt.plot(nComponents, homoScoresEMrpData, label='RP')
	plt.ylabel('Score')
	plt.xlabel('Components')
	plt.legend(loc=0)
	plt.title('EM With DR - Homo')
	plt.savefig(f'{expName}-EM-DR-Homo.png')

	plt.figure()
	plt.plot(nComponents, compScoresEMfaData, label='FA')
	plt.plot(nComponents, compScoresEMpcaData, label='PCA')
	plt.plot(nComponents, compScoresEMicaData, label='ICA')
	plt.plot(nComponents, compScoresEMrpData, label='RP')
	plt.ylabel('Score')
	plt.xlabel('Components')
	plt.legend(loc=0)
	plt.title('EM With DR - Comp')
	plt.savefig(f'{expName}-EM-DR-CO.png')

	plt.figure()
	plt.plot(nComponents, silScoresEMfaData, label='FA')
	plt.plot(nComponents, silScoresEMpcaData, label='PCA')
	plt.plot(nComponents, silScoresEMicaData, label='ICA')
	plt.plot(nComponents, silScoresEMrpData, label='RP')
	plt.ylabel('Score')
	plt.xlabel('Components')
	plt.legend(loc=0)
	plt.title('EM With DR - Sil')
	plt.savefig(f'{expName}-EM-DR-Sil.png')

	plt.figure()
	plt.plot(nComponents, aicScoresEMfaData, label='FA')
	plt.plot(nComponents, aicScoresEMpcaData, label='PCA')
	plt.plot(nComponents, aicScoresEMicaData, label='ICA')
	plt.plot(nComponents, aicScoresEMrpData, label='RP')
	plt.ylabel('Score')
	plt.xlabel('Components')
	plt.legend(loc=0)
	plt.title('EM With DR - AIC')
	plt.savefig(f'{expName}-EM-DR-AIC.png')

	plt.figure()
	plt.plot(nComponents, bicScoresEMfaData, label='FA')
	plt.plot(nComponents, bicScoresEMpcaData, label='PCA')
	plt.plot(nComponents, bicScoresEMicaData, label='ICA')
	plt.plot(nComponents, bicScoresEMrpData, label='RP')
	plt.ylabel('Score')
	plt.xlabel('Components')
	plt.legend(loc=0)
	plt.title('EM With DR - BIC')
	plt.savefig(f'{expName}-EM-DR-BIC.png')



#kmeansPartDiabetes(X_train1, X_test1, y_train1, y_test1,'Diabetes')
#kmeansPartStroke(X_train2, X_test2, y_train2, y_test2,'Stroke')

#expectedMaxPartDiabetes(X_train1, X_test1, y_train1, y_test1,'Diabetes')
#expectedMaxPartDiabetes(X_train2, X_test2, y_train2, y_test2,'Stroke')


clf_nn = MLPClassifier(hidden_layer_sizes=(5, 2), random_state=7, max_iter=1000)
clf_nn.fit(X_train1, y_train1)
y_pred = clf_nn.predict(X_test1)
nn_accuracy = accuracy_score(y_test1, y_pred)
print('Accuracy of neural network without hyperparameter tuning is %.2f%%' % (nn_accuracy * 100))


alpha_range = np.logspace(-1, 2, 5)
lr_range = np.logspace(-5, 0, 6)
tuned_params = {'alpha' : alpha_range, 'learning_rate_init' : lr_range}
clf_nn = GridSearchCV(clf_nn, param_grid=tuned_params, cv=5, n_jobs=-1)
clf_nn.fit(X_train1, y_train1)
best_clf_nn = clf_nn.best_estimator_
best_params = clf_nn.best_params_
print("Best parameters set found on development set:")
print(best_params)



t0 = time.time()
best_clf_nn.fit(X_train1, y_train1)
t1 = time.time()
print('Training time: %f seconds' % (t1 - t0))
t0 = time.time()
y_pred = best_clf_nn.predict(X_test1)
t1 = time.time()
test_time = t1 - t0
print('Inference time on test data: %f seconds' % test_time)
best_accuracy = accuracy_score(y_test1, y_pred)
print('Best accuracy of neural network is %.2f%%' % (best_accuracy * 100))


_, train_scores, test_scores = learning_curve(clf_nn, X_train1, y_train1, train_sizes=np.linspace(0.1, 1.0, 5), cv=5)
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
plt.figure()
plt.title('Learning curve for neural network MLP Classifier')
plt.plot(np.linspace(0.1, 1.0, 5), train_scores_mean, 'o-', label='Training score')
plt.plot(np.linspace(0.1, 1.0, 5), test_scores_mean, 'o-', label='Cross Validation score')
plt.xlabel('training set size')
plt.ylabel("score")
plt.legend(loc="best")
plt.grid()
plt.savefig('NN-learning-curve.png')


#PCA

pca = PCA(n_components=6)
pca.fit(X_train1)
X_train_pca = pca.transform(X_train1)
X_test_pca = pca.transform(X_test1)



clf_nn = MLPClassifier(hidden_layer_sizes=(5, 2), random_state=7, max_iter=1000)
clf_nn.fit(X_train_pca, y_train1)
y_pred = clf_nn.predict(X_test_pca)
nn_accuracy = accuracy_score(y_test1, y_pred)
print('Accuracy of neural network without hyperparameter tuning is %.2f%%' % (nn_accuracy * 100))

# Define grid for grid search after observing validation curves
alpha_range = np.logspace(-1, 2, 5)
lr_range = np.logspace(-5, 0, 6)
tuned_params = {'alpha' : alpha_range, 'learning_rate_init' : lr_range}
clf_nn = GridSearchCV(clf_nn, param_grid=tuned_params, cv=5, n_jobs=-1)
clf_nn.fit(X_train_pca, y_train1)
best_clf_nn = clf_nn.best_estimator_
best_params = clf_nn.best_params_
print("Best parameters set found on development set:")
print(best_params)

t0 = time.time()
best_clf_nn.fit(X_train_pca, y_train1)
t1 = time.time()
print('Training time: %f seconds' % (t1 - t0))
t0 = time.time()
y_pred = best_clf_nn.predict(X_test_pca)
t1 = time.time()
test_time = t1 - t0
print('Inference time on test data: %f seconds' % test_time)
best_accuracy = accuracy_score(y_test1, y_pred)
print('Best accuracy of PCA neural network is %.2f%%' % (best_accuracy * 100))


_, train_scores, test_scores = learning_curve(clf_nn, X_train1, y_train1, train_sizes=np.linspace(0.1, 1.0, 5), cv=5)
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
plt.figure()
plt.title('Learning curve for neural network MLP Classifier')
plt.plot(np.linspace(0.1, 1.0, 5), train_scores_mean, 'o-', label='Training score')
plt.plot(np.linspace(0.1, 1.0, 5), test_scores_mean, 'o-', label='Cross Validation score')
plt.xlabel('training set size')
plt.ylabel("score")
plt.legend(loc="best")
plt.grid()
plt.savefig('NN-PCA-learning-curve.png')


#ICA


ica = FastICA(random_state = 3,n_components=6)
ica.fit(X_train1)
X_train_ica = ica.transform(X_train1)
X_test_ica = ica.transform(X_test1)



clf_nn = MLPClassifier(hidden_layer_sizes=(5, 2), random_state=7, max_iter=1000)
clf_nn.fit(X_train_ica, y_train1)
y_pred = clf_nn.predict(X_test_ica)
nn_accuracy = accuracy_score(y_test1, y_pred)
print('Accuracy of neural network without hyperparameter tuning is %.2f%%' % (nn_accuracy * 100))

# Define grid for grid search after observing validation curves
alpha_range = np.logspace(-1, 2, 5)
lr_range = np.logspace(-5, 0, 6)
tuned_params = {'alpha' : alpha_range, 'learning_rate_init' : lr_range}
clf_nn = GridSearchCV(clf_nn, param_grid=tuned_params, cv=5, n_jobs=-1)
clf_nn.fit(X_train_ica, y_train1)
best_clf_nn = clf_nn.best_estimator_
best_params = clf_nn.best_params_
print("Best parameters set found on development set:")
print(best_params)

t0 = time.time()
best_clf_nn.fit(X_train_ica, y_train1)
t1 = time.time()
print('Training time: %f seconds' % (t1 - t0))
t0 = time.time()
y_pred = best_clf_nn.predict(X_test_ica)
t1 = time.time()
test_time = t1 - t0
print('Inference time on test data: %f seconds' % test_time)
best_accuracy = accuracy_score(y_test1, y_pred)
print('Best accuracy of ICA neural network is %.2f%%' % (best_accuracy * 100))

_, train_scores, test_scores = learning_curve(clf_nn, X_train1, y_train1, train_sizes=np.linspace(0.1, 1.0, 5), cv=5)
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
plt.figure()
plt.title('Learning curve for neural network MLP Classifier')
plt.plot(np.linspace(0.1, 1.0, 5), train_scores_mean, 'o-', label='Training score')
plt.plot(np.linspace(0.1, 1.0, 5), test_scores_mean, 'o-', label='Cross Validation score')
plt.xlabel('training set size')
plt.ylabel("score")
plt.legend(loc="best")
plt.grid()
plt.savefig('NN-ICA-learning-curve.png')

#Random Projection

rpData = random_projection.GaussianRandomProjection(random_state=3, n_components=6)
rpData.fit(X_train1)
X_train_rpData = rpData.transform(X_train1)
X_test_rpData = rpData.transform(X_test1)



clf_nn = MLPClassifier(hidden_layer_sizes=(5, 2), random_state=7, max_iter=1000)
clf_nn.fit(X_train_rpData, y_train1)
y_pred = clf_nn.predict(X_test_rpData)
nn_accuracy = accuracy_score(y_test1, y_pred)
print('Accuracy of neural network without hyperparameter tuning is %.2f%%' % (nn_accuracy * 100))

# Define grid for grid search after observing validation curves
alpha_range = np.logspace(-1, 2, 5)
lr_range = np.logspace(-5, 0, 6)
tuned_params = {'alpha' : alpha_range, 'learning_rate_init' : lr_range}
clf_nn = GridSearchCV(clf_nn, param_grid=tuned_params, cv=5, n_jobs=-1)
clf_nn.fit(X_train_rpData, y_train1)
best_clf_nn = clf_nn.best_estimator_
best_params = clf_nn.best_params_
print("Best parameters set found on development set:")
print(best_params)

t0 = time.time()
best_clf_nn.fit(X_train_rpData, y_train1)
t1 = time.time()
print('Training time: %f seconds' % (t1 - t0))
t0 = time.time()
y_pred = best_clf_nn.predict(X_test_rpData)
t1 = time.time()
test_time = t1 - t0
print('Inference time on test data: %f seconds' % test_time)
best_accuracy = accuracy_score(y_test1, y_pred)
print('Best accuracy of rpData neural network is %.2f%%' % (best_accuracy * 100))

_, train_scores, test_scores = learning_curve(clf_nn, X_train1, y_train1, train_sizes=np.linspace(0.1, 1.0, 5), cv=5)
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
plt.figure()
plt.title('Learning curve for neural network MLP Classifier')
plt.plot(np.linspace(0.1, 1.0, 5), train_scores_mean, 'o-', label='Training score')
plt.plot(np.linspace(0.1, 1.0, 5), test_scores_mean, 'o-', label='Cross Validation score')
plt.xlabel('training set size')
plt.ylabel("score")
plt.legend(loc="best")
plt.grid()
plt.savefig('NN-RP-learning-curve.png')

#Factor Analysis

faData = FactorAnalysis(random_state=3, n_components=6)
faData.fit(X_train1)
X_train_faData = faData.transform(X_train1)
X_test_faData = faData.transform(X_test1)



clf_nn = MLPClassifier(hidden_layer_sizes=(5, 2), random_state=7, max_iter=1000)
clf_nn.fit(X_train_faData, y_train1)
y_pred = clf_nn.predict(X_test_faData)
nn_accuracy = accuracy_score(y_test1, y_pred)
print('Accuracy of neural network without hyperparameter tuning is %.2f%%' % (nn_accuracy * 100))

# Define grid for grid search after observing validation curves
alpha_range = np.logspace(-1, 2, 5)
lr_range = np.logspace(-5, 0, 6)
tuned_params = {'alpha' : alpha_range, 'learning_rate_init' : lr_range}
clf_nn = GridSearchCV(clf_nn, param_grid=tuned_params, cv=5, n_jobs=-1)
clf_nn.fit(X_train_faData, y_train1)
best_clf_nn = clf_nn.best_estimator_
best_params = clf_nn.best_params_
print("Best parameters set found on development set:")
print(best_params)

t0 = time.time()
best_clf_nn.fit(X_train_faData, y_train1)
t1 = time.time()
print('Training time: %f seconds' % (t1 - t0))
t0 = time.time()
y_pred = best_clf_nn.predict(X_test_faData)
t1 = time.time()
test_time = t1 - t0
print('Inference time on test data: %f seconds' % test_time)
best_accuracy = accuracy_score(y_test1, y_pred)
print('Best accuracy of rpData neural network is %.2f%%' % (best_accuracy * 100))


_, train_scores, test_scores = learning_curve(clf_nn, X_train1, y_train1, train_sizes=np.linspace(0.1, 1.0, 5), cv=5)
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
plt.figure()
plt.title('Learning curve for neural network MLP Classifier')
plt.plot(np.linspace(0.1, 1.0, 5), train_scores_mean, 'o-', label='Training score')
plt.plot(np.linspace(0.1, 1.0, 5), test_scores_mean, 'o-', label='Cross Validation score')
plt.xlabel('training set size')
plt.ylabel("score")
plt.legend(loc="best")
plt.grid()
plt.savefig('NN-FA-learning-curve.png')


#KMeans

kmeans = KMeans(n_clusters=10, random_state=random_seed)
kmeans.fit(X_train1)
X_train_kmeans = kmeans.transform(X_train1)
X_test_kmeans = kmeans.transform(X_test1)

clf_nn = MLPClassifier(hidden_layer_sizes=(5, 2), random_state=7, max_iter=1000)
clf_nn.fit(X_train_kmeans, y_train1)
y_pred = clf_nn.predict(X_test_kmeans)
nn_accuracy = accuracy_score(y_test1, y_pred)
print('Accuracy of neural network without hyperparameter tuning is %.2f%%' % (nn_accuracy * 100))


# Define grid for grid search after observing validation curves
alpha_range = np.logspace(-1, 2, 5)
lr_range = np.logspace(-5, 0, 6)
tuned_params = {'alpha' : alpha_range, 'learning_rate_init' : lr_range}
clf_nn = GridSearchCV(clf_nn, param_grid=tuned_params, cv=5, n_jobs=-1)
clf_nn.fit(X_train_kmeans, y_train1)
best_clf_nn = clf_nn.best_estimator_
best_params = clf_nn.best_params_
print("Best parameters set found on development set:")
print(best_params)

t0 = time.time()
best_clf_nn.fit(X_train_kmeans, y_train1)
t1 = time.time()
print('Training time: %f seconds' % (t1 - t0))
t0 = time.time()
y_pred = best_clf_nn.predict(X_test_kmeans)
t1 = time.time()
test_time = t1 - t0
print('Inference time on test data: %f seconds' % test_time)
best_accuracy = accuracy_score(y_test1, y_pred)
print('Best accuracy of neural network is %.2f%%' % (best_accuracy * 100))


_, train_scores, test_scores = learning_curve(clf_nn, X_train1, y_train1, train_sizes=np.linspace(0.1, 1.0, 5), cv=5)
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
plt.figure()
plt.title('Learning curve for neural network MLP Classifier')
plt.plot(np.linspace(0.1, 1.0, 5), train_scores_mean, 'o-', label='Training score')
plt.plot(np.linspace(0.1, 1.0, 5), test_scores_mean, 'o-', label='Cross Validation score')
plt.xlabel('training set size')
plt.ylabel("score")
plt.legend(loc="best")
plt.grid()
plt.savefig('NN-Kmeans-learning-curve.png')

#EM

gmm = GaussianMixture(n_components=7, random_state=random_seed)
gmm.fit(X_train1)
X_train_gmm = gmm.predict_proba(X_train1)
X_test_gmm = gmm.predict_proba(X_test1)


clf_nn = MLPClassifier(hidden_layer_sizes=(5, 2), random_state=7, max_iter=1000)
clf_nn.fit(X_train_gmm, y_train1)
y_pred = clf_nn.predict(X_test_gmm)
nn_accuracy = accuracy_score(y_test1, y_pred)
print('Accuracy of neural network without hyperparameter tuning is %.2f%%' % (nn_accuracy * 100))

# Define grid for grid search after observing validation curves
# alpha_range = np.logspace(-10, -5, 5)
alpha_range = np.asarray([0])
lr_range = np.logspace(-5, 0, 6)
tuned_params = {'alpha' : alpha_range, 'learning_rate_init' : lr_range}
clf_nn = GridSearchCV(clf_nn, param_grid=tuned_params, cv=5, n_jobs=-1)
clf_nn.fit(X_train_gmm, y_train1)
best_clf_nn = clf_nn.best_estimator_
best_params = clf_nn.best_params_
print("Best parameters set found on development set:")
print(best_params)


t0 = time.time()
best_clf_nn.fit(X_train_gmm, y_train1)
t1 = time.time()
print('Training time: %f seconds' % (t1 - t0))
t0 = time.time()
y_pred = best_clf_nn.predict(X_test_gmm)
t1 = time.time()
test_time = t1 - t0
print('Inference time on test data: %f seconds' % test_time)
best_accuracy = accuracy_score(y_test1, y_pred)
print('Best accuracy of neural network is %.2f%%' % (best_accuracy * 100))


_, train_scores, test_scores = learning_curve(clf_nn, X_train1, y_train1, train_sizes=np.linspace(0.1, 1.0, 5), cv=5)
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
plt.figure()
plt.title('Learning curve for neural network MLP Classifier')
plt.plot(np.linspace(0.1, 1.0, 5), train_scores_mean, 'o-', label='Training score')
plt.plot(np.linspace(0.1, 1.0, 5), test_scores_mean, 'o-', label='Cross Validation score')
plt.xlabel('training set size')
plt.ylabel("score")
plt.legend(loc="best")
plt.grid()
plt.savefig('NN-EM-learning-curve.png')

