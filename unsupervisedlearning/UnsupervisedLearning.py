#!/usr/bin/env python3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.decomposition import FactorAnalysis
from scipy.stats import kurtosis 
from sklearn import random_projection
from sklearn.metrics import pairwise
from supervised_learning.data_load import  load_women_diabetes_data, load_heart_stroke_data

import mlrose
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import logging
import time
np.set_printoptions(threshold=sys.maxsize)
"""
References:
https://scikit-learn.org/stable/modules/random_projection.html#gaussian-random-matrix
https://stackoverflow.com/questions/36566844/pca-projection-and-reconstruction-in-scikit-learn
https://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html
https://docs.scipy.org/doc/numpy/reference/generated/numpy.std.html
https://hackr.io/blog/numpy-matrix-multiplication
https://www.geeksforgeeks.org/find-average-list-python/
https://www.geeksforgeeks.org/scipy-stats-kurtosis-function-python/
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.kurtosis.html
https://stackoverflow.com/questions/36566844/pca-projection-and-reconstruction-in-scikit-learn
https://stackabuse.com/implementing-pca-in-python-with-scikit-learn/
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html
https://www.geeksforgeeks.org/adding-new-column-to-existing-dataframe-in-pandas/
https://www.youtube.com/playlist?list=PLBv09BD7ez_7beI0_fuE96lSbsr_8K8YD
https://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition
https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FactorAnalysis.html#sklearn.decomposition.FactorAnalysis
https://stackoverflow.com/questions/4674473/valueerror-setting-an-array-element-with-a-sequence
http://blog.yhat.com/posts/sparse-random-projections.html
https://stackabuse.com/implementing-pca-in-python-with-scikit-learn/
https://stackoverflow.com/questions/31909945/obtain-eigen-values-and-vectors-from-sklearn-pca
https://esigma6.wordpress.com/2018/11/03/2-3-9-3-homogeneity-completeness-and-v-measure/
https://scikit-learn.org/stable/modules/clustering.html
https://www.datasciencecentral.com/profiles/blogs/em-algorithm-explained-in-one-picture
https://www.statisticshowto.datasciencecentral.com/em-algorithm-expectation-maximization/
https://scikit-learn.org/stable/modules/clustering.html
https://sebastianraschka.com/Articles/2015_pca_in_3_steps.html
https://en.wikipedia.org/wiki/Independent_component_analysis
https://scikit-learn.org/stable/modules/random_projection.html#gaussian-random-matrix
https://stackoverflow.com/questions/44915809/k-means-clustering-performance-benchmarking
https://stats.stackexchange.com/questions/81427/aic-guidelines-in-model-selection
"""
"""
To Do Clustering:
1. Read Data in 
2. Split into Train and test sets
3. Optionally apply a standard scaler
4. Expected Maximization
	Loop through the number of components that is set
	1. Create the learner and initialize it
	2. fit the training data
	3. Predict on the testing data
	4. get AIC of Test data
	5. get BIC of Test data
	6. Compute a bunch of different metrics to determine the best number of clusters
	7. Plot them
5. Kmeans
	Loop through the number of components that is set
	1. Create the learner and initialize it
	2. fit the training data
	3. Predict on the testing data
	4. get AIC of Test data
	5. get BIC of Test data
	6. Compute a bunch of different metrics to determine the best number of clusters
	7. Plot them
"""
"""
To Do Other:
1. Run the clustering algorithms on the dataset and describe what I see
2. Apply dimenstionality reduction algorithms to the two datasets and describe what I see
3. Re Run clustering experiments on the newly created data from the dimenstionality reduced data
4. Run Neural Net on data that has been dimenstionality reduced
5. Reapply clustering algos to dimenstionality reduced data and run neural network again
6. Training times
"""
LOG_FILENAME = 'app.log'
logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG)
start = 0
end = 0 
total = 0

def loadDataPoker(trainingPath, testingPath):
	trainingDF = pd.read_csv(trainingPath, sep=',', header=None )
	testingDF = pd.read_csv(testingPath, sep=',', header=None ) 
	dataTraining = trainingDF.iloc[:,:-1]
	dataTesting = testingDF.iloc[:,:-1]
	dataTraining_Class = trainingDF.iloc[:,-1]
	dataTesting_Class = testingDF.iloc[:,-1]
	#logging.debug(dataTraining.shape)
	#logging.debug(dataTraining_Class.shape)
	return dataTraining.head(5000),dataTesting.head(500),dataTraining_Class.head(5000),dataTesting_Class.head(500)

def loadDataDefault(path):
	#70-30
	default = pd.read_excel(path, sep=',')
	default = default.drop(default.index[0])
	tDF = default.iloc[:,:-1]
	ts_Class = default.iloc[:,-1]
	trainingDF,testngDF, training_Class,testing_Class = train_test_split(tDF, ts_Class, test_size = 0.3, random_state = 0)
	training_ClassDF = pd.DataFrame(training_Class)
	testing_ClassDF = pd.DataFrame(testing_Class)
	return trainingDF.drop(trainingDF.columns[0], axis=1), training_ClassDF.iloc[:,-1].astype(int),testngDF.drop(testngDF.columns[0], axis=1),testing_ClassDF.iloc[:,-1].astype(int)

def clustersExperimentsPoker(dataTraining,dataTesting,dataTraining_Class,dataTesting_Class,expName):
	aicScoresEM = []
	bicScoresEM = []
	homoScoresEM = []
	compScoresEM = []
	silScoresEM = []
	scoreEM = []

	homoScoresKM = []
	compScoresKM = []
	silScoresKM = []
	scoreKm = []
	nComponents = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,25,30,40,100,500]
	for item in nComponents:
		em = GaussianMixture(n_components = item, max_iter=1000, covariance_type = 'spherical',\
			tol = .0001, init_params = 'random',random_state = 3,warm_start = False)
		em.fit(dataTraining)
		predictions = em.predict(dataTraining)
		aicScoresEM.append(em.aic(dataTraining))
		bicScoresEM.append(em.bic(dataTraining))
		homoScoresEM.append(metrics.homogeneity_score(dataTraining_Class, predictions))
		compScoresEM.append(metrics.completeness_score(dataTraining_Class, predictions))
		silScoresEM.append(metrics.silhouette_score(dataTraining, predictions, metric='euclidean'))
	
	emActual = GaussianMixture(n_components = 500, max_iter=1000, covariance_type = 'spherical',\
			tol = .0001, init_params = 'random',random_state = 3,warm_start = False)
	emActual.fit(dataTraining)
	predictions=emActual.predict(dataTesting)
	error = 1 - accuracy_score(dataTesting_Class,predictions)
	print (error)
	logging.debug(f'The error is {error}')
	
	for item in nComponents:
		km = KMeans(n_clusters=item,init='k-means++',max_iter=1000,tol=.00001,random_state=3,\
			algorithm='auto')
		km.fit(dataTraining)
		predictions = km.predict(dataTesting)
		homoScoresKM.append(metrics.homogeneity_score(dataTraining_Class, km.labels_))
		compScoresKM.append(metrics.completeness_score(dataTraining_Class, km.labels_))
		scoreKm.append(abs(km.score(dataTraining)))
		silScoresKM.append(metrics.silhouette_score(dataTraining,km.labels_))


	kmActual = KMeans(n_clusters=500,init='k-means++',max_iter=1000,tol=.00001,random_state=3,\
			algorithm='auto')
	kmActual.fit(dataTraining)
	predictions = km.predict(dataTesting)
	error = 1 - accuracy_score(dataTesting_Class,predictions)
	print (error)
	logging.debug(f'The error is {error}')

	q1 = plt.figure(1)
	ax1 = q1.add_subplot(111)
	ax1.plot(nComponents,aicScoresEM,label = 'aic')
	ax1.plot(nComponents,bicScoresEM,label = 'bic')
	ax1.set_ylabel('Scores')
	ax1.set_xlabel('Number of Clusters')
	ax1.legend(loc=0)
	ax1.set_title('AIC Vs BIC')
	q1.savefig(f'{expName}-aicVbic-EM.png')

	q2 = plt.figure(2)
	ax2 = q2.add_subplot(111)
	ax2.plot(nComponents,homoScoresEM,label = 'Homo')
	ax2.plot(nComponents,compScoresEM,label = 'Comp')
	ax2.plot(nComponents,silScoresEM,label = 'Sil')
	ax2.set_ylabel('Scores')
	ax2.set_xlabel('Number of Clusters')
	ax2.legend(loc=0)
	ax2.set_title('Homo vs Comp Scores')
	q2.savefig(f'{expName}-ScoreAnalysis-EM.png')

	q5 = plt.figure(5)
	ax5 = q5.add_subplot(111)
	ax5.plot(nComponents,homoScoresKM,label = 'Homo')
	ax5.plot(nComponents,compScoresKM,label = 'Comp')
	ax5.plot(nComponents,silScoresKM, label = 'Sil')
	ax5.set_ylabel('Scores')
	ax5.set_xlabel('Number of Clusters')
	ax5.legend(loc=0)
	ax5.set_title('Homo vs Comp Scores')
	q5.savefig(f'{expName}-ScoreAnalysis-Kmeans.png')
	
	q6 = plt.figure(6)
	ax6 = q6.add_subplot(111)
	ax6.plot(nComponents,scoreKm,label = 'KM Variance')
	ax6.set_ylabel('Variance')
	ax6.set_xlabel('Number of Clusters')
	ax6.legend(loc=0)
	ax6.set_title('EM vs KM Variance')
	q6.savefig(f'{expName}-VarianceAnalysis-Kmeans.png')

def clustersExperimentsCreditCard(dataTraining,dataTesting,dataTraining_Class,dataTesting_Class,expName):
	aicScores = []
	bicScores = []
	homoScores = []
	compScores = []
	silScoresEM = []

	homoScoresKM = []
	compScoresKM = []
	silScoresKM = []
	scoreKm = []
	nComponents = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,30,40,100,200]
	for item in nComponents:
		em = GaussianMixture(n_components = item, max_iter=1000, covariance_type = 'spherical',\
			tol = .0001, init_params = 'random',random_state = 3,warm_start = False)
		em.fit(dataTraining)
		predictions = em.predict(dataTraining)
		aicScores.append(em.aic(dataTraining))
		bicScores.append(em.bic(dataTraining))
		homoScores.append(metrics.homogeneity_score(dataTraining_Class, predictions))
		compScores.append(metrics.completeness_score(dataTraining_Class, predictions))
		silScoresEM.append(metrics.silhouette_score(dataTraining, predictions, metric='euclidean'))
	
	emActual = GaussianMixture(n_components = 200, max_iter=1000, covariance_type = 'spherical',\
			tol = .0001, init_params = 'random',random_state = 3,warm_start = False)
	emActual.fit(dataTraining)
	predictions=emActual.predict(dataTesting)
	error = 1 - accuracy_score(dataTesting_Class,predictions)
	print (error)
	logging.debug(f'The error is {error}')
	
	for item in nComponents:
		km = KMeans(n_clusters=item,init='k-means++',max_iter=1000,tol=.00001,random_state=3,\
			algorithm='auto')
		km.fit(dataTraining)
		predictions = km.predict(dataTraining)
		homoScoresKM.append(metrics.homogeneity_score(dataTraining_Class, km.labels_))
		compScoresKM.append(metrics.completeness_score(dataTraining_Class, km.labels_))
		silScoresKM.append(metrics.silhouette_score(dataTraining,km.labels_))
		scoreKm.append(abs(km.score(dataTraining)))
	
	kmActual = KMeans(n_clusters=200,init='k-means++',max_iter=1000,tol=.00001,random_state=3,\
			algorithm='auto')
	kmActual.fit(dataTraining)
	predictions = km.predict(dataTesting)
	error = 1 - accuracy_score(dataTesting_Class,predictions)
	print (error)
	logging.debug(f'The error is {error}')

	q3 = plt.figure(3)
	ax3 = q3.add_subplot(111)
	ax3.plot(nComponents,aicScores,label = 'aic')
	ax3.plot(nComponents,bicScores,label = 'bic')
	ax3.set_ylabel('Scores')
	ax3.set_xlabel('Number of Clusters')
	ax3.legend(loc=0)
	ax3.set_title('AIC Vs BIC')
	q3.savefig(f'{expName}-aicVbic.png')

	q4 = plt.figure(4)
	ax4 = q4.add_subplot(111)
	ax4.plot(nComponents,homoScores,label = 'Homo')
	ax4.plot(nComponents,compScores,label = 'Comp')
	ax4.plot(nComponents,silScoresEM,label='Sil')
	ax4.set_ylabel('Scores')
	ax4.set_xlabel('Number of Clusters')
	ax4.legend(loc=0)
	ax4.set_title('Homo vs Comp Scores')
	q4.savefig(f'{expName}-ScoreAnalysis.png')

	q7 = plt.figure(7)
	ax7 = q7.add_subplot(111)
	ax7.plot(nComponents,homoScoresKM,label = 'Homo')
	ax7.plot(nComponents,compScoresKM,label = 'Comp')
	ax7.plot(nComponents,silScoresKM,label='Sil')
	ax7.set_ylabel('Scores')
	ax7.set_xlabel('Number of Clusters')
	ax7.legend(loc=0)
	ax7.set_title('Homo vs Comp Scores')
	q7.savefig(f'{expName}-ScoreAnalysis-Kmeans.png')
	
	q8 = plt.figure(8)
	ax8 = q8.add_subplot(111)
	ax8.plot(nComponents,scoreKm,label = 'Variance')
	ax8.set_ylabel('Variance')
	ax8.set_xlabel('Number of Clusters')
	ax8.legend(loc=0)
	ax8.set_title('KM Variance')
	q8.savefig(f'{expName}-VarianceAnalysis-Kmeans.png')

def principalComponentPoker(dataTraining,dataTesting,dataTraining_Class,dataTesting_Class,expName):
	pca = PCA(random_state=3)
	pca.fit(dataTraining)
	variance = np.cumsum(pca.explained_variance_)
	q9 = plt.figure(9)
	ax9 = q9.add_subplot(111)
	ax9.plot(pca.explained_variance_, variance,label = 'Variance')
	ax9.set_xlabel('Components')
	ax9.set_ylabel('Variance')
	ax10 = ax9.twinx()
	ax10.plot(pca.explained_variance_, pca.explained_variance_,label = 'Eigenvalues')
	ax10.set_ylabel('Eigenvalues')
	plt.title(f"PCA Variance and Eigenvalues: {expName} ")
	q9.savefig(f'{expName}-PCAVarianceandEigenvalues.png')
	blagh = pca.n_components_
	logging.debug(f'Recommended components {blagh}')

def principalComponentCredit(dataTraining,dataTraining_Class,dataTesting,dataTesting_Class,expName):
	pca = PCA(random_state=3)
	pca.fit(dataTraining)
	variance = np.cumsum(pca.explained_variance_)
	q90 = plt.figure(90)
	ax90 = q90.add_subplot(111)
	ax90.plot(pca.explained_variance_, variance,label = 'Variance')
	ax90.set_xlabel('Components')
	ax90.set_ylabel('Variance')
	ax100 = ax90.twinx()
	ax100.plot(pca.explained_variance_, pca.explained_variance_,label = 'Eigenvalues')
	ax100.set_ylabel('Eigenvalues')
	plt.title(f"PCA Variance and Eigenvalues: {expName} ")
	q90.savefig(f'{expName}-PCAVarianceandEigenvalues.png')
	blagh = pca.n_components_
	logging.debug(f'Recommended components {blagh}')

def independentComponentPoker(dataTraining,dataTesting,dataTraining_Class,dataTesting_Class,expName):
	nComponents = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,30,40,100]
	ica = FastICA(random_state = 3)
	kurtosisList = []
	for item in nComponents:
		ica.set_params(n_components=item)
		asdf = ica.fit_transform(dataTraining)
		kurtosisList.append(np.mean(kurtosis(asdf)))

	q11 = plt.figure(11)
	ax11 = q11.add_subplot(111)
	ax11.plot(nComponents,kurtosisList,label = 'Kurtosis')
	ax11.set_ylabel('Kurtosis')
	ax11.set_xlabel('Components')
	ax11.legend(loc=0)
	ax11.set_title('ICA Kurtosis')
	q11.savefig(f'{expName}-Kurtosis-ICA.png')

def independentComponentCredit(dataTraining,dataTraining_Class,dataTesting,dataTesting_Class,expName):
	nComponents = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,30,40,100]
	ica = FastICA(random_state = 3)
	kurtosisList = []
	for item in nComponents:
		ica.set_params(n_components=item)
		asdf = ica.fit_transform(dataTraining)
		kurtosisList.append(np.mean(kurtosis(asdf)))

	q111 = plt.figure(111)
	ax111 = q111.add_subplot(111)
	ax111.plot(nComponents,kurtosisList,label = 'Kurtosis')
	ax111.set_ylabel('Kurtosis')
	ax111.set_xlabel('Components')
	ax111.legend(loc=0)
	ax111.set_title('ICA Kurtosis')
	q111.savefig(f'{expName}-Kurtosis-ICA.png')

def randomProjectionsPoker(dataTraining,dataTesting,dataTraining_Class,dataTesting_Class,expName):
	nComponents = [2,3,4,5,6,7,8,9,10,20]#,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,30,40
	reconError = []
	meanError = []
	stdError = []
	for item in nComponents:
		rp = random_projection.GaussianRandomProjection(n_components = item)
		X_transformed = rp.fit_transform(dataTraining)
		X_projected = (X_transformed.dot(rp.components_)) - dataTraining
		reconError.append(((dataTraining - X_projected) ** 2).mean())
		meanError.append(np.mean(((dataTraining - X_projected) ** 2).mean()))
		stdError.append(np.std(((dataTraining - X_projected) ** 2).mean())) 
	q12 = plt.figure(12)
	ax12 = q12.add_subplot(111)
	ax12.plot(nComponents,meanError, label = 'Mean')
	ax12.plot(nComponents,stdError, label = 'STD')
	ax12.set_ylabel('Error')
	ax12.set_xlabel('Components')
	ax12.legend(loc=0)
	ax12.set_title('Random Projection Error')
	q12.savefig(f'{expName}-Error-RP.png')

def randomProjectionsCredit(dataTraining,dataTraining_Class,dataTesting,dataTesting_Class,expName):
	nComponents = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,23,30]#,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,30,40
	reconError = []
	meanError = []
	stdError = []
	for item in nComponents:
		rp = random_projection.GaussianRandomProjection(n_components = item)
		X_transformed = rp.fit_transform(dataTraining)
		X_projected = (X_transformed.dot(rp.components_)) - dataTraining
		reconError.append(((dataTraining - X_projected) ** 2).mean())
		meanError.append(np.mean(((dataTraining - X_projected) ** 2).mean()))
		stdError.append(np.std(((dataTraining - X_projected) ** 2).mean())) 
	q122 = plt.figure(122)
	ax122 = q122.add_subplot(111)
	ax122.plot(nComponents,meanError, label = 'Mean')
	ax122.plot(nComponents,stdError, label = 'STD')
	ax122.set_ylabel('Error')
	ax122.set_xlabel('Components')
	ax122.legend(loc=0)
	ax122.set_title('Random Projection Error')
	q122.savefig(f'{expName}-Error-RP.png')

def factorAnalysisPoker(dataTraining,dataTesting,dataTraining_Class,dataTesting_Class,expName):
	nComponents = [2,3,4,5,6,7,8,9,10,11,12,13,14,15]#,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,30,40
	reconError = []
	meanError = []
	stdError = []
	for item in nComponents:
		fa = FactorAnalysis(n_components = item)
		X_transformed = fa.fit_transform(dataTraining)
		X_projected = (X_transformed.dot(fa.components_)) - dataTraining
		reconError.append(((dataTraining - X_projected) ** 2).mean())
		meanError.append(np.mean(((dataTraining - X_projected) ** 2).mean()))
		stdError.append(np.std(((dataTraining - X_projected) ** 2).mean())) 
	q13 = plt.figure(13)
	ax13 = q13.add_subplot(111)
	ax13.plot(nComponents,meanError, label = 'Mean')
	ax13.plot(nComponents,stdError, label = 'STD')
	ax13.set_ylabel('Error')
	ax13.set_xlabel('Components')
	ax13.legend(loc=0)
	ax13.set_title('Factor Analysis Error')
	q13.savefig(f'{expName}-Error-FA.png')

def factorAnalysisCredit(dataTraining,dataTraining_Class,dataTesting,dataTesting_Class,expName):
	nComponents = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,30]#,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,30,40
	reconError = []
	meanError = []
	stdError = []
	for item in nComponents:
		fa = FactorAnalysis(n_components = item)
		X_transformed = fa.fit_transform(dataTraining)
		X_projected = (X_transformed.dot(fa.components_)) - dataTraining
		reconError.append(((dataTraining - X_projected) ** 2).mean())
		meanError.append(np.mean(((dataTraining - X_projected) ** 2).mean()))
		stdError.append(np.std(((dataTraining - X_projected) ** 2).mean())) 
	q133 = plt.figure(133)
	ax133 = q133.add_subplot(111)
	ax133.plot(nComponents,meanError, label = 'Mean')
	ax133.plot(nComponents,stdError, label = 'STD')
	ax133.set_ylabel('Error')
	ax133.set_xlabel('Components')
	ax133.legend(loc=0)
	ax133.set_title('Factor Analysis Error')
	q133.savefig(f'{expName}-Error-FA.png')

def kmeansPartPoker(dataTraining,dataTesting,dataTraining_Class,dataTesting_Class,expName):
	"""
	To do:
	add some different eval measures like training and testing set accuracy etc
	get training times as well
	"""
	nComponents = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,30,40,100,200]
	rpData = random_projection.GaussianRandomProjection(random_state=3,n_components = 10)\
	.fit_transform(dataTraining)
	icaData = FastICA(random_state = 3,n_components=40).fit_transform(dataTraining)
	pcaData = PCA(random_state=3,n_components=10).fit_transform(dataTraining)
	faData = FactorAnalysis(random_state=3,n_components = 10).fit_transform(dataTraining)
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
		km = KMeans(n_clusters=item,init='k-means++',max_iter=1000,tol=.00001,random_state=3,\
			algorithm='auto')
		km.fit(rpData)
		predictions = km.predict(rpData)
		homoScoresKMrpData.append(metrics.homogeneity_score(dataTraining_Class, km.labels_))
		compScoresKMrpData.append(metrics.completeness_score(dataTraining_Class, km.labels_))
		scoreKmrpData.append(abs(km.score(rpData)))
		silScoresKMrpData.append(metrics.silhouette_score(rpData,km.labels_))
	for item in nComponents:
		km = KMeans(n_clusters=item,init='k-means++',max_iter=1000,tol=.00001,random_state=3,\
			algorithm='auto')
		km.fit(icaData)
		predictions = km.predict(icaData)
		homoScoresKMicaData.append(metrics.homogeneity_score(dataTraining_Class, km.labels_))
		compScoresKMicaData.append(metrics.completeness_score(dataTraining_Class, km.labels_))
		scoreKmicaData.append(abs(km.score(icaData)))
		silScoresKMicaData.append(metrics.silhouette_score(icaData,km.labels_))
	for item in nComponents:
		km = KMeans(n_clusters=item,init='k-means++',max_iter=1000,tol=.00001,random_state=3,\
			algorithm='auto')
		km.fit(pcaData)
		predictions = km.predict(pcaData)
		homoScoresKMpcaData.append(metrics.homogeneity_score(dataTraining_Class, km.labels_))
		compScoresKMpcaData.append(metrics.completeness_score(dataTraining_Class, km.labels_))
		scoreKmpcaData.append(abs(km.score(pcaData)))
		silScoresKMpcaData.append(metrics.silhouette_score(pcaData,km.labels_))
	for item in nComponents:
		km = KMeans(n_clusters=item,init='k-means++',max_iter=1000,tol=.00001,random_state=3,\
			algorithm='auto')
		km.fit(faData)
		predictions = km.predict(faData)
		homoScoresKMfaData.append(metrics.homogeneity_score(dataTraining_Class, km.labels_))
		compScoresKMfaData.append(metrics.completeness_score(dataTraining_Class, km.labels_))
		scoreKmfaData.append(abs(km.score(faData)))
		silScoresKMfaData.append(metrics.silhouette_score(faData,km.labels_))
	
	q14 = plt.figure(14)
	ax14 = q14.add_subplot(111)
	ax14.plot(nComponents,homoScoresKMpcaData , label = 'PCA')
	ax14.plot(nComponents,homoScoresKMfaData , label = 'FA')
	ax14.plot(nComponents,homoScoresKMicaData , label = 'ICA')
	ax14.plot(nComponents,homoScoresKMrpData , label = 'RP')
	ax14.set_ylabel('Score')
	ax14.set_xlabel('Components')
	ax14.legend(loc=0)
	ax14.set_title('Kmeans With DR - Homo')
	q14.savefig(f'{expName}-Kmeans-DR-Homo.png')
	
	q15 = plt.figure(15)
	ax15 = q15.add_subplot(111)
	ax15.plot(nComponents,compScoresKMfaData , label = 'FA')
	ax15.plot(nComponents,compScoresKMpcaData , label = 'PCA')
	ax15.plot(nComponents,compScoresKMicaData , label = 'ICA')
	ax15.plot(nComponents,compScoresKMrpData , label = 'RP')
	ax15.set_ylabel('Score')
	ax15.set_xlabel('Components')
	ax15.legend(loc=0)
	ax15.set_title('Kmeans With DR - Comp')
	q15.savefig(f'{expName}-Kmeans-DR-CO.png')

	q16 = plt.figure(16)
	ax16 = q16.add_subplot(111)
	ax16.plot(nComponents,silScoresKMfaData , label = 'FA')
	ax16.plot(nComponents,silScoresKMpcaData , label = 'PCA')
	ax16.plot(nComponents,silScoresKMicaData , label = 'ICA')
	ax16.plot(nComponents,silScoresKMrpData , label = 'RP')
	ax16.set_ylabel('Score')
	ax16.set_xlabel('Components')
	ax16.legend(loc=0)
	ax16.set_title('Kmeans With DR - Sil')
	q16.savefig(f'{expName}-Kmeans-DR-Sil.png')

def kmeansPartCredit(dataTraining,dataTraining_Class,dataTesting,dataTesting_Class,expName):
	"""
	To do:
	add some different eval measures like training and testing set accuracy etc
	get training times as well
	"""
	nComponents = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,30,40,100,200]
	rpData = random_projection.GaussianRandomProjection(random_state=3,n_components = 200)\
	.fit_transform(dataTraining)
	icaData = FastICA(random_state = 3,n_components=200).fit_transform(dataTraining)
	pcaData = PCA(random_state=3,n_components=10).fit_transform(dataTraining)
	faData = FactorAnalysis(random_state=3,n_components = 200).fit_transform(dataTraining)
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
		km = KMeans(n_clusters=item,init='k-means++',max_iter=1000,tol=.00001,random_state=3,\
			algorithm='auto')
		km.fit(rpData)
		predictions = km.predict(rpData)
		homoScoresKMrpData.append(metrics.homogeneity_score(dataTraining_Class, km.labels_))
		compScoresKMrpData.append(metrics.completeness_score(dataTraining_Class, km.labels_))
		scoreKmrpData.append(km.score(rpData))
		silScoresKMrpData.append(metrics.silhouette_score(rpData,km.labels_))
	for item in nComponents:
		km = KMeans(n_clusters=item,init='k-means++',max_iter=1000,tol=.00001,random_state=3,\
			algorithm='auto')
		km.fit(icaData)
		predictions = km.predict(icaData)
		homoScoresKMicaData.append(metrics.homogeneity_score(dataTraining_Class, km.labels_))
		compScoresKMicaData.append(metrics.completeness_score(dataTraining_Class, km.labels_))
		scoreKmicaData.append(km.score(icaData))
		silScoresKMicaData.append(metrics.silhouette_score(icaData,km.labels_))
	for item in nComponents:
		km = KMeans(n_clusters=item,init='k-means++',max_iter=1000,tol=.00001,random_state=3,\
			algorithm='auto')
		km.fit(pcaData)
		predictions = km.predict(pcaData)
		homoScoresKMpcaData.append(metrics.homogeneity_score(dataTraining_Class, km.labels_))
		compScoresKMpcaData.append(metrics.completeness_score(dataTraining_Class, km.labels_))
		scoreKmpcaData.append(km.score(pcaData))
		silScoresKMpcaData.append(metrics.silhouette_score(pcaData,km.labels_))
	for item in nComponents:
		km = KMeans(n_clusters=item,init='k-means++',max_iter=1000,tol=.00001,random_state=3,\
			algorithm='auto')
		km.fit(faData)
		predictions = km.predict(faData)
		homoScoresKMfaData.append(metrics.homogeneity_score(dataTraining_Class, km.labels_))
		compScoresKMfaData.append(metrics.completeness_score(dataTraining_Class, km.labels_))
		scoreKmfaData.append(km.score(faData))
		silScoresKMfaData.append(metrics.silhouette_score(faData,km.labels_))
	
	q144 = plt.figure(144)
	ax144 = q144.add_subplot(111)
	ax144.plot(nComponents,homoScoresKMpcaData , label = 'PCA')
	ax144.plot(nComponents,homoScoresKMfaData , label = 'FA')
	ax144.plot(nComponents,homoScoresKMicaData , label = 'ICA')
	ax144.plot(nComponents,homoScoresKMrpData , label = 'RP')
	ax144.set_ylabel('Score')
	ax144.set_xlabel('Components')
	ax144.legend(loc=0)
	ax144.set_title('Kmeans With DR - Homo')
	q144.savefig(f'{expName}-Kmeans-DR-Homo.png')
	
	q155 = plt.figure(155)
	ax155 = q155.add_subplot(111)
	ax155.plot(nComponents,compScoresKMfaData , label = 'FA')
	ax155.plot(nComponents,compScoresKMpcaData , label = 'PCA')
	ax155.plot(nComponents,compScoresKMicaData , label = 'ICA')
	ax155.plot(nComponents,compScoresKMrpData , label = 'RP')
	ax155.set_ylabel('Score')
	ax155.set_xlabel('Components')
	ax155.legend(loc=0)
	ax155.set_title('Kmeans With DR - Comp')
	q155.savefig(f'{expName}-Kmeans-DR-CO.png')

	q166 = plt.figure(166)
	ax166 = q166.add_subplot(111)
	ax166.plot(nComponents,silScoresKMfaData , label = 'FA')
	ax166.plot(nComponents,silScoresKMpcaData , label = 'PCA')
	ax166.plot(nComponents,silScoresKMicaData , label = 'ICA')
	ax166.plot(nComponents,silScoresKMrpData , label = 'RP')
	ax166.set_ylabel('Score')
	ax166.set_xlabel('Components')
	ax166.legend(loc=0)
	ax166.set_title('Kmeans With DR - Sil')
	q166.savefig(f'{expName}-Kmeans-DR-Sil.png')

def expectedMaxPartPoker(dataTraining,dataTesting,dataTraining_Class,dataTesting_Class,expName):
	"""
	To do:
	add some different eval measures like training and testing set accuracy etc
	get training times as well
	why use that distance function
	"""
	nComponents = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,30,40,100,200]
	rpData = random_projection.GaussianRandomProjection(random_state=3,n_components = 40)\
	.fit_transform(dataTraining)
	icaData = FastICA(random_state = 3,n_components=40).fit_transform(dataTraining)
	pcaData = PCA(random_state=3,n_components=10).fit_transform(dataTraining)
	faData = FactorAnalysis(random_state=3,n_components = 40).fit_transform(dataTraining)
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
		em = GaussianMixture(n_components = item, max_iter=1000, covariance_type = 'spherical',\
			tol = .0001, init_params = 'random',random_state = 3,warm_start = False)
		em.fit(rpData)
		predictions = em.predict(rpData)
		aicScoresEMrpData.append(em.aic(rpData))
		bicScoresEMrpData.append(em.bic(rpData))
		homoScoresEMrpData.append(metrics.homogeneity_score(dataTraining_Class, predictions))
		compScoresEMrpData.append(metrics.completeness_score(dataTraining_Class, predictions))
		silScoresEMrpData.append(metrics.silhouette_score(rpData, predictions, metric='euclidean'))
	for item in nComponents:
		em = GaussianMixture(n_components = item, max_iter=1000, covariance_type = 'spherical',\
			tol = .0001, init_params = 'random',random_state = 3,warm_start = False)
		em.fit(icaData)
		predictions = em.predict(icaData)
		aicScoresEMicaData.append(em.aic(icaData))
		bicScoresEMicaData.append(em.bic(icaData))
		homoScoresEMicaData.append(metrics.homogeneity_score(dataTraining_Class, predictions))
		compScoresEMicaData.append(metrics.completeness_score(dataTraining_Class, predictions))
		silScoresEMicaData.append(metrics.silhouette_score(icaData, predictions, metric='euclidean'))
	for item in nComponents:
		em = GaussianMixture(n_components = item, max_iter=1000, covariance_type = 'spherical',\
			tol = .0001, init_params = 'random',random_state = 3,warm_start = False)
		em.fit(pcaData)
		predictions = em.predict(pcaData)
		aicScoresEMpcaData.append(em.aic(pcaData))
		bicScoresEMpcaData.append(em.bic(pcaData))
		homoScoresEMpcaData.append(metrics.homogeneity_score(dataTraining_Class, predictions))
		compScoresEMpcaData.append(metrics.completeness_score(dataTraining_Class, predictions))
		silScoresEMpcaData.append(metrics.silhouette_score(pcaData, predictions, metric='euclidean'))
	for item in nComponents:
		em = GaussianMixture(n_components = item, max_iter=1000, covariance_type = 'spherical',\
			tol = .0001, init_params = 'random',random_state = 3,warm_start = False)
		em.fit(faData)
		predictions = em.predict(faData)
		aicScoresEMfaData.append(em.aic(faData))
		bicScoresEMfaData.append(em.bic(faData))
		homoScoresEMfaData.append(metrics.homogeneity_score(dataTraining_Class, predictions))
		compScoresEMfaData.append(metrics.completeness_score(dataTraining_Class, predictions))
		silScoresEMfaData.append(metrics.silhouette_score(faData, predictions, metric='euclidean'))
	q17 = plt.figure(17)
	ax17 = q17.add_subplot(111)
	ax17.plot(nComponents,homoScoresEMpcaData , label = 'PCA')
	ax17.plot(nComponents,homoScoresEMfaData , label = 'FA')
	ax17.plot(nComponents,homoScoresEMicaData , label = 'ICA')
	ax17.plot(nComponents,homoScoresEMrpData , label = 'RP')
	ax17.set_ylabel('Score')
	ax17.set_xlabel('Components')
	ax17.legend(loc=0)
	ax17.set_title('EM With DR - Homo')
	q17.savefig(f'{expName}-EM-DR-Homo.png')
	
	q18 = plt.figure(18)
	ax18 = q18.add_subplot(111)
	ax18.plot(nComponents,compScoresEMfaData , label = 'FA')
	ax18.plot(nComponents,compScoresEMpcaData , label = 'PCA')
	ax18.plot(nComponents,compScoresEMicaData , label = 'ICA')
	ax18.plot(nComponents,compScoresEMrpData , label = 'RP')
	ax18.set_ylabel('Score')
	ax18.set_xlabel('Components')
	ax18.legend(loc=0)
	ax18.set_title('EM With DR - Comp')
	q18.savefig(f'{expName}-EM-DR-CO.png')

	q19 = plt.figure(19)
	ax19 = q19.add_subplot(111)
	ax19.plot(nComponents,silScoresEMfaData , label = 'FA')
	ax19.plot(nComponents,silScoresEMpcaData , label = 'PCA')
	ax19.plot(nComponents,silScoresEMicaData , label = 'ICA')
	ax19.plot(nComponents,silScoresEMrpData , label = 'RP')
	ax19.set_ylabel('Score')
	ax19.set_xlabel('Components')
	ax19.legend(loc=0)
	ax19.set_title('EM With DR - Sil')
	q19.savefig(f'{expName}-EM-DR-Sil.png')

	q20 = plt.figure(20)
	ax20 = q20.add_subplot(111)
	ax20.plot(nComponents,aicScoresEMfaData , label = 'FA')
	ax20.plot(nComponents,aicScoresEMpcaData , label = 'PCA')
	ax20.plot(nComponents,aicScoresEMicaData , label = 'ICA')
	ax20.plot(nComponents,aicScoresEMrpData , label = 'RP')
	ax20.set_ylabel('Score')
	ax20.set_xlabel('Components')
	ax20.legend(loc=0)
	ax20.set_title('EM With DR - AIC')
	q20.savefig(f'{expName}-EM-DR-AIC.png')

	q21 = plt.figure(21)
	ax21 = q21.add_subplot(111)
	ax21.plot(nComponents,bicScoresEMfaData , label = 'FA')
	ax21.plot(nComponents,bicScoresEMpcaData , label = 'PCA')
	ax21.plot(nComponents,bicScoresEMicaData , label = 'ICA')
	ax21.plot(nComponents,bicScoresEMrpData , label = 'RP')
	ax21.set_ylabel('Score')
	ax21.set_xlabel('Components')
	ax21.legend(loc=0)
	ax21.set_title('EM With DR - BIC')
	q21.savefig(f'{expName}-EM-DR-BIC.png')

def expectedMaxPartCredit(dataTraining,dataTraining_Class,dataTesting,dataTesting_Class,expName):
	"""
	To do:
	add some different eval measures like training and testing set accuracy etc
	get training times as well
	why use that distance function
	"""
	nComponents = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,30,40,100,200]
	rpData = random_projection.GaussianRandomProjection(random_state=3,n_components = 200)\
	.fit_transform(dataTraining)
	icaData = FastICA(random_state = 3,n_components=200).fit_transform(dataTraining)
	pcaData = PCA(random_state=3,n_components=10).fit_transform(dataTraining)
	faData = FactorAnalysis(random_state=3,n_components = 200).fit_transform(dataTraining)
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
		em = GaussianMixture(n_components = item, max_iter=1000, covariance_type = 'spherical',\
			tol = .0001, init_params = 'random',random_state = 3,warm_start = False)
		em.fit(rpData)
		predictions = em.predict(rpData)
		aicScoresEMrpData.append(em.aic(rpData))
		bicScoresEMrpData.append(em.bic(rpData))
		homoScoresEMrpData.append(metrics.homogeneity_score(dataTraining_Class, predictions))
		compScoresEMrpData.append(metrics.completeness_score(dataTraining_Class, predictions))
		silScoresEMrpData.append(metrics.silhouette_score(rpData, predictions, metric='euclidean'))
	for item in nComponents:
		em = GaussianMixture(n_components = item, max_iter=1000, covariance_type = 'spherical',\
			tol = .0001, init_params = 'random',random_state = 3,warm_start = False)
		em.fit(icaData)
		predictions = em.predict(icaData)
		aicScoresEMicaData.append(em.aic(icaData))
		bicScoresEMicaData.append(em.bic(icaData))
		homoScoresEMicaData.append(metrics.homogeneity_score(dataTraining_Class, predictions))
		compScoresEMicaData.append(metrics.completeness_score(dataTraining_Class, predictions))
		silScoresEMicaData.append(metrics.silhouette_score(icaData, predictions, metric='euclidean'))
	for item in nComponents:
		em = GaussianMixture(n_components = item, max_iter=1000, covariance_type = 'spherical',\
			tol = .0001, init_params = 'random',random_state = 3,warm_start = False)
		em.fit(pcaData)
		predictions = em.predict(pcaData)
		aicScoresEMpcaData.append(em.aic(pcaData))
		bicScoresEMpcaData.append(em.bic(pcaData))
		homoScoresEMpcaData.append(metrics.homogeneity_score(dataTraining_Class, predictions))
		compScoresEMpcaData.append(metrics.completeness_score(dataTraining_Class, predictions))
		silScoresEMpcaData.append(metrics.silhouette_score(pcaData, predictions, metric='euclidean'))
	for item in nComponents:
		em = GaussianMixture(n_components = item, max_iter=1000, covariance_type = 'spherical',\
			tol = .0001, init_params = 'random',random_state = 3,warm_start = False)
		em.fit(faData)
		predictions = em.predict(faData)
		aicScoresEMfaData.append(em.aic(faData))
		bicScoresEMfaData.append(em.bic(faData))
		homoScoresEMfaData.append(metrics.homogeneity_score(dataTraining_Class, predictions))
		compScoresEMfaData.append(metrics.completeness_score(dataTraining_Class, predictions))
		silScoresEMfaData.append(metrics.silhouette_score(faData, predictions, metric='euclidean'))
	q177 = plt.figure(177)
	ax177 = q177.add_subplot(111)
	ax177.plot(nComponents,homoScoresEMpcaData , label = 'PCA')
	ax177.plot(nComponents,homoScoresEMfaData , label = 'FA')
	ax177.plot(nComponents,homoScoresEMicaData , label = 'ICA')
	ax177.plot(nComponents,homoScoresEMrpData , label = 'RP')
	ax177.set_ylabel('Score')
	ax177.set_xlabel('Components')
	ax177.legend(loc=0)
	ax177.set_title('EM With DR - Homo')
	q177.savefig(f'{expName}-EM-DR-Homo.png')
	
	q188 = plt.figure(188)
	ax188 = q188.add_subplot(111)
	ax188.plot(nComponents,compScoresEMfaData , label = 'FA')
	ax188.plot(nComponents,compScoresEMpcaData , label = 'PCA')
	ax188.plot(nComponents,compScoresEMicaData , label = 'ICA')
	ax188.plot(nComponents,compScoresEMrpData , label = 'RP')
	ax188.set_ylabel('Score')
	ax188.set_xlabel('Components')
	ax188.legend(loc=0)
	ax188.set_title('EM With DR - Comp')
	q188.savefig(f'{expName}-EM-DR-CO.png')

	q199 = plt.figure(199)
	ax199 = q199.add_subplot(111)
	ax199.plot(nComponents,silScoresEMfaData , label = 'FA')
	ax199.plot(nComponents,silScoresEMpcaData , label = 'PCA')
	ax199.plot(nComponents,silScoresEMicaData , label = 'ICA')
	ax199.plot(nComponents,silScoresEMrpData , label = 'RP')
	ax199.set_ylabel('Score')
	ax199.set_xlabel('Components')
	ax199.legend(loc=0)
	ax199.set_title('EM With DR - Sil')
	q199.savefig(f'{expName}-EM-DR-Sil.png')

	q200 = plt.figure(200)
	ax200 = q200.add_subplot(111)
	ax200.plot(nComponents,aicScoresEMfaData , label = 'FA')
	ax200.plot(nComponents,aicScoresEMpcaData , label = 'PCA')
	ax200.plot(nComponents,aicScoresEMicaData , label = 'ICA')
	ax200.plot(nComponents,aicScoresEMrpData , label = 'RP')
	ax200.set_ylabel('Score')
	ax200.set_xlabel('Components')
	ax200.legend(loc=0)
	ax200.set_title('EM With DR - AIC')
	q200.savefig(f'{expName}-EM-DR-AIC.png')

	q211 = plt.figure(211)
	ax211 = q211.add_subplot(111)
	ax211.plot(nComponents,bicScoresEMfaData , label = 'FA')
	ax211.plot(nComponents,bicScoresEMpcaData , label = 'PCA')
	ax211.plot(nComponents,bicScoresEMicaData , label = 'ICA')
	ax211.plot(nComponents,bicScoresEMrpData , label = 'RP')
	ax211.set_ylabel('Score')
	ax211.set_xlabel('Components')
	ax211.legend(loc=0)
	ax211.set_title('EM With DR - BIC')
	q211.savefig(f'{expName}-EM-DR-BIC.png')

def clusteredNeuralNetwork(dataTraining,dataTraining_Class,dataTesting,dataTesting_Class,expName):
	em = GaussianMixture(n_components = 40, max_iter=1000, covariance_type = 'spherical',\
			tol = .0001, init_params = 'random',random_state = 3,warm_start = False)
	em.fit(dataTraining)
	predictionsEM = em.predict(dataTraining)

	km = KMeans(n_clusters=40,init='k-means++',max_iter=1000,tol=.00001,random_state=3,\
			algorithm='auto')
	km.fit(dataTraining)

	emTest = GaussianMixture(n_components = 40, max_iter=1000, covariance_type = 'spherical',\
			tol = .0001, init_params = 'random',random_state = 3,warm_start = False)
	emTest.fit(dataTesting)
	predictionsEMTest = emTest.predict(dataTesting)
	kmTest = KMeans(n_clusters=40,init='k-means++',max_iter=1000,tol=.00001,random_state=3,\
			algorithm='auto')
	kmTest.fit(dataTesting)
 
	dataTraining['23'] = km.labels_
	dataTrainingKM = dataTraining
	dataTraining = dataTraining.drop(columns = ['23'])

	dataTraining['23'] = predictionsEM
	dataTrainingEM = dataTraining
	dataTraining = dataTraining.drop(columns = ['23'])

	dataTesting['23'] = kmTest.labels_
	dataTestingKM =  dataTesting
	dataTesting = dataTesting.drop(columns = ['23'])

	dataTesting['23'] = predictionsEMTest
	dataTestingEM =  dataTesting
	dataTesting = dataTesting.drop(columns = ['23'])

	logging.debug("_______________Starting Neural Network-CCDefault-KM__________________")
	parameters = {'activation':('identity','logistic','tanh','relu'),\
	'alpha':[0.0001,0.001,0.01],\
	'learning_rate':('constant','invscaling','adaptive'),\
	'learning_rate_init':[0.0001,0.001,0.01,.1],\
	'hidden_layer_sizes':[(5,), (10,), (15,)]}
	cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
	dt = MLPClassifier(max_iter=2000)
	start = time.time()
	clf = GridSearchCV(dt, parameters, cv=5,n_jobs=-1)
	clf.fit(dataTrainingKM, dataTraining_Class)
	end = time.time()
	total = end - start
	logging.debug(f'Neural Network Grid Search Execution Time: {total}')
	logging.debug("-------------Best Estimator NN-------------")
	logging.debug(clf.best_estimator_)
	logging.debug("-------------Best Params NN-------------")
	logging.debug(clf.best_params_)
	logging.debug("-------------Best Score NN-------------")
	logging.debug(clf.best_score_)
	logging.debug("-------------Starting Learning Curve-CCDefault---------------")
	start = time.time()
	train_sizes2, train_scores2, test_scores2 =\
	 learning_curve(clf.best_estimator_,\
	 dataTraining, dataTraining_Class, cv=cv, n_jobs=-1,\
	  train_sizes=[.2,.3,.4,.5,.6,.7,.8,.9])

	#train_sizes=np.arange(start=2,stop=5000)
	train_scores_mean2 = np.mean(train_scores2,axis=1)
	train_scores_std2 = np.std(train_scores2, axis=1)
	test_scores_mean2 = np.mean(test_scores2, axis=1)
	test_scores_std2 = np.std(test_scores2, axis=1)
	
	q48 = plt.figure(48)
	ax48 = q48.add_subplot(111)
	ax48.plot(train_sizes2,1 - train_scores_mean2,label = 'Training Error')
	ax48.plot(train_sizes2,1 - test_scores_mean2,label = 'Cross Validation Error')
	ax48.set_ylabel('Error')
	ax48.set_xlabel('Data Size')
	ax48.set_title('Learning Curve Neural Network - CCDefault')
	ax48.legend(loc=0)
	q48.savefig('LearningCurve-NN-CCDefault-KM.png')
	end = time.time()
	total= end - start
	logging.debug(f'Neural Network Learning Curve Execution Time: {total}')
	prediction = clf.predict(dataTestingKM)
	accuracy = accuracy_score(dataTesting_Class,prediction)
	logging.debug(f'LearningCurve-NN-CCDefault-KM accuracy: {accuracy}')
	logging.debug('')
	logging.debug("_______________Starting Neural Network-CCDefault-EM__________________")
	parameters = {'activation':('identity','logistic','tanh','relu'),\
	'alpha':[0.0001,0.001,0.01],\
	'learning_rate':('constant','invscaling','adaptive'),\
	'learning_rate_init':[0.0001,0.001,0.01,.1],\
	'hidden_layer_sizes':[(5,), (10,), (15,)]}
	cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
	dt = MLPClassifier(max_iter=2000)
	start = time.time()
	clf = GridSearchCV(dt, parameters, cv=5,n_jobs=-1)
	clf.fit(dataTrainingEM, dataTraining_Class)
	end = time.time()
	total = end - start
	logging.debug(f'Neural Network Grid Search Execution Time: {total}')
	logging.debug("-------------Best Estimator NN-------------")
	logging.debug(clf.best_estimator_)
	logging.debug("-------------Best Params NN-------------")
	logging.debug(clf.best_params_)
	logging.debug("-------------Best Score NN-------------")
	logging.debug(clf.best_score_)
	logging.debug("-------------Starting Learning Curve-CCDefault---------------")
	start = time.time()
	train_sizes2, train_scores2, test_scores2 =\
	 learning_curve(clf.best_estimator_,\
	 dataTraining, dataTraining_Class, cv=cv, n_jobs=14,\
	  train_sizes=[.2,.3,.4,.5,.6,.7,.8,.9])

	train_scores_mean2 = np.mean(train_scores2,axis=1)
	train_scores_std2 = np.std(train_scores2, axis=1)
	test_scores_mean2 = np.mean(test_scores2, axis=1)
	test_scores_std2 = np.std(test_scores2, axis=1)

	q49 = plt.figure(49)
	ax49 = q49.add_subplot(111)
	ax49.plot(train_sizes2,1 - train_scores_mean2,label = 'Training Error')
	ax49.plot(train_sizes2,1 - test_scores_mean2,label = 'Cross Validation Error')
	ax49.set_ylabel('Error')
	ax49.set_xlabel('Data Size')
	ax49.set_title('Learning Curve Neural Network - CCDefault - EM')
	ax49.legend(loc=0)
	q49.savefig('LearningCurve-NN-CCDefault-EM.png')
	end = time.time()
	total= end - start
	logging.debug(f'Neural Network Learning Curve Execution Time: {total}') 
	prediction = clf.predict(dataTestingEM)
	accuracy = accuracy_score(dataTesting_Class,prediction)
	logging.debug(f'LearningCurve-NN-CCDefault-EM accuracy: {accuracy}')

def drNeuralNetwork(dataTraining,dataTraining_Class,dataTesting,dataTesting_Class,expName):
	rpData = random_projection.GaussianRandomProjection(random_state=3,n_components = 40).fit_transform(dataTraining)
	icaData = FastICA(random_state = 3,n_components=40).fit_transform(dataTraining)
	pcaData = PCA(random_state=3,n_components=10).fit_transform(dataTraining)
	faData = FactorAnalysis(random_state=3,n_components = 40).fit_transform(dataTraining)
	logging.debug('')
	"""
	logging.debug("_______________Starting Neural Network-CCDefault-RandomProjection__________________")
	parameters = {'activation':('identity','logistic','tanh','relu'),\
	'alpha':[0.0001,0.001,0.01],\
	'learning_rate':('constant','invscaling','adaptive'),\
	'learning_rate_init':[0.0001,0.001,0.01,.1],\
	'hidden_layer_sizes':[(5,), (10,), (15,)]}
	cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
	dt = MLPClassifier(max_iter=2000)
	start = time.time()
	clf = GridSearchCV(dt, parameters, cv=5,n_jobs=-1)
	clf.fit(rpData, dataTraining_Class)
	end = time.time()
	total = end - start
	logging.debug(f'Neural Network Grid Search Execution Time: {total}')
	logging.debug("-------------Best Estimator NN-------------")
	logging.debug(clf.best_estimator_)
	logging.debug("-------------Best Params NN-------------")
	logging.debug(clf.best_params_)
	logging.debug("-------------Best Score NN-------------")
	logging.debug(clf.best_score_)
	logging.debug("-------------Starting Learning Curve-CCDefault-Random Projection---------------")
	start = time.time()
	train_sizes2, train_scores2, test_scores2 =\
	 learning_curve(clf.best_estimator_, rpData, dataTraining_Class, cv=cv, n_jobs=-1,\
	  train_sizes=[.2,.3,.4,.5,.6,.7,.8,.9])

	train_scores_mean2 = np.mean(train_scores2,axis=1)
	train_scores_std2 = np.std(train_scores2, axis=1)
	test_scores_mean2 = np.mean(test_scores2, axis=1)
	test_scores_std2 = np.std(test_scores2, axis=1)
	q43 = plt.figure(43)
	ax43 = q43.add_subplot(111)
	ax43.plot(train_sizes2,1 - train_scores_mean2,label = 'Training Error')
	ax43.plot(train_sizes2,1 - test_scores_mean2,label = 'Cross Validation Error')
	ax43.set_ylabel('Error')
	ax43.set_xlabel('Data Size')
	ax43.set_title('Learning Curve Neural Network - CCDefault-Random Projection')
	ax43.legend(loc=0)
	q43.savefig('LearningCurve-NN-CCDefault-RandomProjection.png')
	end = time.time()
	total= end - start
	logging.debug(f'Neural Network Learning Curve Execution Time: {total}')

	rpDataTest = random_projection.GaussianRandomProjection(random_state=3,n_components =40).fit_transform(dataTesting)
	prediction = clf.predict(rpDataTest)
	accuracy = accuracy_score(dataTesting_Class,prediction)
	logging.debug(f'LearningCurve-NN-CCDefault-Random accuracy: {accuracy}')
	"""
	logging.debug('')
	logging.debug("_______________Starting Neural Network-CCDefault-Independent Component__________________")
	parameters = {'activation':('identity','logistic','tanh','relu'),\
	'alpha':[0.0001,0.001,0.01],\
	'learning_rate':('constant','invscaling','adaptive'),\
	'learning_rate_init':[0.0001,0.001,0.01,.1],\
	'hidden_layer_sizes':[(5,), (10,), (15,)]}
	cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
	dt = MLPClassifier(max_iter=2000)
	start = time.time()
	clf = GridSearchCV(dt, parameters, cv=5,n_jobs=-1)
	clf.fit(icaData, dataTraining_Class)
	end = time.time()
	total = end - start
	logging.debug(f'Neural Network Grid Search Execution Time: {total}')
	logging.debug("-------------Best Estimator NN-------------")
	logging.debug(clf.best_estimator_)
	logging.debug("-------------Best Params NN-------------")
	logging.debug(clf.best_params_)
	logging.debug("-------------Best Score NN-------------")
	logging.debug(clf.best_score_)
	logging.debug("-------------Starting Learning Curve-CCDefault-IndependentComponent---------------")
	start = time.time()
	train_sizes2, train_scores2, test_scores2 =\
	 learning_curve(clf.best_estimator_, icaData, dataTraining_Class, cv=cv, n_jobs=-1,\
	  train_sizes=[.2,.3,.4,.5,.6,.7,.8,.9])

	train_scores_mean2 = np.mean(train_scores2,axis=1)
	train_scores_std2 = np.std(train_scores2, axis=1)
	test_scores_mean2 = np.mean(test_scores2, axis=1)
	test_scores_std2 = np.std(test_scores2, axis=1)
	q44 = plt.figure(44)
	ax44 = q44.add_subplot(111)
	ax44.plot(train_sizes2,1 - train_scores_mean2,label = 'Training Error')
	ax44.plot(train_sizes2,1 - test_scores_mean2,label = 'Cross Validation Error')
	ax44.set_ylabel('Error')
	ax44.set_xlabel('Data Size')
	ax44.set_title('Learning Curve Neural Network - CCDefault-IndependentComponent')
	ax44.legend(loc=0)
	q44.savefig('LearningCurve-NN-CCDefault-IndependentComponent.png')
	end = time.time()
	total= end - start
	logging.debug(f'Neural Network Learning Curve Execution Time: {total}')

	icaDataTest = FastICA(random_state = 3,n_components=40).fit_transform(dataTesting)
	prediction = clf.predict(icaDataTest)
	accuracy = accuracy_score(dataTesting_Class,prediction)
	logging.debug(f'LearningCurve-NN-CCDefault-IndependentComponent accuracy: {accuracy}')
	logging.debug('')
	logging.debug("_______________Starting Neural Network-CCDefault-Principal Component__________________")
	parameters = {'activation':('identity','logistic','tanh','relu'),\
	'alpha':[0.0001,0.001,0.01],\
	'learning_rate':('constant','invscaling','adaptive'),\
	'learning_rate_init':[0.0001,0.001,0.01,.1],\
	'hidden_layer_sizes':[(5,), (10,), (15,)]}
	cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
	dt = MLPClassifier(max_iter=2000)
	start = time.time()
	clf = GridSearchCV(dt, parameters, cv=5,n_jobs=-1)
	clf.fit(pcaData, dataTraining_Class)
	end = time.time()
	total = end - start
	logging.debug(f'Neural Network Grid Search Execution Time: {total}')
	logging.debug("-------------Best Estimator NN-------------")
	logging.debug(clf.best_estimator_)
	logging.debug("-------------Best Params NN-------------")
	logging.debug(clf.best_params_)
	logging.debug("-------------Best Score NN-------------")
	logging.debug(clf.best_score_)
	logging.debug("-------------Starting Learning Curve-CCDefault-PrincipalComponent---------------")
	start = time.time()
	train_sizes2, train_scores2, test_scores2 =\
	 learning_curve(clf.best_estimator_, pcaData, dataTraining_Class, cv=cv, n_jobs=-1,\
	  train_sizes=[.2,.3,.4,.5,.6,.7,.8,.9])

	train_scores_mean2 = np.mean(train_scores2,axis=1)
	train_scores_std2 = np.std(train_scores2, axis=1)
	test_scores_mean2 = np.mean(test_scores2, axis=1)
	test_scores_std2 = np.std(test_scores2, axis=1)
	q45 = plt.figure(45)
	ax45 = q45.add_subplot(111)
	ax45.plot(train_sizes2,1 - train_scores_mean2,label = 'Training Error')
	ax45.plot(train_sizes2,1 - test_scores_mean2,label = 'Cross Validation Error')
	ax45.set_ylabel('Error')
	ax45.set_xlabel('Data Size')
	ax45.set_title('Learning Curve Neural Network - CCDefault-PrincipalComponent')
	ax45.legend(loc=0)
	q45.savefig('LearningCurve-NN-CCDefault-PrincipalComponent.png')
	end = time.time()
	total= end - start
	logging.debug(f'Neural Network Learning Curve Execution Time: {total}')
	pcaDataTest = PCA(random_state=3,n_components=10).fit_transform(dataTesting)
	prediction = clf.predict(pcaDataTest)
	accuracy = accuracy_score(dataTesting_Class,prediction)
	logging.debug(f'LearningCurve-NN-CCDefault-PrincipalComponent accuracy: {accuracy}')
	logging.debug('')
	logging.debug("_______________Starting Neural Network-CCDefault-Factor Analysis__________________")
	parameters = {'activation':('identity','logistic','tanh','relu'),\
	'alpha':[0.0001,0.001,0.01],\
	'learning_rate':('constant','invscaling','adaptive'),\
	'learning_rate_init':[0.0001,0.001,0.01,.1],\
	'hidden_layer_sizes':[(5,), (10,), (15,)]}
	cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
	dt = MLPClassifier(max_iter=2000)
	start = time.time()
	clf = GridSearchCV(dt, parameters, cv=5,n_jobs=-1)
	clf.fit(faData, dataTraining_Class)
	end = time.time()
	total = end - start
	logging.debug(f'Neural Network Grid Search Execution Time: {total}')
	logging.debug("-------------Best Estimator NN-------------")
	logging.debug(clf.best_estimator_)
	logging.debug("-------------Best Params NN-------------")
	logging.debug(clf.best_params_)
	logging.debug("-------------Best Score NN-------------")
	logging.debug(clf.best_score_)
	logging.debug("-------------Starting Learning Curve-CCDefault-FactorAnalysis---------------")
	start = time.time()
	train_sizes2, train_scores2, test_scores2 =\
	 learning_curve(clf.best_estimator_, faData, dataTraining_Class, cv=cv, n_jobs=-1,\
	  train_sizes=[.2,.3,.4,.5,.6,.7,.8,.9])

	train_scores_mean2 = np.mean(train_scores2,axis=1)
	train_scores_std2 = np.std(train_scores2, axis=1)
	test_scores_mean2 = np.mean(test_scores2, axis=1)
	test_scores_std2 = np.std(test_scores2, axis=1)
	q46 = plt.figure(46)
	ax46 = q46.add_subplot(111)
	ax46.plot(train_sizes2,1 - train_scores_mean2,label = 'Training Error')
	ax46.plot(train_sizes2,1 - test_scores_mean2,label = 'Cross Validation Error')
	ax46.set_ylabel('Error')
	ax46.set_xlabel('Data Size')
	ax46.set_title('Learning Curve Neural Network - CCDefault-FactorAnalysis')
	ax46.legend(loc=0)
	q46.savefig('LearningCurve-NN-CCDefault-FactorAnalysis.png')
	end = time.time()
	total= end - start
	logging.debug(f'Neural Network Learning Curve Execution Time: {total}')
	faDataTest = FactorAnalysis(random_state=3,n_components = 40).fit_transform(dataTesting)
	prediction = clf.predict(faDataTest)
	accuracy = accuracy_score(dataTesting_Class,prediction)
	logging.debug(f'LearningCurve-NN-CCDefault-FactorAnalysis accuracy: {accuracy}')    
	logging.debug('')

def baseLineNNet(dataTraining,dataTraining_Class,dataTesting,dataTesting_Class,expName):
	logging.debug("_______________Starting Neural Network-CCDefault - BaseLine__________________")
	parameters = {'activation':('identity','logistic','tanh','relu'),\
	'alpha':[0.0001,0.001,0.01],\
	'learning_rate':('constant','invscaling','adaptive'),\
	'learning_rate_init':[0.0001,0.001,0.01,.1],\
	'hidden_layer_sizes':[(5,), (10,), (15,)]}
	cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
	#,hidden_layer_sizes=(10,)
	dt = MLPClassifier(max_iter=2000)
	start = time.time()
	clf = GridSearchCV(dt, parameters, cv=5,n_jobs=14)
	clf.fit(dataTraining, dataTraining_Class)
	end = time.time()
	total = end - start
	logging.debug(f'Neural Network Grid Search Execution Time: {total}')
	logging.debug("-------------Best Estimator NN-------------")
	logging.debug(clf.best_estimator_)
	logging.debug("-------------Best Params NN-------------")
	logging.debug(clf.best_params_)
	logging.debug("-------------Best Score NN-------------")
	logging.debug(clf.best_score_)
	logging.debug("-------------Starting Learning Curve-CCDefault- BaseLine---------------")
	start = time.time()
	train_sizes2, train_scores2, test_scores2 =\
	 learning_curve(MLPClassifier(activation='identity',alpha=0.0001,max_iter=2000,\
	 	learning_rate='invscaling',learning_rate_init=0.0001,hidden_layer_sizes=(10,)),\
	 dataTraining, dataTraining_Class, cv=cv, n_jobs=14,\
	  train_sizes=[.2,.3,.4,.5,.6,.7,.8,.9])

	train_scores_mean2 = np.mean(train_scores2,axis=1)
	train_scores_std2 = np.std(train_scores2, axis=1)
	test_scores_mean2 = np.mean(test_scores2, axis=1)
	test_scores_std2 = np.std(test_scores2, axis=1)
	
	q47 = plt.figure(47)
	ax47 = q47.add_subplot(111)
	ax47.plot(train_sizes2,1 - train_scores_mean2,label = 'Training Error')
	ax47.plot(train_sizes2,1 - test_scores_mean2,label = 'Cross Validation Error')
	ax47.set_ylabel('Error')
	ax47.set_xlabel('Data Size')
	ax47.set_title('Learning Curve Neural Network - CCDefault')
	ax47.legend(loc=0)
	q47.savefig('LearningCurve-NN-CCDefault-BaseLine.png')
	end = time.time()
	total= end - start
	logging.debug(f'Neural Network Learning Curve Execution Time: {total}')
	prediction = clf.predict(dataTesting)
	accuracy = accuracy_score(dataTesting_Class,prediction)
	logging.debug(f"NN - CCDefault Accuracy Score: {accuracy}")

def main():
	#/Users/JamesMichaels/Desktop/Georgia Tech/MachineLearning-7641/MachineLearning/SupervisedLearning/data/ /home/ec2-user/MachineLearning/SupervisedLearning/data/
	dataTraining,dataTesting,dataTraining_Class,dataTesting_Class = \
	loadDataPoker('/home/ec2-user/MachineLearning/SupervisedLearning/data/poker-hand-training-true.data',\
		'/home/ec2-user/MachineLearning/SupervisedLearning/data/poker-hand-testing.data')
	
	defaultTraining,defaultTraining_Class,defaultTesting,defaultTesting_Class = \
	loadDataDefault(\
		'/home/ec2-user/MachineLearning/SupervisedLearning/data/default_of_credit_card_clients.xls')
	
	clustersExperimentsPoker(dataTraining,dataTesting,dataTraining_Class,dataTesting_Class,'Poker')
	clustersExperimentsCreditCard(defaultTraining,defaultTesting,defaultTraining_Class,\
		defaultTesting_Class,'Credit-Card')

	principalComponentPoker(dataTraining,dataTesting,dataTraining_Class,dataTesting_Class,'Poker')
	principalComponentCredit(defaultTraining,defaultTraining_Class,defaultTesting,defaultTesting_Class,\
		'Credit-Card')
	
	independentComponentPoker(dataTraining,dataTesting,dataTraining_Class,dataTesting_Class,'Poker')
	independentComponentCredit(defaultTraining,defaultTraining_Class,defaultTesting,defaultTesting_Class,\
		'Credit-Card')

	randomProjectionsPoker(dataTraining,dataTesting,dataTraining_Class,dataTesting_Class,'Poker')
	randomProjectionsCredit(defaultTraining,defaultTraining_Class,defaultTesting,defaultTesting_Class,\
		'Credit-Card')
	
	factorAnalysisPoker(dataTraining,dataTesting,dataTraining_Class,dataTesting_Class,'Poker')
	factorAnalysisCredit(defaultTraining,defaultTraining_Class,defaultTesting,defaultTesting_Class,\
		'Credit-Card')
	
	kmeansPartPoker(dataTraining,dataTesting,dataTraining_Class,dataTesting_Class,'Poker')
	kmeansPartCredit(defaultTraining,defaultTraining_Class,defaultTesting,defaultTesting_Class,\
		'Credit-Card')

	expectedMaxPartPoker(dataTraining,dataTesting,dataTraining_Class,dataTesting_Class,'Poker')
	expectedMaxPartCredit(defaultTraining,defaultTraining_Class,defaultTesting,defaultTesting_Class,\
		'Credit-Card')
	
	#drNeuralNetwork(dataTraining,dataTesting,dataTraining_Class,dataTesting_Class,'Poker')
	drNeuralNetwork(defaultTraining,defaultTraining_Class,defaultTesting,defaultTesting_Class,\
		'Credit-Card')
	#clusteredNeuralNetwork(dataTraining,dataTesting,dataTraining_Class,dataTesting_Class,'Poker')
	clusteredNeuralNetwork(defaultTraining,defaultTraining_Class,defaultTesting,defaultTesting_Class,\
		'Credit-Card')

	#baseLineNNet(dataTraining,dataTesting,dataTraining_Class,dataTesting_Class,'Poker')
	baseLineNNet(defaultTraining,defaultTraining_Class,defaultTesting,defaultTesting_Class,\
		'Credit-Card')
	
if __name__=="__main__":
	main()	