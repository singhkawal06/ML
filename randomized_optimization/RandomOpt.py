#!/usr/bin/env python3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
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
https://mlrose.readthedocs.io/en/stable/source/fitness.html
https://mlrose.readthedocs.io/en/stable/source/tutorial3.html
https://mlrose.readthedocs.io/en/stable/source/tutorial1.html
"""
"""
To Do
start = time.time()
end = time.time()
total = end - start
"""
LOG_FILENAME = 'app.log'
logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG)
start = 0
end = 0 
total = 0
queenIterList = [1,2,4,5,6,7,8,9,10,15,20,30,40,80,100,150,200,2560]#,250,300,350,400,450,500,550,600,700,800,900,1000
itersList = [1,2,4,5,6,7,8,9,10,15,20,30,40,80,160,320,640,1280,2560]#,5120,10240
knapSackIterList = [1,2,4,5,6,7,8,9,10,15,20,30,40]
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

def queensExp():
	rhcQeDF = pd.DataFrame(columns =\
	 ['numIters','FitnessValue','Time'])
	gaQeDF = pd.DataFrame(columns =\
	 ['numIters','FitnessValue','Time'])
	saQeDF = pd.DataFrame(columns =\
	 ['numIters','FitnessValue','Time'])
	miQeDF = pd.DataFrame(columns =\
	 ['numIters','FitnessValue','Time'])
	fitness = mlrose.Queens()
	problem = mlrose.DiscreteOpt(length = 8, fitness_fn = fitness, maximize=False, max_val=8)
	schedule = mlrose.ExpDecay(init_temp=10.0, exp_const=0.5, min_temp=0.00001)
	logging.debug('Starting Queens - Simulated_annealing')
	init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])
	for item in queenIterList:
		start = time.time()
		best_state, best_fitness = mlrose.simulated_annealing(problem, \
			schedule = schedule, max_attempts = 1000, max_iters = item,\
			 init_state = init_state,\
			random_state = 1)
		end = time.time()
		total = end - start
		logging.debug(f'The best state found is: {best_state}')
		logging.debug(f'The fitness at the best state is: {best_fitness}')
		logging.debug(f'Total time: {total}')
		logging.debug(' ')
		saQeDF = saQeDF.append({'numIters':item, 'FitnessValue':best_fitness,'Time':total},ignore_index=True)
		logging.debug('------------------------------------------------------')
	logging.debug('Starting Queens - Randomized_hill_climbing')
	for item in queenIterList:
		start = time.time()
		best_state, best_fitness = mlrose.random_hill_climb(problem, \
			max_attempts = 1000, max_iters = item,\
			restarts = 5,\
			random_state = 1)
		end = time.time()
		total = end - start
		logging.debug(f'The best state found is: {best_state}')
		logging.debug(f'The fitness at the best state is: {best_fitness}')
		logging.debug(f'Total time: {total}')
		rhcQeDF = rhcQeDF.append({'numIters':item, 'FitnessValue':best_fitness,'Time':total},ignore_index=True)
		logging.debug(' ')
		logging.debug('------------------------------------------------------')
	logging.debug('Starting Queens - Genetic Algorithm')
	for item in queenIterList:
		start = time.time()
		best_state, best_fitness = mlrose.genetic_alg(problem, \
			max_attempts = 1000, max_iters = item,\
			pop_size = 200,mutation_prob=0.15,\
			random_state = 1)
		end = time.time()
		total = end - start
		logging.debug(f'The best state found is: {best_state}')
		logging.debug(f'The fitness at the best state is: {best_fitness}')
		logging.debug(f'Total time: {total}')
		gaQeDF = gaQeDF.append({'numIters':item, 'FitnessValue':best_fitness,'Time':total},ignore_index=True)
		logging.debug(' ')
		logging.debug('------------------------------------------------------')
	logging.debug('Starting Queens - MIMIC')
	for item in queenIterList:
		start = time.time()
		best_state, best_fitness = mlrose.mimic(problem, \
			max_attempts = 1000, max_iters = item,\
			pop_size = 200,keep_pct=0.2,\
			random_state = 1)
		end = time.time()
		total = end - start
		logging.debug(f'The best state found is: {best_state}')
		logging.debug(f'The fitness at the best state is: {best_fitness}')
		logging.debug(f'Total time: {total}')
		miQeDF = miQeDF.append({'numIters':item, 'FitnessValue':best_fitness,'Time':total},ignore_index=True)
		logging.debug(' ')
		logging.debug('------------------------------------------------------')
	q2 = plt.figure(2)
	ax2 = q2.add_subplot(111)
	ax2.plot(saQeDF['numIters'],saQeDF['Time'],label = 'Simulated Annealing')
	ax2.plot(rhcQeDF['numIters'],rhcQeDF['Time'],label = 'Random Hill Climb')
	ax2.plot(gaQeDF['numIters'],gaQeDF['Time'],label = 'Genetic Algorithm')
	ax2.plot(miQeDF['numIters'],miQeDF['Time'],label = 'MIMIC')
	ax2.set_ylabel('Time')
	ax2.set_xlabel('Iterations')
	ax2.legend(loc=0)
	ax2.set_title('Training Time Vs Iterations')
	q2.savefig('Queens_Time_Opt.png')
	q5 = plt.figure(5)
	ax5 = q5.add_subplot(111)
	ax5.plot(saQeDF['numIters'],saQeDF['FitnessValue'],label = 'Simulated Annealing')
	ax5.plot(rhcQeDF['numIters'],rhcQeDF['FitnessValue'],label = 'Random Hill Climb')
	ax5.plot(gaQeDF['numIters'],gaQeDF['FitnessValue'],label = 'Genetic Algorithm')
	ax5.plot(miQeDF['numIters'],miQeDF['FitnessValue'],label = 'MIMIC')
	ax5.set_ylabel('Fitness Value')
	ax5.set_xlabel('Iterations')
	ax5.legend(loc=0)
	ax5.set_title('Fitness Value Vs Iterations')
	q5.savefig('Queens_FitnessValue_Opt.png')

def fourPeaksExp():
	rhcFpDF = pd.DataFrame(columns =\
	 ['numIters','FitnessValue','Time'])
	gaFpDF = pd.DataFrame(columns =\
	 ['numIters','FitnessValue','Time'])
	saFpDF = pd.DataFrame(columns =\
	 ['numIters','FitnessValue','Time'])
	miFpDF = pd.DataFrame(columns =\
	 ['numIters','FitnessValue','Time'])
	fitness = mlrose.FourPeaks(t_pct=0.15)
	problem = mlrose.DiscreteOpt(length = 1000, fitness_fn = fitness, maximize=False, max_val=2)
	schedule = mlrose.ArithDecay()
	logging.debug('Starting 4 Peaks - Simulated_annealing')
	#init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])
	for item in itersList:
		start = time.time()
		best_state, best_fitness,fitness_curve = mlrose.simulated_annealing(problem, \
			schedule = schedule, max_attempts = 5, max_iters = item,\
				#init_state = init_state,\
				random_state = 1,curve=True)
		end = time.time()
		total = end - start
		print (fitness_curve)
			#logging.debug(f'The best state found is: {best_state}')
		logging.debug(f'The fitness at the best state is: {best_fitness}')
		logging.debug(f'Total time: {total}')
		logging.debug(' ')
		saFpDF = saFpDF.append({'numIters':item, 'FitnessValue':best_fitness,'Time':total},ignore_index=True)
		logging.debug('------------------------------------------------------')
	logging.debug('Starting 4 Peaks - Randomized_hill_climbing')
	for item in itersList:
		start = time.time()
		best_state, best_fitness,fitness_curve = mlrose.random_hill_climb(problem, \
			max_attempts = 150, max_iters = item,\
			restarts = 0,\
			random_state = 1, curve=True)
		end = time.time()
		total = end - start
		print (fitness_curve)
		#logging.debug(f'The best state found is: {best_state}')
		logging.debug(f'The fitness at the best state is: {best_fitness}')
		logging.debug(f'Total time: {total}')
		logging.debug(' ')
		rhcFpDF = rhcFpDF.append({'numIters':item, 'FitnessValue':best_fitness,'Time':total},ignore_index=True)
		logging.debug('------------------------------------------------------')
	logging.debug('Starting 4 Peaks - Genetic Algorithm')
	for item in itersList:
		start = time.time()
		best_state, best_fitness,fitness_curve = mlrose.genetic_alg(problem, \
			max_attempts = 3, max_iters = item,\
			pop_size = 1,mutation_prob=.1,\
			random_state = 1, curve=True)
		end = time.time()
		total = end - start
		#logging.debug(f'The best state found is: {best_state}')
		print (fitness_curve)
		logging.debug(f'The fitness at the best state is: {best_fitness}')
		logging.debug(f'Total time: {total}')
		logging.debug(' ')
		gaFpDF = gaFpDF.append({'numIters':item, 'FitnessValue':best_fitness,'Time':total},ignore_index=True)
		logging.debug('------------------------------------------------------')
	logging.debug('Starting 4 Peaks - MIMIC')
	for item in itersList:
		start = time.time()
		best_state, best_fitness,fitness_curve = mlrose.mimic(problem, \
			max_attempts = 1, max_iters = item,\
			pop_size = 5,keep_pct=.5,\
			random_state = 1, curve=True)
		end = time.time()
		total = end - start
		print (fitness_curve)
		#logging.debug(f'The best state found is: {best_state}')
		logging.debug(f'The fitness at the best state is: {best_fitness}')
		logging.debug(f'Total time: {total}')
		logging.debug(' ')
		miFpDF = miFpDF.append({'numIters':item, 'FitnessValue':best_fitness,'Time':total},ignore_index=True)
		logging.debug('------------------------------------------------------')
	q3 = plt.figure(3)
	ax3 = q3.add_subplot(111)
	ax3.plot(saFpDF['numIters'],saFpDF['Time'],label = 'Simulated Annealing')
	ax3.plot(rhcFpDF['numIters'],rhcFpDF['Time'],label = 'Random Hill Climb')
	ax3.plot(gaFpDF['numIters'],gaFpDF['Time'],label = 'Genetic Algorithm')
	ax3.plot(miFpDF['numIters'],miFpDF['Time'],label = 'MIMIC')
	ax3.set_ylabel('Time')
	ax3.set_xlabel('Iterations')
	ax3.legend(loc=0)
	ax3.set_title('Training Time Vs Iterations')
	q3.savefig('FourPeaks_Time_Opt.png')
	q6 = plt.figure(6)
	ax6 = q6.add_subplot(111)
	ax6.plot(saFpDF['numIters'],saFpDF['FitnessValue'],label = 'Simulated Annealing')
	ax6.plot(rhcFpDF['numIters'],rhcFpDF['FitnessValue'],label = 'Random Hill Climb')
	ax6.plot(gaFpDF['numIters'],gaFpDF['FitnessValue'],label = 'Genetic Algorithm')
	ax6.plot(miFpDF['numIters'],miFpDF['FitnessValue'],label = 'MIMIC')
	ax6.set_ylabel('Fitness Value')
	ax6.set_xlabel('Iterations')
	ax6.legend(loc=0)
	ax6.set_title('Fitness Value Vs Iterations')
	q6.savefig('FourPeaks_FitnessValue_Opt.png')

def knapsackExp():
	rhcKsDF = pd.DataFrame(columns =\
	 ['numIters','FitnessValue','Time'])
	gaKsDF = pd.DataFrame(columns =\
	 ['numIters','FitnessValue','Time'])
	saKsDF = pd.DataFrame(columns =\
	 ['numIters','FitnessValue','Time'])
	miKsDF = pd.DataFrame(columns =\
	 ['numIters','FitnessValue','Time'])
	"""np.random.rand(1000,1)#"""
	"""np.random.rand(1000,1)#"""
	weights = [10, 5, 2, 8, 15,9,2,7]
	values = [1, 2, 3, 4, 5, 9, 2, 7]
	fitness = mlrose.Knapsack(weights,values,max_weight_pct=0.95)
	problem = mlrose.DiscreteOpt(length = 8, fitness_fn = fitness, maximize=False, max_val=2)
	schedule = mlrose.ArithDecay()
	logging.debug('Starting KnapSack - Simulated_annealing')
	#init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])
	for item in knapSackIterList:
		start = time.time()
		best_state, best_fitness,fitness_curve = mlrose.simulated_annealing(problem, \
			schedule = schedule, max_attempts = 10, max_iters = item,\
			 #init_state = init_state,\
			random_state = 1, curve= True)
		end = time.time()
		total = end - start
		#print (fitness_curve)
		logging.debug(f'The best state found is: {best_state}')
		logging.debug(f'The fitness at the best state is: {best_fitness}')
		logging.debug(f'Total time: {total}')
		logging.debug(' ')
		saKsDF = saKsDF.append({'numIters':item, 'FitnessValue':best_fitness,'Time':total},ignore_index=True)
		logging.debug('------------------------------------------------------')
	logging.debug('Starting KnapSack - Randomized_hill_climbing')
	for item in knapSackIterList:
		start = time.time()
		best_state, best_fitness,fitness_curve = mlrose.random_hill_climb(problem, \
			max_attempts = 10, max_iters = item,\
			restarts = 0,\
			random_state = 1, curve = True)
		end = time.time()
		total = end - start
		logging.debug(f'The best state found is: {best_state}')
		logging.debug(f'The fitness at the best state is: {best_fitness}')
		logging.debug(f'Total time: {total}')
		logging.debug(' ')
		rhcKsDF = rhcKsDF.append({'numIters':item, 'FitnessValue':best_fitness,'Time':total},ignore_index=True)
		logging.debug('------------------------------------------------------')
	logging.debug('Starting KnapSack - Genetic Algorithm')
	for item in knapSackIterList:
		start = time.time()
		best_state, best_fitness,fitness_curve = mlrose.genetic_alg(problem, \
			max_attempts = 3, max_iters = item,\
			pop_size = 50,mutation_prob=0.01,\
			random_state = 1, curve = True)
		end = time.time()
		total = end - start
		logging.debug(f'The best state found is: {best_state}')
		logging.debug(f'The fitness at the best state is: {best_fitness}')
		logging.debug(f'Total time: {total}')
		logging.debug(' ')
		gaKsDF = gaKsDF.append({'numIters':item, 'FitnessValue':best_fitness,'Time':total},ignore_index=True)
		logging.debug('------------------------------------------------------')
	logging.debug('Starting KnapSack - MIMIC')
	for item in knapSackIterList:
		start = time.time()
		best_state, best_fitness = mlrose.mimic(problem, \
			max_attempts = 1, max_iters = item,\
			pop_size = 50,keep_pct=.00005,\
			random_state = 1)
		end = time.time()
		total = end - start
		logging.debug(f'The best state found is: {best_state}')
		logging.debug(f'The fitness at the best state is: {best_fitness}')
		logging.debug(f'Total time: {total}')
		logging.debug(' ')
		miKsDF = miKsDF.append({'numIters':item, 'FitnessValue':best_fitness,'Time':total},ignore_index=True)
		logging.debug('------------------------------------------------------')
	q4 = plt.figure(4)
	ax4 = q4.add_subplot(111)
	ax4.plot(saKsDF['numIters'],saKsDF['Time'],label = 'Simulated Annealing')
	ax4.plot(rhcKsDF['numIters'],rhcKsDF['Time'],label = 'Random Hill Climb')
	ax4.plot(gaKsDF['numIters'],gaKsDF['Time'],label = 'Genetic Algorithm')
	ax4.plot(miKsDF['numIters'],miKsDF['Time'],label = 'MIMIC')
	ax4.set_ylabel('Time')
	ax4.set_xlabel('Iterations')
	ax4.legend(loc=0)
	ax4.set_title('Training Time Vs Iterations')
	q4.savefig('Knapsack_Time_Opt.png')
	q7 = plt.figure(7)
	ax7 = q7.add_subplot(111)
	ax7.plot(saKsDF['numIters'],saKsDF['FitnessValue'],label = 'Simulated Annealing')
	ax7.plot(rhcKsDF['numIters'],rhcKsDF['FitnessValue'],label = 'Random Hill Climb')
	ax7.plot(gaKsDF['numIters'],gaKsDF['FitnessValue'],label = 'Genetic Algorithm')
	ax7.plot(miKsDF['numIters'],miKsDF['FitnessValue'],label = 'MIMIC')
	ax7.set_ylabel('Fitness Value')
	ax7.set_xlabel('Iterations')
	ax7.legend(loc=0)
	ax7.set_title('Fitness Value Vs Iterations')
	q7.savefig('Knapsack_FitnessValue_Opt.png')

def creditCardDefaultExp(dataTraining,dataTesting,dataTraining_Class,dataTesting_Class):
	one_hot = OneHotEncoder()
	scaler = MinMaxScaler()
	#print (dataTraining_Class.shape)
	#print (dataTesting_Class)
	dataTrainingScaled = scaler.fit_transform(dataTraining)
	dataTestingScaled = scaler.transform(dataTesting)
	dataTraining_ClassHotEncoded = one_hot.fit_transform(dataTraining_Class.values.reshape(-1, 1))\
	.todense()
	dataTesting_ClassHotEncoded = one_hot.transform(dataTesting_Class.values.reshape(-1, 1))\
	.todense()
	#print (dataTraining_ClassHotEncoded.shape)
	#print (dataTesting_ClassHotEncoded.shape)
	rhcNNDF = pd.DataFrame(columns =\
	 ['numIters','Error','Time','TestingError'])
	saNNDF = pd.DataFrame(columns =\
	 ['numIters','Error','Time','TestingError'])
	gaNNDF = pd.DataFrame(columns =\
	 ['numIters','Error','Time','TestingError'])
	gdNNDF = pd.DataFrame(columns =\
	 ['numIters','Error','Time','TestingError'])
	rhcNN = np.zeros(shape=(2,len(itersList)))
	saNN = np.zeros(shape=(2,len(itersList)))
	gaNN = np.zeros(shape=(2,len(itersList)))
	gdNN = np.zeros(shape=(2,len(itersList)))
	for item in itersList:
		logging.debug(f'Starting CCDefault - Random Hill climb {item}')
		start = time.time()
		neuralNetRHC = mlrose.NeuralNetwork(hidden_nodes = [20], activation = 'tanh',\
			algorithm = 'random_hill_climb', max_iters = item,\
			bias = True, is_classifier = True, learning_rate = 0.3,\
			early_stopping = True, clip_max = 25, max_attempts = 200,\
			random_state = 3,restarts = 5)
		neuralNetRHC.fit(dataTrainingScaled,dataTraining_ClassHotEncoded)
		end = time.time()
		total = end - start
		logging.debug(f'Total time: {total}')
		predictionTraining = neuralNetRHC.predict(dataTrainingScaled)
		accuracyTraining = accuracy_score(dataTraining_ClassHotEncoded,predictionTraining)
		logging.debug(f'Training Score: {accuracyTraining}')
		prediction = neuralNetRHC.predict(dataTestingScaled)
		#print (dataTesting_ClassHotEncoded)
		#print (prediction)
		accuracy = accuracy_score(dataTesting_ClassHotEncoded,prediction)
		logging.debug(f'Testing Score: {accuracy}')
		rhcNN = np.append(rhcNN,[item,1-accuracyTraining])
		rhcNNDF = rhcNNDF.append({'numIters':item, 'Error':1-accuracyTraining,'Time':total,'TestingError':1-accuracy},ignore_index=True)
		logging.debug(' ')
		logging.debug('-------------------------------------------')
	for item in itersList:
		logging.debug(f'Starting CCDefault - Simulated_annealing {item}')
		start = time.time()
		schedule = mlrose.GeomDecay(init_temp=1.0, decay=0.99, min_temp=0.001)
		neuralNetSA = mlrose.NeuralNetwork(hidden_nodes = [20], activation = 'tanh',\
			algorithm = 'simulated_annealing', max_iters = item,\
			bias = True, is_classifier = True, learning_rate = 0.2,\
			early_stopping = True, clip_max = 25, max_attempts = 100,\
			random_state = 3,schedule = schedule)#,schedule = schedule 
		neuralNetSA.fit(dataTrainingScaled,dataTraining_ClassHotEncoded)
		end = time.time()
		total = end - start
		logging.debug(f'Total time: {total}')
		predictionTraining = neuralNetSA.predict(dataTrainingScaled)
		accuracyTraining = accuracy_score(dataTraining_ClassHotEncoded,predictionTraining)
		logging.debug(f'Training Score: {accuracyTraining}')
		prediction = neuralNetSA.predict(dataTestingScaled)
		accuracy = accuracy_score(dataTesting_ClassHotEncoded,prediction)
		logging.debug(f'Testing Score: {accuracy}')
		saNN = np.append(saNN,[item,1-accuracyTraining])
		saNNDF = saNNDF.append({'numIters':item, 'Error':1-accuracyTraining,'Time':total,'TestingError':1-accuracy},ignore_index=True)
		logging.debug(' ')
		logging.debug('-------------------------------------------')
	for item in itersList:
		logging.debug(f'Starting CCDefault - genetic_alg {item}')
		start = time.time()
		neuralNetGA = mlrose.NeuralNetwork(hidden_nodes = [20], activation = 'tanh',\
			algorithm = 'genetic_alg', max_iters = item,\
			bias = True, is_classifier = True, learning_rate = 0.1,\
			early_stopping = True, clip_max = 25, max_attempts = 200,\
			random_state = 3,pop_size = 300)
		neuralNetGA.fit(dataTrainingScaled,dataTraining_ClassHotEncoded)
		end = time.time()
		total = end - start
		logging.debug(f'Total time: {total}')
		predictionTraining = neuralNetGA.predict(dataTrainingScaled)
		accuracyTraining = accuracy_score(dataTraining_ClassHotEncoded,predictionTraining)
		logging.debug(f'Training Score: {accuracyTraining}')
		prediction = neuralNetGA.predict(dataTestingScaled)
		accuracy = accuracy_score(dataTesting_ClassHotEncoded,prediction)
		logging.debug(f'Testing Score: {accuracy}')
		gaNN = np.append(gaNN,[item,1-accuracyTraining])
		gaNNDF = gaNNDF.append({'numIters':item, 'Error':1-accuracyTraining,'Time':total,'TestingError':1-accuracy},ignore_index=True)
		logging.debug(' ')
		logging.debug('-------------------------------------------')
	for item in itersList:
		logging.debug(f'Starting CCDefault - Gradient Descent {item}')
		start = time.time()
		neuralNetGD = mlrose.NeuralNetwork(hidden_nodes = [20], activation = 'tanh',\
			algorithm = 'gradient_descent', max_iters = item,\
			bias = True, is_classifier = True, learning_rate = 0.0001,\
			early_stopping = True, clip_max = 25, max_attempts = 100,\
			random_state = 3)
		neuralNetGD.fit(dataTrainingScaled,dataTraining_ClassHotEncoded)
		end = time.time()
		total = end - start
		logging.debug(f'Total time: {total}')
		predictionTraining = neuralNetGD.predict(dataTrainingScaled)
		accuracyTraining = accuracy_score(dataTraining_ClassHotEncoded,predictionTraining)
		logging.debug(f'Training Score: {accuracyTraining}')
		prediction = neuralNetGD.predict(dataTestingScaled)
		accuracy = accuracy_score(dataTesting_ClassHotEncoded,prediction)
		logging.debug(f'Testing Score: {accuracy}')
		gdNN = np.append(gdNN,[item,1-accuracyTraining])
		gdNNDF = gdNNDF.append({'numIters':item, 'Error':1-accuracyTraining,'Time':total,'TestingError':1-accuracy},ignore_index=True)
		logging.debug(' ')
		logging.debug('-------------------------------------------')
	print (rhcNNDF)
	print (saNNDF)
	print (gaNNDF)
	print (gdNNDF) 
	q1 = plt.figure(1)
	ax1 = q1.add_subplot(111)
	ax1.plot(rhcNNDF['numIters'],rhcNNDF['Error'],label = 'Random Hill Climb')
	ax1.plot(saNNDF['numIters'],saNNDF['Error'],label = 'Simulated Annealing')
	ax1.plot(gaNNDF['numIters'],gaNNDF['Error'],label = 'Genetic Algorithm')
	ax1.plot(gdNNDF['numIters'],gdNNDF['Error'],label = 'Gradient Descent Algorithm')
	ax1.set_ylabel('Error')
	ax1.set_xlabel('Iterations')
	ax1.legend(loc=0)
	ax1.set_title('Comparison of Error Rates based on Algs - Training')
	q1.savefig('AlgComps_TrainingError.png')

	q11 = plt.figure(11)
	ax11 = q11.add_subplot(111)
	ax11.plot(rhcNNDF['numIters'],rhcNNDF['Error'],label = 'Random Hill Climb Training Error')
	ax11.plot(rhcNNDF['numIters'],rhcNNDF['TestingError'],label = 'Random Hill Climb Testing Error')
	ax11.set_ylabel('Error')
	ax11.set_xlabel('Iterations')
	ax11.legend(loc=0)
	ax11.set_title('Comparison of Error Rates - RHC')
	q11.savefig('AlgComps_RHC.png')

	q12 = plt.figure(12)
	ax12 = q12.add_subplot(111)
	ax12.plot(saNNDF['numIters'],saNNDF['Error'],label = 'Simulated Annealing Training Error')
	ax12.plot(saNNDF['numIters'],saNNDF['TestingError'],label = 'Simulated Annealing Testing Error')
	ax12.set_ylabel('Error')
	ax12.set_xlabel('Iterations')
	ax12.legend(loc=0)
	ax12.set_title('Comparison of Error Rates - SA')
	q12.savefig('AlgComps_SA.png')

	q13 = plt.figure(13)
	ax13 = q13.add_subplot(111)
	ax13.plot(np.linspace(0, 200,19),gaNNDF['Error'],label = 'Genetic Algorithm Training Error')
	ax13.plot(np.linspace(0, 200,19),gaNNDF['TestingError'],label = 'Genetic Algorithm Testing Error')
	ax13.set_ylabel('Error')
	ax13.set_xlabel('Iterations')
	ax13.legend(loc=0)
	ax13.set_title('Comparison of Error Rates - GA')
	q13.savefig('AlgComps_GA.png')

	q14 = plt.figure(14)
	ax14 = q14.add_subplot(111)
	ax14.plot(gdNNDF['numIters'],gdNNDF['Error'],label = 'Gradient Descent Algorithm Training Error')
	ax14.plot(gdNNDF['numIters'],gdNNDF['TestingError'],label = 'Gradient Descent Algorithm Testing Error')
	ax14.set_ylabel('Error')
	ax14.set_xlabel('Iterations')
	ax14.legend(loc=0)
	ax14.set_title('Comparison of Error Rates - GD')
	q14.savefig('AlgComps_GD.png')

	q8 = plt.figure(8)
	ax8 = q8.add_subplot(111)
	ax8.plot(rhcNNDF['numIters'],rhcNNDF['Time'],label = 'Random Hill Climb')
	ax8.plot(saNNDF['numIters'],saNNDF['Time'],label = 'Simulated Annealing')
	ax8.plot(gaNNDF['numIters'],gaNNDF['Time'],label = 'Genetic Algorithm')
	ax8.plot(gdNNDF['numIters'],gdNNDF['Time'],label = 'Gradient Descent Algorithm')
	ax8.set_ylabel('Error')
	ax8.set_xlabel('Iterations')
	ax8.legend(loc=0)
	ax8.set_title('Comparison of Error Rates based on Algs - Time')
	q8.savefig('AlgComps_Training_Time.png')
	
	q9 = plt.figure(9)
	ax9 = q9.add_subplot(111)
	ax9.plot(rhcNNDF['numIters'],rhcNNDF['TestingError'],label = 'Random Hill Climb')
	ax9.plot(saNNDF['numIters'],saNNDF['TestingError'],label = 'Simulated Annealing')
	ax9.plot(gaNNDF['numIters'],gaNNDF['TestingError'],label = 'Genetic Algorithm')
	ax9.plot(gdNNDF['numIters'],gdNNDF['TestingError'],label = 'Gradient Descent Algorithm')
	ax9.set_ylabel('Error')
	ax9.set_xlabel('Iterations')
	ax9.legend(loc=0)
	ax9.set_title('Comparison of Error Rates based on Algs - Testing')
	q9.savefig('AlgComps_TestingError.png') 

def main():
	#/Users/JamesMichaels/Desktop/Georgia Tech/MachineLearning-7641/MachineLearning/SupervisedLearning/data/ /home/ec2-user/MachineLearning/SupervisedLearning/data/
	dataTraining,dataTesting,dataTraining_Class,dataTesting_Class = \
	loadDataPoker('/home/ec2-user/MachineLearning/SupervisedLearning/data/poker-hand-training-true.data',\
		'/home/ec2-user/MachineLearning/SupervisedLearning/data/poker-hand-testing.data')
	queensExp()
	knapsackExp()
	creditCardDefaultExp(dataTraining,dataTesting,dataTraining_Class,dataTesting_Class)
	fourPeaksExp()
	

if __name__=="__main__":
	main()