import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score, f1_score
import mlrose_hiive
from supervised_learning.data_load import  load_women_diabetes_data
import numpy as np
import time
import matplotlib.pyplot as plt

save_path = '../charts/'


# load our dataset
X, y = load_women_diabetes_data('../data/diabetes_in_women/data.csv')

# shuffle to avoid underlying distributions
X, y = shuffle(X, y, random_state=26)


itersList = [1,2,4,8,16,32,64,128,256,512,1024,2048]

#itersList = [1,2,4,8]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=26)
X_train = scale(X_train)
X_test = scale(X_test)

"""
nn_model_rhc = mlrose_hiive.NeuralNetwork(hidden_nodes=[30], activation='relu',
                                    algorithm='random_hill_climb', max_iters=1000,
                                    bias=True, is_classifier=True, learning_rate=1,
                                    early_stopping=True, clip_max=5, max_attempts=100,
                                    random_state=3)
nn_model_sa = mlrose_hiive.NeuralNetwork(hidden_nodes=[30], activation='relu',
                                   algorithm='simulated_annealing', max_iters=1000,
                                   bias=True, is_classifier=True, learning_rate=1,
                                   early_stopping=True, clip_max=5, max_attempts=100,
                                   random_state=3)
nn_model_ga = mlrose_hiive.NeuralNetwork(hidden_nodes=[30], activation='relu',
                                   algorithm='genetic_alg', max_iters=1000,
                                   bias=True, is_classifier=True, learning_rate=1,
                                   early_stopping=True, clip_max=5, max_attempts=100,
                                   random_state=3)

print(nn_model_rhc)
neural_nets = [nn_model_rhc, nn_model_sa, nn_model_ga]



for nn in neural_nets:
    t = time.time()
    nn.fit(X_train, y_train)

    y_train_pred = nn.predict(X_train)
    y_train_accuracy = accuracy_score(y_train, y_train_pred)
    print("Train accuracy for {}: {}".format(nn, y_train_accuracy))

    y_test_pred = nn.predict(X_test)
    y_test_accuracy = accuracy_score(y_test, y_test_pred)
    print("Test accuracy for {}: {}".format(nn, y_test_accuracy))
    print("Time needed: {}".format(time.time()-t))

"""

rhcNNDF = pd.DataFrame(columns =\
	 ['numIters','TrainF1Score','TestF1Score','TrainError','TrainAccuracy','TestAccuracy','Time','TestingError'])

saNNDF = pd.DataFrame(columns =\
	 ['numIters','TrainF1Score','TestF1Score','TrainError','TrainAccuracy','TestAccuracy','Time','TestingError'])

gaNNDF = pd.DataFrame(columns =\
	 ['numIters','TrainF1Score','TestF1Score','TrainError','TrainAccuracy','TestAccuracy','Time','TestingError'])


for itr in itersList:
    nn_model_rhc = mlrose_hiive.NeuralNetwork(hidden_nodes=[30], activation='relu',
                                              algorithm='random_hill_climb', max_iters=itr,
                                              bias=True, is_classifier=True, learning_rate=1,
                                              early_stopping=True, clip_max=5, max_attempts=100,
                                              random_state=3)
    t = time.time()
    nn_model_rhc.fit(X_train, y_train)

    y_train_pred = nn_model_rhc.predict(X_train)
    y_train_accuracy = accuracy_score(y_train, y_train_pred)
    y_train_f1 = f1_score(y_train, y_train_pred)
    print("Train accuracy for itr {} {}: {}".format(itr,nn_model_rhc, y_train_accuracy))
    print("Train F1 Score for itr {} {}: {}".format(itr,nn_model_rhc, y_train_f1))

    y_test_pred = nn_model_rhc.predict(X_test)
    y_test_accuracy = accuracy_score(y_test, y_test_pred)
    y_test_f1 = f1_score(y_test, y_test_pred)
    print("Test accuracy for {}: {}".format(itr,nn_model_rhc, y_test_accuracy))
    print("Test F1 Score for itr {} {}: {}".format(itr,nn_model_rhc, y_train_f1))
    print("Time needed for itr {}:  {}".format(itr,time.time() - t))
    rhcNNDF = rhcNNDF.append(
        {'numIters': itr,'TrainF1Score':y_train_f1,'TestF1Score':y_test_f1,'TrainError': 1 - y_train_accuracy,'TrainAccuracy':y_train_accuracy,'TestAccuracy':y_test_accuracy, 'Time': time.time() - t, 'TestingError': 1 - y_test_accuracy},
        ignore_index=True)


for itr in itersList:
    nn_model_sa = mlrose_hiive.NeuralNetwork(hidden_nodes=[30], activation='relu',
                                             algorithm='simulated_annealing', max_iters=itr,
                                             bias=True, is_classifier=True, learning_rate=1,
                                             early_stopping=True, clip_max=5, max_attempts=100,
                                             random_state=3)
    t = time.time()
    nn_model_sa.fit(X_train, y_train)

    y_train_pred = nn_model_sa.predict(X_train)
    y_train_accuracy = accuracy_score(y_train, y_train_pred)
    y_train_f1 = f1_score(y_train, y_train_pred)
    print("Train accuracy for itr {} {}: {}".format(itr,nn_model_sa, y_train_accuracy))
    print("Train F1 Score for itr {} {}: {}".format(itr,nn_model_sa, y_train_f1))

    y_test_pred = nn_model_sa.predict(X_test)
    y_test_accuracy = accuracy_score(y_test, y_test_pred)
    y_test_f1 = f1_score(y_test, y_test_pred)
    print("Test accuracy for {}: {}".format(itr,nn_model_sa, y_test_accuracy))
    print("Test F1 Score for itr {} {}: {}".format(itr,nn_model_sa, y_train_f1))
    print("Time needed for itr {}:  {}".format(itr,time.time() - t))
    saNNDF = saNNDF.append(
        {'numIters': itr,'TrainF1Score':y_train_f1,'TestF1Score':y_test_f1,'TrainError': 1 - y_train_accuracy,'TrainAccuracy':y_train_accuracy,'TestAccuracy':y_test_accuracy, 'Time': time.time() - t, 'TestingError': 1 - y_test_accuracy},
        ignore_index=True)


for itr in itersList:
    nn_model_ga = mlrose_hiive.NeuralNetwork(hidden_nodes=[30], activation='relu',
                                             algorithm='genetic_alg', max_iters=itr,
                                             bias=True, is_classifier=True, learning_rate=1,
                                             early_stopping=True, clip_max=5, max_attempts=100,
                                             random_state=3)
    t = time.time()
    nn_model_ga.fit(X_train, y_train)

    y_train_pred = nn_model_ga.predict(X_train)
    y_train_accuracy = accuracy_score(y_train, y_train_pred)
    y_train_f1 = f1_score(y_train, y_train_pred)
    print("Train accuracy for itr {} {}: {}".format(itr,nn_model_ga, y_train_accuracy))
    print("Train F1 Score for itr {} {}: {}".format(itr,nn_model_ga, y_train_f1))

    y_test_pred = nn_model_ga.predict(X_test)
    y_test_accuracy = accuracy_score(y_test, y_test_pred)
    y_test_f1 = f1_score(y_test, y_test_pred)
    print("Test accuracy for {}: {}".format(itr,nn_model_ga, y_test_accuracy))
    print("Test F1 Score for itr {} {}: {}".format(itr,nn_model_ga, y_train_f1))
    print("Time needed for itr {}:  {}".format(itr,time.time() - t))
    gaNNDF = gaNNDF.append(
        {'numIters': itr,'TrainF1Score':y_train_f1,'TestF1Score':y_test_f1,'TrainError': 1 - y_train_accuracy,'TrainAccuracy':y_train_accuracy,'TestAccuracy':y_test_accuracy, 'Time': time.time() - t, 'TestingError': 1 - y_test_accuracy},
        ignore_index=True)


plt.figure()
plt.title('Comparison of Error Rates - RHC')
plt.plot(rhcNNDF['numIters'], rhcNNDF['TrainError'], label='Random Hill Climb Training Error')
plt.plot(rhcNNDF['numIters'], rhcNNDF['TestingError'], label='Random Hill Climb Testing Error')
plt.xlabel('Iterations')
plt.ylabel("Error")
plt.legend(loc="best")
plt.grid()
plt.savefig(save_path + 'NueralNetwork_Error_RHC.png')

plt.figure()
plt.title('Comparison of Accruacy  - RHC')
plt.plot(rhcNNDF['numIters'], rhcNNDF['TrainAccuracy'], label='Random Hill Climb Training Accuracy')
plt.plot(rhcNNDF['numIters'], rhcNNDF['TestAccuracy'], label='Random Hill Climb Testing Accuracy')
plt.xlabel('Iterations')
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.grid()
plt.savefig(save_path + 'NueralNetwork_Accuracy_RHC.png')

plt.figure()
plt.title('Comparison of F1 Scores - RHC')
plt.plot(rhcNNDF['numIters'], rhcNNDF['TrainF1Score'], label='Random Hill Climb Training F1 Score')
plt.plot(rhcNNDF['numIters'], rhcNNDF['TestF1Score'], label='Random Hill Climb Testing F1 Score')
plt.xlabel('Iterations')
plt.ylabel("F1 Score")
plt.legend(loc="best")
plt.grid()
plt.savefig(save_path + 'NueralNetwork_F1Score_RHC.png')



plt.figure()
plt.title('Comparison of Error Rates - SA')
plt.plot(saNNDF['numIters'], saNNDF['TrainError'], label='Simulated Annealing Training Error')
plt.plot(saNNDF['numIters'], saNNDF['TestingError'], label='Simulated Annealing Testing Error')
plt.xlabel('Iterations')
plt.ylabel("Error")
plt.legend(loc="best")
plt.grid()
plt.savefig(save_path + 'NueralNetwork_Error_SA.png')

plt.figure()
plt.title('Comparison of Accruacy  - SA')
plt.plot(saNNDF['numIters'], saNNDF['TrainAccuracy'], label='Simulated Annealing Training Accuracy')
plt.plot(saNNDF['numIters'], saNNDF['TestAccuracy'], label='Simulated Annealing Testing Accuracy')
plt.xlabel('Iterations')
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.grid()
plt.savefig(save_path + 'NueralNetwork_Accuracy_SA.png')

plt.figure()
plt.title('Comparison of F1 Scores - SA')
plt.plot(saNNDF['numIters'], saNNDF['TrainF1Score'], label='Simulated Annealing Training F1 Score')
plt.plot(saNNDF['numIters'], saNNDF['TestF1Score'], label='Simulated Annealing Testing F1 Score')
plt.xlabel('Iterations')
plt.ylabel("F1 Score")
plt.legend(loc="best")
plt.grid()
plt.savefig(save_path + 'NueralNetwork_F1Score_SA.png')



plt.figure()
plt.title('Comparison of Error Rates - GA')
plt.plot(gaNNDF['numIters'], gaNNDF['TrainError'], label='Genetic Algo Training Error')
plt.plot(gaNNDF['numIters'], gaNNDF['TestingError'], label='Genetic Algo Testing Error')
plt.xlabel('Iterations')
plt.ylabel("Error")
plt.legend(loc="best")
plt.grid()
plt.savefig(save_path + 'NueralNetwork_Error_GA.png')

plt.figure()
plt.title('Comparison of Accruacy  - GA')
plt.plot(gaNNDF['numIters'], gaNNDF['TrainAccuracy'], label='Genetic Algo Training Accuracy')
plt.plot(gaNNDF['numIters'], gaNNDF['TestAccuracy'], label='Genetic Algo Testing Accuracy')
plt.xlabel('Iterations')
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.grid()
plt.savefig(save_path + 'NueralNetwork_Accuracy_GA.png')

plt.figure()
plt.title('Comparison of F1 Scores - GA')
plt.plot(gaNNDF['numIters'], gaNNDF['TrainF1Score'], label='Genetic Algo Training F1 Score')
plt.plot(gaNNDF['numIters'], gaNNDF['TestF1Score'], label='Genetic Algo Testing F1 Score')
plt.xlabel('Iterations')
plt.ylabel("F1 Score")
plt.legend(loc="best")
plt.grid()
plt.savefig(save_path + 'NueralNetwork_F1Score_GA.png')


plt.figure()
plt.title('Comparison of Algorithms Training F1 Scores')
plt.plot(rhcNNDF['numIters'], rhcNNDF['TrainF1Score'], label='Random Hill Climb Training F1 Score',color="navy")
plt.plot(saNNDF['numIters'], saNNDF['TrainF1Score'], label='Simulated Annealing Training F1 Score',color="red")
plt.plot(gaNNDF['numIters'], gaNNDF['TrainF1Score'], label='Genetic Algo Training F1 Score',color="green")
plt.xlabel('Iterations')
plt.ylabel("F1 Score")
plt.legend(loc="best")
plt.grid()
plt.savefig(save_path + 'NueralNetwork_F1Score_comparison.png')

plt.figure()
plt.title('Comparison of Algorithms Training F1 Scores')
plt.plot(rhcNNDF['numIters'], rhcNNDF['TrainF1Score'], label='Random Hill Climb Training F1 Score',color="navy")
plt.plot(saNNDF['numIters'], saNNDF['TrainF1Score'], label='Simulated Annealing Training F1 Score',color="red")
plt.plot(gaNNDF['numIters'], gaNNDF['TrainF1Score'], label='Genetic Algo Training F1 Score',color="green")
plt.xlabel('Iterations')
plt.ylabel("F1 Score")
plt.legend(loc="best")
plt.grid()
plt.savefig(save_path + 'NueralNetwork_TrainF1Score_comparison.png')

plt.figure()
plt.title('Comparison of Algorithms Test F1 Scores')
plt.plot(rhcNNDF['numIters'], rhcNNDF['TestF1Score'], label='Random Hill Climb Test F1 Score',color="navy")
plt.plot(saNNDF['numIters'], saNNDF['TestF1Score'], label='Simulated Annealing Test F1 Score',color="red")
plt.plot(gaNNDF['numIters'], gaNNDF['TestF1Score'], label='Genetic Algo Test F1 Score',color="green")
plt.xlabel('Iterations')
plt.ylabel("F1 Score")
plt.legend(loc="best")
plt.grid()
plt.savefig(save_path + 'NueralNetwork_TestF1Score_comparison.png')

plt.figure()
plt.title('Comparison of Algorithms Training Error')
plt.plot(rhcNNDF['numIters'], rhcNNDF['TrainError'], label='Random Hill Climb Train Error',color="navy")
plt.plot(saNNDF['numIters'], saNNDF['TrainError'], label='Simulated Annealing Train Error',color="red")
plt.plot(gaNNDF['numIters'], gaNNDF['TrainError'], label='Genetic Algo Train Error',color="green")
plt.xlabel('Iterations')
plt.ylabel("Error")
plt.legend(loc="best")
plt.grid()
plt.savefig(save_path + 'NueralNetwork_TrainError_comparison.png')

plt.figure()
plt.title('Comparison of Algorithms Test Error')
plt.plot(rhcNNDF['numIters'], rhcNNDF['TestingError'], label='Random Hill Climb Testing Error',color="navy")
plt.plot(saNNDF['numIters'], saNNDF['TestingError'], label='Simulated Annealing Testing Error',color="red")
plt.plot(gaNNDF['numIters'], gaNNDF['TestingError'], label='Genetic Algo Testing Error',color="green")
plt.xlabel('Iterations')
plt.ylabel("Error")
plt.legend(loc="best")
plt.grid()
plt.savefig(save_path + 'NueralNetwork_TestingError_comparison.png')

plt.figure()
plt.title('Comparison of Algorithms Time')
plt.plot(rhcNNDF['numIters'], rhcNNDF['Time'], label='Random Hill Climb Time',color="navy")
plt.plot(saNNDF['numIters'], saNNDF['Time'], label='Simulated Annealing Time',color="red")
plt.plot(gaNNDF['numIters'], gaNNDF['Time'], label='Genetic Algo Time',color="green")
plt.xlabel('Iterations')
plt.ylabel("Time")
plt.legend(loc="best")
plt.grid()
plt.savefig(save_path + 'NueralNetwork_Time_comparison.png')


rhcNNDF.to_csv('RHC.csv',index=False)
saNNDF.to_csv('SA.csv',index=False)
gaNNDF.to_csv('GA.csv',index=False)