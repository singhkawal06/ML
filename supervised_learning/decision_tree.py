
from supervised_learning.data_load import  load_heart_stroke_data, load_women_diabetes_data
from datetime import time
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
import time
import matplotlib.pyplot as plt

import numpy as np

X, y = load_heart_stroke_data('../data/heart-stroke/data.csv')
#X, y = load_women_diabetes_data('../data/diabetes_in_women/data.csv')


X = preprocessing.scale(X)

img_path = '../images/'


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=12)


dt_classifer = tree.DecisionTreeClassifier(random_state=7)
dt_classifer.fit(X_train, y_train)
prediction_y_dt = dt_classifer.predict(X_test)
dt_accuracy = accuracy_score(y_test, prediction_y_dt)
print('Accuracy of decision tree classifier without hyperparameter tuning is %.2f%%' % (dt_accuracy * 100))

param_range = np.arange(1, 30)
train_scores, test_scores = validation_curve(tree.DecisionTreeClassifier(random_state=7), X_train, y_train,
                                             param_name="max_depth", param_range=param_range, cv=5)
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
plt.figure()
plt.title('Validation curve for decision tree')
plt.plot(param_range,train_scores_mean, label='Training score',color="darkorange")
plt.plot(param_range, test_scores_mean, label='Cross Validation score',color="navy")
plt.xlabel('max_depth')
plt.ylabel("score")
plt.legend(loc="best")
plt.grid()
plt.savefig(img_path + 'dt_validation_curve_graph_1.png')

param_range_2 = np.arange(2, 30)
train_scores, test_scores = validation_curve(tree.DecisionTreeClassifier(random_state=7), X_train, y_train,
                                             param_name="min_samples_split", param_range=param_range_2, cv=5)
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.figure()
plt.plot(param_range_2, train_scores_mean, label='Training score',color="darkorange")
plt.plot(param_range_2, test_scores_mean, label='Cross Validation score',color="navy")
plt.title('Validation curve for decision tree')
plt.xlabel('min_samples_split')
plt.ylabel("score")
plt.legend(loc="best")
plt.grid()
plt.savefig(img_path + 'dt_validation_curve_graph_2.png')

parameters = {'max_depth': param_range, 'min_samples_split': param_range_2}
best_dt_classifer = GridSearchCV(dt_classifer, param_grid=parameters, cv=5)
time_start = time.time()
best_dt_classifer.fit(X_train, y_train)
time_end = time.time()
time_difference = time_end - time_start
print("Training Time:")
print(time_difference)
print("Best parameters:")
print(best_dt_classifer.best_params_)
time_start = time.time()
prediction_y_dt = best_dt_classifer.predict(X_test)
time_end = time.time()
time_difference = time_end - time_start
print("Testing Time:")
print(time_difference)
dt_accuracy = accuracy_score(y_test, prediction_y_dt)
print('Accuracy of decision tree classifier is %.2f%%' % (dt_accuracy * 100))

_, train_scores, test_scores = learning_curve(best_dt_classifer, X_train, y_train, train_sizes=np.linspace(0.1, 1.0, 5), cv=5)
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
plt.figure()
plt.title('Learning curve for decision tree classifier')
plt.plot(np.linspace(0.1, 1.0, 5), train_scores_mean, 'o-', label='Training score',color="darkorange")
plt.plot(np.linspace(0.1, 1.0, 5), test_scores_mean, 'o-', label='Cross Validation score',color="navy")
plt.xlabel('training set size')
plt.ylabel("score")
plt.legend(loc="best")
plt.grid()
plt.savefig(img_path + 'dt_learning_curve_graph.png')



#Boosting




dt_stump = tree.DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
boost_classifier = AdaBoostClassifier(base_estimator=dt_stump, n_estimators=100,random_state=7)
boost_classifier.fit(X_train, y_train)
prediction_y_boost = boost_classifier.predict(X_test)
boosted_accuracy = accuracy_score(y_test, prediction_y_boost)
print('Accuracy of Adaboost without hyperparameter tuning is %.2f%%' % (boosted_accuracy * 100))


num_folds = 4
kf = KFold(n_splits=num_folds,shuffle=False, random_state=None)
train_scores = np.zeros((100, num_folds))
val_scores = np.zeros((100, num_folds))
for idx, (train_index, test_index) in enumerate(kf.split(X_train)):
    boost_classifier.fit(X_train[train_index], y_train[train_index])
    train_scores[:, idx] = np.asarray(list(boost_classifier.staged_score(X_train[train_index], y_train[train_index])))
    val_scores[:, idx] = np.asarray(list(boost_classifier.staged_score(X_train[test_index], y_train[test_index])))

n_estimators_range = np.arange(100) + 1
plt.figure()
plt.plot(n_estimators_range, np.mean(train_scores, axis=1), label='Training score')
plt.plot(n_estimators_range, np.mean(val_scores, axis=1), label='Cross-validation score')
plt.title('Cross-validation curve for AdaBoost')
plt.xlabel('Number of weak learners')
plt.ylabel("Classification score")
plt.legend(loc="best")
plt.grid()
plt.savefig(img_path + 'boost_validation_curve.png')


parameters = {
    "n_estimators": [5, 10, 15, 20, 25, 50, 75, 100],
    "learning_rate": [0.001, 0.01, 0.1, 1.],
}


clf = GridSearchCV(
    boost_classifier,
    parameters,
    cv=5,
    scoring='accuracy',
)
time_start = time.time()
clf.fit(X_train, y_train)
time_end = time.time()
time_difference = time_end - time_start
print("Training Time:")
print(time_difference)
print("Best parameters:")
print(clf.best_params_)
time_start = time.time()
prediction_y_boost = clf.best_estimator_.predict(X_test)
time_end = time.time()
time_difference = time_end - time_start
print("Testing Time:")
print(time_difference)
dt_accuracy = accuracy_score(y_test, prediction_y_boost)

print('Accuracy of ADABoost  is %.2f%%' % (dt_accuracy * 100))


_, train_scores, test_scores = learning_curve(clf, X_train, y_train, train_sizes=np.linspace(0.1, 1.0, 5), cv=5)
mean_train_scores = np.mean(train_scores, axis=1)
mean_test_scores = np.mean(test_scores, axis=1)

plt.figure()
plt.title('Learning curve for AdaBoost Classifier')

plt.plot(np.linspace(0.1, 1.0, 5), mean_train_scores, 'o-', label='Training score')
plt.plot(np.linspace(0.1, 1.0, 5), mean_test_scores, 'o-', label='Cross-validation score')
plt.xlabel('training set size')
plt.ylabel("score")
plt.legend(loc="best")
plt.grid()
plt.savefig(img_path + 'boost_learning_curve.png')


#Neural Networks

mlp_classifier = MLPClassifier(hidden_layer_sizes=(7, 4), random_state=7, early_stopping=True, max_iter=300,validation_fraction=0.2)
mlp_classifier.fit(X_train, y_train)
prediction_y_nn = mlp_classifier.predict(X_test)
nn_accuracy = accuracy_score(y_test, prediction_y_nn)
print('Accuracy of neural network MLP Classifier is %.2f%%' % (nn_accuracy * 100))

train_scores, test_scores = validation_curve(mlp_classifier, X_train, y_train, param_name="alpha", param_range=np.logspace(-3, 3, 7), cv=5)
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
plt.figure()
plt.title('Validation curve for neural network MLP Classifier ')
plt.semilogx(np.logspace(-3, 3, 7), train_scores_mean, label='Training score')
plt.semilogx(np.logspace(-3, 3, 7), test_scores_mean, label='Cross Validation score')
plt.xlabel('alpha')
plt.ylabel("Classification score")
plt.legend(loc="best")
plt.grid()
plt.savefig(img_path + 'nn_validation_curve_1.png')

lr_range = np.logspace(-5, 0, 5)
train_scores, test_scores = validation_curve(mlp_classifier, X_train, y_train, param_name="learning_rate_init", param_range=np.logspace(-5, 0, 5),
                                             cv=5)
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
plt.figure()
plt.title('Validation curve for neural network MLP Classifier ')
plt.semilogx(np.logspace(-5, 0, 5), train_scores_mean, label='Training score')
plt.semilogx(np.logspace(-5, 0, 5), test_scores_mean, label='Cross Validation score')
plt.xlabel('Learning rate')
plt.ylabel("Classification score")
plt.legend(loc="best")
plt.grid()
plt.savefig(img_path + 'nn_validation_curve_2.png')



# Define grid for grid search after observing validation curves
alpha_range = np.logspace(-3, 2, 5)
lr_range = np.logspace(-5, 0, 6)
tuned_params = {'alpha' : alpha_range, 'learning_rate_init' : lr_range}
best_mlp_classifier = GridSearchCV(mlp_classifier, param_grid=tuned_params, cv=5)
time_start = time.time()
best_mlp_classifier.fit(X_train, y_train)
time_end = time.time()
time_difference = time_end - time_start
print("Training Time:")
print(time_difference)
best_parameters = best_mlp_classifier.best_params_
time_start = time.time()
prediction_y_nn = best_mlp_classifier.best_estimator_.predict(X_test)
time_end = time.time()
time_difference = time_end - time_start
print("Testing Time:")
print(time_difference)
dt_accuracy = accuracy_score(y_test, prediction_y_nn)
print('Accuracy of MLP Classifier tree is %.2f%%' % (dt_accuracy * 100))

print("Best parameters:")
print(best_parameters)


_, train_scores, test_scores = learning_curve(mlp_classifier, X_train, y_train, train_sizes=np.linspace(0.1, 1.0, 5), cv=5)
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
plt.savefig(img_path + 'nn_learning_curve.png')

#Loss Curve

mlp_classifier = MLPClassifier(hidden_layer_sizes=(5, 2), random_state=7, max_iter=1, warm_start=True)
mlp_classifier.set_params(alpha=best_parameters['alpha'], learning_rate_init=best_parameters['learning_rate_init'])

X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_train, y_train, test_size=0.2, random_state=12)

num_epochs = 500
training_loss = np.empty(num_epochs)
training_scores = np.empty(num_epochs)
scores = np.empty(num_epochs)
for i in range(num_epochs):
    mlp_classifier.fit(X_train_new, y_train_new)
    training_loss[i] = mlp_classifier.loss_
    training_scores[i] = accuracy_score(y_train_new, mlp_classifier.predict(X_train_new))
    scores[i] = accuracy_score(y_test_new, mlp_classifier.predict(X_test_new))

prediction_y_nn = mlp_classifier.predict(X_test)
nn_accuracy = accuracy_score(y_test, prediction_y_nn)
print('Accuracy of neural network MLP Classifier is %.2f%%' % (nn_accuracy * 100))

xrange = np.arange(num_epochs) + 1
plt.figure()
plt.title('Training loss curve for neural network MLP Classifier')
plt.plot(xrange, training_loss)
plt.xlabel('Epochs')
plt.ylabel("Loss")
plt.grid()
plt.savefig(img_path + 'nn_train_loss.png')

plt.figure()
plt.plot(xrange, training_scores, label='Training score')
plt.plot(xrange, scores, label='Validation score')
plt.title('Training and validation score curve for neural network MLP Classifier')
plt.xlabel('Epochs')
plt.ylabel("score")
plt.grid()
plt.legend(loc="best")
plt.savefig(img_path + 'nn_score_curve.png')





#SVM
svm_classifier_poly = svm.SVC(kernel='poly')
svm_classifier_poly.fit(X_train, y_train)
prediction_y_svm = svm_classifier_poly.predict(X_test)
svm_poly_accuracy = accuracy_score(y_test, prediction_y_svm)
print('Accuracy of SVM Classifier with polynomial kernel is %.2f%%' % (svm_poly_accuracy * 100))


svm_classifier_linear = svm.SVC(kernel='linear')
svm_classifier_linear.fit(X_train, y_train)
prediction_y_svm = svm_classifier_linear.predict(X_test)
svm_linear_accuracy = accuracy_score(y_test, prediction_y_svm)
print('Accuracy of SVM Classifier with linear kernel is %.2f%%' % (svm_linear_accuracy * 100))


svm_classifier_rbf = svm.SVC(kernel='rbf')
svm_classifier_rbf.fit(X_train, y_train)
prediction_y_svm = svm_classifier_rbf.predict(X_test)
svm_rbf_accuracy = accuracy_score(y_test, prediction_y_svm)
print('Accuracy of SVM Classifier with RBF kernel is %.2f%%' % (svm_rbf_accuracy * 100))


train_scores, test_scores = validation_curve(svm_classifier_poly, X_train, y_train, param_name="C", param_range=np.logspace(-3, 3, 7), cv=5)

plt.figure()
plt.title('Validation curve for SVM (poly kernel)')

plt.semilogx(np.logspace(-3, 3, 7), np.mean(train_scores, axis=1), label='Training score')
plt.semilogx(np.logspace(-3, 3, 7), np.mean(test_scores, axis=1), label='Cross-validation score')
plt.xlabel('C')
plt.ylabel("Classification score")
plt.legend(loc="best")
plt.grid()
plt.savefig(img_path + 'SVM_validation_curve_1.png')

tuned_params = {'C' : np.logspace(-3, 1, 10)}
best_svm_classifier = GridSearchCV(svm_classifier_poly, param_grid=tuned_params, cv=5)
time_start = time.time()
best_svm_classifier.fit(X_train, y_train)
time_end = time.time()
time_difference = time_end - time_start
print("Training Time:")
print(time_difference)
best_parameters = best_svm_classifier.best_params_
print("Best parameters :" )
print(best_parameters)
time_start = time.time()
prediction_y_svm = best_svm_classifier.predict(X_test)
time_end = time.time()
time_difference = time_end - time_start
print("Testing Time:")
print(time_difference)
accuracy_score = accuracy_score(y_test, prediction_y_svm)
print('Best accuracy with SVM Classifier (poly kernel) is %.2f%%' % (accuracy_score * 100))




_, train_scores, test_scores = learning_curve(best_svm_classifier, X_train, y_train, train_sizes=np.linspace(0.1, 1.0, 5), cv=5)
mean_train_scores = np.mean(train_scores, axis=1)
mean_test_scores = np.mean(test_scores, axis=1)

plt.figure()
plt.plot(np.linspace(0.1, 1.0, 5), mean_train_scores, 'o-', label='Training score')
plt.plot(np.linspace(0.1, 1.0, 5), mean_test_scores, 'o-', label='Cross Validation score')
plt.title('Learning curve  SVM (poly kernel)')
plt.xlabel('training set size')
plt.ylabel("score")
plt.legend(loc="best")
plt.grid()
plt.savefig(img_path + 'svm_learning_curve.png')

#KNN
"""


for k in range(1, 20):
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train, y_train)
    prediction_y_knn = knn_classifier.predict(X_test)
    knn_classifier_accuracy = accuracy_score(y_test, prediction_y_knn)
    print('Accuracy of KNN Classifier with k value = %d is %.2f%%' % (k, knn_classifier_accuracy * 100))

train_scores, test_scores = validation_curve(KNeighborsClassifier(), X_train, y_train, param_name="n_neighbors",
                                             param_range= np.arange(1, 50), cv=5)

mean_train_scores = np.mean(train_scores, axis=1)
mean_test_scores = np.mean(test_scores, axis=1)
plt.figure()
plt.title('Validation curve for kNN Classifier')
plt.plot(np.arange(1, 50), mean_train_scores, label='Training score')
plt.plot(np.arange(1, 50), mean_test_scores, label='Cross Validation score')
plt.xlabel('k value')
plt.ylabel("score")
plt.legend(loc="best")
plt.grid()
plt.savefig(img_path + 'knn_validation_curve_1.png')


optimal_k = np.argmax(np.mean(test_scores, axis=1)) + 1
print('Optimal value of k: %d' % optimal_k)
best_knn_classifier = KNeighborsClassifier(n_neighbors=optimal_k)
time_start = time.time()
best_knn_classifier.fit(X_train, y_train)
time_end = time.time()
time_difference = time_end - time_start
print("Training Time:")
print(time_difference)

time_start = time.time()
prediction_y_knn = best_knn_classifier.predict(X_test)
time_end = time.time()
time_difference = time_end - time_start
print("Testing Time:")
print(time_difference)
accuracy_score = accuracy_score(y_test, prediction_y_knn)
print('Accuracy of kNN with k = %d is %.2f%%' % (optimal_k, accuracy_score * 100))


_, train_scores, test_scores = learning_curve(best_knn_classifier, X_train, y_train, train_sizes=np.linspace(0.1, 1.0, 5), cv=5)
mean_train_scores = np.mean(train_scores, axis=1)
mean_test_scores = np.mean(test_scores, axis=1)

plt.figure()
plt.title('Learning curve for kNN')
plt.plot(np.linspace(0.1, 1.0, 5), mean_train_scores, 'o-', label='Training score')
plt.plot(np.linspace(0.1, 1.0, 5), mean_test_scores, 'o-', label='Cross Validation score')
plt.xlabel('training set size')
plt.ylabel("score")
plt.legend(loc="best")
plt.grid()
plt.savefig(img_path + 'knn_learning_curve.png')


"""