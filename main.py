import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve, validation_curve, cross_val_predict
from sklearn.linear_model import Perceptron

#---------------------------#
#       REFERENCIAS         #
#---------------------------#
# Referencia (Missing values): https://machinelearningmastery.com/handle-missing-data-python/

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")

    print(train_scores_mean, test_scores_mean)
    return plt

def plot_validation(X, y, clf, param_name, param_range):
    train_scores, test_scores = validation_curve(
        clf, X, y, param_name=param_name, param_range=param_range,
        cv=10, scoring="accuracy", n_jobs=4)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    print(train_scores_mean)
    plt.title("Validation Curve")
    plt.xlabel("$learning rate$")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.semilogx(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()

#---------------------------#
#     PREPROCESAMIENTO      #
#---------------------------#
# 1. Manejar data faltante.
file_name = "cleveland.txt"
data = pd.read_csv(file_name, header=None)
data.fillna(data.mean(), inplace=True)

# 2. Normalizar datos
min_max_scaler = preprocessing.MinMaxScaler()   # Función utilizada para normalización min-max
X = data.values #returns a numpy array
y = X[:,-1]
y_bin = np.array([1 if int(i)>0.1 else 0 for i in X[:,-1]]) # Para clasificador binario.
X_scaled = min_max_scaler.fit_transform(X)

# 3. Ajustar clases (Clasificador binario)
#---------------------------#
#       CLASIFICADORES      #
#---------------------------#
clf_mlp = MLPClassifier(solver='sgd', alpha=1e-5,hidden_layer_sizes=(5,3), 
    random_state=1, learning_rate_init=0.001, max_iter=3000, momentum= 0.5,
    activation='tanh' )
clf_svm = SVC(gamma='auto', kernel='sigmoid')
clf_perceptron = Perceptron(tol=1e-3, random_state=0)
models = [clf_mlp, clf_svm]

#---------------------------#
#           TESTING         #
#---------------------------#
n_splits   = 10  # Número de particiones para el K-fold
skf = StratifiedKFold(n_splits=n_splits)
skf.get_n_splits(X_scaled, y)
StratifiedKFold(n_splits=n_splits, random_state=None, shuffle=True)

# Pruebas utilizando MLP
def runMLP(X_scaled, y, name):
    activations = ['tanh', 'relu']
    layers = [(3,), (4,),(3,2), (4,3), (5,2), (5,3)]  # (5,3)
    learning_rs = [0.001, 0.005, 0.01, 0.05]
    momentums = [0, 0.5]
    for m in momentums:
        if(m > 0):
            f = open(name + "Momentum.txt", "w")
        else:
            f = open(name+".txt", "w")
        for a in activations:
            for layer in layers:
                for lr in learning_rs:
                    model = MLPClassifier(solver='sgd', alpha=1e-5,hidden_layer_sizes=layer, 
                    random_state=1, learning_rate_init=lr, max_iter=3000, momentum= m,
                    activation=a)
                    training_acc, validation_acc = 0, 0
                    for train_index, test_index in skf.split(X_scaled, y):
                        # Separar conjuntos de prueba y entrenamiento.
                        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
                        y_train, y_test = y[train_index], y[test_index]
                        
                        # Entrenar clasificador
                        model.fit(X_train, y_train)

                        # Almacenar precision de cada ejecucion
                        training_acc += model.score(X_train, y_train)
                        validation_acc += model.score(X_test, y_test)

                    # Presentar resultados
                    p_training, p_valid = training_acc/n_splits, validation_acc/n_splits
                    result = a + "\t" + str(len(layer)) + "\t" + str(layer) + "\t" + str(lr) + "\t"
                    result = result + str(round(p_training,4)*100) +"\t" + str(round(p_valid,4)*100) +"\n"
                    print(result)
                    f.write(result)
        f.close()

def runSVM(X_scaled, y, name):
    kernels = ['sigmoid', 'rbf']
    f = open(name + ".txt", "w")
    for k in kernels:
        model = SVC(gamma='auto', kernel=k)
        training_acc, validation_acc = 0, 0
        for train_index, test_index in skf.split(X_scaled, y):
            # Separar conjuntos de prueba y entrenamiento.
            X_train, X_test = X_scaled[train_index], X_scaled[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Entrenar modelos
            model.fit(X_train, y_train)

            # Almacenar precision de cada ejecucion
            training_acc += model.score(X_train, y_train)
            validation_acc += model.score(X_test, y_test)

        # Presentar resultados
        p_training, p_valid = training_acc/n_splits, validation_acc/n_splits
        result =k + "\t"
        result = result + str(round(p_training,4)*100) +"\t" + str(round(p_valid,4)*100) +"\n"
        print(result)
        f.write(result)

runMLP(X_scaled, y, "MLP")
runMLP(X_scaled, y_bin, "MLPBin")
runSVM(X_scaled, y, "SVM")
runSVM(X_scaled, y_bin, "SVMBin")

