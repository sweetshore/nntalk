from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from optimization import gradient
from random import shuffle

iris_names = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

def logistic_model(feat, coeff):
    """Calculates our estimate of probability that object belongs to class 1.
    
    Input:
    feat -- vector of object features
    coeff -- vector of model coefficients
    """
    val = 0.
    for n in range(len(feat)):
        val += feat[n] * coeff[n]
    val += coeff[-1] # bias term -- last coefficient
    e = np.exp(val)
    return e / (e + 1)

def vec_norm(vec):
    squares = 0.
    for val in vec:
        squares += val * val
    return np.sqrt(squares)

def loss(label, model_fun, feat, coeff):
    """loss function"""
    return np.abs(float(label) - model_fun(feat, coeff)) #+ 0.01 * vec_norm(coeff)

def sgd(loss_fun, model_fun, data, labels, coeff_start, rate, num_epochs):
    """Stochastic gradient descent"""
    coeff = coeff_start
    sample_indices = list(range(len(labels)))
    for epoch in range(num_epochs):
        shuffle(sample_indices)
        for nsample in sample_indices:
            label = labels[nsample] # current label
            feat = data[nsample] # current feature vector

            fun = lambda x: loss_fun(label, model_fun, feat, x) # define new function of x: fun(x)
            grad = gradient(fun, coeff, 0.01)
            coeff -= rate * grad # update model coefficients

        accuracy, loss = test(coeff, model_fun, loss_fun, data, labels)
        print("epoch:", epoch, "accuracy:", accuracy, "loss:", loss)
        
    return coeff

def predict(feat, model_fun, coeff):
    """Returns predicted class index for given features"""
    if model_fun(feat, coeff) > 0.5:
        return 1
    else:
        return 0

def test(coeff, model_fun, loss_fun, data, labels):
    correct = 0
    loss = 0.
    for n in range(len(labels)):
        corr_label = labels[n]
        feat = data[n]
        pred_label = predict(feat, logistic_model, coeff)
        loss += loss_fun(corr_label, model_fun, feat, coeff)
        if pred_label == corr_label:
            correct += 1

    percent_correct = float(correct) / float(len(labels))
    average_loss = loss / float(len(labels))
    return percent_correct, average_loss



def read_iris_data(fname='./data/iris2.txt', cols = [2, 3]):
    """Generate feature vectors and labels from iris dataset file"""
    # https://archive.ics.uci.edu/ml/machine-learning-databases/iris/

    iris_classes = {}
    for n, name in enumerate(iris_names):
        iris_classes[name] = n

    data = []
    labels = []
    for line in open(fname, 'rt'):
        words = line.split(',')
        vec = np.zeros(len(cols))
        for n, col in enumerate(cols):
            vec[n] = float(words[col])
        data.append(vec)
        labels.append(iris_classes[words[4][:-1]])

    return data, labels

def plot_iris_data(data, labels, line_coeff=None):
    styles = ['or', 'ob', 'og']
    #ax = plt.figure()
    for curr_label, name in enumerate(iris_names):
        x = []
        y = []
        for n, label in enumerate(labels):
            if label == curr_label:
                x.append(data[n][0])
                y.append(data[n][1])
        if len(x) > 0:
            plt.plot(x, y, styles[curr_label], label=iris_names[curr_label])
    plt.legend()

    if line_coeff is not None:
        fun = lambda x: -1. / line_coeff[1] * (line_coeff[0] * x + line_coeff[2])
        x = [0., 5.]
        y = [fun(x[0]), fun(x[1])]
        plt.plot(x, y, '-k')

    plt.show()

if __name__ == "__main__":
    data, labels = read_iris_data()
    #plot_iris_data(data, labels)

    coeff_start = np.array([1., 1., 1.])
    coeff = sgd(loss, logistic_model, data, labels, coeff_start, 0.1, 10)
    print(coeff)
    plot_iris_data(data, labels, coeff)
