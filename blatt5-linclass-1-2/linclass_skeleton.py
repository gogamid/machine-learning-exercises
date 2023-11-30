import numpy as np
import requests
import os


class LinClass(object):

    ## allocate weight and bias arrays and store ref 2 train data/labels
    def __init__(self, n_in, n_out, X, T):
        self.W = np.ones([n_in, n_out]) * 0.1
        self.b = np.ones([1, n_out]) * 0.1
        self.X = X
        self.T = T
        self.N = X.shape[0]

    # normal cross-entropy loss. Fill in your code here! [b5p4-cross-entropy
    def loss(self, Y, T):
        return - (np.log(Y) * T).sum(axis=1).mean()

    def dLdb(self, Y, T):
        return -(T / Y).sum(axis=1).mean()

    # fill in your code here!
    def dLdW(self, X, Y, T):
        print()

    # softmax: fill in your code here! [b5p2-softmax]
    def S(self, X):
        ex = np.exp(X)
        return ex / ex.sum(axis=1).reshape(-1, 1)

    # dummy model, fill in your code! [b5p3-lin-classifier]
    def lin_class(self, X):
        return self.S(np.matmul(X, self.W) + self.b)

    # performs a single gradient descent step
    # works with any size of X and T
    def train_step(self, X, T, eps):
        print()

    # perform multiple gradient descent steps and display loss. Does it go down??
    def train(self, max_it, eps):
        print()


if __name__ == "__main__":

    # download MNIST if not present in current dir!
    if not os.path.exists("./mnist.npz"):
        print("Downloading MNIST...")
        fname = 'mnist.npz'
        url = 'http://www.gepperth.net/alexander/downloads/'
        r = requests.get(url + fname)
        open(fname, 'wb').write(r.content)

    # read it into
    data = np.load("mnist.npz")
    traind = data["arr_0"]
    trainl = data["arr_2"]
    traind = traind.reshape(60000, 784)
    lc = LinClass(784, 10, traind, trainl)
    f = lc.lin_class(traind[0, :])
    Y_test = np.array([[0, 0, 0, 0, 0.5, 0, 0.3, 0, 0.2, 0, 0]]) + 0.00000000001;
    T_test = np.array([[0, 0, 0, 0, 1., 0, 0, 0, 0, 0, 0]])
    l = lc.loss(Y_test, T_test)
    print(l)
