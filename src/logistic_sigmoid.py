import logging
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle

from src.util import getBinaryData, sigmoid, sigmoid_cost, error_rate

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

# create file handler which logs even debug messages
fh = logging.FileHandler('../logs/logistic_sigmoid_output.log')
fh.setLevel(logging.DEBUG)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s,%(levelname)s,%(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)

# add the handlers to logger
logger.addHandler(ch)
logger.addHandler(fh)


class LogisticModel(object):
    def __init__(self):
        pass

    def fit(self, X, Y, learning_rate=10e-7, reg=0*10e-22, epochs=10000, show_fig=False):
        X, Y = shuffle(X, Y)
        Xvalid, Yvalid = X[-1000:], Y[-1000:]
        X, Y = X[:-1000], Y[:-1000]

        N, D = X.shape
        self.W = np.random.randn(D) / np.sqrt(D)
        self.b = 0

        costs = []
        best_validation_error = 1
        for i in range(epochs):
                # forward propagation and cost calculation
                pY = self.forward(X)

                # gradient descent step
                self.W -= learning_rate*(X.T.dot(pY - Y) + reg*self.W)
                self.b -= learning_rate*((pY - Y).sum() + reg*self.b)

                if i % 20 == 0:
                    pYvalid = self.forward(Xvalid)
                    c = sigmoid_cost(Yvalid, pYvalid)
                    costs.append(c)
                    e = error_rate(Yvalid, np.round(pYvalid))
                    logger.info("i: {itr} cost: {cost} error: {error}".format(itr=i, cost=c, error=e))
                    if e < best_validation_error:
                        best_validation_error = e
        logger.info("best_validation_error: {error}".format(error=best_validation_error))

        if show_fig:
            plt.plot(costs)
            plt.show()

    def forward(self, X):
        return sigmoid(X.dot(self.W) + self.b)

    def predict(self, X):
        pY = self.forward(X)
        return np.round(pY)

    def score(self, X, Y):
        prediction = self.predict(X)
        return 1 - error_rate(Y, prediction)


def main():

    logger.info("getting binary Data...")
    X, Y = getBinaryData()

    X0 = X[Y==0, :]
    X1 = X[Y==1, :]
    X1 = np.repeat(X1, 9, axis=0)
    X = np.vstack([X0, X1])
    Y = np.array([0]*len(X0) + [1]*len(X1))

    N, D = X.shape
    logging.info("N:{}".format(N))

    # logging.info("p(Y=0):", np.sum(Y == 0) / float(N), "p(Y=1):", np.sum(Y == 1) / float(N).format(N))
    # logging.info("p(Y=0): {.5f} | p(Y=1): {.5f}".format(np.sum(Y == 0) / float(N), np.sum(Y == 1) / float(N).format(N)))

    model = LogisticModel()
    model.fit(X, Y, show_fig=True)
    model.score(X, Y)

    # scores = cross_val_score(model, X, Y, cv=5)
    # # print "score mean:", np.mean(scores), "stdev:", np.std(scores)
    # logging.info("score mean: {} stdev: {}".format(np.mean(scores), np.std(scores)))

if __name__ == '__main__':
    main()
o