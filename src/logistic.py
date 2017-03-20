import logging

import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle

from src.util import getData, softmax, cost, y2indicator, error_rate

OUTPUT_FILE = 'logistic_regression_output'

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

# create file handler which logs even debug messages
fh = logging.FileHandler('../logs/' + OUTPUT_FILE + '.log')
fh.setLevel(logging.DEBUG)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)

# add the handlers to logger
logger.addHandler(ch)
logger.addHandler(fh)


class LogisticModel(object):
    def __init__(self):
        pass

    def fit(self, X, Y, learning_rate=10e-8, reg=10e-12, epochs=10000, show_fig=True):
        """
        fit
        :param X:
        :param Y:
        :param learning_rate:
        :param reg:
        :param epochs:
        :param show_fig:
        :return:
        """
        X, Y = shuffle(X, Y)
        Xvalid, Yvalid = X[-1000:], Y[-1000:]
        Tvalid = y2indicator(Yvalid)
        X, Y = X[:-1000], Y[:-1000]

        N, D = X.shape
        K = len(set(Y))
        T = y2indicator(Y)

        logger.info("**************************************************************")
        logger.info("init weights")
        logger.info("**************************************************************")
        self.W = np.random.randn(D, K) / np.sqrt(D + K)
        self.b = np.zeros(K)

        costs = []
        best_validation_error = 1

        for i in range(epochs):
            # forward propagation and cost calculation
            pY = self.forward(X)

            # gradient descent step
            self.W -= learning_rate*(X.T.dot(pY - T) + reg*self.W)
            self.b -= learning_rate*((pY - T).sum(axis=0) + reg*self.b)

            if i % 10 == 0:
                pYvalid = self.forward(Xvalid)
                c = cost(Tvalid, pYvalid)
                costs.append(c)
                e = error_rate(Yvalid, np.argmax(pYvalid, axis=1))
                print("i: {itr}, cost: {cost}, error: {error}".format(itr=i, cost=c, error=e))
                logger.info("i: {itr}, cost: {cost}, error: {error}".format(itr=i, cost=c, error=e))
                if e < best_validation_error:
                    best_validation_error = e
        logger.info("best_validation_error:{best}".format(best=best_validation_error))

        self.showFig(costs, show_fig)

    def showFig(self, costs, show_fig, file_name=OUTPUT_FILE):
        """

        :param costs:
        :param show_fig:
        :return:
        """
        if show_fig:
            plt.plot(costs)
            plt.savefig(file_name, bbox_inches='tight')
            plt.show()

    def forward(self, X):
        """
        forward - using softmax
        :param X:
        :return:
        """
        return softmax(X.dot(self.W) + self.b)

    def predict(self, X):
        """

        :param X:
        :return:
        """
        pY = self.forward(X)
        return np.argmax(pY, axis=1)

    def score(self, X, Y):
        """
        score
        :param X:
        :param Y:
        :return:
        """
        prediction = self.predict(X)
        return 1 - error_rate(Y, prediction)


def main():
    """main"""

    logger.info("**************************************************************")
    logger.info("getting Data")
    logger.info("**************************************************************")
    X, Y = getData()

    logger.info("**************************************************************")
    logger.info("Generating Model")
    logger.info("**************************************************************")
    model = LogisticModel()

    logger.info("**************************************************************")
    logger.info("Fitting the model")
    logger.info("**************************************************************")
    model.fit(X, Y, show_fig=True)
    logger.info("score: {}".format(model.score(X, Y)))

if __name__ == '__main__':
    main()
