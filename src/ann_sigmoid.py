import matplotlib.pyplot as plt
import numpy as np
import logging

from sklearn.utils import shuffle

from src.util import getBinaryData, sigmoid, sigmoid_cost, error_rate

OUTPUT_FILE_NAME = 'ann_sigmoid_output'

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

# create file handler which logs even debug messages
fh = logging.FileHandler('../logs/' + OUTPUT_FILE_NAME + '.log')
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


class ANN(object):
    """ANN"""

    def __init__(self, M):
        """

        :param M: number of hidden units
        """
        self.M = M

    def fit(self, X, Y, learning_rate=5*10e-7, reg=1.0, epochs=10000, show_fig=False):
        """fit"""

        X, Y = shuffle(X, Y)
        Xvalid, Yvalid = X[-1000:], Y[-1000:]
        X, Y = X[:-1000], Y[:-1000]

        N, D = X.shape

        logger.info("**************************************************************")
        logger.info("init weights")
        logger.info("**************************************************************")
        self.W1 = np.random.randn(D, self.M) / np.sqrt(D + self.M)
        self.W2 = np.random.randn(self.M) / np.sqrt(self.M)

        logger.info("**************************************************************")
        logger.info("calculating bias term")
        logger.info("**************************************************************")
        self.b1 = np.zeros(self.M)
        self.b2 = 0

        costs = []
        best_validation_error = 1

        logger.info("**************************************************************")
        logger.info("beginning for loop")
        logger.info("**************************************************************")
        for i in range(epochs):

            # forward propagation and cost calculation
            pY, Z = self.forward(X)

            # gradient descent step
            pY_Y = pY - Y
            self.W2 -= learning_rate*(Z.T.dot(pY_Y) + reg*self.W2)
            self.b2 -= learning_rate*((pY_Y).sum() + reg*self.b2)

            dZ = np.outer(pY_Y, self.W2) * (1 - Z*Z)
            self.W1 -= learning_rate*(X.T.dot(dZ) + reg*self.W1)
            self.b1 -= learning_rate*(np.sum(dZ, axis=0) + reg*self.b1)

            if i % 20 == 0:
                pYvalid, _ = self.forward(Xvalid)
                c = sigmoid_cost(Yvalid, pYvalid)
                costs.append(c)
                e = error_rate(Yvalid, np.round(pYvalid))
                print("i:", i, "cost:", c, "error:", e)
                logger.info("i: {itr}, cost: {cost}, error: {error}".format(itr=i, cost=c, error=e))
                if e < best_validation_error:
                    best_validation_error = e
        logger.info("best_validation_error:{best}".format(best=best_validation_error))

        self.showFig(costs, show_fig)

    def showFig(self, costs, show_fig=True, fileName=OUTPUT_FILE_NAME):
        """

        :param costs:
        :param show_fig:
        :param fileName:
        :return:
        """
        if show_fig:
            logger.info("**************************************************************")
            logger.info("Generating pyplot.")
            logger.info("**************************************************************")
            plt.plot(costs)
            plt.savefig(fileName, bbox_inches='tight')
            plt.show()

    def forward(self, X):
        """
        forward propagationn using tanh()
        :param X:
        :return:
        """
        Z = np.tanh(X.dot(self.W1) + self.b1)
        return sigmoid(Z.dot(self.W2) + self.b2), Z

    def predict(self, X):
        """
        predict
        :param X:
        :return:
        """
        pY = self.forward(X)
        return np.round(pY)

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

    logger.info("**************************************************************")
    logger.info("getting binary data")
    logger.info("**************************************************************")
    X, Y = getBinaryData()

    X0 = X[Y==0, :]
    X1 = X[Y==1, :]

    logger.info("**************************************************************")
    logger.info("duplicating X1 so both sets of data are roughly equal")
    logger.info("**************************************************************")
    X1 = np.repeat(X1, 9, axis=0)
    X = np.vstack([X0, X1])
    Y = np.array([0]*len(X0) + [1]*len(X1))

    logger.info("**************************************************************")
    logger.info("Running ANN")
    logger.info("**************************************************************")
    model = ANN(100)

    logger.info("**************************************************************")
    logger.info("fitting")
    logger.info("**************************************************************")
    model.fit(X, Y, show_fig=True)
    logger.info("model score: {}".format(model.score(X, Y)))

    scores = cross_val_score(model, X, Y, cv=5)
    logger.info("score mean: {score} stdev: {stdev}".format(score=np.mean(scores), stdev=np.std(scores)))


if __name__ == '__main__':
    main()
