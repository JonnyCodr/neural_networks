import logging
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.utils import shuffle

from src.util import getData, softmax, cost2, y2indicator, error_rate

OUTPUT_FILE_NAME = 'ann_softmax_output'

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
    """ANN class"""

    def __init__(self, M):
        """
        init
        :param M: number of hidden units
        """
        self.M = M

    # learning rate 10e-6 is too large
    def fit(self, X, Y, learning_rate=10e-7, reg=10e-7, epochs=1000, show_fig=False):
        """fit
            can be used with relu or tanh

            :X inputs
            :Y outputs
            :learning_rate
            :reg
            :epochs number of
            :show_fig
        """
        X, Y = shuffle(X, Y)

        logger.info("split data into training and validation sets")
        Xvalid, Yvalid = X[-1000:], Y[-1000:]
        # Tvalid = y2indicator(Yvalid)
        X, Y = X[:-1000], Y[:-1000]

        N, D = X.shape
        K = len(set(Y))
        T = y2indicator(Y)

        logger.info("**************************************************************")
        logger.info("init weights")
        logger.info("**************************************************************")
        self.W1 = np.random.randn(D, self.M) / np.sqrt(D + self.M)
        self.W2 = np.random.randn(self.M, K) / np.sqrt(self.M + K)

        logger.info("**************************************************************")
        logger.info("init bias term")
        logger.info("**************************************************************")
        self.b1 = np.zeros(self.M)
        self.b2 = np.zeros(K)

        costs = []
        best_validation_error = 1

        logger.info("**************************************************************")
        logger.info("beginning for loop")
        logger.info("**************************************************************")
        for i in range(epochs):

            logger.info("forward propagation and cost calculation")
            pY, Z = self.forward(X)

            logger.info("gradient descent step")
            pY_T = pY - T
            self.W2 -= learning_rate*(Z.T.dot(pY_T) + reg*self.W2)
            self.b2 -= learning_rate*(pY_T.sum(axis=0) + reg*self.b2)
            # dZ = pY_T.dot(self.W2.T) * (Z > 0) # relu
            dZ = pY_T.dot(self.W2.T) * (1 - Z*Z) # tanh
            self.W1 -= learning_rate*(X.T.dot(dZ) + reg*self.W1)
            self.b1 -= learning_rate*(dZ.sum(axis=0) + reg*self.b1)

            if i % 10 == 0:
                pYvalid, _ = self.forward(Xvalid)
                c = cost2(Yvalid, pYvalid)
                costs.append(c)
                e = error_rate(Yvalid, np.argmax(pYvalid, axis=1))
                print("i:", i, "cost:", c, "error:", e)
                logger.info("i: {itr} cost: {cost} error: {error}".format(itr=i, cost=c, error=e))
                if e < best_validation_error:
                    best_validation_error = e
        logger.info("best_validation_error: {}".format(best_validation_error))
        self.showFig(costs, show_fig)

    def showFig(self, costs, show_fig):
        """

        :param costs:
        :param show_fig:
        :return:
        """
        if show_fig:
            logger.info("**************************************************************")
            logger.info("Generating pyplot.")
            logger.info("**************************************************************")
            plt.plot(costs)
            plt.savefig(OUTPUT_FILE_NAME, bbox_inches='tight')
            plt.show()

    def forward(self, X):
        """

        :param X:
        :return:
        """
        # Z = relu(X.dot(self.W1) + self.b1)
        Z = np.tanh(X.dot(self.W1) + self.b1)
        return softmax(Z.dot(self.W2) + self.b2), Z

    def predict(self, X):
        """

        :param X:
        :return:
        """
        pY, _ = self.forward(X)
        return np.argmax(pY, axis=1)

    def score(self, X, Y):
        """

        :param X:
        :param Y:
        :return:
        """
        prediction = self.predict(X)
        return 1 - error_rate(Y, prediction)


def main():
    logger.info("**************************************************************")
    logger.info("Get Data")
    logger.info("**************************************************************")
    X, Y = getData()

    logger.info("**************************************************************")
    logger.info("Generating Model")
    logger.info("**************************************************************")
    model = ANN(200)

    logger.info("**************************************************************")
    logger.info("fitting the model")
    logger.info("**************************************************************")
    model.fit(X, Y, reg=0, show_fig=True)
    logger.info("model score: {}".format(model.score(X, Y)))

    scores = cross_val_score(model, X, Y, cv=5)
    logger.info("score mean: {score} stdev: {stdev}".format(score=np.mean(scores), stdev=np.std(scores)))

if __name__ == '__main__':
    main()
