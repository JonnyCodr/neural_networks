import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T
from sklearn.utils import shuffle
import logging

from src.util import getData, error_rate, relu, init_weight_and_bias, getBinaryData

OUTPUT_FILE_NAME = 'ann_theano_output'

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


class HiddenLayer(object):
    """
    Hidden Layer
    """
    def __init__(self, M1, M2, an_id):
        """
        init
        :param M1:
        :param M2:
        :param an_id:
        """
        self.id = an_id
        self.M1 = M1
        self.M2 = M2
        W, b = init_weight_and_bias(M1, M2)
        self.W = theano.shared(W, 'W_%s' % self.id)
        self.b = theano.shared(b, 'b_%s' % self.id)
        self.params = [self.W, self.b]

    def forward(self, X):
        return relu(X.dot(self.W) + self.b)


class ANN(object):
    """
        ANN
    """
    def __init__(self, hidden_layer_sizes):
        """
        init
        :param hidden_layer_sizes:
        """
        self.hidden_layer_sizes = hidden_layer_sizes

    def fit(self, X, Y, learning_rate=10e-7, mu=0.99, decay=0.999, reg=10e-12, eps=10e-10, epochs=400, batch_sz=100, show_fig=False):
        """
        fit
        :param X:
        :param Y:
        :param learning_rate:
        :param mu:
        :param decay:
        :param reg:
        :param eps:
        :param epochs:
        :param batch_sz:
        :param show_fig:
        :return:
        """
        learning_rate = np.float32(learning_rate)
        mu = np.float32(mu)
        decay = np.float32(decay)
        reg = np.float32(reg)
        eps = np.float32(eps)

        # make a validation set
        X, Y = shuffle(X, Y)
        X = X.astype(np.float32)
        Y = Y.astype(np.int32)
        Xvalid, Yvalid = X[-1000:], Y[-1000:]
        X, Y = X[:-1000], Y[:-1000]

        # initialize hidden layers
        N, D = X.shape
        K = len(set(Y))
        self.hidden_layers = []
        M1 = D
        count = 0
        for M2 in self.hidden_layer_sizes:
            h = HiddenLayer(M1, M2, count)
            self.hidden_layers.append(h)
            M1 = M2
            count += 1
        W, b = init_weight_and_bias(M1, K)
        self.W = theano.shared(W, 'W_logreg')
        self.b = theano.shared(b, 'b_logreg')

        # collect params for later use
        self.params = [self.W, self.b]
        for h in self.hidden_layers:
            self.params += h.params

        # for momentum
        dparams = [theano.shared(np.zeros(p.get_value().shape, dtype=np.float32)) for p in self.params]

        # for rmsprop
        cache = [theano.shared(np.zeros(p.get_value().shape, dtype=np.float32)) for p in self.params]

        # set up theano functions and variables
        thX = T.fmatrix('X')
        thY = T.ivector('Y')
        pY = self.forward(thX)

        rcost = reg*T.sum([(p*p).sum() for p in self.params])
        cost = -T.mean(T.log(pY[T.arange(thY.shape[0]), thY])) + rcost
        prediction = self.predict(thX)

        cost_predict_op = theano.function(inputs=[thX, thY], outputs=[cost, prediction])

        updates = [
            (c, decay*c + (np.float32(1)-decay)*T.grad(cost, p)*T.grad(cost, p)) for p, c in zip(self.params, cache)
        ] + [
            (p, p + mu*dp - learning_rate*T.grad(cost, p)/T.sqrt(c + eps)) for p, c, dp in zip(self.params, cache, dparams)
        ] + [
            (dp, mu*dp - learning_rate*T.grad(cost, p)/T.sqrt(c + eps)) for p, c, dp in zip(self.params, cache, dparams)
        ]

        # momentum only
        # updates = [
        #     (p, p + mu*dp - learning_rate*T.grad(cost, p)) for p, dp in zip(self.params, dparams)
        # ] + [
        #     (dp, mu*dp - learning_rate*T.grad(cost, p)) for p, dp in zip(self.params, dparams)
        # ]

        train_op = theano.function(
            inputs=[thX, thY],
            updates=updates
        )

        n_batches = N / batch_sz
        costs = []
        for i in range(epochs):
            X, Y = shuffle(X, Y)
            for j in range(int(n_batches)):
                Xbatch = X[j*batch_sz:(j*batch_sz+batch_sz)]
                Ybatch = Y[j*batch_sz:(j*batch_sz+batch_sz)]

                train_op(Xbatch, Ybatch)

                if j % 20 == 0:
                    c, p = cost_predict_op(Xvalid, Yvalid)
                    costs.append(c)
                    e = error_rate(Yvalid, p)
                    print("i:", i, "j:", str(j), "nb:", n_batches, "cost:", c, "error rate:", e)
                    logger.info("i: {itr} j: {j} nb: {nb} cost: {cost} error rate {error}"
                                .format(itr=str(i), j=str(j), nb=str(n_batches), cost=str(c), error=str(e)))

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
        forward
        :param X:
        :return:
        """
        Z = X
        for h in self.hidden_layers:
            Z = h.forward(Z)
        return T.nnet.softmax(Z.dot(self.W) + self.b)

    def predict(self, X):
        """
        predict
        :param X:
        :return:
        """
        pY = self.forward(X)
        return T.argmax(pY, axis=1)


def main():
    logger.info("**************************************************************")
    logger.info("Get Data")
    logger.info("**************************************************************")
    # X, Y = getData()
    X, Y = getBinaryData()

    logger.info("**************************************************************")
    logger.info("Generating Model")
    logger.info("**************************************************************")
    model = ANN([2000, 1000])

    logger.info("**************************************************************")
    logger.info("fitting the model")
    logger.info("**************************************************************")
    model.fit(X, Y, show_fig=True)

if __name__ == '__main__':
    main()
