import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

from src.util import getData, y2indicator, error_rate, init_weight_and_bias, getBinaryData
import logging

OUTPUT_FILE_NAME = 'ann_tensorflow_output'

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
        self.id = an_id
        self.M1 = M1
        self.M2 = M2
        W, b = init_weight_and_bias(M1, M2)
        self.W = tf.Variable(W.astype(np.float32))
        self.b = tf.Variable(b.astype(np.float32))
        self.params = [self.W, self.b]

    def forward(self, X):
        return tf.nn.relu(tf.matmul(X, self.W) + self.b)


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

    def fit(self, X, Y, learning_rate=10e-7, mu=0.99, decay=0.999, reg=10e-3, epochs=400, batch_sz=100, show_fig=False):
        """
        fit
        :param X: 
        :param Y: 
        :param learning_rate: 
        :param mu: 
        :param decay: 
        :param reg: 
        :param epochs: 
        :param batch_sz: 
        :param show_fig: 
        :return: 
        """
        K = len(set(Y)) # won't work later b/c we turn it into indicator

        # make a validation set
        X, Y = shuffle(X, Y)
        X = X.astype(np.float32)
        Y = y2indicator(Y).astype(np.float32)
        # Y = Y.astype(np.int32)
        Xvalid, Yvalid = X[-1000:], Y[-1000:]
        Yvalid_flat = np.argmax(Yvalid, axis=1) # for calculating error rate
        X, Y = X[:-1000], Y[:-1000]

        # initialize hidden layers
        N, D = X.shape
        
        self.hidden_layers = []
        M1 = D
        count = 0
        for M2 in self.hidden_layer_sizes:
            h = HiddenLayer(M1, M2, count)
            self.hidden_layers.append(h)
            M1 = M2
            count += 1
        W, b = init_weight_and_bias(M1, K)
        self.W = tf.Variable(W.astype(np.float32))
        self.b = tf.Variable(b.astype(np.float32))

        # collect params for later use
        self.params = [self.W, self.b]
        for h in self.hidden_layers:
            self.params += h.params

        # set up theano functions and variables
        tfX = tf.placeholder(tf.float32, shape=(None, D), name='X')
        tfT = tf.placeholder(tf.float32, shape=(None, K), name='T')
        act = self.forward(tfX)

        rcost = reg*sum([tf.nn.l2_loss(p) for p in self.params])
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(act, tfT)) + rcost
        prediction = self.predict(tfX)
        train_op = tf.train.RMSPropOptimizer(learning_rate, decay=decay, momentum=mu).minimize(cost)

        n_batches = N / batch_sz
        costs = []
        init = tf.initialize_all_variables()
        with tf.Session() as session:
            session.run(init)

            logger.info("**************************************************************")
            logger.info("Iterating....")
            logger.info("**************************************************************")
            for i in range(epochs):
                X, Y = shuffle(X, Y)
                for j in range(int(n_batches)):
                    Xbatch = X[j*batch_sz:(j*batch_sz+batch_sz)]
                    Ybatch = Y[j*batch_sz:(j*batch_sz+batch_sz)]

                    session.run(train_op, feed_dict={tfX: Xbatch, tfT: Ybatch})

                    if j % 20 == 0:
                        c = session.run(cost, feed_dict={tfX: Xvalid, tfT: Yvalid})
                        costs.append(c)

                        p = session.run(prediction, feed_dict={tfX: Xvalid, tfT: Yvalid})
                        e = error_rate(Yvalid_flat, p)
                        print("i: {itr} j: {jtr} nb: {nb} cost: {cost} error rate: {error}"
                                    .format(itr=i, jtr=j, nb=n_batches, cost=c, error=e))
                        logger.info("i: {itr} j: {jtr} nb: {nb} cost: {cost} error rate: {error}"
                                    .format(itr=i, jtr=j, nb=n_batches, cost=c, error=e))

        if show_fig:
            logger.info("**************************************************************")
            logger.info("Generating pyplot.")
            logger.info("**************************************************************")
            plt.plot(costs)
            plt.savefig(OUTPUT_FILE_NAME, bbox_inches='tight')
            plt.show()

    def forward(self, X):
        Z = X
        for h in self.hidden_layers:
            Z = h.forward(Z)
        return tf.matmul(Z, self.W) + self.b

    def predict(self, X):
        act = self.forward(X)
        return tf.argmax(act, 1)


def main():
    logger.info("**************************************************************")
    logger.info("Get binary data.")
    logger.info("**************************************************************")
    # X, Y = getData()
    X, Y = getBinaryData()

    logger.info("**************************************************************")
    logger.info("Generating Model.")
    logger.info("**************************************************************")
    model = ANN([2000, 1000, 500])

    logger.info("**************************************************************")
    logger.info("Fitting Model.")
    logger.info("**************************************************************")
    model.fit(X, Y, show_fig=True)

if __name__ == '__main__':
    main()
