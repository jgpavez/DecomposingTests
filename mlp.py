__docformat__ = 'restructedtext en'


import cPickle
import gzip
import os
import sys
import time
import pdb

import numpy

import theano
import theano.tensor as T
import copy
from utils import loadData

from logistic_sgd import LogisticRegression

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None
                       else activation(lin_output))
        # parameters of the model
        self.params = [self.W, self.b]


class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softamx layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        self.hiddenLayer = HiddenLayer(rng=rng, input=input,
                                       n_in=n_in, n_out=n_hidden,
                                       activation=T.tanh)

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out)

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = abs(self.hiddenLayer.W).sum() \
                + abs(self.logRegressionLayer.W).sum()

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (self.hiddenLayer.W ** 2).sum() \
                    + (self.logRegressionLayer.W ** 2).sum()

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        self.y_out = self.logRegressionLayer.p_y_given_x
        self.predictions = self.logRegressionLayer.y_pred
        self.logit = T.log(self.y_out) - T.log(1. - self.y_out)

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params

    def save_model(self, best_params, filename="params.pkl"):
        ''' Save parameters of the model '''

        print '...saving model'
        #if not os.path.isdir(save_dir):
        #    os.makedirs(save_dir)
        #save_file= open(os.path.join(save_dir, filename),'wb')
        save_file= open(filename,'wb')
        cPickle.dump(best_params, save_file, protocol=cPickle.HIGHEST_PROTOCOL)
        save_file.close()

    def load_model(self, filename="params.pkl"):
        ''' Save parameters of the model '''

        #print '...loading model'

        save_file = open(filename, 'r')
        params = cPickle.load(save_file)
        save_file.close()
            
        self.hiddenLayer.W.set_value(params[0].get_value(), borrow=True)
        self.hiddenLayer.b.set_value(params[1].get_value(), borrow=True)
        self.logRegressionLayer.W.set_value(params[2].get_value(), borrow=True)
        self.logRegressionLayer.b.set_value(params[3].get_value(), borrow=True)

def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asmatrix(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')

def make_predictions(dataset, learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=10,
              batch_size=20, n_hidden=40,in_size=1,out_size=2,
              model_file='model/mlp/adaptive_0_1.pkl'):

    test_set_x = dataset
    in_size = test_set_x.shape[1] if len(test_set_x.shape) > 1 else 1
    if in_size == 1:
      test_set_x = test_set_x.reshape(test_set_x.shape[0],1)
    else:
      test_set_x = test_set_x.reshape(test_set_x.shape[0],in_size)
    # quick fix to avoid more change of code, have to change it
    test_set_y = numpy.ones(test_set_x.shape[0])
    test_set_x, test_set_y = shared_dataset((test_set_x, test_set_y))


    ######################
    # BUILD ACTUAL MODEL #
    ######################

    #print '... building the model'

    # allocate symbolic variables for the data
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    classifier = MLP(rng=rng, input=x, n_in=in_size,
                     n_hidden=n_hidden, n_out=out_size)


    classifier.load_model(filename=model_file)
    test_model = theano.function([], [classifier.predictions,classifier.y_out],
                            givens={x: test_set_x})

    predictions, probs = test_model()

    return probs

def train_mlp(datasets,learning_rate=0.01, L1_reg=0.000, L2_reg=0.0001, n_epochs=100,
             batch_size=50, n_hidden=40,in_size=1,out_size=2,
              save_file='model/mlp/adaptive_0_1.pkl'):


    train_set_x, train_set_y = datasets
    in_size = train_set_x.shape[1] if len(train_set_x.shape) > 1 else 1
    rng = numpy.random.RandomState(1234)
    indices = rng.permutation(train_set_x.shape[0])
    train_set_x = train_set_x[indices]
    train_set_y = train_set_y[indices]
    if in_size == 1:
      train_set_x = train_set_x.reshape(train_set_x.shape[0],1)
    else:
      train_set_x = train_set_x.reshape(train_set_x.shape[0],in_size)
    train_set_x , train_set_y = shared_dataset((train_set_x, train_set_y))

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels


    # construct the MLP class
    classifier = MLP(rng=rng, input=x, n_in=in_size,
                     n_hidden=n_hidden, n_out=out_size)

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = classifier.negative_log_likelihood(y) \
         + L1_reg * classifier.L1 \
         + L2_reg * classifier.L2_sqr

    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = []
    for param in classifier.params:
        gparam = T.grad(cost, param)
        gparams.append(gparam)

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs
    updates = []
    # given two list the zip A = [a1, a2, a3, a4] and B = [b1, b2, b3, b4] of
    # same length, zip generates a list C of same size, where each element
    # is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    for param, gparam in zip(classifier.params, gparams):
        updates.append((param, param - learning_rate * gparam))

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(inputs=[index], outputs=cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size:(index + 1) * batch_size],
                y: train_set_y[index * batch_size:(index + 1) * batch_size]})

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    best_params = None
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False
  
    while (epoch < n_epochs) and (not done_looping):
        minibatch_avg_cost = 0.
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost += train_model(minibatch_index)/batch_size
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index
        print 'Epoch: {0}, cost: {1}'.format(
             epoch, minibatch_avg_cost)  

    best_params = copy.deepcopy(classifier.params)
    classifier.save_model(best_params, save_file)


# This is a wrapper in order to make use on decomposing test easier
class MLPTrainer():
    def __init__(self,n_hidden=40, L2_reg=0.001):
        self.n_hidden = n_hidden
        self.L2_reg = L2_reg

    def fit(self, X, y,save_file=''):
        train_mlp((X,y),
                  save_file=save_file,
                  n_hidden = self.n_hidden, L2_reg = self.L2_reg)
    def predict_proba(self, X, model_file='model'):
        return make_predictions(dataset=X, model_file=model_file, n_hidden = self.n_hidden)

 