import argparse
import sys
import os
import importlib
import mytools
importlib.reload(mytools)

if __name__ == "__main__" and "-f" not in sys.argv:
    params = mytools.get_CLI_parameters(sys.argv)
else:
    params = mytools.get_CLI_parameters("".split())

# the import statements aren't all at the beginning because some things are
# imported based on the command line inputs
import time
import numpy as np
import theano
import theano.tensor as T
theano.config.optimizer = "None"
import lasagne
import matplotlib
if params["remote"]:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.linear_model import LogisticRegression
import copy
import pickle
import pdb
na = np.newaxis

def one_hot_encoding(target, n_classes):
    """ IMPORTANT: if there are only two classes, this doesn't return a
        one hot encoding but instead just leaves the target vector as is
    """
    if n_classes == 1:
        return target
    target = target.flatten()
    target_mat = np.eye(n_classes)[target]
    return target_mat


def one_minus_one_encoding(target, n_classes):
    target_vec = np.ones((len(target), n_classes)) * (-1)
    target_vec[range(len(target)), target] = 1
    return target_vec


def transform_target(target, params):
    """ Transforms the target from an int value to other target choices
        like one-hot
    """
    if params["loss_choice"] == "categorical_crossentropy":
        return one_hot_encoding(target, params["n_classes"])
    elif params["loss_choice"] == "MSE":
        return one_minus_one_encoding(target, params["n_classes"])


def data_with_dims(X, input_shape):
    """ Restores the dimensions of the flattened data
        this assumes that the original data are 2-d."""
    X = X.reshape((X.shape[0],
                   1,
                   input_shape[0],
                   input_shape[1]))
    return X


def get_horseshoe_pattern(horseshoe_distractors):
    if horseshoe_distractors:
        A = np.empty((params["input_dim"], 8))
    else:
        A = np.empty((params["input_dim"], 4))

    # create the patterns for the actual classes
    pic = np.zeros((10, 10))
    pic[3:7, 2] = 1
    pic[3:7, 7] = 1
    pic[2, 3:7] = 1
    pic[7, 3:7] = 1
    for target in np.arange(4):
        current_pic = pic.copy()
        if target == 0:
            current_pic[2:8, 2] = 0
        elif target == 1:
            current_pic[2:8, 7] = 0
        elif target == 2:
            current_pic[2, 2:8] = 0
        elif target == 3:
            current_pic[7, 2:8] = 0
        A[:, target] = current_pic.flatten()

    if not horseshoe_distractors:
        return A

    # create the patterns for the distractors
    pic = np.zeros((10, 10))
    pic[4, 2] = 1
    pic[3, 3] = 1
    pic[2, 4] = 1
    pic[2, 5] = 1
    pic[3, 6] = 1
    pic[4, 7] = 1
    pic[5, 7] = 1
    pic[6, 6] = 1
    pic[7, 5] = 1
    pic[7, 4] = 1
    pic[6, 3] = 1
    pic[5, 2] = 1
    for distractor in np.arange(4, 8):
        current_pic = pic.copy()
        if distractor == 4:
            current_pic[4, 2] = 0
            current_pic[3, 3] = 0
            current_pic[2, 4] = 0
        elif distractor == 5:
            current_pic[2, 5] = 0
            current_pic[3, 6] = 0
            current_pic[4, 7] = 0
        elif distractor == 6:
            current_pic[5, 7] = 0
            current_pic[6, 6] = 0
            current_pic[7, 5] = 0
        elif distractor == 7:
            current_pic[7, 4] = 0
            current_pic[6, 3] = 0
            current_pic[5, 2] = 0
        A[:, distractor] = current_pic.flatten()

    return A


def create_data(params, N):
    if params["data"] == "horseshoe":
        return create_horseshoe_data(params, N)
    elif params["data"] == "ring":
        return create_ring_data(params, N)
    else:
        raise("Requested datatype unknown")


def create_horseshoe_data(params, N):
    A = get_horseshoe_pattern(params["horseshoe_distractors"])

    if params["specific_dataclass"] is not None:
        # this will/should only be triggered if N=1, in this case the user
        # requests a datapoint of a specific class
        y_true = np.array([params["specific_dataclass"]])[:, na]
    else:
        # if no specific class is requested, generate classes randomly
        y_true = np.random.randint(low=0, high=4, size=N)[:, na]

    y_onehot = one_hot_encoding(y_true, params["n_classes"]).T

    if params["horseshoe_distractors"]:
        y_dist = np.random.randint(low=0, high=4, size=N)[:, na]
        y_dist_onehot = one_hot_encoding(y_dist, params["n_classes"])
        y_onehot = np.concatenate((y_onehot, y_dist_onehot.T), axis=0)

    # create X by multiplying the target vector with the patterns,
    # and tranpose because we want the data to be in [samples, features] form
    X = np.dot(A, y_onehot).T

    for idx in range(X.shape[0]):
        X[idx, :] += (np.random.normal(size=(100))*params["noise_scale"])

    y = y_true.astype(np.int32)
    return X, y


def create_ring_data(params, N):
    """
    Creates 2d data aligned in clusters aligned on a ring
    """
    n_centers = 10
    n_per_center = int(np.ceil(N / n_centers))
    C = .01*np.eye(params["input_dim"])
    radius = 1
    class_means = radius*np.array([[np.cos(i*2.*np.pi/n_centers),np.sin(i*2.*np.pi/n_centers)] for i in range(n_centers)])

    X = np.empty((n_centers * n_per_center, params["input_dim"]))
    y = np.empty((n_centers * n_per_center, 1), dtype=np.int32)
    idx = 0
    while idx < n_centers:
        curr_data = np.random.multivariate_normal((0,0), C, size=n_per_center) + class_means[idx, :]
        X[idx*n_per_center:idx*n_per_center+n_per_center, :] = curr_data
        y[idx*n_per_center:idx*n_per_center+n_per_center] = int(idx%params["n_classes"])
        idx += 1

    if params["bias_in_data"]:
        onesvec = np.atleast_2d(np.ones((X.shape[0]))).T
        X = np.hstack((X, onesvec))
    return X[:N], y[:N]


def build_mlp(params, input_var=None):
    if params["bias_in_data"]:
        bias = None
    else:
        bias = lasagne.init.Constant(0.)

    # Input layer
    if params["bias_in_data"]:
        current_layer = lasagne.layers.InputLayer(shape=(None, params["input_dim"]+1),
                                        input_var=input_var, b=bias)
    else:
        current_layer = lasagne.layers.InputLayer(shape=(None, params["input_dim"]),
                                        input_var=input_var, b=bias)
    # Hidden layers
    for layer_size in params["layer_sizes"]:
        if layer_size == 0:
            print("Zero layer requested, ignoring...")
            continue
        current_layer = lasagne.layers.DenseLayer(
            current_layer, num_units=layer_size,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(),
            b=bias)


    # Output layer
    if params["loss_choice"] == "categorical_crossentropy":
        if params["n_classes"] == 2:
            print("Binary classification problem detected, using one output neuron")
            l_out = lasagne.layers.DenseLayer(
                    current_layer, num_units=1,
                    nonlinearity=lasagne.nonlinearities.sigmoid,
                    b=bias)
        else:
            l_out = lasagne.layers.DenseLayer(
                    current_layer, num_units=params["n_classes"],
                    nonlinearity=lasagne.nonlinearities.softmax,
                    b=bias)
    elif params["loss_choice"] == "MSE":
        l_out = lasagne.layers.DenseLayer(
                current_layer, num_units=params["n_classes"],
                nonlinearity=lasagne.nonlinearities.linear)
    return l_out


def build_cnn(params, input_var):
    # Input layer
    current_layer = lasagne.layers.InputLayer(shape=(None,
                                                    1,
                                                    params["input_shape"][0],
                                                    params["input_shape"][1]),
                                            input_var=input_var)
    # Hidden layers
    n_filters = 5
    current_layer = lasagne.layers.Conv2DLayer(current_layer,
                        num_filters=n_filters,
                        filter_size=(3, 3),
                        pad="same",
                        nonlinearity=lasagne.nonlinearities.rectify)
    n_filters = 4
    current_layer = lasagne.layers.Conv2DLayer(current_layer,
                        num_filters=n_filters,
                        filter_size=(3, 3),
                        pad="same",
                        nonlinearity=lasagne.nonlinearities.rectify)

    # Output layer
    l_out = lasagne.layers.DenseLayer(
            current_layer, num_units=params["n_classes"],
            nonlinearity=lasagne.nonlinearities.softmax)
    return l_out


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def train_network(params):
    params = copy.deepcopy(params)
    X_train, y_train_int = create_data(params, params["N_train"])
    X_val, y_val_int = create_data(params, params["N_val"])
    X_test, y_test_int = create_data(params, params["N_test"])
    y_train = transform_target(y_train_int, params)
    y_val = transform_target(y_val_int, params)
    y_test = transform_target(y_test_int, params)

    target_int_var = T.matrix('target_int')

    if params["loss_choice"] == "categorical_crossentropy":
        print("Building model and compiling functions...")
        if params["model"] == 'mlp':
            # Prepare Theano variables for inputs and targets
            input_var = T.matrix('mlp_inputs')
            target_var = T.matrix('targets')
            network = build_mlp(params, input_var)
        elif params["model"] == "cnn":
            # Prepare Theano variables for inputs and targets
            input_var = T.tensor4('cnn_inputs')
            target_var = T.matrix('targets')
            network = build_cnn(params, input_var)

        output = lasagne.layers.get_output(network)
        if params["n_classes"] == 2:
            loss = lasagne.objectives.binary_crossentropy(output, target_var)
        else:
            loss = lasagne.objectives.categorical_crossentropy(output, target_var)
        loss = loss.mean()

        # Create update expressions for training, i.e., how to modify the
        # parameters at each training step. Here, we'll use Stochastic Gradient
        # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
        network_params = lasagne.layers.get_all_params(network, trainable=True)
        updates = lasagne.updates.nesterov_momentum(
                loss, network_params, learning_rate=0.02, momentum=0.9)

        # Create a loss expression for validation/testing. The crucial difference
        # here is that we do a deterministic forward pass through the network,
        # disabling dropout layers.
        test_loss = lasagne.objectives.categorical_crossentropy(output,
                                                                target_var)
        test_loss = test_loss.mean()

    if params["loss_choice"] == "MSE":
        # Prepare Theano variables for inputs and targets
        input_var = T.matrix('inputs')
        target_var = T.dmatrix('targets')

        print("Building model and compiling functions...")
        if params["model"] == 'mlp':
            network = build_mlp(params, input_var)

        output = lasagne.layers.get_output(network)

        loss = lasagne.objectives.squared_error(output, target_var)
        loss = loss.mean()

        # Create update expressions for training, i.e., how to modify the
        # parameters at each training step. Here, we'll use Stochastic Gradient
        # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
        network_params = lasagne.layers.get_all_params(network, trainable=True)
        updates = lasagne.updates.nesterov_momentum(
                loss, network_params, learning_rate=0.01, momentum=0.9)

        # Create a loss expression for validation/testing. The crucial difference
        # here is that we do a deterministic forward pass through the network,
        # disabling dropout layers.
        test_loss = lasagne.objectives.squared_error(output, target_var)
        test_loss = test_loss.mean()

    # save some useful variables and functions into the params dict
    params["input_var"] = input_var
    params["target_var"] = target_var
    params["output_var"] = output
    params["get_output"] = theano.function([input_var], output)


    # As a bonus, also create an expression for the classification accuracy:
    if params["n_classes"] == 2:
        prediction = T.round(output)
    else:
        prediction = T.shape_padaxis(T.argmax(output, axis=1), 1)

    params["prediction_func"] = theano.function([input_var], prediction)
    test_acc = T.mean(T.eq(prediction, target_int_var),
                      dtype=theano.config.floatX)


    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_int_var, target_var], [test_loss, test_acc])


    ############################################################################
    # TRAINING LOOP
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(params["epochs"]):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, params["minibatch_size"], shuffle=True):
            inputs, targets = batch
            if params["model"] == "cnn":
                inputs = data_with_dims(inputs, params["input_shape"])
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        if epoch%10 == 0:
            val_err = 0
            val_acc = 0
            val_batches = 0
            for batch in iterate_minibatches(X_val, y_val_int, params["minibatch_size"], shuffle=False):
                inputs, targets_int = batch
                targets = one_hot_encoding(targets_int, params["n_classes"])
                if params["model"] == "cnn":
                    inputs = data_with_dims(inputs, params["input_shape"])
                err, acc = val_fn(inputs, targets_int, targets)
                val_err += err
                val_acc += acc
                val_batches += 1

            if params["verbose"]:
                # Then we print the results for this epoch:
                print("Epoch {} of {} took {:.3f}s".format(
                    epoch + 1, params["epochs"], time.time() - start_time))
                print("  training loss:\t\t{:.6f}".format(train_err / train_batches))


                print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
                print("  validation accuracy:\t\t{:.2f} %".format(
                    val_acc / val_batches * 100))

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test_int, params["minibatch_size"], shuffle=False):
        inputs, targets_int = batch
        targets = one_hot_encoding(targets_int, params["n_classes"])
        if params["model"] == "cnn":
            inputs = data_with_dims(inputs, params["input_shape"])
        err, acc = val_fn(inputs, targets_int, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))
    return network, params


def get_target_title(target):
    if target == 0:
        target_title = "left open"
    elif target == 1:
        target_title = "right open"
    elif target == 2:
        target_title = "top open"
    elif target == 3:
        target_title = "bottom open"
    elif target == "ones":
        target_title = "just ones"
    return target_title


def plot_heatmap(R_i, axis=None, title=""):
    if axis == None:
        fig, axis = plt.subplots(1, 1, figsize=(15, 10))

    plot = axis.pcolor(R_i, cmap="viridis", vmin=-np.max(abs(R_i)), vmax=np.max(abs(R_i)))
    axis.set_title(title)
    axis.invert_yaxis()
    plt.colorbar(plot, ax=axis)


def forward_pass(func_input, network, input_var):
    """
    IMPORTANT: THIS FUNCTION CAN ONLY BE CALLED WITH A SINGLE INPUT SAMPLE
    the function expects a row vector
    """
    get_activations = theano.function([input_var],
                lasagne.layers.get_output(lasagne.layers.get_all_layers(network)))
    # in the following, the dimension has to be expanded because we're
    # performing a forward pass of just one input sample here but the network
    # expects several
    if func_input.ndim == 1:
        activations = get_activations(np.expand_dims(func_input, axis=0))
    elif func_input.ndim == 2:
        if func_input.shape[0] > 1:
            raise("ERROR: only pass a single datapoint to forward_pass()!")
        activations = get_activations(func_input)
    elif activations.ndim > 2:
        raise("Input data had too many dimensions")

    # Transpose all activation vectors so that they are column vectors
    for idx in range(len(activations)):
        activations[idx] = activations[idx].T
    return activations


def get_network_parameters(network, bias_in_data):
    # --- get paramters and activations for the input ---
    all_params = lasagne.layers.get_all_param_values(network)
    if bias_in_data:
        W_mats = all_params
    else:
        W_mats = all_params[0::2]
        biases = all_params[1::2]

    for idx in range(len(W_mats)):
        W_mats[idx] = W_mats[idx].T
        if not params["bias_in_data"]:
            biases[idx] = np.atleast_2d(biases[idx]).T

    if bias_in_data:
        return W_mats
    else:
        return W_mats, biases


def compute_relevance(func_input, network, output_neuron, params, epsilon = .01):
    W_mats, biases = get_network_parameters(network, params["bias_in_data"])
    activations = forward_pass(func_input, network, params["input_var"])

    # --- forward propagation to compute preactivations ---
    preactivations = []
    for W, b, X in zip(W_mats, biases, activations):
        preactivation = np.dot(W, X) + np.expand_dims(b, 1)
        preactivations.append(preactivation)

    # --- relevance backpropagation ---
    # the first backpass is special so it can't be in the loop
    R_over_z = activations[-1][output_neuron] / (preactivations[-1][output_neuron] + epsilon)
    R = np.multiply(W_mats[-1][output_neuron,:].T, activations[-2].T) * R_over_z
    for idx in np.arange(2, len(activations)):
        R_over_z = np.divide(R, preactivations[-idx].squeeze() + epsilon)
        Z_ij = np.multiply(W_mats[-idx], activations[-idx-1].T + epsilon)
        R = np.sum(np.multiply(Z_ij.T, R_over_z), axis=1)
    R = R.reshape((10, 10))

    return R


def manual_classification(X):
    """ Performs a manual classification of the horseshoe data
        using knowledge of the data creation process
        Input: X of shape (samples, width, height)
    """
    def single_classification(func_input):
        left_bar = np.sum(func_input[3:7, 2])
        right_bar = np.sum(func_input[3:7, 7])
        upper_bar = np.sum(func_input[2, 3:7])
        lower_bar = np.sum(func_input[7, 3:7])

        left_open = right_bar + upper_bar + lower_bar
        right_open = left_bar + upper_bar + lower_bar
        up_open = left_bar + right_bar + lower_bar
        low_open = left_bar + right_bar + upper_bar
        return np.argmax([left_open, right_open, up_open, low_open])

    manual_prediction = np.empty((len(X)))
    for idx in range(len(X)):
        # do manual classification by summing over bars
        manual_prediction[idx] = single_classification(X[idx])

    # bring into column vector shape and cast to int
    manual_prediction = manual_prediction[:, na].astype(int)
    return manual_prediction


def compute_w(X, network, params):
    activations = forward_pass(X, network, params["input_var"])
    W_mats, biases = get_network_parameters(network, params["bias_in_data"])
    S_mats = copy.deepcopy(W_mats) # this makes an actual copy of W_mats

    # --- forward propagation to compute preactivations ---
    # this is not strictly necessary to compute, I only use it to verify that
    # what the output of the computed weight vector is correct
    preactivations = []
    for W, b, a in zip(W_mats, biases, activations):
        preactivation = np.dot(W, a) + b
        preactivations.append(preactivation)
    # -----------------------------------------------------

    for idx in range(len(S_mats)):
        # extend the weight matrices with a row of zeroes below (last element a 1),
        # and a column to the right in which there are the biases of the next layer.
        S_mats[idx] = np.vstack((S_mats[idx], np.zeros(S_mats[idx].shape[1])[na, :]))
        bias_and_one = np.vstack((biases[idx], 1))
        S_mats[idx] = np.hstack((S_mats[idx], bias_and_one))

        # extend all activations by a 1
        activations[idx+1] = np.vstack((activations[idx+1], 1))

        # set the rows in the weight matrix to zero where the activation of the
        # neuron in the layer that this matrix produced was zero
        if idx < len(S_mats)-1:
            S_mats[idx][activations[idx+1].squeeze() < 0.000001, :] = 0

    # for the weight matrix to the last layer we don't need to incorporate the bias
    S_mats[-1] = S_mats[-1][:-1, :]

    s = S_mats[0]
    for idx in range(1, len(S_mats)):
        s = np.dot(S_mats[idx], s)

    X_ext = np.vstack((X.T, 1)).T # the double transpose is due to weird behavior of vstack
    if not np.allclose(np.dot(s, X_ext.T), preactivations[-1]):
        # checking whether the computed w gives the same result as the network
        # gave
        raise("There was an error computing the weight vector")

    w = s[:, :-1]
    print(w.shape)
    return w


def compute_accuracy(y, y_hat):
    """ Compute the percentage of correct classifications """
    if y.shape == y_hat.shape:
        n_correct = np.sum(y == y_hat)
        p_correct = n_correct / y.size
        return p_correct
    else:
        raise("The two inputs didn't have the same shape")


# regular mlp
params["model"] = "mlp"
params["layer_sizes"] = [100, 100, 10] # as requested by pieter-jan
mlp, mlp_params = train_network(params)
mlp_prediction_func = mlp_params["prediction_func"]

#params["model"] = "cnn"
#cnn, cnn_params = train_network(params)
#cnn_prediction_func = cnn_params["prediction_func"]


## compare prediction scores
## some more data
#X, y = create_data(params, 500)
#
## predict with mlp
#mlp_prediction = mlp_prediction_func(X)
#mlp_score = compute_accuracy(y, mlp_prediction)
#
#
## predict with cnn
#X = data_with_dims(X, cnn_params["input_shape"])
#cnn_prediction = cnn_prediction_func(X)
#cnn_score = compute_accuracy(y, cnn_prediction)
#
## manually predict
#man_prediction = manual_classification(X[:, 0, :, :])
#man_score = compute_accuracy(y, man_prediction)
#
#
#print("MLP score: " + str(mlp_score))
#print("CNN score: " + str(cnn_score))
#print("manual score: " + str(man_score))

params["specific_dataclass"] = 0
X, y = create_data(params, 1)
w_mlp = compute_w(X, mlp, mlp_params)
plot_heatmap(w_mlp[0].reshape((10, 10)))
plt.show()
mlp_params["input_var"]
X.shape

# get A via Haufe method
params["specific_dataclass"] = None
X_train, y_train = create_data(params, 500)
y_train
A = get_horseshoe_pattern(params["horseshoe_distractors"])
y = one_hot_encoding(y_train, params["n_classes"])
params["n_classes"]
np.eye(params["n_classes"])[y_train.flatten()]
Sigma_s = np.cov(y.T)
Sigma_X = np.cov(X_train.T)
A_haufe = np.dot(np.dot(Sigma_X, w_mlp.T), Sigma_s)

fig, axes = plt.subplots(1, 4)
plot_heatmap(A[:, 0].reshape((10, 10)), axis=axes[0], title="True A")
plot_heatmap(X.reshape((10, 10)), axis=axes[1], title="input point")
plot_heatmap(w_mlp.T[:, 0].reshape((10, 10)), axis=axes[2], title="W")
plot_heatmap(A_haufe[:, 0].reshape((10, 10)), axis=axes[3], title="A Haufe 2013")
plt.show()
#cnn_wmats, cnn_biases = get_network_parameters(cnn, params["bias_in_data"])
#cnn_wmats[1].shape
#all_layers = lasagne.layers.get_all_layers(cnn)
#lasagne.layers.get_output_shape(all_layers[-2])
#fig, axes = plt.subplots(1, 5, figsize=(15, 10))
#for filter_idx in np.arange(5):
#    plot_heatmap(cnn_wmats[0][:,:,0,filter_idx], axes[filter_idx])
#    plt.subplots_adjust(wspace=.5)
#    plt.savefig(open("relevance.png", "w"))
#plt.show()


mlp_params["output_var"]
gradient = T.grad(mlp_params["output_var"][0, 0], mlp_params["input_var"])
compute_grad = theano.function([mlp_params["input_var"]], gradient)
mlp_gradient = compute_grad(X)

# normalize the gradient and w_flat
w_mlp[0, :] /= np.linalg.norm(w_mlp[0, :])
mlp_gradient /= np.linalg.norm(mlp_gradient)

# plot both the gradient and w_flat
plot_heatmap(mlp_gradient.reshape((10, 10)))
plot_heatmap(w_mlp[0, :].reshape((10, 10)))
plt.show()

# verify that the two are equal numerically
if np.allclose(mlp_gradient, w_mlp[0, :]):
    print("The gradient and w_flat are equal!")

w_mlp_z[0, :10]
w_mlp[0, :10]

a = mlp_params["output_var"][0]

#get_output = theano.function([params["input_var"]], lasagne.layers.get_output(network))
#theano.printing.debugprint(mlp_params["output_var"])

# create another example
#x, y = create_data(params, 4)
#
#fig, axes = plt.subplots(1, 5, figsize=(15, 10))
## first plotting the input image
#title = "input image"
#x_withdims = np.reshape(x[params["dataset"]], (10, 10))
#plot_heatmap(x_withdims, y[params["dataset"]], axis=axes[0], title=title)
#for output_neuron in np.arange(4):
#    title = get_target_title(output_neuron)
#    x[params["dataset"]].shape
#    r = compute_relevance(x[params["dataset"]], network, output_neuron, params)
#    plot_heatmap(r, output_neuron, axes[output_neuron+1], title)
#    plt.subplots_adjust(wspace=.5)
#    plt.savefig(open("relevance.png", "w"))


######## logistic regression
#print("performing logistic regression")
#x_train, y_train = create_data(params, params["n_train"])
#logreg = logisticregression()
#logreg.fit(x_train, np.ravel(y_train))
#coefs = logreg.coef_
#coefs = np.reshape(coefs, (coefs.shape[0], 10,-1))
#print("finished logistic regression")
#
## plot a random input sample
#plot_heatmap(X_train[0].reshape((10, 10)), title="training image")
#
#fig, axes = plt.subplots(1, 4, figsize=(15, 10))
#for output_neuron in np.arange(4):
#    title = get_target_title(output_neuron)
#    plot_heatmap(coefs[output_neuron], axes[output_neuron], title=title)
##    plt.savefig(open(params["plots_dir"] + "/coefs.png", "w"), dpi=400)
#
## plot the result of W.T @ A (the patterns)
#W = LogReg.coef_.T
#plot_heatmap(np.dot(W.T, A[:, :4]))
#
#
# plot one of the distractor patterns
#y_distractor = np.array([[0,0,0,0,0,0,0,1]]).T
#X_distractor = np.dot(A, y_distractor)
#plot_heatmap(X_distractor.reshape((10,10)))
#
# get A via Haufe method
#y = one_hot_encoding(y_train.squeeze())
#Sigma_s = np.cov(y.T)
#Sigma_X = np.cov(X_train.T)
#A_haufe = np.dot(np.dot(Sigma_X, W), Sigma_s)

# check whether the input data looks correct
#y_class = np.array([[1,0,0,0,0,0,0,0]]).T
#X_class = np.dot(A, y_class)
#plot_heatmap(X_class.reshape((10,10)))

#fig, axes = plt.subplots(1, 3, figsize=(15, 10))
#plot_heatmap(A[:, 0].reshape((10, 10)), axis=axes[0], title="true A")
#plot_heatmap(W[:, 0].reshape((10, 10)), axis=axes[1], title="LogReg weights")
#plot_heatmap(A_haufe[:, 0].reshape((10, 10)), axis=axes[2], title="A haufe")
#plt.show()

# comparing manual classification with network output
#X, y = create_data(params, 200)

#manual_output = manual_classification(X)
#manual_score = np.sum(manual_output == y)
#network_output = lasagne.layers.get_output(network)
#get_network_output = theano.function([params["input_var"]], network_output)
#network_prediction = np.argmax(get_network_output(X), axis=1)
#network_score = np.sum(network_prediction == y)
#logreg_prediction = LogReg.predict(np.reshape(X, (len(X),100)))
#logreg_score = np.sum(logreg_prediction ==y)
#print("Manual classification score: " + str(manual_score))
#print("Network classification score: " + str(network_score))
#print("LogReg classification score: " + str(logreg_score))


if params["do_plotting"]:
    plt.show()
