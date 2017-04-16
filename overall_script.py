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
    if target.shape[1] > 1:
        return target
    target = target.flatten()
    target_mat = np.eye(n_classes)[target]
    return target_mat


def one_minus_one_encoding(target, n_classes):
    target = target.squeeze()
    target_vec = np.ones((len(target), n_classes)) * (-1)
    target_vec[range(len(target)), target] = 1
    return target_vec


def transform_target(target, params):
    """ Transforms the target from an int value to other target choices
        like one-hot
    """
    if "n_output_units" in params:
        n_output_units = params["n_output_units"]
    else:
        n_output_units = params["n_classes"]

    if params["loss_choice"] == "categorical_crossentropy":
        return one_hot_encoding(target, n_output_units)
    elif params["loss_choice"] == "MSE":
        return one_minus_one_encoding(target, n_output_units)


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
        if N != 1:
            raise("specific_dataclass is set so N should be 1")
        y_true = np.array([params["specific_dataclass"]])[:, na]
    else:
        # if no specific class is requested, generate classes randomly
        y_true = np.random.randint(low=0, high=4, size=N)[:, na]

    y = one_hot_encoding(y_true, params["n_output_units"]).T

    if params["horseshoe_distractors"]:
#        y_dist = np.random.randint(low=0, high=4, size=N)[:, na]
#        y_dist_onehot = one_hot_encoding(y_dist, params["n_output_units"])
        y_dist = np.random.normal(size=(4, N))
        y = np.concatenate((y, y_dist), axis=0)

    # create X by multiplying the target vector with the patterns,
    # and tranpose because we want the data to be in [samples, features] form
    X = np.dot(A, y).T

    for idx in range(X.shape[0]):
        X[idx, :] += (np.random.normal(size=(100))*params["noise_scale"])

    y = y_true.astype(np.int32)
    return X, y


def create_ring_data(params, N):
    """
    Creates 2d data aligned in clusters aligned on a ring
    """
    n_centers = 8
    n_per_center = int(np.ceil(N / n_centers))
    C = .02*np.eye(params["input_dim"])
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

        l_out = lasagne.layers.DenseLayer(
                current_layer, num_units=params["n_output_units"],
                nonlinearity=lasagne.nonlinearities.softmax)
    return l_out


def build_cnn(params, input_var):
    # Input layer
    current_layer = lasagne.layers.InputLayer(shape=(None,
                                                    1,
                                                    params["input_shape"][0],
                                                    params["input_shape"][1]),
                                            input_var=input_var)
    # Hidden layers
    n_filters = 15
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
    X_train, y_train= create_data(params, params["N_train"])
    X_val, y_val= create_data(params, params["N_val"])
    X_test, y_test= create_data(params, params["N_test"])
    if params["n_output_units"] > 1:
        y_train = transform_target(y_train, params)
        y_val = transform_target(y_val, params)
        y_test = transform_target(y_test, params)

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

        output_var = lasagne.layers.get_output(network)
        if params["n_classes"] == 2:
            loss = lasagne.objectives.binary_crossentropy(output_var, target_var)
        else:
            loss = lasagne.objectives.categorical_crossentropy(output_var, target_var)
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
        test_loss = lasagne.objectives.categorical_crossentropy(output_var,
                                                                target_var)
        test_loss = test_loss.mean()

    if params["loss_choice"] == "MSE":
        # Prepare Theano variables for inputs and targets
        input_var = T.matrix('inputs')
        target_var = T.dmatrix('targets')

        print("Building model and compiling functions...")
        if params["model"] == 'mlp':
            network = build_mlp(params, input_var)

        output_var = lasagne.layers.get_output(network)

        loss = lasagne.objectives.squared_error(output_var, target_var)
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
        test_loss = lasagne.objectives.squared_error(output_var, target_var)
        test_loss = test_loss.mean()

    # save some useful variables and functions into the params dict
    params["input_var"] = input_var
    params["target_var"] = target_var
    params["output_var"] = output_var
    params["get_output"] = theano.function([input_var], output_var, allow_input_downcast=True)


    # Create an expression for the classification accuracy:
    if params["n_output_units"] == 2:
        prediction_var = T.round(output_var)
    else:
        prediction_var = T.shape_padaxis(T.argmax(output_var, axis=1), 1)

    params["prediction_func"] = theano.function([input_var], prediction_var, allow_input_downcast=True)


    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates, allow_input_downcast=True)

    # Compile a second function computing the validation loss and accuracy:
    def val_fn(X, y):
        y_hat = params["prediction_func"](X)

        # transform prediction from one-hot into scalar int if necessary
        # remember that both the prediction y_hat and the targets y 
        # are in one hot encoding
        if y_hat.shape[1] > 1:
            y_hat = np.argmax(y_hat, axis=1)
            y = np.argmax(y, axis=1)

        return np.sum(y_hat == y) / y.shape[0]


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
            val_acc = 0
            val_batches = 0
            for batch in iterate_minibatches(X_val, y_val, params["minibatch_size"], shuffle=False):
                inputs, targets = batch
                if params["model"] == "cnn":
                    inputs = data_with_dims(inputs, params["input_shape"])
                acc = val_fn(inputs, targets)
                val_acc += acc
                val_batches += 1

            if params["verbose"]:
                # Then we print the results for this epoch:
                print("Epoch {} of {} took {:.3f}s".format(
                    epoch + 1, params["epochs"], time.time() - start_time))
                print("  training loss:\t\t{:.6f}".format(train_err / train_batches))


                print("  validation accuracy:\t\t{:.2f} %".format(
                    val_acc / val_batches * 100))

    # After training, we compute and print the test error:
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, params["minibatch_size"], shuffle=False):
        inputs, targets = batch
        targets = one_hot_encoding(targets, params["n_classes"])
        if params["model"] == "cnn":
            inputs = data_with_dims(inputs, params["input_shape"])
        acc = val_fn(inputs, targets)
        test_acc += acc
        test_batches += 1
    print("Final results:")
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


def forward_pass(func_input, network, input_var, params):
    """
    IMPORTANT: THIS FUNCTION CAN ONLY BE CALLED WITH A SINGLE INPUT SAMPLE
    the function expects a row vector
    """
    get_activations = theano.function([input_var],
                lasagne.layers.get_output(lasagne.layers.get_all_layers(network)), 
                allow_input_downcast=True)
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
    activations = forward_pass(func_input, network, params["input_var"], params)

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



def compute_accuracy(y, y_hat):
    """ Compute the percentage of correct classifications """
    if y.shape == y_hat.shape:
        n_correct = np.sum(y == y_hat)
        p_correct = n_correct / y.size
        return p_correct
    else:
        raise("The two inputs didn't have the same shape")



def get_W_from_gradients(X, params):
    """
    Returns W of shape [n_features, n_output_units]
    In other words, W contains the gradients in the columns
    (and there are as many gradients as there are output units)
    """

    # input shape handling
    input_shape = params["input_shape"]
    # the shape of output_var is [1, n_output_units]
    if len(params["input_shape"]) == 1:
        n_features = input_shape[0]
    elif len(input_shape) == 2:
        n_features = input_shape[0] * input_shape[1]
        X = data_with_dims(X, params["input_shape"])
    else:
        raise("Unexpected input shape")

    W = np.empty((n_features, params["n_output_units"]))
    for output_idx in range(params["n_output_units"]):
        gradient_var = T.grad(params["output_var"][0, output_idx], params["input_var"])
        compute_grad = theano.function([params["input_var"]], gradient_var, allow_input_downcast=True)
        gradient = compute_grad(X)
        W[:, output_idx] = gradient.flatten()
    return W




def plot_background():
    # create some data for scatterplot
    X, y = create_ring_data(params, params["N_train"])
    # create a mesh to plot in
    h = .01 # step size in the mesh
    x_min, x_max = -2, 2
    y_min, y_max = -2, 2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    mesh = np.c_[xx.ravel(), yy.ravel()]

    output = lasagne.layers.get_output(network)
    get_output = theano.function([params["input_var"]], output, allow_input_downcast=True)
    Z = get_output(mesh)
    Z = Z[:, OUTPUT_NEURON_SELECTED]

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)

    #plt.scatter(X[:,0], X[:,1], c=y, cmap="gray", s=40)
    plt.imshow(Z, interpolation="nearest", cmap=cm.gray, alpha=0.4,
               extent=[x_min, x_max, y_min, y_max])
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


def plot_w_or_patterns(what_to_plot):
    # create a mesh to plot in
    h = .5 # step size in the mesh
    x_min, x_max = -2, 2 + h
    y_min, y_max = -2, 2 + h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    mesh = np.c_[xx.ravel(), yy.ravel()]

    my_linewidth = 1.5
    all_vecs = np.empty((len(mesh), 4))
    # get A via Haufe method
    X_train, y_train = create_data(params, 5000)
    y = one_hot_encoding(y_train, params["n_classes"])
    Sigma_s = np.cov(y, rowvar=False)

    #Sigma_s_inv = np.linalg.inv(np.cov(y.T))
    Sigma_X = np.cov(X_train.T)
    for idx in range(len(mesh)):
        if idx%100 == 0:
            print("Computing weight vector nr " + str(idx) + " out of " + str(len(mesh)))
        X_pos = mesh[idx][na, :]
        W = get_W_from_gradients(X_pos, params)

        if what_to_plot == "gradients":
            plot_vector = W[:, OUTPUT_NEURON_SELECTED]
            plot_vector /= np.linalg.norm(plot_vector) * VECTOR_ADJUST_CONSTANT
        elif what_to_plot == "patterns":
            A_haufe = np.dot(np.dot(Sigma_X, W), np.linalg.pinv(Sigma_s))
#            A_haufe = np.dot(Sigma_X, W)
            plot_vector = A_haufe[:, OUTPUT_NEURON_SELECTED]
            plot_vector /= np.linalg.norm(plot_vector) * VECTOR_ADJUST_CONSTANT
        else:
            raise Exception("Choose gradients or patterns for what_to_plot")
        plt.quiver(X_pos[0, 0], X_pos[0, 1], plot_vector[0], plot_vector[1], scale=None)



# RING DATA
# train MLP on ring data
params["layer_sizes"] = [20, 20]
params["data"] = "ring"
params["model"] = "mlp"
params["input_dim"] = 2
params["input_shape"] = (2,)
params["n_classes"] = 2
params["n_output_units"] = 2
network, params = train_network(params)
OUTPUT_NEURON_SELECTED = 1
VECTOR_ADJUST_CONSTANT = 3




# GRADIENTS
plt.figure()
plot_w_or_patterns(what_to_plot="gradients")
# plot background
plot_background()
plt.title("gradients")

# PATTERNS
plt.figure()
plot_w_or_patterns(what_to_plot="patterns")
# plot background
plot_background()
plt.title("patterns")






## regular mlp
#params["model"] = "mlp"
#params["layer_sizes"] = [100, 100, 10] # as requested by pieter-jan
#mlp, mlp_params = train_network(params)
#mlp_params["input_shape"] = [100]
#mlp_prediction_func = mlp_params["prediction_func"]
#
#params["model"] = "cnn"
#cnn, cnn_params = train_network(params)
#cnn_prediction_func = cnn_params["prediction_func"]


## ### compare prediction scores ###
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




## computing the gradient of the inputs of the MLP
#mlp_gradient = T.grad(mlp_params["output_var"][0, 0], mlp_params["input_var"])
#compute_grad_mlp = theano.function([mlp_params["input_var"]], mlp_gradient)
#mlp_gradient = compute_grad_mlp(X)
## normalize the gradient
#mlp_gradient /= np.linalg.norm(mlp_gradient)
#
## computing the gradient of the inputs of the CNN
#cnn_gradient = T.grad(cnn_params["output_var"][0, 0], cnn_params["input_var"])
#compute_grad_cnn = theano.function([cnn_params["input_var"]], cnn_gradient)
#cnn_gradient = compute_grad_cnn(data_with_dims(X, cnn_params["input_shape"]))
#cnn_gradient = cnn_gradient.reshape((1, 100))
## normalize the gradient
#cnn_gradient /= np.linalg.norm(cnn_gradient)




## get an input point for which we want the weights / patterns
#params["specific_dataclass"] = 0
#params["input_shape"] = [100]
#X, y = create_data(params, 1)
#X.shape
#
#len(mlp_params["input_shape"])
#
## get A via Haufe method
#params["specific_dataclass"] = None
#X_train, y_train = create_data(params, 500)
#A = get_horseshoe_pattern(params["horseshoe_distractors"])
#y = one_hot_encoding(y_train, params["n_classes"])
#Sigma_s = np.cov(y.T)
#Sigma_X = np.cov(X_train.T)
#W_mlp = get_W_from_gradients(X, mlp_params)
#A_haufe_mlp = np.dot(np.dot(Sigma_X, W_mlp), Sigma_s)
#W_cnn = get_W_from_gradients(data_with_dims(X, cnn_params["input_shape"]), cnn_params)
#A_haufe_cnn = np.dot(np.dot(Sigma_X, W_cnn), Sigma_s)
#
## plot real pattern, input point, weights and haufe pattern for MLP
#grad_mlp = W_mlp[:, 0]
#fig, axes = plt.subplots(1, 4)
#plot_heatmap(A[:, 0].reshape((10, 10)), axis=axes[0], title="True A")
#plot_heatmap(X.reshape((10, 10)), axis=axes[1], title="input point")
#plot_heatmap(grad_mlp.reshape((10, 10)), axis=axes[2], title="W")
#plot_heatmap(A_haufe_mlp[:, 0].reshape((10, 10)), axis=axes[3], title="A Haufe 2013")
#plt.suptitle("MLP", size=16)
#plt.show()
#
## plot real pattern, input point, weights and haufe pattern for CNN
#grad_cnn = W_cnn[:, 0]
#fig, axes = plt.subplots(1, 4)
#plot_heatmap(A[:, 0].reshape((10, 10)), axis=axes[0], title="True A")
#plot_heatmap(X.reshape((10, 10)), axis=axes[1], title="input point")
#plot_heatmap(grad_cnn.reshape((10, 10)), axis=axes[2], title="W")
#plot_heatmap(A_haufe_cnn[:, 0].reshape((10, 10)), axis=axes[3], title="A Haufe 2013")
#plt.suptitle("CNN", size=16)
#plt.show()





if params["do_plotting"]:
    plt.show()
