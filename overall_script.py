import argparse
import sys
import os
import importlib
import mytools
from mytools import *
importlib.reload(mytools)

if __name__ == "__main__" and "-f" not in sys.argv:
    params = get_CLI_parameters(sys.argv)
else:
    params = get_CLI_parameters("".split())

# the import statements aren't all at the beginning because some things are
# imported based on the command line inputs
import time
import numpy as np
import theano
import theano.tensor as T
import lasagne
import matplotlib
if params["remote"]:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import copy
import pickle
na = np.newaxis

def transform_target(target, loss_choice):
    """ Transforms the target from an int value to other target choices
        like one-hot
    """
    if loss_choice == "categorical_crossentropy":
        return target
    elif loss_choice == "MSE":
        target_vec = np.ones((len(target), 4)) * (-1)
        target_vec[range(len(target)), target] = 1
        return target_vec


def create_data(params, N):
    if params["data"] == "horseshoe":
        return create_horseshoe_data(params, N)
    elif params["data"] == "ring":
        return create_ring_data(params, N)
    else:
        raise("Requested datatype unknown")


def create_horseshoe_data(params, N):
    pic = np.zeros((10, 10))
    pic[3:7, 2] = 1
    pic[3:7, 7] = 1
    pic[2, 3:7] = 1
    pic[7, 3:7] = 1
    overall_idx = 0
    X = np.empty((N, params["input_dim"]))
    y = np.empty(N, dtype=np.int32)
    while overall_idx < N:
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
            current_pic += (np.random.normal(size=(10, 10))*params["noise_scale"])
            X[overall_idx, :] = np.reshape(current_pic, (-1))
            y[overall_idx] = target
            overall_idx += 1
            if overall_idx >= N:
                break
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
    y = np.empty(n_centers * n_per_center, dtype=np.int32)
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

    X_train, y_train = create_data(params, params["N_train"])
    X_val, y_val = create_data(params, params["N_val"])
    X_test, y_test = create_data(params, params["N_test"])
    y_train = transform_target(y_train, params["loss_choice"])
    y_val = transform_target(y_val, params["loss_choice"])
    y_test = transform_target(y_test, params["loss_choice"])

    if params["loss_choice"] == "categorical_crossentropy":
        # Prepare Theano variables for inputs and targets
        input_var = T.matrix('inputs')
        target_var = T.ivector('targets')

        print("Building model and compiling functions...")
        if params["model"] == 'mlp':
            network = build_mlp(params, input_var)

        prediction = lasagne.layers.get_output(network)
        if params["n_classes"] == 2:
            loss = lasagne.objectives.binary_crossentropy(prediction, target_var)
        else:
            loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
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
        test_prediction = lasagne.layers.get_output(network, deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                                target_var)
        test_loss = test_loss.mean()

        # As a bonus, also create an expression for the classification accuracy:
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                        dtype=theano.config.floatX)

    if params["loss_choice"] == "MSE":
        # Prepare Theano variables for inputs and targets
        input_var = T.matrix('inputs')
        target_var = T.dmatrix('targets')

        print("Building model and compiling functions...")
        if params["model"] == 'mlp':
            network = build_mlp(params, input_var)

        prediction = lasagne.layers.get_output(network)

        loss = lasagne.objectives.squared_error(prediction, target_var)
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
        test_prediction = lasagne.layers.get_output(network, deterministic=True)
        test_loss = lasagne.objectives.squared_error(test_prediction,
                                                        target_var)
        test_loss = test_loss.mean()

        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), T.argmax(target_var, axis=1)),
                        dtype=theano.config.floatX)

    params["input_var"] = input_var
    params["target_var"] = target_var

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

#    import pdb; pdb.set_trace()

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(params["epochs"]):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, params["minibatch_size"], shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, params["minibatch_size"], shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
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
    for batch in iterate_minibatches(X_test, y_test, params["minibatch_size"], shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
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

    return manual_prediction


def predict(Xl, network):
    """ Predicts the class of the input

    Input
    X: [samples, features]
    """
    get_predictions = theano.function([params["input_var"]], lasagne.layers.get_output(network))
    output = get_predictions(X)
    return np.argmax(output, axis=1)


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
    w[0] /= np.linalg.norm(w[0])
    w[1] /= np.linalg.norm(w[1])

    return w




network, params = train_network(params)
get_output = theano.function([params["input_var"]], lasagne.layers.get_output(network))

X_pos = np.array([1, 0])[na, :]
w = compute_w(X_pos, network, params)



length = 2
my_linewidth = 3
plt.figure()
plt.plot([0, w[0, 0]], [0, w[0, 1]], linewidth=my_linewidth)
plt.plot([0, w[1, 0]], [0, w[1, 1]], linewidth=my_linewidth)

# create some data for scatterplot
X, y = create_ring_data(params, params["N_train"])
# create a mesh to plot in
h = .01 # step size in the mesh
x_min, x_max = -2, 2
y_min, y_max = -2, 2
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
mesh = np.c_[xx.ravel(), yy.ravel()]

Z = predict(mesh, network)

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.scatter(X[:,0], X[:,1], c=y, cmap="gray", s=40)
plt.scatter(X_pos[0, 0], X_pos[0, 1], s = 200)
plt.contour(xx, yy, Z, cmap="gray", alpha=0.8)
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())


# draw the analytical decision boundary
#n_centers = 10
#radius = 10
#additive_term = 2*np.pi / (2*n_centers)
#end_points = radius*np.array([[np.cos(i*2.*np.pi/n_centers + additive_term), np.sin(i*2.*np.pi/n_centers + additive_term)] for i in range(n_centers)])
#for idx in np.arange(n_centers):
#    plt.plot([0, end_points[idx, 0]], [0, end_points[idx, 1]], color="red", linewidth=2.0)
#plt.show()


if params["do_plotting"]:
    plt.show()