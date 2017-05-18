import sys
import time
import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.nonlinearities import softmax
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#from sklearn.linear_model import LogisticRegression
import copy
na = np.newaxis

# imports from this project
import mytools
import networks
theano.config.optimizer = "None"

if __name__ == "__main__" and "-f" not in sys.argv:
    params = mytools.get_CLI_parameters(sys.argv)
else:
    params = mytools.get_CLI_parameters([])


def one_hot_encoding(target, n_classes):
    if len(target.shape) > 1:
        raise ValueError("target should be of shape (n_samples,)")
    target = target.flatten()
    encoding = np.eye(n_classes)[target]
    return encoding


def get_horseshoe_pattern(horseshoe_distractors):
    if horseshoe_distractors:
        A = np.empty((100, 8))
    else:
        A = np.empty((100, 4))

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
    """create_data
    creates data based on params["data"]. Returned data will be shuffled.

    :param params: dict, parameter dictionary
    :param N: scalar, number of samples
    """
    if params["data"] == "horseshoe":
        X, y = create_horseshoe_data(params, N)
    elif params["data"] == "ring":
        X, y = create_ring_data(params, N)
    else:
        raise("Requested datatype unknown")
    permutation = np.random.permutation(N)
    X = X[permutation]
    y = y[permutation]
    return X, y


def create_horseshoe_data(params, N):
    A = get_horseshoe_pattern(params["horseshoe_distractors"])

    if params["specific_dataclass"] is not None:
        # this should only be triggered if N=1, in this case the user
        # requests a datapoint of a specific class
        if N != 1:
            raise("specific_dataclass is set so N should be 1")
        y = np.array([params["specific_dataclass"]])
    else:
        # if no specific class is requested, generate classes randomly
        y = np.random.randint(low=0, high=4, size=N)

    y_onehot = one_hot_encoding(y, params["n_classes"]).T

    if params["horseshoe_distractors"]:
        y_dist = np.random.normal(size=(4, N))
        y_onehot = np.concatenate((y_onehot, y_dist), axis=0)

    # create X by multiplying the target vector with the patterns,
    # and tranpose because we want the data to be in [samples, features] form
    X = np.dot(A, y_onehot).T
    for idx in range(X.shape[0]):
        X[idx, :] += (np.random.normal(size=(100))*params["noise_scale"])
    return X, y.astype(np.int32)


def create_ring_data(params, N):
    """
    Creates 2d data aligned in clusters aligned on a ring
    """
    n_centers = 8
    n_per_center = int(np.ceil(N / n_centers))
    C = .02*np.eye(2)
    radius = 1
    class_means = radius*np.array([[np.cos(i*2.*np.pi/n_centers),np.sin(i*2.*np.pi/n_centers)] for i in range(n_centers)])

    X = np.random.multivariate_normal((0, 0), C, size=n_per_center) + class_means[0, :]
    y = np.ones((n_per_center,)) * int(0 % params["n_classes"])
    for idx in range(1, n_centers):
        X_part = np.random.multivariate_normal((0, 0), C, size=n_per_center) + class_means[idx, :]
        y_part = np.ones((n_per_center,)) * int(idx % params["n_classes"])
        X = np.concatenate((X, X_part), 0)
        y = np.concatenate((y, y_part), 0)
    y = y.astype(np.int)

    # in case N is not a multiple of n_centers, we just created a few too many datapoints
    if N < X.shape[0]:
        X = X[:N]
        y = y[:N]

    if params["bias_in_data"]:
        onesvec = np.atleast_2d(np.ones((X.shape[0]))).T
        X = np.hstack((X, onesvec))
    return X, y


def iterate_minibatches(inputs, targets, batchsize, shuffle=True):
    """
    this function makes no assumption about the shape of inputs and 
    targets. It just assumes that they have the same length.
    """
    assert len(inputs) == len(targets)
    indices = np.arange(len(inputs))
    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = indices[start_idx:start_idx + batchsize]
        yield inputs[excerpt], targets[excerpt]


def train_network(params):
    params = copy.deepcopy(params)
    X_train, y_train = create_data(params, params["N_train"])
    X_val, y_val = create_data(params, params["N_val"])
    X_test, y_test = create_data(params, params["N_test"])

    # if we're training the CNN, we need extra dimensions
    if params["model"] == "cnn":
        X_train = X_train.reshape(params["network_input_shape"])
        X_val = X_val.reshape(params["network_input_shape"])
        X_test = X_test.reshape(params["network_input_shape"])

    if not params["network_input_shape"][1:] == X_train.shape[1:]:
        raise ValueError("parameter network_input_shape didn't fit train data")

    print("Building model and compiling functions...")
    input_var = T.matrix('inputs')
    target_var = T.matrix('targets')
    if params["model"] == 'mlp':
        network = networks.build_mlp(params, input_var)
    elif params["model"] == "cnn":
        input_var = T.tensor4("inputs")
        network = networks.build_cnn(params, input_var)
    elif params["model"] == "custom":
        network = networks.build_custom_ringpredictor(params, input_var)
    output_var = lasagne.layers.get_output(network)

    loss = lasagne.objectives.categorical_crossentropy(output_var, target_var)

    # average loss expressions
    loss = loss.mean()

    network_params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, network_params, learning_rate=params["lr"], momentum=0.9)

    # save some useful variables and functions into the params dict
    params["input_var"] = input_var
    params["target_var"] = target_var
    params["output_var"] = output_var
    params["output_func"] = theano.function(
            [input_var], output_var, allow_input_downcast=True)

    # Create an expression for the classification accuracy:
    prediction_var = T.argmax(output_var, axis=1)
    params["prediction_func"] = theano.function(
            [input_var], prediction_var, allow_input_downcast=True)

    if params["model"] == "custom":
        return network, params

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var],
                               loss, updates=updates, allow_input_downcast=True)

    # TRAINING LOOP
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(params["epochs"]):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, params["minibatch_size"]):
            inputs, targets = batch
            train_err += train_fn(inputs, one_hot_encoding(targets, params["n_classes"]))
            train_batches += 1

        # And a full pass over the validation data:
        if epoch % 10 == 0:
            val_acc = 0
            val_batches = 0
            for batch in iterate_minibatches(X_val, y_val, params["minibatch_size"]):
                X_val_batch, y_val_batch = batch
                y_val_hat = params["prediction_func"](X_val_batch)
                val_acc += compute_accuracy(y_val_batch, y_val_hat)
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
        X_test_batch, y_test_batch = batch
        y_test_hat = params["prediction_func"](X_test_batch)
        test_acc += compute_accuracy(y_test_batch, y_test_hat)
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
    """plot_heatmap

    :param R_i: array, shape (width, height)
    :param axis: matplotlib subplot axis object
    :param title: string, title of plot
    """
    if axis is None:
        fig, axis = plt.subplots(1, 1, figsize=(15, 10))
    axis.pcolor(R_i, cmap="viridis", vmin=-np.max(abs(R_i)), vmax=np.max(abs(R_i)))
    axis.set_title(title)
    axis.invert_yaxis()
    axis.set_xticks([])
    axis.set_yticks([])


def forward_pass(X, network, input_var, params):
    """
    IMPORTANT: THIS FUNCTION CAN ONLY BE CALLED WITH A SINGLE INPUT SAMPLE
    the function expects a row vector
    """
    get_activations = theano.function(
        [input_var],
        lasagne.layers.get_output(lasagne.layers.get_all_layers(network)),
        allow_input_downcast=True)
    activations = get_activations(X)
    for i in range(len(activations)):
        activations[i] = activations[i].squeeze()
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

    if bias_in_data:
        return W_mats
    else:
        return W_mats, biases


def LRP(X, network, output_neuron, params, rule="epsilon", epsilon=.01, alpha=0.5):
    """LRP
    Compute the layerwise relevance propagation either with the epislon- or with the
    alphabeta rule

    :param X: array, shape network_input_shape
    :param network: lasagne network
    :param output_neuron: scalar, the output neuron with respect to which the
        LRP is to be computed
    :param params: dict, parameter dictionary
    :param rule: string, either "epsilon" or "alphabeta"
    :param epsilon: scalar, epsilon parameter for epsilon rule
    :param alpha: scalar, alpha parameter for alphabeta rule (beta is inferred)

    :returns: relevance, array of shape X.shape
    """

    W_mats, biases = get_network_parameters(network, params["bias_in_data"])
    activations = forward_pass(X, network, params["input_var"], params)
    # reshape the input activations to be a shapeless vector
    activations[0] = activations[0].reshape((-1,))

    # --- relevance backpropagation ---
    # the first backpass is special so it can't be in the loop. Here we "compute" the
    # relevance in the last layer (set relevance to activation)
    R = np.array([activations[-1][output_neuron], ])
    # extract the relevant row from the last weight matrix and save it into the
    # list that holds all weight matrices
    W_mats[-1] = W_mats[-1][[output_neuron]]
    # loop from end to beginning starting at the second last layer
    for idx in np.arange(len(activations)-2, -1, -1):
        if rule is "epsilon":
            R = epsilon_rule(R, W_mats[idx], biases[idx], activations[idx], epsilon)
        elif rule == "alphabeta":
            R = alphabeta_rule(R, W_mats[idx], biases[idx], activations[idx], alpha)
        else:
            raise Exception("Unrecognized rule selected")

    R = R.reshape((X.shape))
    return R


def easyLRP(X, network, output_neuron, params, rule="epsilon", epsilon=.01, alpha=0.5):
    """easyLRP
    Compute the layerwise relevance propagation either with the epislon- or with the
    alphabeta rule

    :param X: array, shape network_input_shape
    :param network: lasagne network
    :param output_neuron: scalar, the output neuron with respect to which the
        LRP is to be computed
    :param params: dict, parameter dictionary
    :param rule: string, either "epsilon" or "alphabeta"
    :param epsilon: scalar, epsilon parameter for epsilon rule
    :param alpha: scalar, alpha parameter for alphabeta rule (beta is inferred)

    :returns: relevance, array of shape X.shape + (n_classes,)
    """
    # get all gradients and select the gradients with respect to the output_neuron desired
    gradients = get_gradients(X, network, params)
    relevance = gradients * X[..., np.newaxis]
    return relevance


def epsilon_rule(R, W, b, activations_current_layer, epsilon):
    """epsilon_rule

    :param R: shape (n_neurons_next,)
    :param W: weight matrix shape (n_neurons_next, n_neurons_current)
        weight matrix with the weights for a particular neuron in the next layer in one row
    :param b: bias vector for next layer of shape (n_neurons_next,)
    :param activations_current_layer: shape (n_neurons_current,)
    """

    # input checks
    if not (len(R.shape) == 1 and
            len(activations_current_layer.shape) == 1):
        raise ValueError("Relevances and activations must all have shape (n_neurons,)")
    if not len(W.shape) == 2:
        raise ValueError("W must have shape (n_neurons_next, n_neurons_current)")

    # R_message_matrix of shape (n_current_layer, n_next_layer)
    R_message_matrix = np.empty(shape=W.T.shape)

    for j in range(R_message_matrix.shape[1]):
        # compute the inputs to j in the next layer and j's preactivation
        inputs_to_j = W[j] * activations_current_layer
        z_j = np.sum(inputs_to_j) + b[j]

        # compute the relevance messages that neuron j sends
        if z_j >= 0:
            R_messages_from_j = inputs_to_j / (z_j + epsilon) * R[j]
        elif z_j < 0:
            R_messages_from_j = inputs_to_j / (z_j - epsilon) * R[j]

        # insert the relevance messages into the relevance message matrix
        R_message_matrix[:, j] = R_messages_from_j

    # sum over axis 1. Every row holds relevance messages from all neurons in the next layer
    # the a specific neuron in the current layer
    R_current = np.sum(R_message_matrix, axis=1)

    # make sure our computed relevance has the same shape as the activations in the current layer
    assert R_current.shape == activations_current_layer.shape, "computed relevance has " + \
                                                               "incorrect shape"
    return R_current


def alphabeta_rule(R, W, b, activations_current_layer, alpha):
    """alphabeta_rule

    :param R: shape (n_neurons_next,)
    :param W: weight matrix shape (n_neurons_next, n_neurons_current)
    :param b: bias vector for next layer of shape (n_neurons_next,)
    :param activations_current_layer: shape (n_neurons_current,)
    """

    # alpha and beta have to add up to 1, so we can compute beta
    beta = 1 - alpha

    # input checks
    if not (len(R.shape) == 1 and
            len(activations_current_layer.shape) == 1):
        raise ValueError("Relevances and activations must all have shape (n_neurons,)")
    if not len(W.shape) == 2:
        raise ValueError("W must have shape (n_neurons_next, n_neurons_current)")

    # R_message_matrix of shape (n_current_layer, n_next_layer)
    R_message_matrix = np.empty(shape=W.T.shape)

    for j in range(R_message_matrix.shape[1]):
        # compute the inputs to j in the next layer and j's preactivation
        inputs_to_j = W[j] * activations_current_layer

        # get the indicies of the positive and negative inputs
        positive_inputs = inputs_to_j >= 0
        negative_inputs = ~positive_inputs

        if b[j] >= 0:
            positive_bias = b[j]
            negative_bias = 0
        else:
            positive_bias = 0
            negative_bias = b[j]

        positive_preactivation = np.sum(inputs_to_j[positive_inputs]) + positive_bias
        negative_preactivation = np.sum(inputs_to_j[negative_inputs]) + negative_bias

        R_messages_from_j = R[j] * (
                    alpha * positive_inputs / positive_preactivation +
                    beta  * negative_inputs / negative_preactivation)

        # insert the relevance messages into the relevance message matrix
        R_message_matrix[:, j] = R_messages_from_j

    # sum over axis 1. Every row holds relevance messages from all neurons in the next layer
    # the a specific neuron in the current layer
    R_current = np.sum(R_message_matrix, axis=1)

    # make sure our computed relevance has the same shape as the activations in the current layer
    assert R_current.shape == activations_current_layer.shape, "computed relevance has " + \
                                                               "incorrect shape"
    return R_current


def compute_accuracy(y, y_hat):
    """compute_accuracy
    Compute the percentage of correct classifications

    :param y: array, shape (n_samples,)
    :param y_hat: array, shape (n_samples,)
    """
    if not len(y.shape) == 1 or not len(y_hat.shape) == 1:
        raise ValueError("Both inputs need to have shape (n_samples,)")
    if not y.shape == y_hat.shape:
        raise ValueError("The two inputs didn't have the same shape")

    n_correct = np.sum(y == y_hat)
    p_correct = n_correct / y.size
    return p_correct


def get_gradients(X, network, params):
    """get_gradients

    :param X: array, shape network_input_shape
         X contains the data for which to compute the gradients
    :param params: parameter dict

    :returns: gradients, array of shape X.shape + (n_classes,)
         the first axis contains samples, the last axis contains the
         class indices with regard to which the gradient is taken
    """
    replaced_nonlinearity = False
    if hasattr(network, "nonlinearity") and network.nonlinearity == softmax:
        network.nonlinearity = lambda x: x
        replaced_nonlinearity = True

    gradients = np.empty(X.shape + (params["n_classes"],))
    for output_idx in range(params["n_classes"]):
        gradient_var = T.grad(params["output_var"][0, output_idx], params["input_var"])
        compute_grad = theano.function(
            [params["input_var"]], gradient_var, allow_input_downcast=True)
        for sample_idx in range(X.shape[0]):
            gradients[sample_idx, ..., output_idx] = compute_grad(X[[sample_idx]])
    gradients.reshape(X.shape + (-1,))

    if replaced_nonlinearity:
        network.nonlinearity = softmax
    return gradients


def get_patterns(gradients, Sigma_X, Sigma_s_inv):
    """get_patterns
    Compute the Patterns according to Haufe 2015 given the gradients

    :param gradients: array, shape (n_samples, [...,] feature dimensions, n_classes)
    :param Sigma_X: array, shape (n_features, n_features)
        Covariance matrix of the input
    :param Sigma_s_inv: array, shape (n_classes, n_classes)
        Covariance matrix of the classes

    :returns: patterns, array of shape gradients.shape
    """
    orig_shape = gradients.shape
    gradients = gradients.reshape((gradients.shape[0], -1, gradients.shape[-1]))
    patterns = np.einsum('jk,ikl->ijl', Sigma_X, gradients)
    patterns = patterns.reshape(orig_shape)
    return patterns


def plot_background(params):
    """
    This function is for the ring data example only
    """
    # create some data for scatterplot
    X, y = create_data(params, 2000)
    y = params["prediction_func"](X)

    # create a mesh to plot in
    h = .01  # step size in the mesh
    x_min, x_max = -2, 2
    y_min, y_max = -2, 2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    mesh = np.c_[xx.ravel(), yy.ravel()]

    # compute output activations
    Z = params["output_func"](mesh)
    Z = Z[:, OUTPUT_NEURON_SELECTED]
    # reshape to mesh shape
    Z = Z.reshape(xx.shape)
    Z = np.flipud(Z)

    # due to annoying matplotlib behavior (matplotlib plots lines despite
    # marker="o"), we have to loop here.
    class_1_mask = (y == 0).squeeze()
    class_2_mask = (y == 1).squeeze()
    for idx in range(X[:500].shape[0]):
        plt.plot(X[class_1_mask, 0][idx], X[class_1_mask, 1][idx],
                 color="white",
                 marker="o",
                 fillstyle="full",
                 markeredgecolor="black")
        plt.plot(X[class_2_mask, 0][idx], X[class_2_mask, 1][idx],
                 color="black",
                 marker="o",
                 fillstyle="full",
                 markeredgecolor="black")
    plt.imshow(Z, interpolation="nearest", cmap=cm.gray, alpha=0.4,
               extent=[x_min, x_max, y_min, y_max])
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


def create_2d_mesh(interval=0.2):
    # create a mesh to plot in
    x_min, x_max = -2, 2 + interval
    y_min, y_max = -2, 2 + interval
    xx, yy = np.meshgrid(np.arange(x_min, x_max, interval),
                         np.arange(y_min, y_max, interval))
    mesh = np.c_[xx.ravel(), yy.ravel()]
    return mesh


def normalize_arrows(arrows, length=0.3):
    """normalize_arrows

    :param arrows: array of shape (n_samples, 2)
    :param length: scalar, desired Euclidean length of arrows
    """
    # normalize arrows to one using broadcasting (that's why we transpose twice)
    arrows = (arrows.T / np.linalg.norm(arrows, axis=1)).T
    # make all arrows length length
    arrows *= length
    return arrows


######################################################################
# RING DATA
# train MLP on ring data
params["layer_sizes"] = [8]
params["data"] = "ring"
params["model"] = "mlp"
params["n_classes"] = 2
params["network_input_shape"] = (-1, 2)
params["epochs"] = 30
network, params = train_network(params)
OUTPUT_NEURON_SELECTED = 0
VECTOR_ADJUST_CONSTANT = 3

X, y = create_data(params, 5000)

Sigma_X = np.cov(X, rowvar=False)
Sigma_s = np.cov(one_hot_encoding(y, params["n_classes"]), rowvar=False)
Sigma_s_inv = np.linalg.pinv(Sigma_s)

mesh = create_2d_mesh()

# GRADIENTS
# get all the gradients (this will have the output neurons in the last dimension)
# gradients has shape (n_samples, n_features, n_classes)
gradients = get_gradients(mesh, network, params)
# select gradients for only one output neuron and normalize their length
gradients_plotting = normalize_arrows(gradients[..., OUTPUT_NEURON_SELECTED])

plt.figure()
plot_background(params)
plt.quiver(mesh[:, 0], mesh[:, 1], gradients_plotting[:, 0], gradients_plotting[:, 1], scale=None)
plt.title("gradients")


# PATTERNS
# compute A from the Haufe paper.
# The columns of A are the activation patterns, i. e. A has shape (n_features, n_classes)
patterns = get_patterns(gradients, Sigma_X, Sigma_s_inv)
patterns_plotting = normalize_arrows(patterns[..., OUTPUT_NEURON_SELECTED])
plt.figure()
plot_background(params)
plt.quiver(mesh[:, 0], mesh[:, 1], patterns_plotting[:, 0], patterns_plotting[:, 1], scale=None)
plt.title("patterns")

# RELEVANCE
relevance = easyLRP(mesh, network, OUTPUT_NEURON_SELECTED, params, rule="alphabeta", alpha=2)
relevance_plotting = normalize_arrows(relevance[..., OUTPUT_NEURON_SELECTED])
plt.figure()
plot_background(params)
plt.quiver(mesh[:, 0], mesh[:, 1], relevance_plotting[:, 0], relevance_plotting[:, 1], scale=None)
plt.title("relevance")
plt.show()

######################################################################
# HORSESHOE DATA
OUTPUT_NEURON_SELECTED = 0

# regular mlp
params["model"] = "mlp"
params["data"] = "horseshoe"
params["n_classes"] = 4
params["network_input_shape"] = (-1, 100)
params["layer_sizes"] = [100, 10]  # as requested by pieter-jan
mlp, mlp_params = train_network(params.copy())
mlp_prediction_func = mlp_params["prediction_func"]

params["model"] = "cnn"
params["network_input_shape"] = (-1, 1, 10, 10)
params["epochs"] = 1
cnn, cnn_params = train_network(params.copy())
cnn_prediction_func = cnn_params["prediction_func"]

##### compare prediction scores ###
# some more data
X, y = create_data(params, 500)

Sigma_X = np.cov(X, rowvar=False)
Sigma_s = np.cov(one_hot_encoding(y, params["n_classes"]), rowvar=False)
Sigma_s_inv = np.linalg.pinv(Sigma_s)

# predict with mlp
mlp_prediction = mlp_prediction_func(X)
mlp_score = compute_accuracy(y, mlp_prediction)

# predict with cnn
cnn_prediction = cnn_prediction_func(X.reshape(params["network_input_shape"]))
cnn_score = compute_accuracy(y, cnn_prediction)

print("MLP score: " + str(mlp_score))
print("CNN score: " + str(cnn_score))

######
# get an input point for which we want the weights / patterns
params["specific_dataclass"] = 0
X, y = create_data(params, 1)
params["specific_dataclass"] = None
A = get_horseshoe_pattern(params["horseshoe_distractors"])

# MLP
W_mlp = get_gradients(X, mlp, mlp_params)
A_haufe_mlp = get_patterns(W_mlp, Sigma_X, Sigma_s_inv)
relevance_mlp = easyLRP(X, mlp, OUTPUT_NEURON_SELECTED, mlp_params, epsilon=0.00001)[..., OUTPUT_NEURON_SELECTED]

# plot real pattern, input point, weights and haufe pattern for MLP
W_mlp = W_mlp[..., OUTPUT_NEURON_SELECTED]
A_haufe_mlp = A_haufe_mlp[..., OUTPUT_NEURON_SELECTED]
fig, axes = plt.subplots(1, 5, figsize=(15, 5))
plot_heatmap(A[:, 0].reshape((10, 10)), axis=axes[0], title="True pattern")
plot_heatmap(X.reshape((10, 10)), axis=axes[1], title="input point")
plot_heatmap(W_mlp.reshape((10, 10)), axis=axes[2], title="gradient")
plot_heatmap(A_haufe_mlp.reshape((10, 10)), axis=axes[3], title="pattern Haufe 2014")
plot_heatmap(relevance_mlp.reshape((10, 10)), axis=axes[4], title="LRP")
plt.suptitle("MLP", size=16)


# CNN
W_cnn = get_gradients(X.reshape(cnn_params["network_input_shape"]), cnn, cnn_params)
A_haufe_cnn = get_patterns(W_cnn, Sigma_X, Sigma_s_inv)
relevance_cnn = easyLRP(
        X.reshape(cnn_params["network_input_shape"]), cnn, OUTPUT_NEURON_SELECTED, cnn_params)[..., OUTPUT_NEURON_SELECTED]

# plot real pattern, input point, weights and haufe pattern for CNN
W_cnn = W_cnn[..., OUTPUT_NEURON_SELECTED]
A_haufe_cnn = A_haufe_cnn[..., OUTPUT_NEURON_SELECTED]
fig, axes = plt.subplots(1, 5, figsize=(15, 5))
plot_heatmap(A[:, 0].reshape((10, 10)), axis=axes[0], title="True pattern")
plot_heatmap(X.reshape((10, 10)), axis=axes[1], title="input point")
plot_heatmap(W_cnn.reshape((10, 10)), axis=axes[2], title="gradient")
plot_heatmap(A_haufe_cnn.reshape((10, 10)), axis=axes[3], title="pattern Haufe 2014")
plot_heatmap(relevance_cnn.reshape((10, 10)), axis=axes[4], title="LRP")
plt.suptitle("CNN", size=16)

#######
if params["do_plotting"]:
    plt.show()
