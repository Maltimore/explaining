import time
import numpy as np
import theano
import theano.tensor as T
import lasagne
import matplotlib.pyplot as plt


# parameters
params = {}
params["N_train"] = 500
params["N_val"] = 200
params["N_test"] = 200
params["model"] = 'mlp'
params["num_epochs"] = 50
params["minibatch_size"] = 20
params["noise_scale"] = .1
params["loss_choice"] = "MSE"
params["layer_sizes"] = [200, 400]
params["INPUT_DIM"] = 10
params["do_plotting"] = True
params["output_neuron"] = 1 # for which output neuron to compute the relevance
params["dataset"] = 402 # which dataset to use


def get_target(target, loss_choice):
    if loss_choice == "categorical_crossentropy":
        return target
    elif loss_choice == "MSE":
        target_vec = np.ones(4) * (-1)
        target_vec[target] = 1
        return target_vec

def get_category(target):
    try:
        # if the target is a one-hot vector, return argmax
        return np.argmax(target)
    except TypeError:
        # otherwise the target is already the category
        return target

def create_N_examples(params, N):
    pic = np.zeros((10, 10))
    pic[2:8, 2] = 1
    pic[2:8, 7] = 1
    pic[2, 2:8] = 1
    pic[7, 2:8] = 1
    overall_idx = 0
    X = np.empty((N, 1, 10, 10))
    if params["loss_choice"] == "categorical_crossentropy":
        y = np.empty(N, dtype=np.int32)
    elif params["loss_choice"] == "MSE":
        y = np.empty((N, 4), dtype=np.int32)
    else:
        raise("Unrecognized Loss choice")
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
            current_pic += (np.random.normal(size=(params["INPUT_DIM"], params["INPUT_DIM"]))*params["noise_scale"] -1) *2
            X[overall_idx, 0, :, :] = current_pic
            y[overall_idx] = get_target(target, params["loss_choice"])
            overall_idx += 1
    return X, y


def load_dataset(params):
    X_train, y_train = create_N_examples(params, params["N_train"])
    X_val, y_val = create_N_examples(params, params["N_val"])
    X_test, y_test = create_N_examples(params, params["N_test"])
    return X_train, y_train, X_val, y_val, X_test, y_test


def build_mlp(params, input_var=None):
    # Input layer
    l_in = lasagne.layers.InputLayer(shape=(None, 1, params["INPUT_DIM"], params["INPUT_DIM"]),
                                     input_var=input_var)
    # Hidden layers
    for layer_size in params["layer_sizes"]:
        current_hidden_layer = lasagne.layers.DenseLayer(
            l_in, num_units=layer_size,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    # Output layer
    if params["loss_choice"] == "categorical_crossentropy":
        l_out = lasagne.layers.DenseLayer(
                current_hidden_layer, num_units=4,
                nonlinearity=lasagne.nonlinearities.softmax)
    elif params["loss_choice"] == "MSE":
        l_out = lasagne.layers.DenseLayer(
                current_hidden_layer, num_units=4,
                nonlinearity=lasagne.nonlinearities.tanh)
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


# Load the dataset
print("Loading data...")
X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(params)

if params["loss_choice"] == "categorical_crossentropy":
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    if params["model"] == 'mlp':
        network = build_mlp(params, input_var)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)

    # loss option 1: categorical crossentropy
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()


    # We could add some weight decay as well here, see lasagne.regularization.

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
    input_var = T.tensor4('inputs')
    target_var = T.dmatrix('targets')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    if params["model"] == 'mlp':
        network = build_mlp(params, input_var)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)

    # loss option 2: MSE with {-1,1} targets
    loss = lasagne.objectives.squared_error(prediction, target_var)
    loss = loss.mean()


    # We could add some weight decay as well here, see lasagne.regularization.

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

# Compile a function performing a training step on a mini-batch (by giving
# the updates dictionary) and returning the corresponding training loss:
train_fn = theano.function([input_var, target_var], loss, updates=updates)

# Compile a second function computing the validation loss and accuracy:
val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

# Finally, launch the training loop.
print("Starting training...")
# We iterate over epochs:
for epoch in range(params["num_epochs"]):
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

    # Then we print the results for this epoch:
    print("Epoch {} of {} took {:.3f}s".format(
        epoch + 1, params["num_epochs"], time.time() - start_time))
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
print(test_batches)
print("Final results:")
print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
print("  test accuracy:\t\t{:.2f} %".format(
    test_acc / test_batches * 100))


# ------------------------------------------------------------------------------
# Relevance backpropagation
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def get_target_title(target):
    target = get_category(target)
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


def plot_heatmap(R_i, output_neuron, axis=None, title=""):
    if axis == None:
        fig, axis = plt.subplots(1, 2, figsize=(15, 10))

    plot = axis.pcolor(R_i, cmap="viridis")
    axis.set_title(title)
    axis.invert_yaxis()
    plt.colorbar(plot, ax=axis)


def compute_relevance(Input, network, output_neuron, epsilon = .01):
    # --- get paramters and activations for the input ---
    all_params = lasagne.layers.get_all_params(network)
    W_mats = all_params[0::2]
    biases = all_params[1::2]
    get_activations = theano.function([input_var],
                lasagne.layers.get_output(lasagne.layers.get_all_layers(network)))
    activations = get_activations(np.expand_dims(np.expand_dims(Input, axis=0), axis=0))
    # loop over W_mats, biases and activations to extract values and
    # flatten/transpose
    idx = 0
    for W, b, X in zip(W_mats, biases, activations):
        W_mats[idx] = W.get_value().T
        biases[idx] = b.get_value()
        activations[idx] = X[0].flatten()
        idx += 1
    # since there is one more layer than weight matrices/biases, extract last
    # layers values
    activations[idx] = activations[idx][0].flatten()


    # --- forward propagation to compute preactivations ---
    preactivations = []
    for W, b, X in zip(W_mats, biases, activations):
        preactivation = np.dot(W, X) + b
        preactivations.append(preactivation)

    # --- relevance backpropagation ---
    # the first backpass is special so it can't be in the loop
    R_over_z = activations[-1][output_neuron] / (preactivations[-1][output_neuron] + epsilon)
    R = np.multiply(W_mats[-1][output_neuron,:].T, activations[-2]) * R_over_z
    for idx in np.arange(2, len(activations)):
        R_over_z = np.divide(R, preactivations[-idx] + epsilon)
        Z_ij = np.multiply(W_mats[-idx], activations[-idx-1] + epsilon)
        R = np.sum(np.multiply(Z_ij.T, R_over_z), axis=1)
    R = R.reshape((10, 10))

    return R


fig, axes = plt.subplots(1, 5, figsize=(15, 10))
# first plotting the input image
title = "Input image"
plot_heatmap(X_train[params["dataset"]][0], y_train[params["dataset"]], axis=axes[0], title=title)
for output_neuron in np.arange(4):
    title = "Relevance for target " + get_target_title(output_neuron)
    R = compute_relevance(X_train[params["dataset"]][0], network, output_neuron)
    plot_heatmap(R, output_neuron, axes[output_neuron+1], title)
if params["do_plotting"]:
    plt.show()
