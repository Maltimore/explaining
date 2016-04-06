import time
import sys
import numpy as np
import theano
import theano.tensor as T
import lasagne
import matplotlib.pyplot as plt
import argparse
from mytools import softmax
from sklearn.linear_model import LogisticRegression

def get_parameters(argv):
    # command line interface
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--loss_choice", default="MSE")
    parser.add_argument("-n", "--noise_scale", default=0.6, type=float)
    parser.add_argument("-e", "--epochs", default=50, type=int)
    parser.add_argument("-m", "--model", default="mlp")
    parser.add_argument("--layer_sizes", default="200,200")
    parser.add_argument("-p", "--do_plotting", default=False)
    parser.add_argument("--verbose", default=False)

    params = vars(parser.parse_args(argv[1:]))
    params["N_train"] = 500
    params["N_val"] = 200
    params["N_test"] = 200
    params["minibatch_size"] = 20
    params["INPUT_DIM"] = 10
    params["output_neuron"] = 1 # for which output neuron to compute the
                                # relevance (choice 0..3)
    params["dataset"] = 2 # which dataset to use (choice 0..3)
    print(type(params["noise_scale"]))
    # extract layer sizes from input string
    layer_list = []
    for size in params["layer_sizes"].split(","):
        layer_list.append(int(size))
    params["layer_sizes"] = layer_list

    return params


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


def create_N_examples(params, N):
    pic = np.zeros((10, 10))
    pic[3:7, 2] = 1
    pic[3:7, 7] = 1
    pic[2, 3:7] = 1
    pic[7, 3:7] = 1
    overall_idx = 0
    X = np.empty((N, 10, 10))
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
            current_pic += (np.random.normal(size=(params["INPUT_DIM"], params["INPUT_DIM"]))*params["noise_scale"])
            X[overall_idx, :, :] = current_pic
            y[overall_idx] = target
            overall_idx += 1
            if overall_idx >= N:
                break
    return X, y


def build_mlp(params, input_var=None):
    # Input layer
    current_layer = lasagne.layers.InputLayer(shape=(None, 1, params["INPUT_DIM"], params["INPUT_DIM"]),
                                     input_var=input_var)
    # Hidden layers
    for layer_size in params["layer_sizes"]:
        if layer_size == 0:
            print("Zero layer requested, ignoring...")
            continue
        current_layer = lasagne.layers.DenseLayer(
            current_layer, num_units=layer_size,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    # Output layer
    if params["loss_choice"] == "categorical_crossentropy":
        l_out = lasagne.layers.DenseLayer(
                current_layer, num_units=4,
                nonlinearity=lasagne.nonlinearities.softmax)
    elif params["loss_choice"] == "MSE":
        l_out = lasagne.layers.DenseLayer(
                current_layer, num_units=4,
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

    X_train, y_train = create_N_examples(params, params["N_train"])
    X_val, y_val = create_N_examples(params, params["N_val"])
    X_test, y_test = create_N_examples(params, params["N_test"])
    X_train = np.expand_dims(X_train, 1)
    X_val = np.expand_dims(X_val, 1)
    X_test = np.expand_dims(X_test, 1)
    y_train = transform_target(y_train, params["loss_choice"])
    y_val = transform_target(y_val, params["loss_choice"])
    y_test = transform_target(y_test, params["loss_choice"])

    if params["loss_choice"] == "categorical_crossentropy":
        # Prepare Theano variables for inputs and targets
        input_var = T.tensor4('inputs')
        target_var = T.ivector('targets')

        print("Building model and compiling functions...")
        if params["model"] == 'mlp':
            network = build_mlp(params, input_var)

        prediction = lasagne.layers.get_output(network)

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
        input_var = T.tensor4('inputs')
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
    print(test_batches)
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


def plot_heatmap(R_i, output_neuron, axis=None, title=""):
    if axis == None:
        fig, axis = plt.subplots(1, 1, figsize=(15, 10))

    plot = axis.pcolor(R_i, cmap="viridis")
    axis.set_title(title)
    axis.invert_yaxis()
    plt.colorbar(plot, ax=axis)


def forward_pass(func_input, network, input_var):
    get_activations = theano.function([input_var],
                lasagne.layers.get_output(lasagne.layers.get_all_layers(network)))
    activations = get_activations(np.expand_dims(np.expand_dims(func_input, axis=0), axis=0))
    return activations


def compute_relevance(func_input, network, output_neuron, params, epsilon = .01):

    # --- get paramters and activations for the input ---
    all_params = lasagne.layers.get_all_params(network)
    W_mats = all_params[0::2]
    biases = all_params[1::2]
    activations = forward_pass(func_input, network, params["input_var"])

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


def manual_classification(func_input):
    left_bar = np.sum(func_input[3:7, 2])
    right_bar = np.sum(func_input[3:7, 7])
    upper_bar = np.sum(func_input[2, 3:7])
    lower_bar = np.sum(func_input[7, 3:7])

    left_open = right_bar + upper_bar + lower_bar
    right_open = left_bar + upper_bar + lower_bar
    up_open = left_bar + right_bar + lower_bar
    low_open = left_bar + right_bar + upper_bar
    return np.argmax([left_open, right_open, up_open, low_open])


if __name__ == "__main__" and "-f" not in sys.argv:
    params = get_parameters(sys.argv)
else:
    params = get_parameters("".split())

network, params = train_network(params)
# create another example
X, y = create_N_examples(params, 4)

fig, axes = plt.subplots(1, 5, figsize=(15, 10))
# first plotting the input image
title = "Input image"
plot_heatmap(X[params["dataset"]], y[params["dataset"]], axis=axes[0], title=title)
for output_neuron in np.arange(4):
    title = "Relevance for target " + get_target_title(output_neuron)
    R = compute_relevance(X[params["dataset"]], network, output_neuron, params)
    plot_heatmap(R, output_neuron, axes[output_neuron+1], title)
if params["do_plotting"]:
    plt.show()



####### logistic regression
print("Performing Logistic Regression")
X_train, y_train = create_N_examples(params, 500)
X_train = np.reshape(X_train, (500, -1), order="C")

LogReg = LogisticRegression()
LogReg.fit(X_train, y_train)
coefs = LogReg.coef_
coefs = np.reshape(coefs, (coefs.shape[0], params["INPUT_DIM"],-1))

params["dataset"] = 1
title = "Coefs for " + str(get_target_title(params["dataset"]))
plot_heatmap(coefs[params["dataset"]], y[params["dataset"]], title=title)
if params["do_plotting"]:
    plt.show()




# comparing manual classification with network output
X, y = create_N_examples(params, 200)
manual_score = 0
network_score = 0
for idx in range(len(X)):
    if idx%10 == 0:
        print("Processing item " + str(idx))
    # do manual classification by summing over bars
    manual_prediction = manual_classification(X[idx])

    if manual_prediction == y[idx]:
        manual_score += 1

network_output = lasagne.layers.get_output(network)
get_network_output = theano.function([params["input_var"]], network_output)
network_prediction = np.argmax(get_network_output(np.expand_dims(X, 1)), axis=1)
network_score = np.sum(network_prediction == y)
logreg_prediction = LogReg.predict(np.reshape(X, (len(X),100)))
logreg_score = np.sum(logreg_prediction ==y)
print("Manual classification score: " + str(manual_score))
print("Network classification score: " + str(network_score))
print("LogReg classification score: " + str(logreg_score))
