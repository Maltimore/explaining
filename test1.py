import time
import numpy as np
import theano
import theano.tensor as T
import lasagne


# parameters
model = 'mlp'
num_epochs = 10
minibatch_size = 20
noise_scale = .3

# some global variables
INPUT_DIM = 10


def load_dataset(noise_scale=.6):
    def create_N_examples(N=500, noise_scale=.6):
        pic = np.zeros((10, 10))
        pic[2:8, 2] = 1
        pic[2:8, 7] = 1
        pic[2, 2:8] = 1
        pic[7, 2:8] = 1
        overall_idx = 0
        X = np.empty((N, 1, 10, 10))
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
                current_pic += np.random.normal(size=(INPUT_DIM, INPUT_DIM))*noise_scale
                X[overall_idx, 0, :, :] = current_pic
                y[overall_idx] = target
                overall_idx += 1
        return X, y

    X_train, y_train = create_N_examples(500, noise_scale)
    X_val, y_val = create_N_examples(100, noise_scale)
    X_test, y_test = create_N_examples(100, noise_scale)
    return X_train, y_train, X_val, y_val, X_test, y_test


def build_mlp(input_var=None):
    # This creates an MLP of two hidden layers of 800 units each, followed by
    # a softmax output layer of 10 units. It applies 20% dropout to the input
    # data and 50% dropout to the hidden layers.

    # Input layer, specifying the expected input shape of the network
    # (unspecified batchsize, 1 channel, 28 rows and 28 columns) and
    # linking it to the given Theano variable `input_var`, if any:
    l_in = lasagne.layers.InputLayer(shape=(None, 1, INPUT_DIM, INPUT_DIM),
                                     input_var=input_var)

    # Add a fully-connected layer of 800 units, using the linear rectifier, and
    # initializing weights with Glorot's scheme (which is the default anyway):
    l_hid1 = lasagne.layers.DenseLayer(
            l_in, num_units=200,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    # Finally, we'll add the fully-connected output layer, of 10 softmax units:
    l_out = lasagne.layers.DenseLayer(
            l_hid1, num_units=4,
            nonlinearity=lasagne.nonlinearities.softmax)

    # Each layer is linked to its incoming layer(s), so we only need to pass
    # the output layer to give access to a network in Lasagne:
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
X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(noise_scale)

# Prepare Theano variables for inputs and targets
input_var = T.tensor4('inputs')
target_var = T.ivector('targets')

# Create neural network model (depending on first command line parameter)
print("Building model and compiling functions...")
if model == 'mlp':
    network = build_mlp(input_var)
all_layers = lasagne.layers.get_all_layers(network)

# Create a loss expression for training, i.e., a scalar objective we want
# to minimize (for our multi-class problem, it is the cross-entropy loss):
prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss = loss.mean()
# We could add some weight decay as well here, see lasagne.regularization.

# Create update expressions for training, i.e., how to modify the
# parameters at each training step. Here, we'll use Stochastic Gradient
# Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=0.01, momentum=0.9)

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

# Compile a function performing a training step on a mini-batch (by giving
# the updates dictionary) and returning the corresponding training loss:
train_fn = theano.function([input_var, target_var], loss, updates=updates)

# Compile a second function computing the validation loss and accuracy:
val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

# Finally, launch the training loop.
print("Starting training...")
# We iterate over epochs:
for epoch in range(num_epochs):
    # In each epoch, we do a full pass over the training data:
    train_err = 0
    train_batches = 0
    start_time = time.time()
    for batch in iterate_minibatches(X_train, y_train, minibatch_size, shuffle=True):
        inputs, targets = batch
        train_err += train_fn(inputs, targets)
        train_batches += 1

    # And a full pass over the validation data:
    val_err = 0
    val_acc = 0
    val_batches = 0
    for batch in iterate_minibatches(X_val, y_val, minibatch_size, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        val_err += err
        val_acc += acc
        val_batches += 1

    # Then we print the results for this epoch:
    print("Epoch {} of {} took {:.3f}s".format(
        epoch + 1, num_epochs, time.time() - start_time))
    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
    print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
    print("  validation accuracy:\t\t{:.2f} %".format(
        val_acc / val_batches * 100))

# After training, we compute and print the test error:
test_err = 0
test_acc = 0
test_batches = 0
for batch in iterate_minibatches(X_test, y_test, minibatch_size, shuffle=False):
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



def compute_relevance(X, network):
    W1_var, b1_var, W2_var, b2_var = lasagne.layers.get_all_params(network)
    W1 = W1_var.get_value().T
    W2 = W2_var.get_value().T
    b1 = b1_var.get_value().T
    b2 = b2_var.get_value().T
    # propagate
    X = X.flatten()
    hid1_z = np.dot(W1, X) + b1
    hid1_act = np.copy(hid1_z)
    hid1_act[hid1_act < 0] = 0
    output_layer_z = np.dot(W2, hid1_act.flatten()) + b2
    output_layer_act = softmax(output_layer_z)
    output_layer_act

    W_z_mat1 = np.multiply(W1, X.T)
    #W_z_mat2 = np.multiply(W2, hid1_act.T)

    R_k_over_z_k = output_layer_act[0] / output_layer_z[0]
    R_j = np.multiply(W2[0,:].T, hid1_act) * R_k_over_z_k

    R_j_over_z_j = np.divide(R_j, hid1_z)
    R_i = np.sum(np.multiply(W_z_mat1.T, R_j_over_z_j), axis=1)

    R_i = R_i.reshape((10, 10))
    return R_i

R_i = compute_relevance(X_train[0][0], network)
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(15,10))
plot1 = axes[0].pcolor(R_i)
axes[0].set_title("Relevance per pixel")
plt.colorbar(plot1, ax=axes[0])
plot2 = axes[1].pcolor(X_train[0][0])
plt.colorbar(plot2, ax=axes[1])
axes[1].set_title("Input image")
plt.show()



#output = theano.function([input_var], lasagne.layers.get_output(all_layers[-1]))
#my ut = output(X_train[np.newaxis, 0])
#myout
