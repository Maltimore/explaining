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
import main_methods
theano.config.optimizer = "None"

params = mytools.get_CLI_parameters(sys.argv)

# train MLP on ring data
params["layer_sizes"] = [8]
params["data"] = "ring"
params["model"] = "mlp"
params["n_classes"] = 2
params["network_input_shape"] = (-1, 2)
params["epochs"] = 30

# CREATE DATA
X_train, y_train = main_methods.create_data(params, params["N_train"])
X_val, y_val = main_methods.create_data(params, params["N_val"])
X_test, y_test = main_methods.create_data(params, params["N_test"])
data = (X_train, y_train,
        X_val, y_val,
        X_test, y_test)

network, params = main_methods.train_network(data, params)
OUTPUT_NEURON_SELECTED = 0
VECTOR_ADJUST_CONSTANT = 3

if hasattr(network, "nonlinearity") and network.nonlinearity == softmax:
    network.nonlinearity = lambda x: x
output_var_network = lasagne.layers.get_output(network)
raw_output_network_f = theano.function([params["input_var"]],
    output_var_network, allow_input_downcast=True)

X, y = main_methods.create_data(params, 5000)

y_hat = raw_output_network_f(X)

Sigma_X = np.cov(X, rowvar=False)
Sigma_s = np.cov(y_hat, rowvar=False)
Sigma_s_inv = np.linalg.pinv(Sigma_s)

mesh = main_methods.create_2d_mesh()

# GRADIENTS
# get all the gradients (this will have the output neurons in the last dimension)
# gradients has shape (n_samples, n_features, n_classes)
gradients = main_methods.get_gradients(mesh, network, OUTPUT_NEURON_SELECTED, params)
# select gradients for only one output neuron and normalize their length
gradients_plotting = main_methods.normalize_arrows(gradients)

plt.figure()
main_methods.plot_background(OUTPUT_NEURON_SELECTED, params)
plt.quiver(mesh[:, 0], mesh[:, 1], gradients_plotting[:, 0], gradients_plotting[:, 1], scale=None)
plt.title("gradients")


# PATTERNS
# compute A from the Haufe paper.
# The columns of A are the activation patterns, i. e. A has shape (n_features, n_classes)
patterns = main_methods.get_patterns(mesh, network, OUTPUT_NEURON_SELECTED, params, Sigma_X, Sigma_s_inv)
patterns_plotting = main_methods.normalize_arrows(patterns)
plt.figure()
main_methods.plot_background(OUTPUT_NEURON_SELECTED, params)
plt.quiver(mesh[:, 0], mesh[:, 1], patterns_plotting[:, 0], patterns_plotting[:, 1], scale=None)
plt.title("patterns")

# RELEVANCE
relevance = main_methods.easyLRP(mesh, network, OUTPUT_NEURON_SELECTED, params, rule="alphabeta", alpha=2)
relevance_plotting = main_methods.normalize_arrows(relevance)
plt.figure()
main_methods.plot_background(OUTPUT_NEURON_SELECTED, params)
plt.quiver(mesh[:, 0], mesh[:, 1], relevance_plotting[:, 0], relevance_plotting[:, 1], scale=None)
plt.title("relevance")
plt.show()
