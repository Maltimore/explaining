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
import load_mnist
theano.config.optimizer = "None"

params = mytools.get_CLI_parameters(sys.argv)

params["model"] = "mlp"
params["data"] = "mnist"
params["n_classes"] = 10
params["network_input_shape"] = (-1, 28 * 28)
params["epochs"] = 3
heatmap_shape = (28, 28)

X_train, y_train, X_val, y_val, X_test, y_test = load_mnist.load_dataset()
mean = np.mean(X_train)
std = np.std(X_train)
X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std
data = (X_train, y_train,
        X_val, y_val,
        X_test, y_test)

# MLP TRAINING
mlp, mlp_params = main_methods.train_network(data, params)
mlp_prediction_func = mlp_params["prediction_func"]
if hasattr(mlp, "nonlinearity") and mlp.nonlinearity == softmax:
    mlp.nonlinearity = lambda x: x
output_var_mlp = lasagne.layers.get_output(mlp)
raw_output_mlp_f = theano.function([mlp_params["input_var"]],
    output_var_mlp, allow_input_downcast=True)

## CNN TRAINING
#params["model"] = "cnn"
#params["network_input_shape"] = (-1, 1, 28, 28)
#params["epochs"] = 1
##X_train = X_train.reshape(params["network_input_shape"])
##X_val = X_val.reshape(params["network_input_shape"])
##X_test = X_test.reshape(params["network_input_shape"])
#data = (X_train.reshape(params["network_input_shape"]), y_train,
#        X_val.reshape(params["network_input_shape"]), y_val,
#        X_test.reshape(params["network_input_shape"]), y_test)
#cnn, cnn_params = main_methods.train_network(data, params)
#
#cnn_prediction_func = cnn_params["prediction_func"]
#if hasattr(cnn, "nonlinearity") and cnn.nonlinearity == softmax:
#    cnn.nonlinearity = lambda x: x
#output_var_cnn = lasagne.layers.get_output(cnn)
#raw_output_cnn_f = theano.function([cnn_params["input_var"]],
#   output_var_cnn, allow_input_downcast=True)

# some more data
X, y = (X_val[:1000], y_val[:1000])

y_hat_mlp = raw_output_mlp_f(X)
#y_hat_cnn = raw_output_cnn_f(X.reshape(cnn_params["network_input_shape"]))

Sigma_X = np.cov(X, rowvar=False)
Sigma_s_mlp = np.cov(y_hat_mlp, rowvar=False)
Sigma_s_mlp_inv = np.linalg.pinv(Sigma_s_mlp)
#Sigma_s_cnn = np.cov(y_hat_cnn, rowvar=False)
#Sigma_s_cnn_inv = np.linalg.pinv(Sigma_s_cnn)

# predict with mlp
mlp_prediction = mlp_prediction_func(X)
mlp_score = main_methods.compute_accuracy(y, mlp_prediction)

# predict with cnn
#cnn_prediction = cnn_prediction_func(X.reshape(params["network_input_shape"]))
#cnn_score = main_methods.compute_accuracy(y, cnn_prediction)

print("MLP score: " + str(mlp_score))
#print("CNN score: " + str(cnn_score))

######
# get an input point for which we want the weights / patterns
params["specific_dataclass"] = 0
X, y = (X_val[[0]], y_val[[0]])
OUTPUT_NEURON_SELECTED = y_val[0]
#params["specific_dataclass"] = None
#A = main_methods.get_horseshoe_pattern(params["horseshoe_distractors"])

# MLP
W_mlp = main_methods.get_gradients(X, mlp, OUTPUT_NEURON_SELECTED, mlp_params)
A_haufe_mlp = main_methods.get_patterns(X, mlp, OUTPUT_NEURON_SELECTED, mlp_params, Sigma_X, Sigma_s_mlp_inv)
relevance_mlp = main_methods.LRP(X, mlp, OUTPUT_NEURON_SELECTED, mlp_params,
    rule="alphabeta", alpha=2, epsilon=0.01)

# plot real pattern, input point, weights and haufe pattern for MLP
fig, axes = plt.subplots(1, 4, figsize=(15, 5))
main_methods.plot_heatmap(X.reshape(heatmap_shape), axis=axes[0], title="input point")
main_methods.plot_heatmap(W_mlp.reshape(heatmap_shape), axis=axes[1], title="gradient")
main_methods.plot_heatmap(A_haufe_mlp.reshape(heatmap_shape), axis=axes[2], title="pattern Haufe 2014")
main_methods.plot_heatmap(relevance_mlp.reshape(heatmap_shape), axis=axes[3], title="LRP")
plt.suptitle("MLP", size=16)


## CNN
#W_cnn = main_methods.get_gradients(
#    X.reshape(cnn_params["network_input_shape"]), cnn, OUTPUT_NEURON_SELECTED, cnn_params)
#A_haufe_cnn = main_methods.get_patterns(
#    X.reshape(cnn_params["network_input_shape"]), cnn, OUTPUT_NEURON_SELECTED, cnn_params,
#    Sigma_X, Sigma_s_cnn_inv)
#relevance_cnn = main_methods.easyLRP(
#    X.reshape(cnn_params["network_input_shape"]), cnn, OUTPUT_NEURON_SELECTED, cnn_params)
#
## plot real pattern, input point, weights and haufe pattern for CNN
#fig, axes = plt.subplots(1, 4, figsize=(15, 5))
#main_methods.plot_heatmap(X.reshape(heatmap_shape), axis=axes[0], title="input point")
#main_methods.plot_heatmap(W_cnn.reshape(heatmap_shape), axis=axes[1], title="gradient")
#main_methods.plot_heatmap(A_haufe_cnn.reshape(heatmap_shape), axis=axes[2], title="pattern Haufe 2014")
#main_methods.plot_heatmap(relevance_cnn.reshape(heatmap_shape), axis=axes[3], title="LRP")
#plt.suptitle("CNN", size=16)
plt.show()
