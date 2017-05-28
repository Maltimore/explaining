import sys
import time
import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.nonlinearities import softmax
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.linear_model import LogisticRegression
import copy
na = np.newaxis

# imports from this project
import mytools
import main_methods
theano.config.optimizer = "None"
# HORSESHOE DATA
OUTPUT_NEURON_SELECTED = 0

params = mytools.get_CLI_parameters(sys.argv)

# set custom parameters
params["model"] = "mlp"
params["data"] = "horseshoe"
params["n_classes"] = 4
params["network_input_shape"] = (-1, 100)
params["layer_sizes"] = [100, 10]  # as requested by pieter-jan
params["epochs"] = 30
params["noise_scale"] = 0.2

# CREATE DATA
X_train, y_train = main_methods.create_data(params, params["N_train"])
X_val, y_val = main_methods.create_data(params, params["N_val"])
X_test, y_test = main_methods.create_data(params, params["N_test"])
data = (X_train, y_train,
        X_val, y_val,
        X_test, y_test)

# LOGISTIC REGRESSION TRAINING
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print("LogReg score: " + str(logreg.score(X_test, y_test)))


# MLP TRAINING
mlp, mlp_params = main_methods.train_network(data, params)
mlp_prediction_func = mlp_params["prediction_func"]
if hasattr(mlp, "nonlinearity") and mlp.nonlinearity == softmax:
    mlp.nonlinearity = lambda x: x
output_var_mlp = lasagne.layers.get_output(mlp)
raw_output_mlp_f = theano.function([mlp_params["input_var"]],
    output_var_mlp, allow_input_downcast=True)

# CNN TRAINING
params["model"] = "cnn"
params["network_input_shape"] = (-1, 1, 10, 10)
params["epochs"] = 1

X_train = X_train.reshape(params["network_input_shape"])
X_val = X_val.reshape(params["network_input_shape"])
X_test = X_test.reshape(params["network_input_shape"])
data = (X_train, y_train,
        X_val, y_val,
        X_test, y_test)
cnn, cnn_params = main_methods.train_network(data, params)

cnn_prediction_func = cnn_params["prediction_func"]
if hasattr(cnn, "nonlinearity") and cnn.nonlinearity == softmax:
    cnn.nonlinearity = lambda x: x
output_var_cnn = lasagne.layers.get_output(cnn)
raw_output_cnn_f = theano.function([cnn_params["input_var"]],
   output_var_cnn, allow_input_downcast=True)

# some more data
X, y = main_methods.create_data(params, 5000)

y_hat_logreg = logreg.predict_proba(X)
y_hat_mlp = raw_output_mlp_f(X)
y_hat_cnn = raw_output_cnn_f(X.reshape(cnn_params["network_input_shape"]))

Sigma_X = np.cov(X, rowvar=False)
Sigma_s_logreg = np.cov(y_hat_logreg, rowvar=False)
Sigma_s_logreg_inv = np.linalg.pinv(Sigma_s_logreg)
Sigma_s_mlp = np.cov(y_hat_mlp, rowvar=False)
Sigma_s_mlp_inv = np.linalg.pinv(Sigma_s_mlp)
Sigma_s_cnn = np.cov(y_hat_cnn, rowvar=False)
Sigma_s_cnn_inv = np.linalg.pinv(Sigma_s_cnn)

# predict with mlp
mlp_prediction = mlp_prediction_func(X)
mlp_score = main_methods.compute_accuracy(y, mlp_prediction)

# predict with cnn
cnn_prediction = cnn_prediction_func(X.reshape(params["network_input_shape"]))
cnn_score = main_methods.compute_accuracy(y, cnn_prediction)

print("MLP score: " + str(mlp_score))
print("CNN score: " + str(cnn_score))

######
# get an input point for which we want the weights / patterns
params["sample_of_class"] = OUTPUT_NEURON_SELECTED
X, y = main_methods.create_data(params, 1)
A = main_methods.get_horseshoe_patterns(params["horseshoe_distractors"])

# PLOTTING ############################
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
main_methods.plot_heatmap(A[:, OUTPUT_NEURON_SELECTED].reshape((10, 10)), axis=axes[0], title="pattern")
main_methods.plot_heatmap(A[:, OUTPUT_NEURON_SELECTED + 4].reshape((10, 10)), axis=axes[1], title="distractor")
main_methods.plot_heatmap(X.reshape((10, 10)), axis=axes[2], title="final sample")
#######################################

# LOGISTIC REGRESSION
W_logreg = logreg.coef_.T
A_haufe_logreg = np.einsum('jk,kl,lm->jm', Sigma_X, W_logreg, Sigma_s_logreg_inv)[..., OUTPUT_NEURON_SELECTED]

# plot real pattern, input point, weights and haufe pattern for MLP
fig, axes = plt.subplots(1, 4, figsize=(10, 5))
main_methods.plot_heatmap(A[:, OUTPUT_NEURON_SELECTED].reshape((10, 10)), axis=axes[0], title="True pattern")
main_methods.plot_heatmap(X.reshape((10, 10)), axis=axes[1], title="sample")
main_methods.plot_heatmap(W_logreg[..., OUTPUT_NEURON_SELECTED].reshape((10, 10)), axis=axes[2], title="gradient")
main_methods.plot_heatmap(A_haufe_logreg.reshape((10, 10)), axis=axes[3], title="pattern Haufe 2014")
plt.suptitle("Logistic regression", size=16)

# MLP
W_mlp = main_methods.get_gradients(X, mlp, OUTPUT_NEURON_SELECTED, mlp_params)
A_haufe_mlp = main_methods.get_patterns(X, mlp, OUTPUT_NEURON_SELECTED, mlp_params, Sigma_X, Sigma_s_mlp_inv)
relevance_mlp = main_methods.easyLRP(X, mlp, OUTPUT_NEURON_SELECTED, mlp_params, epsilon=0.00001)

# plot real pattern, input point, weights and haufe pattern for MLP
fig, axes = plt.subplots(1, 5, figsize=(15, 5))
main_methods.plot_heatmap(A[:, OUTPUT_NEURON_SELECTED].reshape((10, 10)), axis=axes[0], title="True pattern")
main_methods.plot_heatmap(X.reshape((10, 10)), axis=axes[1], title="sample")
main_methods.plot_heatmap(W_mlp.reshape((10, 10)), axis=axes[2], title="gradient")
main_methods.plot_heatmap(A_haufe_mlp.reshape((10, 10)), axis=axes[3], title="pattern Haufe 2014")
main_methods.plot_heatmap(relevance_mlp.reshape((10, 10)), axis=axes[4], title="LRP")
plt.suptitle("MLP", size=16)


# CNN
W_cnn = main_methods.get_gradients(
    X.reshape(cnn_params["network_input_shape"]), cnn, OUTPUT_NEURON_SELECTED, cnn_params)
A_haufe_cnn = main_methods.get_patterns(
    X.reshape(cnn_params["network_input_shape"]), cnn, OUTPUT_NEURON_SELECTED, cnn_params,
    Sigma_X, Sigma_s_cnn_inv)
relevance_cnn = main_methods.easyLRP(
    X.reshape(cnn_params["network_input_shape"]), cnn, OUTPUT_NEURON_SELECTED, cnn_params)

# plot real pattern, input point, weights and haufe pattern for CNN
fig, axes = plt.subplots(1, 5, figsize=(15, 5))
main_methods.plot_heatmap(A[:, OUTPUT_NEURON_SELECTED].reshape((10, 10)), axis=axes[0], title="True pattern")
main_methods.plot_heatmap(X.reshape((10, 10)), axis=axes[1], title="sample")
main_methods.plot_heatmap(W_cnn.reshape((10, 10)), axis=axes[2], title="gradient")
main_methods.plot_heatmap(A_haufe_cnn.reshape((10, 10)), axis=axes[3], title="pattern Haufe 2014")
main_methods.plot_heatmap(relevance_cnn.reshape((10, 10)), axis=axes[4], title="LRP")
plt.suptitle("CNN", size=16)
plt.show()
