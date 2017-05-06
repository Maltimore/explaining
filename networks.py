import lasagne
import numpy as np


def build_custom_ringpredictor(params, input_var=None):
    # Input layer
    current_layer = lasagne.layers.InputLayer(
            shape=(None,) + params["network_input_shape"][1:], input_var=input_var)

    # create coordinates of blob centers on ring
    n_centers = 8
    radius = 1
    class_means = radius*np.array(
        [[np.cos(i*2.*np.pi/n_centers), np.sin(i*2.*np.pi/n_centers)] for i in range(n_centers)])
    class_1_centers = class_means[0::2]
    class_2_centers = class_means[1::2]
    precomputed_W = np.concatenate((class_1_centers, class_2_centers)).T

    # layer that computes cos(input_point, class_center)
    current_layer = lasagne.layers.DenseLayer(
                                current_layer,
                                num_units=8,
                                W=precomputed_W)
    # layer that figures
    current_layer = lasagne.layers.FeaturePoolLayer(current_layer, pool_size=4)
    l_out = lasagne.layers.FeatureWTALayer(current_layer, pool_size=2)
    return l_out


def build_mlp(params, input_var=None):
    current_layer = lasagne.layers.InputLayer(
            shape=(None,) + params["network_input_shape"][1:], input_var=input_var)
    # Hidden layers
    for layer_size in params["layer_sizes"]:
        if layer_size == 0:
            raise ValueError("Layer size 0 requested")
        current_layer = lasagne.layers.DenseLayer(
            current_layer, num_units=layer_size,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    l_out = lasagne.layers.DenseLayer(
            current_layer, num_units=params["n_classes"],
            nonlinearity=lasagne.nonlinearities.softmax)
    return l_out


def build_cnn(params, input_var):
    # Input layer
    current_layer = lasagne.layers.InputLayer(
            shape=(None,) + params["network_input_shape"][1:], input_var=input_var)
    # Hidden layers
    n_filters = 15
    current_layer = lasagne.layers.Conv2DLayer(
        current_layer, num_filters=n_filters, filter_size=(3, 3), pad="same",
        nonlinearity=lasagne.nonlinearities.rectify)
    n_filters = 4
    current_layer = lasagne.layers.Conv2DLayer(
        current_layer, num_filters=n_filters, filter_size=(3, 3),
        pad="same", nonlinearity=lasagne.nonlinearities.rectify)

    # Output layer
    l_out = lasagne.layers.DenseLayer(
            current_layer, num_units=params["n_classes"],
            nonlinearity=lasagne.nonlinearities.softmax)
    return l_out
