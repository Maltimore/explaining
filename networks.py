import lasagne
import numpy as np


def build_custom_ringpredictor(params, input_var=None):
    # Input layer
    current_layer = lasagne.layers.InputLayer(shape=(None, params["input_dim"]),
                                    input_var=input_var)

    # create coordinates of blob centers on ring
    n_centers = 8
    radius = 1
    class_means = radius*np.array([[np.cos(i*2.*np.pi/n_centers),np.sin(i*2.*np.pi/n_centers)] for i in range(n_centers)])
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
