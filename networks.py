import lasagne


def build_custom_ringpredictor(params, input_var=None):
    # Input layer
    current_layer = lasagne.layers.InputLayer(shape=(None, params["input_dim"]),
                                    input_var=input_var)

    # Hidden layer
    current_layer = lasagne.layers.DenseLayer(
                                current_layer,
                                num_units=2,
                                W=precomputed_W)

    l_out = lasagne.layers.FeatureWTALayer(current_layer, pool_size=1)
    return l_out
