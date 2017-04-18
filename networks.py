import lasagne


def build_custom_ringpredictor(params, input_var=None):
    # Input layer
    if params["bias_in_data"]:
        bias = None
        current_layer = lasagne.layers.InputLayer(shape=(None, params["input_dim"]+1),
                                        input_var=input_var, b=bias)
    else:
        bias = lasagne.init.Constant(0.)
        current_layer = lasagne.layers.InputLayer(shape=(None, params["input_dim"]),
                                        input_var=input_var, b=bias)

    # Hidden layer
    current_layer = lasagne.layers.DenseLayer(
        current_layer, num_units=2,
        W=precomputed_W
        b=None)

    l_out = lasagne.layers.FeatureWTALayer(current_layer, pool_size=1)
    return l_out
