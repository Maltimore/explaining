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
        current_layer, num_units=layer_size,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(),
        b=bias)

    l_out = lasagne.layers.DenseLayer(
            current_layer, num_units=params["n_output_units"],
            nonlinearity=lasagne.nonlinearities.linear)
    return l_out
