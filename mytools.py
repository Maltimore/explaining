import numpy as np
import argparse
import sys
import os


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))



def get_CLI_parameters(argv):
    # command line interface
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--loss_choice", default="categorical_crossentropy")
    parser.add_argument("--noise_scale", default=0.4, type=float)
    parser.add_argument("-e", "--epochs", default=50, type=int)
    parser.add_argument("-m", "--model", default="cnn")
    parser.add_argument("--layer_sizes", default="20, 20")
    parser.add_argument("-p", "--do_plotting", default=True)
    parser.add_argument("--verbose", default=True)
    parser.add_argument("-d", "--data", default="horseshoe")
    parser.add_argument("-c", "--n_classes", default="4", type=int)
    parser.add_argument("--n_output_units", default="1", type=int)
    parser.add_argument("-b", "--bias_in_data", default=False)
    parser.add_argument("-r", "--remote", default=False)
    parser.add_argument("-n", "--name", default="default")

    params = vars(parser.parse_args(argv[1:]))
    params["N_train"] = 5000
    params["N_val"] = 200
    params["N_test"] = 200
    params["minibatch_size"] = 30
    params["input_dim"] = 100
    params["input_shape"] = [10, 10]
    params["horseshoe_distractors"] = True
    params["specific_dataclass"] = None

    # extract layer sizes from input string
    layer_list = []
    for size in params["layer_sizes"].split(","):
        layer_list.append(int(size))
    params["layer_sizes"] = layer_list

    # determine the directories to save data
    params["program_dir"] = os.getcwd()
    params["results_dir"] = params["program_dir"] + "/results/" + params["name"]
    params["plots_dir"] = params["program_dir"] + "/plots/" + params["name"]
    if not os.path.exists(params["results_dir"]):
        os.makedirs(params["results_dir"])
    if not os.path.exists(params["plots_dir"]):
        os.makedirs(params["plots_dir"])

    return params

