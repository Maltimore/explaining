import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
na = np.newaxis

simulation_name = "two_output_neurons"
resultfile = "angles.dump"
resultpath = os.getcwd() + "/results/" + simulation_name + "/"
resultpath

results = pickle.load(open(resultpath + resultfile, "rb"))

