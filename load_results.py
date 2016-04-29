import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
na = np.newaxis

colors = ["blue", "red", "orange", "green"]

simulation_name = "big_datasets"
resultfile = "angles.dump"
resultpath = os.getcwd() + "/results/" + simulation_name + "/"
resultpath

results = pickle.load(open(resultpath + resultfile, "rb"))
results

N_vals = results["N_vals"]
angles = results["angles"] * 360/(2*np.pi)
params = results["params"]

plt.figure()
for idx in np.arange(params["n_classes"]):
    plt.scatter(N_vals, angles[idx, :], c=colors[idx], s=100)
plt.xlabel("N training points")
plt.ylabel("angle of one weight vector")
plt.show()
