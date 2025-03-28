import cupy as cp
import numpy as np

# Atomic Form Factor map for C and CL atoms, pre-fitted Gaussian functions in Fourier space
def generate_AFF_map(exp_func):
    return {
        "C": lambda S: 22.71820056 * exp_func(-S**2 / (2.93693829**2)) + 8.95935118 * exp_func(-S**2 / (8.59208848**2)),
        "CL": lambda S: 44.84903889 * exp_func(-S**2 / (3.10005611**2)) + 17.27260847 * exp_func(-S**2 / (9.6431349**2)),
    }

AFF_map_cp = generate_AFF_map(cp.exp)
AFF_map_np = generate_AFF_map(np.exp)