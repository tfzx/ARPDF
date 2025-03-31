import cupy as cp
import numpy as np
from utils.utils import ArrayType

AFF = {
    "C": lambda exp_func, S: 22.71820056 * exp_func(-S**2 / (2.93693829**2)) + 8.95935118 * exp_func(-S**2 / (8.59208848**2)),
    "CL": lambda exp_func, S: 44.84903889 * exp_func(-S**2 / (3.10005611**2)) + 17.27260847 * exp_func(-S**2 / (9.6431349**2)),
}

# Atomic Form Factor map for C and CL atoms, pre-fitted Gaussian functions in Fourier space
def calc_AFF(atom_type: str, S: ArrayType) -> ArrayType:
    xp = cp.get_array_module(S)
    return AFF[atom_type](xp.exp, S)