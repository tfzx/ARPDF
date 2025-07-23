from typing import TypeVar
from utils.utils import ArrayType, get_array_module

ArrayOrTensor = TypeVar("ArrayOrTensor", ArrayType, "torch.Tensor") # type: ignore

AFF = {
    "C": lambda exp_func, S: 22.71820056 * exp_func(-S**2 / (2.93693829**2)) + 8.95935118 * exp_func(-S**2 / (8.59208848**2)),
    "CL": lambda exp_func, S: 44.84903889 * exp_func(-S**2 / (3.10005611**2)) + 17.27260847 * exp_func(-S**2 / (9.6431349**2)),
    "N": lambda exp_func, S: 21.01362962 * exp_func(-S**2 / (3.53511542**2)) + 7.06608494 * exp_func(-S**2 / (10.94238199**2)),
}

# Atomic Form Factor map for C and CL atoms, pre-fitted Gaussian functions in Fourier space
def calc_AFF(atom_type: str, S: ArrayOrTensor) -> ArrayOrTensor:
    xp = get_array_module(S)
    return AFF[atom_type.upper()](xp.exp, S)

ATOM_Crossection = {
    "C": 1.0,
    "CL": 2.0
    # 可以继续添加其他原子
}

# Cross section for C and CL atoms, pre-fitted Gaussian functions in Fourier space
def get_crossection(atom_type: str) -> float:
    return ATOM_Crossection[atom_type.upper()]