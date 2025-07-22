from typing import TypeVar
from utils.utils import ArrayType, get_array_module

ArrayOrTensor = TypeVar("ArrayOrTensor", ArrayType, "torch.Tensor") # type: ignore

ATOM_Crossection = {
    "C": 1.0,
    "CL": 2.0
    # 可以继续添加其他原子
}

# Cross section for C and CL atoms, pre-fitted Gaussian functions in Fourier space
def get_crossection(atom_type: str) -> float:
    return ATOM_Crossection[atom_type.upper()]
