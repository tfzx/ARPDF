import abel
import cupy as cp
import numpy as np
from typing import List, Optional, Tuple, TypeVar


ArrayType = TypeVar("ArrayType", cp.ndarray, np.ndarray)

def box_shift(dx: ArrayType, box: Optional[List[float]] = None) -> ArrayType:
    """
    Shift the coordinates (dx) into the box. Both cupy and numpy array are supported.

    Return
    -----
    shifted_dx: xp.ndarray
    """
    if box is None:
        return dx
    xp = cp.get_array_module(dx)
    def box_to_cell(box: List[float]):
        lx, ly, lz, alpha, beta, gamma = box
        alpha_rad = np.radians(alpha)
        beta_rad = np.radians(beta)
        gamma_rad = np.radians(gamma)

        ax = lx
        bx, by = ly * np.cos(gamma_rad), ly * np.sin(gamma_rad)
        cx, cy = lz * np.cos(beta_rad), lz * (np.cos(alpha_rad) - np.cos(beta_rad) * np.cos(gamma_rad)) / np.sin(gamma_rad)
        cz = np.sqrt(np.clip(lz**2 - cx**2 - cy**2, 0, None))
        cell = xp.array([
            [ax, 0,  0 ],
            [bx, by, 0 ],
            [cx, cy, cz]
        ])
        return cell
    cell = box_to_cell(box)
    return dx - xp.matmul(xp.round(xp.matmul(dx, xp.linalg.inv(cell))), cell)

def generate_grids(rmax: float, N: int = 512, module_name: str = "cp") -> Tuple[ArrayType, ArrayType]:
    """
    Generate a 2D grid of N x N points in the range [-rmax, rmax]^2.

    Returns:
        (X, Y): Grid coordinates.
    """
    module_name = module_name.strip().lower()
    if module_name == "cp":
        xp = cp
    elif module_name == "np":
        xp = np
    else:
        raise ValueError("module_name must be 'cp' or 'np'")
    x = xp.linspace(-rmax, rmax, N, dtype=xp.float32)
    y = xp.linspace(-rmax, rmax, N, dtype=xp.float32)
    return xp.meshgrid(x, y)



abel_inv_mat_cache = {}

def abel_inversion(image: ArrayType) -> ArrayType:
    xp = cp.get_array_module(image)
    global abel_inv_mat_cache
    def get_matrix(n, xp_name: str):
        if (n, xp_name) in abel_inv_mat_cache:
            return abel_inv_mat_cache[(n, xp_name)]
        trans_mat = abel.Transform(np.eye(n), method='basex', direction='inverse', transform_options={"verbose": False}).transform
        trans_mat = xp.array(trans_mat, dtype=xp.float32)
        abel_inv_mat_cache[(n, xp_name)] = trans_mat
        return trans_mat
    return image @ get_matrix(image.shape[1], xp.__name__)

def cosine_similarity(ARPDF1: ArrayType, ARPDF2: ArrayType) -> float:
    """
    Compute the cosine similarity between two ARPDF.
    """
    xp = cp.get_array_module(ARPDF1)
    _x1 = xp.array(ARPDF1, dtype=xp.float32)
    _x2 = xp.array(ARPDF2, dtype=xp.float32)
    return xp.vdot(_x1, _x2) / (xp.linalg.norm(_x1) * xp.linalg.norm(_x2) + 1e-8)

if __name__ == "__main__":
    # Test box_shift
    arr_np = np.random.rand(5, 3)
    arr_cp = cp.random.rand(5, 3)
    res_np = box_shift(arr_np, box=[10, 10, 10, 90, 90, 90])
    res_cp = box_shift(arr_cp, box=[10, 10, 10, 90, 90, 90])
    assert cp.get_array_module(res_np) is np
    assert cp.get_array_module(res_cp) is cp
    # Test generate_grids
    X_cp, Y_cp = generate_grids(10, 512)
    X_np, Y_np = generate_grids(10, 512, "np")
    assert cp.get_array_module(X_cp) is cp
    assert cp.get_array_module(X_np) is np
    # Test abel_inversion
    image_np = np.random.rand(512, 512)
    image_cp = cp.random.rand(512, 512)
    res_np = abel_inversion(image_np)
    assert (512, "numpy") in abel_inv_mat_cache
    res_cp = abel_inversion(image_cp)
    assert (512, "cupy") in abel_inv_mat_cache
    assert cp.get_array_module(res_np) is np
    assert cp.get_array_module(res_cp) is cp
    res_np = abel_inversion(image_np)
    res_cp = abel_inversion(image_cp)
    assert cp.get_array_module(res_np) is np
    assert cp.get_array_module(res_cp) is cp
    # Test cosine_similarity
    cosine_similarity(res_np, res_np)
    cosine_similarity(res_cp, res_cp)
