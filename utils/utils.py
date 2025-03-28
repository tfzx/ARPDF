import abel
import cupy as cp
import numpy as np
from typing import List, Optional, Tuple

def box_shift(dx: cp.ndarray, box: Optional[List[float]] = None) -> cp.ndarray:
    """
    Shift the coordinates (dx) into the box.

    Return
    -----
    shifted_dx: cp.ndarray
    """
    if box is None:
        return dx
    
    def box_to_cell(box: List[float]):
        lx, ly, lz, alpha, beta, gamma = box
        alpha_rad = np.radians(alpha)
        beta_rad = np.radians(beta)
        gamma_rad = np.radians(gamma)

        ax = lx
        bx, by = ly * np.cos(gamma_rad), ly * np.sin(gamma_rad)
        cx, cy = lz * np.cos(beta_rad), lz * (np.cos(alpha_rad) - np.cos(beta_rad) * np.cos(gamma_rad)) / np.sin(gamma_rad)
        cz = np.sqrt(np.clip(lz**2 - cx**2 - cy**2, 0, None))
        cell = cp.array([
            [ax, 0,  0 ],
            [bx, by, 0 ],
            [cx, cy, cz]
        ])
        return cell
    cell = box_to_cell(box)
    return dx - cp.matmul(cp.round(cp.matmul(dx, cp.linalg.inv(cell))), cell)

def generate_grids(rmax: float, N: int = 512) -> Tuple[Tuple[cp.ndarray, cp.ndarray], float]:
    """
    Generate a 2D grid of N x N points in the range [-rmax, rmax]^2.

    Returns:
        (X, Y): Grid coordinates.
    """
    x = cp.linspace(-rmax, rmax, N, dtype=cp.float32)
    y = cp.linspace(-rmax, rmax, N, dtype=cp.float32)
    return cp.meshgrid(x, y)



abel_inv_mat_cache = {}

def abel_inversion(image: cp.ndarray):
    global abel_inv_mat_cache
    def get_matrix(n):
        if n in abel_inv_mat_cache:
            return abel_inv_mat_cache[n]
        trans_mat = abel.Transform(np.eye(n), method='basex', direction='inverse', transform_options={"verbose": False}).transform
        trans_mat = cp.array(trans_mat, dtype=cp.float32)
        abel_inv_mat_cache[n] = trans_mat
        return trans_mat
    return image @ get_matrix(image.shape[1])

def cosine_similarity(ARPDF1, ARPDF2):
    """
    Compute the cosine similarity between two ARPDF.
    """
    _x1 = cp.array(ARPDF1, dtype=cp.float32)
    _x2 = cp.array(ARPDF2, dtype=cp.float32)
    return cp.vdot(_x1, _x2) / (cp.linalg.norm(_x1) * cp.linalg.norm(_x2) + 1e-8)
