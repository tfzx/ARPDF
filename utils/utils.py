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

def generate_grids(xy_range, N: int = 512, M: Optional[int] = None, use_cupy: bool = False) -> Tuple[ArrayType, ArrayType]:
    """
    Generate a 2D grid of N x N points in the range [-rmax, rmax]^2.

    Returns:
        (X, Y): Grid coordinates.
    """
    if use_cupy:
        xp = cp
    else:
        xp = np
    if isinstance(xy_range, (float, int)):
        xmin, xmax, ymin, ymax = [-xy_range, xy_range, -xy_range, xy_range]
    elif len(xy_range) == 2:
        xmin, xmax = xy_range
        ymin, ymax = xy_range
    else:
        xmin, xmax, ymin, ymax = xy_range
    if M is None:
        M = N
    x = xp.linspace(xmin, xmax, N, dtype=xp.float32)
    y = xp.linspace(ymin, ymax, M, dtype=xp.float32)
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

def _to_cupy(*args) -> list:
    out = []
    for arg in args:
        if isinstance(arg, cp.ndarray):
            out.append(arg)
        elif isinstance(arg, np.ndarray):
            out.append(cp.array(arg))
        elif isinstance(arg, list):
            out.append(_to_cupy(*arg))
        elif isinstance(arg, tuple):
            out.append(tuple(_to_cupy(*arg)))
        elif isinstance(arg, dict):
            out.append({k: _to_cupy(v)[0] for k, v in arg.items()})
        else:
            out.append(arg)
    return out

def _to_numpy(*args) -> list:
    out = []
    for arg in args:
        if isinstance(arg, cp.ndarray):
            out.append(arg.get())
        elif isinstance(arg, np.ndarray):
            out.append(arg)
        elif isinstance(arg, list):
            out.append(_to_numpy(*arg))
        elif isinstance(arg, tuple):
            out.append(tuple(_to_numpy(*arg)))
        elif isinstance(arg, dict):
            out.append({k: _to_numpy(v)[0] for k, v in arg.items()})
        else:
            out.append(arg)
    return out

def to_cupy(*args):
    out = _to_cupy(*args)
    return out[0] if len(out) == 1 else out

def to_numpy(*args):
    out = _to_numpy(*args)
    return out[0] if len(out) == 1 else out

def resize_ARPDF(ARPDF_exp, original_grids, grid_size = None):
    raise NotImplementedError("Not implemented yet")

def preprocess_ARPDF(
        ARPDF_raw: np.ndarray, 
        original_range, 
        rmax = None, 
        new_grid_size = None, 
        max_intensity = 1.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    预处理ARPDF数据，包括范围限制、网格重采样和强度归一化。
    
    参数:
    ARPDF_raw: 原始ARPDF数据数组。
    original_range: 原始数据的范围。
    rmax: 数据处理的最大范围，如果未提供，则默认为original_range的最小绝对值。
    new_grid_size: 新的网格尺寸，如果提供，则重采样数据到新的网格尺寸。
    max_intensity: 最大强度值，用于数据归一化，默认为1.0。
    
    返回:
    X: 处理后的X轴网格数据。
    Y: 处理后的Y轴网格数据。
    ARPDF: 处理后的ARPDF数据。
    """
    if rmax is None:
        rmax = np.min(np.abs(original_range))
    N, M = ARPDF_raw.shape
    X_ori, Y_ori = generate_grids(original_range, N, M, use_cupy=False)
    x_mask = np.abs(X_ori[0, :]) <= rmax
    y_mask = np.abs(Y_ori[:, 0]) <= rmax
    X = X_ori[y_mask, :][:, x_mask]
    Y = Y_ori[y_mask, :][:, x_mask]
    ARPDF = ARPDF_raw[y_mask, :][:, x_mask]
    ARPDF = ARPDF_raw / np.max(np.abs(ARPDF_raw)) * max_intensity
    if new_grid_size is not None:
        X, Y, ARPDF = resize_ARPDF(ARPDF, (X, Y), new_grid_size)
    return X, Y, ARPDF

if __name__ == "__main__":
    # Test box_shift
    arr_np = np.random.rand(5, 3)
    arr_cp = cp.random.rand(5, 3)
    res_np = box_shift(arr_np, box=[10, 10, 10, 90, 90, 90])
    res_cp = box_shift(arr_cp, box=[10, 10, 10, 90, 90, 90])
    assert cp.get_array_module(res_np) is np
    assert cp.get_array_module(res_cp) is cp
    # Test generate_grids
    X_np, Y_np = generate_grids(10, 512)
    X_cp, Y_cp = generate_grids(10, 512, use_cupy=True)
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
    arr_np = np.arange(3)
    arr_cp = cp.arange(3, 6)
    print(_to_cupy(arr_np))
    assert isinstance(_to_cupy(arr_np)[0], cp.ndarray)
    print(_to_cupy(arr_np, arr_cp))
    print(_to_cupy((arr_np, arr_cp)))
    print(_to_cupy([arr_np, arr_cp]))
    res = _to_cupy({"a": arr_np, "b": arr_cp}, arr_np)
    print(res)
    assert isinstance(res[0]["a"], cp.ndarray)
    assert isinstance(res[1], cp.ndarray)
    res = _to_cupy({"c": ({"a": arr_np, "b": arr_cp}, arr_cp), "d": [arr_np, arr_cp]})
    print(res)
    assert isinstance(res[0]["c"][0]["a"], cp.ndarray)
    assert isinstance(res[0]["d"][0], cp.ndarray)
    print(to_cupy([arr_np, arr_cp]))