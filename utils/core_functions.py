import sys
import numpy as np
from typing import TypeVar, Optional
import abel
from scipy.special import i0, i0e

ArrayType = TypeVar("ArrayType", np.ndarray, "cupy.ndarray") # type: ignore

def compute_rice_weights(R_grid, r_vals, sigma):
    """
    预计算 Rice 分布权重矩阵，输出 shape = (grid_size, num_atoms)
    """
    R_grid = np.asarray(R_grid).reshape(-1)      # shape = (grid_size,)
    r_vals = np.asarray(r_vals).reshape(-1)      # shape = (num_atoms,)
    
    R = R_grid[:, None]                          # shape = (grid_size, 1)
    r = r_vals[None, :]                          # shape = (1, num_atoms)

    delta_R = (R - r)**2
    I0_term = i0(R * r / sigma**2)
    weights = np.exp(-delta_R / (2 * sigma**2)) * I0_term

    return weights.astype(np.float32).reshape(-1)  # flatten 后传给 CUDA

def get_array_module(x):
    module_name = x.__class__.__module__.split('.')[0]
    if module_name not in ["cupy", "torch", "numpy"]:
        raise ValueError(f"Unsupported array type: {module_name}")
    return sys.modules[module_name]

def to_cupy(*args):
    import cupy as cp
    def _to_cupy(x):
        if isinstance(x, cp.ndarray):
            return x
        elif isinstance(x, np.ndarray):
            return cp.array(x)
        elif isinstance(x, list):
            return [_to_cupy(xi) for xi in x]
        elif isinstance(x, tuple):
            return tuple(_to_cupy(xi) for xi in x)
        elif isinstance(x, dict):
            return {k: _to_cupy(v) for k, v in x.items()}
        else:
            return x
    out = _to_cupy(args)
    return out[0] if len(out) == 1 else out

def to_numpy(*args):
    if "cupy" not in sys.modules:
        return args[0] if len(args) == 1 else args
    import cupy as cp
    def _to_numpy(x):
        if isinstance(x, np.ndarray):
            return x
        elif isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
        elif isinstance(x, list):
            return [_to_numpy(xi) for xi in x]
        elif isinstance(x, tuple):
            return tuple(_to_numpy(xi) for xi in x)
        elif isinstance(x, dict):
            return {k: _to_numpy(v) for k, v in x.items()}
        else:
            return x
    out = _to_numpy(args)
    return out[0] if len(out) == 1 else out


abel_inv_mat_cache = {}

def abel_inversion_basex(image: ArrayType) -> ArrayType:
    xp = get_array_module(image)
    global abel_inv_mat_cache
    def get_matrix(n, xp_name: str):
        key = (n, xp_name)
        if key in abel_inv_mat_cache:
            return abel_inv_mat_cache[key]
        trans_mat = abel.Transform(np.eye(n), method='basex', direction='inverse', transform_options={"verbose": False}).transform
        trans_mat = xp.array(trans_mat, dtype=xp.float32)
        abel_inv_mat_cache[key] = trans_mat
        return trans_mat
    return image @ get_matrix(image.shape[1], xp.__name__)

def abel_inversion_rbasex(image: ArrayType) -> ArrayType:
    xp = get_array_module(image)
    inverse_Abel, _ = abel.rbasex.rbasex_transform(to_numpy(image), direction='inverse', order=2)
    return xp.array(inverse_Abel, dtype=xp.float32)

def abel_inversion(image: ArrayType) -> ArrayType:
    return abel_inversion_rbasex(image)




_generate_field_C = r'''
extern "C" __global__
void generate_field_kernel(float* X, float* Y, float* r_vals, float* theta_vals, float* output,
                           int num_atoms, int grid_size, float delta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= grid_size) return;

    float x = X[idx];
    float y = Y[idx];
    float R = sqrtf(x * x + y * y);
    float sum_field = 0.0f;

    if (R >= 1e-10f) {
        float r1 = fmaxf(R - delta / 2, 0.0f);
        float r2 = fminf(R + delta / 2, 2 * R);
        float denom = (r2 - r1) * R;

        for (int i = 0; i < num_atoms; i++) {
            float r = r_vals[i];
            float theta = theta_vals[i];

            float C = 0.5f * (3 * cosf(theta) * cosf(theta) - 1);
            float F1 = sqrtf(fmaxf(r * r - r1 * r1, 0.0f));
            float F2 = sqrtf(fmaxf(r * r - r2 * r2, 0.0f));

            sum_field += (1 + C * (3 * (y / r) * (y / r) - 1)) * (F1 - F2) / (denom * r);
        }
    } else {
        for (int i = 0; i < num_atoms; i++) {
            float r = r_vals[i];
            float theta = theta_vals[i];
            
            float C = 0.5f * (3 * cosf(theta) * cosf(theta) - 1);
            
            sum_field += (1 + C * (3 * (y / r) * (y / r) - 1)) / (r * r);
        }
    }
    output[idx] = sum_field;
}
'''

generate_field_kernel = None


def generate_field_cuda(X, Y, r_vals, theta_vals, delta):
    global generate_field_kernel
    import cupy as cp
    if generate_field_kernel is None:
        generate_field_kernel = cp.RawKernel(_generate_field_C, 'generate_field_kernel')
    X = cp.array(X, dtype=cp.float32)
    Y = cp.array(Y, dtype=cp.float32)
    r_vals = cp.array(r_vals, dtype=cp.float32)
    theta_vals = cp.array(theta_vals, dtype=cp.float32)
    delta = cp.array(delta, dtype=cp.float32).get()

    grid_size = X.size
    output = cp.zeros_like(X, dtype=cp.float32)

    threads_per_block = 256
    blocks_per_grid = (grid_size + threads_per_block - 1) // threads_per_block


    generate_field_kernel(
        (blocks_per_grid,), (threads_per_block,),
        (X, Y, r_vals, theta_vals, output, r_vals.size, grid_size, delta)
    )
    return output

def f_smooth(r: np.ndarray, rmax: np.ndarray, delta=0.01) -> np.ndarray:
    """
    Smoothed version of `f(r) = 1 / sqrt(max(rmax^2 - r^2, 0))`

    :param r: np.ndarray, radius
    :param rmax: np.ndarray, maximum radius
    :param delta: float, smoothing width, default is 0.01
    :return values: np.ndarray, smoothed function value
    """
    r1 = np.maximum(r - delta/2, 0.0)
    r2 = np.minimum(r + delta/2, 2*r)
    EPS = 1e-10
    F = lambda x: np.sqrt(np.clip(rmax**2 - np.clip(x, 0, None)**2, EPS, None))
    vals = (F(r1) - F(r2)) / np.clip((r2 - r1) * r, 2*EPS**2, None)
    vals = np.where(r <= EPS, 1 / rmax, vals)
    return vals

def generate_field_numpy(
        X: np.ndarray, 
        Y: np.ndarray, 
        r_vals: np.ndarray, 
        theta_vals: np.ndarray, 
        delta: float = 0.01, 
        batch_size: int = 128
    ) -> np.ndarray:
    """
    Generate the field function on the grid defined by R and Y:
    `f(x, y; r, \\theta) = {1 + [3(cos \\theta)^2 - 1][3(y / r)^2 - 1] / 2} / sqrt[max(r^2 - x^2 - y^2, 0)] / r`.
    And sum over all (r, \\theta).

    parameters
    ----------
    X, Y: np.ndarray
        coordinate of the grids
    r_vals: np.ndarray
        radius values
    theta_vals: np.ndarray
        theta values
    delta: float
        smoothing width, default is 0.01.
    batch_size: int
        batch size for processing the field, default is 128.
    
    Returns
    -------
    field: np.ndarray
        The sum of the fields over all (r, \\theta).
    """
    final_field = np.zeros_like(Y)
    num_atoms = r_vals.shape[0]
    r_batches = np.split(r_vals, range(batch_size, num_atoms, batch_size), axis=0)
    theta_batches = np.split(theta_vals, range(batch_size, num_atoms, batch_size), axis=0)
    
    R_view = np.sqrt(X**2 + Y**2).reshape(1, -1)
    Y_view = Y.reshape(1, -1)
    
    for r, theta in zip(r_batches, theta_batches):
        r = r.reshape(-1, 1)
        theta = theta.reshape(-1, 1)
        C = 0.5 * (3 * np.cos(theta)**2 - 1)
        contribution = ((1 + C * (3 * (Y_view / r)**2 - 1)) / r * f_smooth(R_view, r, delta=delta)).sum(axis=0)
        final_field += contribution.reshape(Y.shape)
    return final_field

def generate_field(X: ArrayType, Y: ArrayType, r_vals: ArrayType, theta_vals: ArrayType, delta: np.float32) -> ArrayType:
    """
    Wrapper for field generation.
    """
    xp = get_array_module(X)
    
    # Get quadrant size
    nx, ny = X.shape
    qx, qy = (nx + 1) // 2, (ny + 1) // 2
    rx, ry = nx - qx, ny - qy
    
    # Calculate for first quadrant only
    X_quad = X[:qx, :qy]
    Y_quad = Y[:qx, :qy]
    
    if xp.__name__ == "numpy":
        quad_result = generate_field_numpy(X_quad, Y_quad, r_vals, theta_vals, delta, batch_size=128)
    elif xp.__name__ == "cupy":
        quad_result = generate_field_cuda(X_quad, Y_quad, r_vals, theta_vals, delta)
    else:
        raise ValueError(f"Unsupported array type: {xp.__name__}")
        
    # Reconstruct full result using symmetry
    Top = xp.concatenate((quad_result, xp.fliplr(quad_result[:, :ry])), axis=1)
    Bottom = xp.flipud(Top[:rx, :])
    return xp.concatenate((Top, Bottom), axis=0)


_generate_field_polar_C = r'''
extern "C" __global__
void generate_field_polar_kernel(float* R_grid, float* Phi_grid, float* r_vals, float* theta_vals,
                                 float* weights, float* output,
                                 int num_atoms, int grid_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= grid_size) return;

    float R = R_grid[idx];
    float phi = Phi_grid[idx];

    float sum_field = 0.0f;

    for (int i = 0; i < num_atoms; i++) {
        float r = r_vals[i];                   
        float theta = theta_vals[i];
        float weight = weights[idx * num_atoms + i]; 

        float C = 0.5f * (3.0f * cosf(theta) * cosf(theta) - 1.0f);
        float phi_term = 0.5f * (3.0f * cosf(phi) * cosf(phi) - 1.0f);

        sum_field += weight * (1.0f + 2.0f * C * phi_term) * r;
    }

    output[idx] = sum_field;
}
'''

'''
def prepare_field_polar_cache(R, Phi, r_vals_all, delta):
    import cupy as cp
    import numpy as np
    from scipy.special import i0e

    global _polar_cache
    R_gpu = cp.array(R, dtype=cp.float32)
    Phi_gpu = cp.array(Phi, dtype=cp.float32)

    R_np = R_gpu.get().reshape(-1, 1)
    r_np = r_vals_all.get().reshape(1, -1)

    delta = float(delta)
    weights_np = np.exp(- (R_np - r_np)**2 / (2 * delta**2)) * i0e(R_np * r_np / delta)
    weights_np = weights_np.astype(np.float32).ravel()
    weights_gpu = cp.array(weights_np)

    _polar_cache = {
        "R": R_gpu,
        "Phi": Phi_gpu,
        "weights": weights_gpu,
        "offset": 0  # 用来记录偏移量
    }

def approx_I0e(x):
    import cupy as cp
    x2 = x * x
    return cp.exp(-x) * (
        1 +
        x2 / 4 +
        x2**2 / 64 +
        x2**3 / 2304 +
        x2**4 / 147456 +
        x2**5 / 14745600  # 添加到 x^10 项
    )
'''
def generate_field_polar_cuda(R, Phi, r_vals, theta_vals, sigma0):
    import cupy as cp
    global generate_field_polar_kernel
    if 'generate_field_polar_kernel' not in globals():
        generate_field_polar_kernel = cp.RawKernel(_generate_field_polar_C, 'generate_field_polar_kernel')

    R = cp.array(R, dtype=cp.float32)
    Phi = cp.array(Phi, dtype=cp.float32)
    r_vals = cp.array(r_vals, dtype=cp.float32)
    theta_vals = cp.array(theta_vals, dtype=cp.float32)

    #import numpy as np
    #from scipy.special import i0e
    #R_np = R.get().reshape(-1, 1)  # (grid_size, 1)
    #r_np = r_vals.get().reshape(1, -1)  # (1, num_atoms)

    #delta = float(delta) 

    #weights_np = np.exp(- (R_np - r_np)**2 / (2 * delta**2)) * i0e(R_np * r_np / (delta))
    #weights_np = weights_np.astype(np.float32).ravel()
    #weights = cp.array(weights_np)

    #output = cp.zeros_like(R, dtype=cp.float32)
    #grid_size = R.size
    #num_atoms = r_vals.size

    sigma0 = cp.float32(sigma0)

    # 在 GPU 上直接计算权重（不再 get 到 CPU）
    R_mat = R.reshape(-1, 1)  # shape: (grid_size, 1)
    r_mat = r_vals.reshape(1, -1)  # shape: (1, num_atoms)
    sigma02 = sigma0 ** 2

    exp_term = cp.exp(- (R_mat - r_mat) ** 2 / (2 * sigma02))
    #i0_term = approx_I0e(R_mat * r_mat / delta)  # ✅ 替代 scipy.special.i0e
    #weights = (exp_term * i0_term).astype(cp.float32).reshape(-1)
    weights = exp_term.astype(cp.float32).reshape(-1)

    # 继续在 GPU 上准备输出和参数
    output = cp.zeros_like(R, dtype=cp.float32)
    grid_size = R.size
    num_atoms = r_vals.size

    threads_per_block = 256
    blocks_per_grid = (grid_size + threads_per_block - 1) // threads_per_block


    generate_field_polar_kernel(
        (blocks_per_grid,), (threads_per_block,),
        (R, Phi, r_vals, theta_vals, weights, output, num_atoms, grid_size)
    )

    return output

'''
def generate_field_polar_cuda(R, Phi, r_vals, theta_vals, delta):
    import cupy as cp
    global generate_field_polar_kernel, _polar_cache

    if 'generate_field_polar_kernel' not in globals():
        generate_field_polar_kernel = cp.RawKernel(_generate_field_polar_C, 'generate_field_polar_kernel')

    if '_polar_cache' not in globals():
        raise RuntimeError("Must call prepare_field_polar_cache(...) before generate_field_polar_cuda")

    R_gpu = _polar_cache["R"]
    Phi_gpu = _polar_cache["Phi"]
    weights_gpu = _polar_cache["weights"]

    n = len(r_vals)
    offset = _polar_cache["offset"]
    _polar_cache["offset"] += n  # 更新偏移量

    r_vals_gpu = cp.array(r_vals, dtype=cp.float32)
    theta_vals_gpu = cp.array(theta_vals, dtype=cp.float32)
    weights = weights_gpu[offset:offset + n]

    output = cp.zeros_like(R_gpu, dtype=cp.float32)
    grid_size = R_gpu.size
    num_atoms = r_vals_gpu.size

    threads_per_block = 256
    blocks_per_grid = (grid_size + threads_per_block - 1) // threads_per_block

    generate_field_polar_kernel(
        (blocks_per_grid,), (threads_per_block,),
        (R_gpu, Phi_gpu, r_vals_gpu, theta_vals_gpu, weights, output, num_atoms, grid_size)
    )

    return output
'''

import numpy as np

def generate_field_polar_numpy(
    R_grid, Phi_grid, r_vals, theta_vals, sigma0, grid_shape=None, batch_size=128
):
    """
    CPU 版本：极坐标场计算，支持分批计算，类似于 CUDA 版本

    Args:
        R_grid: ndarray (M,) 或 (Nr, Nphi) - 极坐标半径网格
        Phi_grid: ndarray (M,) 或 (Nr, Nphi) - 极坐标角度网格
        r_vals: ndarray (N,) - 每个原子的极坐标半径
        theta_vals: ndarray (N,) - 每个原子的极坐标角度
        sigma0: float - 高斯展开的宽度参数
        grid_shape: tuple or None - 如果 R_grid 和 Phi_grid 是展平的一维数组，需指定二维网格形状
        batch_size: int - 每次处理的原子数量，防止内存占用过大

    Returns:
        output: ndarray，二维数组，shape = grid_shape 或 R_grid.shape（如果R_grid本身是二维）
    """

    R_grid = np.asarray(R_grid, dtype=np.float32)
    Phi_grid = np.asarray(Phi_grid, dtype=np.float32)
    r_vals = np.asarray(r_vals, dtype=np.float32)
    theta_vals = np.asarray(theta_vals, dtype=np.float32)

    sigma2 = sigma0 ** 2

    # 如果输入是二维网格，展平便于广播计算
    if R_grid.ndim == 2 and Phi_grid.ndim == 2:
        Nr, Nphi = R_grid.shape
        R_flat = R_grid.reshape(-1)  # (M,)
        Phi_flat = Phi_grid.reshape(-1)  # (M,)
    else:
        R_flat = R_grid
        Phi_flat = Phi_grid
        if grid_shape is None:
            raise ValueError("grid_shape must be provided if input grids are 1D")

        Nr, Nphi = grid_shape

    M = R_flat.size
    N = r_vals.size

    output_flat = np.zeros(M, dtype=np.float32)

    # 分批处理原子，防止内存过大
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        r_batch = r_vals[start:end].reshape(1, -1)      # (1, batch_size)
        theta_batch = theta_vals[start:end].reshape(1, -1)  # (1, batch_size)

        # 计算权重
        R_mat = R_flat.reshape(-1, 1)                   # (M,1)
        weights = np.exp(- (R_mat - r_batch) ** 2 / (2 * sigma2))  # (M, batch_size)

        # 计算角度因子
        C_theta = 0.5 * (3 * np.cos(theta_batch) ** 2 - 1)  # (1, batch_size)
        C_phi = 0.5 * (3 * np.cos(Phi_flat) ** 2 - 1)       # (M,)

        factor = 1.0 + 2.0 * (C_phi.reshape(-1,1) * C_theta)  # (M, batch_size)

        # 计算场贡献
        field_batch = weights * factor * r_batch               # (M, batch_size)

        # 累加所有原子批次贡献
        output_flat += np.sum(field_batch, axis=1)

    # 还原成二维网格
    output = output_flat.reshape(Nr, Nphi)

    return output




def generate_field_polar(R: ArrayType, Phi: ArrayType, r_vals: ArrayType, theta_vals: ArrayType, delta: np.float32) -> ArrayType:
    xp = get_array_module(R)
    if xp.__name__ == "numpy":
        return generate_field_polar_numpy(R, Phi, r_vals, theta_vals, delta)
    elif xp.__name__ == "cupy":
        return generate_field_polar_cuda(R, Phi, r_vals, theta_vals, delta)
    else:
        raise ValueError(f"Unsupported array type: {xp.__name__}")



if __name__ == "__main__":
    import cupy as cp
    # Test abel_inversion
    image_np = np.random.rand(512, 512)
    image_cp = cp.random.rand(512, 512)
    res_np = abel_inversion(image_np)
    assert (512, "numpy") in abel_inv_mat_cache
    res_cp = abel_inversion(image_cp)
    assert (512, "cupy") in abel_inv_mat_cache
    assert get_array_module(res_np) is np
    assert get_array_module(res_cp) is cp
    res_np = abel_inversion(image_np)
    res_cp = abel_inversion(image_cp)
    assert get_array_module(res_np) is np
    assert get_array_module(res_cp) is cp
    arr_np = np.arange(3)
    arr_cp = cp.arange(3, 6)
    print(to_cupy(arr_np))
    assert isinstance(to_cupy(arr_np), cp.ndarray)
    print(to_cupy(arr_np, arr_cp))
    print(to_cupy((arr_np, arr_cp)))
    print(to_cupy([arr_np, arr_cp]))
    res = to_cupy({"a": arr_np, "b": arr_cp}, arr_np)
    print(res)
    assert isinstance(res[0]["a"], cp.ndarray)
    assert isinstance(res[1], cp.ndarray)
    res = to_cupy({"c": ({"a": arr_np, "b": arr_cp}, arr_cp), "d": [arr_np, arr_cp]})
    print(res)
    assert isinstance(res["c"][0]["a"], cp.ndarray)
    assert isinstance(res["d"][0], cp.ndarray)
    print(to_cupy([arr_np, arr_cp]))