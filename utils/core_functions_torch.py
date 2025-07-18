import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from typing import Optional, Tuple, Union
import abel

ScalarType = Union[int, float]

def toTensor(input, *args, **kwargs):
    def _to_Tensor(x):
        if isinstance(x, torch.Tensor):
            return x.to(*args, **kwargs)
        elif isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(*args, **kwargs)
        elif isinstance(x, list):
            return [_to_Tensor(xi) for xi in x]
        elif isinstance(x, tuple):
            return tuple(_to_Tensor(xi) for xi in x)
        elif isinstance(x, dict):
            return {k: _to_Tensor(v) for k, v in x.items()}
        else:
            return x
    return _to_Tensor(input)


def generate_gaussian_kernel(sigma: Union[ScalarType, Tuple[ScalarType, ScalarType]], truncate=4.0):
    """
    Generate a 2D Gaussian kernel.
    
    Parameters
    ----------
    sigma: float or tuple
        Standard deviation of the Gaussian kernel. If it's a tuple, the format is (sigma_y, sigma_x).
    truncate: float
        Truncation distance, in units of sigma. Default is 4.0.
        
    Returns
    -------
    kernel: torch.Tensor
        A 2D Gaussian kernel with shape (1, 1, kernel_h, kernel_w).
    """
    if isinstance(sigma, (int, float)):
        sigma = (sigma, sigma)
    elif len(sigma) != 2:
        raise ValueError("sigma must be a float or a tuple of two floats.")
    sigma_y, sigma_x = sigma

    def get_1d_kernel(sigma_i):
        if sigma_i <= 0:
            return torch.ones((1, ), dtype=torch.float32)
        radius = int(truncate * sigma_i + 0.5)
        x = torch.arange(-radius, radius + 1, dtype=torch.float32)
        kernel = torch.exp(-x**2 / (2 * sigma_i**2))
        kernel /= kernel.sum()
        return kernel

    kernel_y = get_1d_kernel(sigma_y)
    kernel_x = get_1d_kernel(sigma_x)
    kernel_2d = torch.outer(kernel_y, kernel_x)
    return kernel_2d.unsqueeze(0).unsqueeze(0)

def gaussian_filter(kernel: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
    input_ = input.unsqueeze(0).unsqueeze(0)
    kH, kW = kernel.shape[2:4]
    output = F.conv2d(input_, kernel, padding=(kH //2, kW // 2))
    return output.squeeze(0).squeeze(0)

def get_abel_trans_mat(n: int) -> torch.Tensor:
    trans_mat = abel.Transform(np.eye(n), method='basex', direction='inverse', transform_options={"verbose": False}).transform
    trans_mat = torch.from_numpy(trans_mat).float()
    return trans_mat


def f_smooth(r: torch.Tensor, rmax: torch.Tensor, delta=0.01) -> torch.Tensor:
    """
    Smoothed version of `f(r) = 1 / sqrt(max(rmax^2 - r^2, 0))`

    :param r: torch.Tensor, radius
    :param rmax: torch.Tensor, maximum radius
    :param delta: float, smoothing width, default is 0.01
    :return values: torch.Tensor, smoothed function value
    """
    r1 = torch.maximum(r - delta/2, torch.as_tensor(0.0))
    r2 = torch.minimum(r + delta/2, 2*r)
    EPS = 1e-10
    F = lambda x: torch.sqrt(torch.clip(rmax**2 - torch.clip(x, 0, None)**2, EPS, None))
    vals = (F(r1) - F(r2)) / torch.clip((r2 - r1) * r, 2*EPS**2, None)
    vals = torch.where(r <= EPS, 1 / rmax, vals)
    return vals

def generate_field(
        R: torch.Tensor, 
        Y: torch.Tensor, 
        r_vals: torch.Tensor, 
        cos_theta_vals: torch.Tensor, 
        delta: float = 0.01, 
        batch_size: int = 128
    ) -> torch.Tensor:
    """
    Generate the field function on the grid defined by R and Y:
    `f(x, y; r, \\theta) = {1 + [3(cos \\theta)^2 - 1][3(y / r)^2 - 1] / 2} / sqrt[max(r^2 - x^2 - y^2, 0)] / r`.
    And sum over all (r, \\theta).

    parameters
    ----------
    R: torch.Tensor
        radius of the grids
    Y: torch.Tensor
        y coordinate of the grids
    r_vals: torch.Tensor
        radius values
    cos_theta_vals: torch.Tensor
        cosine of theta values
    delta: float
        smoothing width, default is 0.01.
    batch_size: int
        batch size for processing the field, default is 128.
    
    Returns
    -------
    field: torch.Tensor
        The sum of the fields over all (r, \\theta).
    """
    final_field = torch.zeros_like(Y)
    r_batches = torch.split(r_vals, batch_size, dim=0)
    cos_theta_batches = torch.split(cos_theta_vals, batch_size, dim=0)
    
    R_view = R.view(1, -1)
    Y_view = Y.view(1, -1)
    
    for r, cos_theta in zip(r_batches, cos_theta_batches):
        r = r.view(-1, 1)
        cos_theta = cos_theta.view(-1, 1)
        C = 0.5 * (3 * cos_theta**2 - 1)
        contribution = ((1 + C * (3 * (Y_view / r)**2 - 1)) / r * f_smooth(R_view, r, delta=delta)).sum(dim=0)
        final_field += contribution.view(Y.shape)
    return final_field



class GND():
    def __init__(self, optimizer: optim.Optimizer, s=0.1, f_lb=0.0, gamma=0.999):
        self.optimizer = optimizer
        self.s = s
        self.f_lb = f_lb
        self.f_min = np.inf
        self.gamma = gamma

    @torch.no_grad()  # 确保不会计算梯度
    def step(self, f_val: torch.Tensor, freeze_lb=False):
        """
        Performs the noise step.
        """
        f_val = f_val.item()
        f_min = min(self.f_min, f_val)
        f_lb = self.f_lb
        sigma_1 = np.sqrt(self.s * np.clip(f_val - f_lb, 0.0, None))
        for group in self.optimizer.param_groups:
            lr = group["lr"]
            sigma = np.sqrt(lr) * sigma_1

            for param in group["params"]:
                d_p = torch.randn_like(param) * sigma / np.sqrt(param.numel())

                # 更新参数
                param.data.add_(d_p)
        if not freeze_lb:
            self.f_lb = min(f_lb * self.gamma + f_min * (1 - self.gamma), f_min)
        self.f_min = f_min
        return f_val