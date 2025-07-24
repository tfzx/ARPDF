from typing import Optional
import numpy as np
from utils.core_functions import ArrayType, get_array_module, to_cupy, to_numpy
import scipy.special as special

def weighted_similarity(w, v1, v2):
    xp = get_array_module(v1)
    def _reshape(arr, shape):
        if xp.__name__ == "torch":
            return arr.view(shape)
        else:
            return arr.reshape(shape)
    _batch_shape = w.shape[:-v1.ndim]
    w_flat = _reshape(w, (-1, np.prod(v1.shape)))
    v1_flat = v1.flatten()
    v2_flat = v2.flatten()
    def inner(w, x, y):
        return w @ (x * y)
    C = 0.005
    def weighted_sim(w, x, y):
        return (inner(w, x, y) / (xp.sqrt(inner(w, x, x)) + 1e-8) + C) / (xp.sqrt(inner(w, y, y)) + C)
    return _reshape(weighted_sim(w_flat, v1_flat, v2_flat), _batch_shape)

def cosine_similarity(ARPDF1: ArrayType, ARPDF2: ArrayType, weight: Optional[ArrayType] = None) -> float:
    """
    Compute the cosine similarity between two ARPDF.
    """
    xp = get_array_module(ARPDF1)
    if weight is None:
        weight = xp.ones_like(ARPDF1)
    return weighted_similarity(weight[None, :, :], ARPDF1, ARPDF2).squeeze(0)

def get_angular_filters(R_grids, r0, sigma):
    """
    Get the 2D Rice ditribution at radius r0 with sigma. This can be used to filter the angular components at radius r0.
    """
    xp = get_array_module(R_grids)
    _r0 = xp.asarray(r0)
    _R = R_grids.reshape((1,) * r0.ndim + R_grids.shape)
    _r0 = _r0.reshape(_r0.shape + (1,) * R_grids.ndim)
    # !Note: Don't use cupyx.scipy.special.i0e, because it's instable for large values (may lead to nan)
    _i0e = special.i0e if xp.__name__ == "numpy" else lambda x: to_cupy(special.i0e(to_numpy(x)))
    return xp.exp(-(_R-_r0)**2/(2*sigma**2)) * _i0e(_r0*_R/sigma)

def get_gaussian_filters(R_grids, r0, sigma):
    """
    Get the 2D Gaussian ditribution at radius r0 with sigma. This can be used to filter the angular components at radius r0.
    """
    xp = get_array_module(R_grids)
    _r0 = xp.asarray(r0)
    _R = R_grids.reshape((1,) * r0.ndim + R_grids.shape)
    _r0 = _r0.reshape(_r0.shape + (1,) * R_grids.ndim)

    return xp.exp(-(_R-_r0)**2/(2*sigma**2)) 

#def angular_similarity(ARPDF1: ArrayType, ARPDF2: ArrayType, angular_filters: ArrayType, r_weight: Optional[ArrayType] = None) -> float:
#    xp = get_array_module(ARPDF1)
#    if r_weight is None:
#        r_weight = xp.ones_like(angular_filters[:, 0, 0])
#    return xp.vdot(r_weight, weighted_similarity(angular_filters, ARPDF1, ARPDF2))

def angular_similarity(ARPDF1, ARPDF2, angular_filters, r_weight=None):
    xp = get_array_module(ARPDF1)
    if r_weight is None:
        r_weight = xp.ones_like(angular_filters[:, 0, 0])
    return xp.vdot(r_weight, weighted_similarity(angular_filters, ARPDF1, ARPDF2))

def strength_similarity(ARPDF1: ArrayType, ARPDF2: ArrayType, angular_filters: ArrayType, r_weight: Optional[ArrayType] = None) -> float:
    xp = get_array_module(ARPDF1)
    if r_weight is None:
        r_weight = xp.ones_like(angular_filters[:, 0, 0])
    def get_strength(img):
        return xp.sqrt(xp.einsum("ijk,jk->i", angular_filters, img**2))
    return weighted_similarity(r_weight, get_strength(ARPDF1), get_strength(ARPDF2))


def oneD_similarity(ARPDF1: ArrayType, ARPDF2: ArrayType, axis: int = 0, weight: Optional[ArrayType] = None) -> float:
    """
    Compute cosine similarity along a central 1D axis of two ARPDFs.
    
    Args:
        ARPDF1: First ARPDF 2D array.
        ARPDF2: Second ARPDF 2D array.
        axis: Which axis to slice along (0 for y-axis/mid-column, 1 for x-axis/mid-row).
        weight: Optional weight for each pixel in the 1D line.
        
    Returns:
        Cosine similarity between the selected 1D lines of ARPDF1 and ARPDF2.
    """

    xp = get_array_module(ARPDF1)

    if weight is not None:
        ARPDF1 = ARPDF1 * weight
        ARPDF2 = ARPDF2 * weight

    if axis == 0:
        center_idx = ARPDF1.shape[1] // 2
        line1 = ARPDF1[:, center_idx]
        line2 = ARPDF2[:, center_idx]
    else:
        center_idx = ARPDF1.shape[0] // 2
        line1 = ARPDF1[center_idx, :]
        line2 = ARPDF2[center_idx, :]

    dot = xp.sum(line1 * line2)
    norm1 = xp.sqrt(xp.sum(line1 * line1)) + 1e-8
    norm2 = xp.sqrt(xp.sum(line2 * line2)) + 1e-8
    similarity = dot / (norm1 * norm2)

    return similarity

def angular_average_similarity(
    ARPDF1: ArrayType,
    ARPDF2: ArrayType,
    angle_range=(np.pi/4, 3*np.pi/4),
    weight: Optional[ArrayType] = None,
    bin_width: float = 0.1
) -> float:
    xp = get_array_module(ARPDF1)
    assert ARPDF1.shape == ARPDF2.shape, "ARPDF1 and ARPDF2 must have same shape"
    H, W = ARPDF1.shape
    cx, cy = W // 2, H // 2

    # Coordinate grid
    x = xp.arange(W) - cx
    y = xp.arange(H) - cy
    X, Y = xp.meshgrid(x, y)
    R = xp.sqrt(X**2 + Y**2)
    Theta = xp.arctan2(Y, X)

    # Angular mask
    theta_min, theta_max = angle_range
    mask = (Theta >= theta_min) & (Theta <= theta_max)

    # Radial bin index per pixel
    R_flat = R[mask]
    ARPDF1_flat = ARPDF1[mask]
    ARPDF2_flat = ARPDF2[mask]
    bin_indices = xp.floor(R_flat / bin_width).astype(int)

    if weight is not None:
        weight_flat = weight[mask]
        w1 = weight_flat * ARPDF1_flat
        w2 = weight_flat * ARPDF2_flat
        sum_w = xp.bincount(bin_indices, weights=weight_flat)
        sum_w1 = xp.bincount(bin_indices, weights=w1)
        sum_w2 = xp.bincount(bin_indices, weights=w2)
        profile1 = sum_w1 / (sum_w + 1e-8)
        profile2 = sum_w2 / (sum_w + 1e-8)
        final_weight = sum_w
    else:
        profile1 = xp.bincount(bin_indices, weights=ARPDF1_flat)
        profile2 = xp.bincount(bin_indices, weights=ARPDF2_flat)
        counts = xp.bincount(bin_indices)
        profile1 /= (counts + 1e-8)
        profile2 /= (counts + 1e-8)
        final_weight = counts

    # Cosine similarity
    dot = xp.sum(profile1 * profile2)
    norm1 = xp.sqrt(xp.sum(profile1**2)) + 1e-8
    norm2 = xp.sqrt(xp.sum(profile2**2)) + 1e-8
    similarity = dot / (norm1 * norm2)
    return similarity
