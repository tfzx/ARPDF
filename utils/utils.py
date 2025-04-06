import os
# import cupy as cp
from matplotlib import pyplot as plt
import numpy as np
from typing import Any, List, Optional, Tuple, Iterable
import json
import MDAnalysis as mda
from utils.core_functions import ArrayType, get_array_module


def box_shift(dx: ArrayType, box: Optional[List[float]] = None) -> ArrayType:
    """
    Shift the coordinates (dx) into the box. Both cupy and numpy array are supported.

    Return
    -----
    shifted_dx: xp.ndarray
    """
    if box is None:
        return dx
    xp = get_array_module(dx)
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
    return dx - xp.round(dx @ xp.linalg.inv(cell)) @ cell

def get_xy_range(xy_range) -> list:
    if isinstance(xy_range, (float, int)):
        return [-xy_range, xy_range, -xy_range, xy_range]
    elif len(xy_range) == 2:
        xmin, xmax = xy_range
        return [xmin, xmax, xmin, xmax]
    else:
        return list(xy_range)

def generate_grids(xy_range, N: int = 512, M: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a 2D grid of N x N points in the range [-rmax, rmax]^2.

    Returns:
        (X, Y): Grid coordinates.
    """
    xmin, xmax, ymin, ymax = get_xy_range(xy_range)
    if M is None:
        M = N
    x = np.linspace(xmin, xmax, N, dtype=np.float32)
    y = np.linspace(ymin, ymax, M, dtype=np.float32)
    return np.meshgrid(x, y)

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
    Preprocess ARPDF data, including range restriction, grid resampling, and intensity normalization.
    
    Parameters
    ----------
    ARPDF_raw: ndarray
        Original ARPDF data.
    original_range: float or list of float
        Original data range.
    rmax: float, optional
        The maximum range for data processing. If not provided, it defaults to the minimum absolute value of original_range.
    new_grid_size: Tuple[int, int], optional
        The new grid size for resampling. If provided, the data will be resampled to the new grid size.
        !Note: This function is not implemented yet.
    max_intensity: float, optional
        The maximum intensity value for normalization. Default is 1.0.
    
    Returns
    -------
    X: ndarray
        The processed X-axis grid data.
    Y: ndarray
        The processed Y-axis grid data.
    ARPDF: ndarray
        The processed ARPDF data.
    """
    if rmax is None:
        rmax = np.min(np.abs(original_range))
    N, M = ARPDF_raw.shape
    X_ori, Y_ori = generate_grids(original_range, N, M)
    x_mask = np.abs(X_ori[0, :]) <= rmax
    y_mask = np.abs(Y_ori[:, 0]) <= rmax
    X = X_ori[y_mask, :][:, x_mask]
    Y = Y_ori[y_mask, :][:, x_mask]
    ARPDF = np.copy(ARPDF_raw[y_mask, :][:, x_mask])
    ARPDF = ARPDF / np.max(np.abs(ARPDF)) * max_intensity
    if new_grid_size is not None:
        X, Y, ARPDF = resize_ARPDF(ARPDF, (X, Y), new_grid_size)
    return np.copy(X), np.copy(Y), np.copy(ARPDF)

def show_images(
        images: Iterable[Tuple[Any, np.ndarray]], 
        plot_range = 10.0, 
        show_range = None, 
        c_range = None,
        colorbar = "last",
        cmap = 'inferno', 
        title = None, 
        xlabel="X-Axis",
        ylabel="Y-Axis",
        clabel="Intensity",
        **kwargs
    ):
    """
    Visualize multiple images.

    Parameters:
        images : Iterable of (title, image) tuples to plot
        plot_range : Image range in units
        show_range : Range to show in units
        c_range : Range to colorbar in units
        colorbar : Colorbar setting. "last", "none", "align", or "all"
        cmap : Color map
        title : Title for subplot
        xlabel : X-axis label
        ylabel : Y-axis label
        clabel : Colorbar label
    """
    def get_c_range(c_range):
        if colorbar == "align" and c_range is None:
            # Compute the global vmin and vmax for all subplots. Make sure the colormap is the same.
            all_values = np.array([[image.min(), image.max()] for _, image in images])
            c_range = all_values[:, 0].min(), all_values[:, 1].max()
        if c_range is not None:
            # broadcast to each image if needed
            c_range = np.array(c_range)
            if c_range.ndim == 0:
                c_range = np.tile([-c_range, c_range], (N, 1)) # set [-c_range, c_range] if c_range is a scalar
            elif c_range.ndim == 1:
                c_range = np.tile(c_range, (N, 1))
        return c_range
    def get_title_map(title):
        if title is None:
            return lambda key: str(key)
        elif isinstance(title, str):
            return lambda key: f"{title} {key}"
        else:
            return title

    N = len(images)
    plot_range = get_xy_range(plot_range)
    show_range = get_xy_range(show_range) if show_range is not None else plot_range
    colorbar = colorbar.strip().lower()
    c_range = get_c_range(c_range)  # None or ndarray of shape (N, 2)
    title_map = get_title_map(title)
    
    W, H = 5 * N + 1, 5
    fig, axs = plt.subplots(1, N, figsize=(W, H), sharey=(colorbar != "all"))   # share y axis if colorbar is not "all"
    axs = np.atleast_1d(axs)  # Ensure axs is always an iterable
    imgs = []
    for (i, ax), (key, image) in zip(enumerate(axs), images):
        img = ax.imshow(image, origin='lower', extent=plot_range, cmap=cmap, **kwargs)
        imgs.append(img)
        ax.set_title(title_map(key))
        ax.set_xlim(show_range[0:2])
        ax.set_ylim(show_range[2:])
        ax.set_xlabel(xlabel)
        if c_range is not None:
            img.set_clim(c_range[i])
    
    # Set y labels
    if colorbar != "all":
        axs[0].set_ylabel(ylabel)
    else:
        for ax in axs:
            ax.set_ylabel(ylabel)

    if len(images) > 1:
        fig.tight_layout()
        if colorbar in ["align", "last"]:
            fig.subplots_adjust(right=(W - 1) / W)  # Adjust layout to leave space for colorbar
            ax_pos = fig.axes[-1].get_position().bounds
            cbar_ax = fig.add_axes([(W - 0.8) / W, ax_pos[1], 0.25 / W, ax_pos[3]])  # Define colorbar position
            fig.colorbar(img, cax=cbar_ax, label=clabel)  # Attach colorbar to the last image
        elif colorbar == "all":
            for i, img in enumerate(imgs):
                fig.colorbar(img, ax=axs[i], label=None if i < N else clabel) # Attach colorbar to all images
    elif colorbar != "none":
        fig.colorbar(img, ax=axs[0], label=clabel)
    return fig


def load_exp_data(data_dir: str, rmax: float = 10.0, new_grid_size = None, max_intensity = 1.0):
    """
    Load experimental data from a directory.
    The directory should contain a metadata file "metadata.json".
    The metadata file should contain the following keys:
    - "expdata_info":
        - "expdata_name": str, the name of the experimental data file. Should be a `.npy` file
        - "xy_range": float or list of float
    The experimental data will be loaded from "expdata_name".

    Parameters
    ----------
    dir : str
        The directory containing the experimental data file and metadata file.
    rmax, new_grid_size, max_intensity : ...
        The same parameters as `preprocess_ARPDF`

    Returns
    -------
    X : ndarray
        The X-axis grid data.
    Y : ndarray
        The Y-axis grid data.
    ARPDF : ndarray
        The processed ARPDF data.
    """
    with open(os.path.join(data_dir, "metadata.json"), "r") as f:
        expdata_info = json.load(f)["expdata_info"]
    filename = os.path.join(data_dir, expdata_info["expdata_name"])
    expdata: np.ndarray = np.load(filename)
    return preprocess_ARPDF(expdata, expdata_info["xy_range"], rmax, new_grid_size, max_intensity)

def load_structure_data(data_dir: str):
    """
    Load structure data from a directory.
    The directory should contain a metadata file "metadata.json".
    The metadata file should contain the following keys:
    - "structure_info":
        - "u1_name": str, the name of the initial structure file.
        - "u2_name": str, the name of the modified structure file. This is optional.
        - "polar_axis": list of float, the polar axis of the second structure
        - "modified_atoms": list of int, the indices of the modified atoms in the second structure

    Parameters
    ----------
    dir : str
        The directory containing the structure data file and metadata file.
    
    Returns
    -------
    u1: mda.Universe
        the initial structure
    u2: mda.Universe
        the modified structure. None if no modified structure is provided.
    polar_axis: list of float
    modified_atoms: list of int
        the indices of the modified atoms in the second structure
    """
    with open(os.path.join(data_dir, "metadata.json"), "r") as f:
        structure_info = json.load(f)["structure_info"]
    u1 = mda.Universe(os.path.join(data_dir, structure_info["u1_name"]))
    u2 = None
    if "u2_name" in structure_info:
        u2 = mda.Universe(os.path.join(data_dir, structure_info["u2_name"]))
    modified_atoms: List[int] = structure_info["modified_atoms"]
    polar_axis: List[float] = structure_info["polar_axis"]
    return u1, u2, modified_atoms, polar_axis

if __name__ == "__main__":
    import cupy as cp
    # Test box_shift
    arr_np = np.random.rand(5, 3)
    arr_cp = cp.random.rand(5, 3)
    res_np = box_shift(arr_np, box=[10, 10, 10, 90, 90, 90])
    res_cp = box_shift(arr_cp, box=[10, 10, 10, 90, 90, 90])
    assert get_array_module(res_np) is np
    assert get_array_module(res_cp) is cp
    # Test generate_grids
    X_np, Y_np = generate_grids(10, 512)
    assert get_array_module(X_np) is np