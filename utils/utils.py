import os
# import cupy as cp
from matplotlib import pyplot as plt
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
from typing import Any, List, Optional, Tuple, Iterable
import json
import MDAnalysis as mda
from MDAnalysis.analysis import align
import MDAnalysis.analysis.distances as mda_dist
from numpy._typing._array_like import NDArray
from utils.core_functions import ArrayType, get_array_module

def compute_all_atom_pairs(
        universe: mda.Universe, 
        cutoff: float = 10.0, 
        modified_atoms: List[int] = None, 
        polar_axis = (0, 0, 1),
        periodic = False
    ) -> Tuple[Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]], int]:
    """
    Compute (r, theta) for all atom pairs within a cutoff distance and group them by atom types.

    Parameters:
        universe (mda.Universe): The MDAnalysis Universe object.
        cutoff (float): Cutoff distance for neighbor search.
        modified_atoms (List[int], optional): Atom indices to limit pair search.
        polar_axis (tuple): Axis to compute theta angle against.
        periodic (bool): if True, consider periodic boundary condition.

    Returns:
        (atom_pairs, num_selected):
        atom_pairs (dict): Mapping from (atom_type_1, atom_type_2) -> (r_values, theta_values).
        num_selected (int): Number of selected atoms.
    """
    # Step 1: Select atoms
    if modified_atoms is not None:
        center_group = universe.atoms[modified_atoms]
        selected_group = center_group + universe.select_atoms(f"around {cutoff} group center", center=center_group, periodic=periodic)
    else:
        center_group = universe.atoms
        selected_group = universe.atoms

    # Step 2: Compute pairwise distance matrix
    box = universe.dimensions if periodic else None
    dist_box = np.array(
        mda_dist.distance_array(center_group.positions, selected_group.positions, box),
        dtype=np.float32
    )

    # Step 3: Mask valid pairs (distance < cutoff and i < j)
    mask = (dist_box < cutoff) & np.triu(np.ones_like(dist_box, dtype=np.bool_), k=1)
    i_idx, j_idx = np.nonzero(mask)

    # Step 4: Compute r values and theta values
    r_vals = dist_box[mask]
    vectors = box_shift(np.array(selected_group.positions)[j_idx] - np.array(center_group.positions)[i_idx], box)
    polar_axis = np.asarray(polar_axis, dtype=np.float32)
    polar_axis /= np.linalg.norm(polar_axis)  # normalize
    theta_vals = np.arccos(np.clip(np.sum(vectors * polar_axis, axis=1) / np.linalg.norm(vectors, axis=1), -1.0, 1.0))

    # Step 5: Prepare atom type pairs
    all_atom_types = sorted(set(universe.atoms.types))
    atom_types_center = np.array(center_group.types, dtype="<U4")
    atom_types_around = np.array(selected_group.types, dtype="<U4")
    atom_pair_types = np.stack([atom_types_center[i_idx], atom_types_around[j_idx]], axis=1)
    atom_pair_types.sort(axis=1)  # enforce type1 <= type2

    # Step 6: Organize into dictionary grouped by atom pair types
    atom_pairs = {}
    for i, type1 in enumerate(all_atom_types):
        for type2 in all_atom_types[i:]:
            pair_mask = np.all(atom_pair_types == [type1, type2], axis=1)
            atom_pairs[(type1, type2)] = (r_vals[pair_mask], theta_vals[pair_mask])

    return atom_pairs, len(selected_group)

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

def select_nbr_mols(universe: mda.Universe, center_atoms: List[int], nbr_distance: float | None = None, periodic = True):
    """
    Select molecules within the specified distance from the center atoms.
    """
    center_group = universe.atoms[center_atoms]
    if nbr_distance is not None:
        center_group = center_group + universe.select_atoms(f"around {nbr_distance} group center", center=center_group, periodic=periodic)
    mask = np.isin(universe.atoms.resids, np.unique(center_group.resids))
    return np.nonzero(mask)[0]

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
    return np.meshgrid(x, y, indexing="ij")

# def generate_grids_polar(
#     r_range: Tuple[float, float] = (0.0, 10.0),
#     theta_range: Tuple[float, float] = (0.0, 2 * np.pi),
#     Nr: int = 512,
#     Ntheta: int = 512
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Generate polar coordinate grids (R, Phi) for use in radial projection.

#     Parameters
#     ----------
#     r_range : (rmin, rmax)
#         Radial distance range (in Å).
#     theta_range : (theta_min, theta_max)
#         Angular range (in radians).
#     Nr : int
#         Number of radial bins.
#     Ntheta : int
#         Number of angular bins.

#     Returns
#     -------
#     R, Phi : np.ndarray, np.ndarray
#         2D meshgrids in polar coordinates.
#     """
#     rmin, rmax = r_range
#     tmin, tmax = theta_range

#     r_vals = np.linspace(rmin, rmax, Nr, dtype=np.float32)
#     theta_vals = np.linspace(tmin, tmax, Ntheta, dtype=np.float32)

#     return np.meshgrid(r_vals, theta_vals, indexing="ij")


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
    x_mask = np.abs(X_ori[:, 0]) <= rmax
    y_mask = np.abs(Y_ori[0, :]) <= rmax
    X = X_ori[x_mask, :][:, y_mask]
    Y = Y_ori[x_mask, :][:, y_mask]
    ARPDF = np.copy(ARPDF_raw[x_mask, :][:, y_mask])
    ARPDF = ARPDF / np.max(np.abs(ARPDF)) * max_intensity
    if new_grid_size is not None:
        X, Y, ARPDF = resize_ARPDF(ARPDF, (X, Y), new_grid_size)
    return np.copy(X), np.copy(Y), np.copy(ARPDF)

def show_images(
        images: Iterable[Tuple[Any, np.ndarray]], 
        plot_range = 10.0, 
        show_range = None, 
        c_range = None,
        colorbar_type = "last",
        cmap = 'inferno', 
        title = None, 
        xlabel="X-Axis",
        ylabel="Y-Axis",
        clabel="Intensity",
        transpose: bool = True,
        **kwargs
    ):
    """
    Visualize multiple images.

    Parameters:
        images : Iterable of (title, image) tuples to plot
        plot_range : Image range in units
        show_range : Range to show in units
        c_range : Range to colorbar in units
        colorbar_type : Colorbar setting. "last", "none", "align", or "all"
        cmap : Color map
        title : Title for subplot
        xlabel : X-axis label
        ylabel : Y-axis label
        clabel : Colorbar label
    """
    def get_c_range(c_range):
        if colorbar_type == "align" and c_range is None:
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
    colorbar_type = colorbar_type.strip().lower()
    c_range = get_c_range(c_range)  # None or ndarray of shape (N, 2)
    title_map = get_title_map(title)
    
    W, H = 5 * N + 1, 5
    fig, axs = plt.subplots(1, N, figsize=(W, H), sharey=(colorbar_type != "all"))   # share y axis if colorbar is not "all"
    axs = np.atleast_1d(axs)  # Ensure axs is always an iterable
    imgs = []
    for (i, ax), (key, image) in zip(enumerate(axs), images):
        image = image.T if transpose else image
        img = ax.imshow(image, origin='lower', extent=plot_range, cmap=cmap, **kwargs)
        imgs.append(img)
        ax.set_title(title_map(key))
        ax.set_xlim(show_range[0:2])
        ax.set_ylim(show_range[2:])
        ax.set_xlabel(xlabel)
        if c_range is not None:
            img.set_clim(c_range[i])
    
    # Set y labels
    if colorbar_type != "all":
        axs[0].set_ylabel(ylabel)
    else:
        for ax in axs:
            ax.set_ylabel(ylabel)

    if len(images) > 1:
        fig.tight_layout()
        if colorbar_type in ["align", "last"]:
            fig.subplots_adjust(right=(W - 1) / W)  # Adjust layout to leave space for colorbar
            ax_pos = fig.axes[-1].get_position().bounds
            cbar_ax = fig.add_axes([(W - 0.8) / W, ax_pos[1], 0.25 / W, ax_pos[3]])  # Define colorbar position
            fig.colorbar(img, cax=cbar_ax, label=clabel)  # Attach colorbar to the last image
        elif colorbar_type == "all":
            for i, img in enumerate(imgs):
                fig.colorbar(img, ax=axs[i], label=None if i < N else clabel) # Attach colorbar to all images
    elif colorbar_type != "none":
        fig.colorbar(img, ax=axs[0], label=clabel)
    return fig

# def show_images_polar(
#         images: Iterable[Tuple[Any, np.ndarray]], 
#         r_range: Tuple[float, float] = (0.0, 1.0),
#         phi_range: Tuple[float, float] = (0.0, 2 * np.pi),
#         c_range: Optional[Tuple[float, float]] = None,
#         colorbar: str = "last",
#         cmap: str = 'inferno', 
#         title = None, 
#         clabel: str = "Intensity",
#         **kwargs
#     ):
#     """
#     Visualize multiple images with R (radius) as x-axis and phi (angle) as y-axis.

#     Parameters:
#         images     : Iterable of (title, image) tuples to plot
#         r_range    : Tuple for radial range (min, max) along x-axis
#         phi_range  : Tuple for angular range (min, max) along y-axis
#         c_range    : Tuple or list of (vmin, vmax) for color scale, or None
#         colorbar   : "last", "all", "none", or "align"
#         cmap       : Colormap
#         title      : Title or function of key
#         clabel     : Colorbar label
#         kwargs     : Additional imshow() kwargs
#     """
#     def get_title_map(title):
#         if title is None:
#             return lambda key: str(key)
#         elif isinstance(title, str):
#             return lambda key: f"{title} {key}"
#         else:
#             return title

#     images = list(images)
#     N = len(images)
#     colorbar = colorbar.strip().lower()
#     title_map = get_title_map(title)

#     # Compute common color range if needed
#     if colorbar == "align" and c_range is None:
#         all_vals = np.array([[img.min(), img.max()] for _, img in images])
#         c_range = (all_vals[:, 0].min(), all_vals[:, 1].max())

#     # Figure setup
#     W, H = 5 * N + 1, 5
#     fig, axs = plt.subplots(1, N, figsize=(W, H), sharey=(colorbar != "all"))
#     axs = np.atleast_1d(axs)
#     imgs = []

#     for i, ((key, image), ax) in enumerate(zip(images, axs)):
#         extent = [r_range[0], r_range[1], phi_range[0], phi_range[1]]
#         image = image.T 
#         img = ax.imshow(image, origin='lower', extent=extent, aspect='auto', cmap=cmap, **kwargs)
#         if c_range is not None:
#             img.set_clim(*c_range)
#         imgs.append(img)
#         ax.set_title(title_map(key))
#         ax.set_xlabel("R")
#         if i == 0 or colorbar == "all":
#             ax.set_ylabel("φ")

#     if colorbar != "none":
#         if colorbar in ["last", "align"]:
#             fig.subplots_adjust(right=0.85)
#             ax_pos = axs[-1].get_position().bounds
#             cax = fig.add_axes([0.87, ax_pos[1], 0.02, ax_pos[3]])
#             fig.colorbar(imgs[-1], cax=cax, label=clabel)
#         elif colorbar == "all":
#             for ax, im in zip(axs, imgs):
#                 fig.colorbar(im, ax=ax, label=clabel)

#     fig.tight_layout()
#     return fig


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
    modified_atoms: list of int
        the indices of the modified atoms in the second structure
    polar_axis: list of float
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

def rotation_matrix(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Calculate rotation matrix to align vector v1 with v2.
    
    Args:
        v1 (np.ndarray): Source vector
        v2 (np.ndarray): Target vector
        
    Returns:
        np.ndarray: 3x3 rotation matrix
    """
    u = v1 - v2
    if np.linalg.norm(u) < 1e-8:
        return np.eye(3)
    u /= np.linalg.norm(u)
    return np.eye(3) - 2 * np.outer(u, u)

def copy_atom_group(atom_group: mda.AtomGroup) -> mda.AtomGroup:
    """
    Copy an atom group to a new universe.
    """
    new_universe = mda.Merge(atom_group)
    new_universe.dimensions = atom_group.dimensions
    return new_universe.atoms

def calculate_rmsd(
    mobile: mda.Universe,
    reference: mda.Universe,
    selection: List[int],
    subselection: Optional[List[int]] = None,
) -> float:
    """
    Calculate RMSD between two structures.
    
    Args:
        mobile (mda.Universe): The mobile structure
        reference (mda.Universe): The reference structure
        selection (List[int]): Atom indices to calculate RMSD
        subselection (Optional[List[int]]): Atom indices to use for alignment (if None, use selection)
        
    Returns:
        float: RMSD value
    """
    # Get selected atoms
    mobile_pos = mobile.atoms[selection].positions
    reference_pos = reference.atoms[selection].positions

    # handle periodic boundary conditions
    mobile_pos = box_shift(mobile_pos - mobile_pos[[0]], mobile.dimensions)
    reference_pos = box_shift(reference_pos - reference_pos[[0]], reference.dimensions)
    
    # Get subselection atoms for alignment if provided
    if subselection is not None:
        mobile_sub_pos = mobile.atoms[subselection].positions
        reference_sub_pos = reference.atoms[subselection].positions
    else:
        mobile_sub_pos = mobile_pos
        reference_sub_pos = reference_pos
    
    # Calculate center of mass
    mobile_com = mobile_sub_pos.mean(axis=0, keepdims=True)
    reference_com = reference_sub_pos.mean(axis=0, keepdims=True)
    
    # Center coordinates
    mobile_sub_pos_centered = mobile_sub_pos - mobile_com
    reference_sub_pos_centered = reference_sub_pos - reference_com
    
    # Calculate rotation matrix using MDAnalysis
    R, rmsd_rot = align.rotation_matrix(
        mobile_sub_pos_centered, reference_sub_pos_centered
    )
    if subselection is None:
        return rmsd_rot
    
    # Calculate RMSD
    diff = (mobile_pos - mobile_com) @ R.T - (reference_pos - reference_com)
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    
    return rmsd


def update_metadata(dir_path: str, metadata: dict) -> None:
    """
    Update and save metadata to a JSON file. If the file already exists, the new metadata will be merged with the existing one.

    Args:
        dir_path (str): Directory to save the metadata file.
        metadata (dict): New metadata to be saved.
    """
    filepath = os.path.join(dir_path, "metadata.json")
    existing_metadata = {}

    # Try to read existing metadata if the file exists
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            try:
                existing_metadata = json.load(f)
                if not isinstance(existing_metadata, dict):
                    existing_metadata = {}
            except json.JSONDecodeError:
                pass  # If the file is corrupted, ignore and overwrite

    # Merge existing metadata with new metadata
    existing_metadata.update(metadata)

    # Save the merged metadata
    with open(filepath, "w") as f:
        json.dump(existing_metadata, f, indent=4)

from scipy.ndimage import map_coordinates

#def polar_to_cartesian(f_rphi, r_max=1.0, grid_shape=(256, 256)): 
#    """
#    将极坐标函数 f(R, phi) 转换为直角坐标函数 f(X, Y)。
#    f_rphi: 2D numpy array，表示函数值，shape 为 (n_phi, n_r)
#    r_max: 极坐标半径最大值
#    grid_shape: 输出直角网格的形状 (height, width)
#
#    返回：
#        f_xy: 直角坐标下的二维函数值数组
#        X, Y: 对应的坐标网格
#    """
#    n_phi, n_r = f_rphi.shape
#    height, width = grid_shape

#    # 创建输出直角坐标网格，中心为 (0,0)
#    x = np.linspace(-r_max, r_max, width)
#    y = np.linspace(-r_max, r_max, height)
#    X, Y = np.meshgrid(x, y)

    # 将直角坐标转换为极坐标
#    R = np.sqrt(X**2 + Y**2)
#    Phi = np.arctan2(Y, X)
#    Phi = np.mod(Phi, 2 * np.pi)

#    # 映射到原始 f_rphi 的坐标范围（注意反转轴顺序）
#    r_coords = (R / r_max) * (n_r - 1)
#    phi_coords = (Phi / (2 * np.pi)) * (n_phi - 1)

    # 构造采样点 (phi_idx, r_idx)，注意 axis 0 是 phi，axis 1 是 r
#    coords = np.array([phi_coords.flatten(), r_coords.flatten()])

    # 插值
#    f_xy = map_coordinates(f_rphi, coords, order=1, mode='nearest').reshape(grid_shape)

#    return f_xy, X, Y


def polar_to_cartesian(ARPDF_rphi, RPhi, XY):
    R, phi = RPhi
    X, Y = XY

    Nr, Nphi = ARPDF_rphi.shape
    r_min, r_max = R.min(), R.max()
    phi_min, phi_max = phi.min(), phi.max()

    # 分辨率
    dr = (r_max - r_min) / (Nr - 1)
    dphi = (phi_max - phi_min) / (Nphi - 1)

    # 将笛卡尔坐标转为极坐标
    r_sample = np.sqrt(X**2 + Y**2)
    phi_sample = np.arctan2(Y, X)

    # 将 phi_sample wrap 到 [phi_min, phi_max]
    phi_sample_wrapped = (phi_sample - phi_min) % (2 * np.pi) + phi_min

    # 判断是否在范围内
    mask = (
        (r_sample >= r_min) & (r_sample <= r_max) &
        (phi_sample_wrapped >= phi_min) & (phi_sample_wrapped <= phi_max)
    )

    # 转为插值索引
    i_sample = (r_sample - r_min) / dr
    j_sample = (phi_sample_wrapped - phi_min) / dphi

    # 越界点设为 -1 再用 map_coordinates + cval 处理
    i_sample_safe = np.where(mask, i_sample, -1)
    j_sample_safe = np.where(mask, j_sample, -1)

    coords = np.vstack([i_sample_safe.ravel(), j_sample_safe.ravel()])
    sampled = map_coordinates(ARPDF_rphi, coords, order=1, mode='constant', cval=0.0)

    return sampled.reshape(X.shape)


#def cartesian_to_polar(f_xy, r_max=1.0, grid_shape=(256, 256)):
#    """
#    将直角坐标函数 f(X, Y) 转换为极坐标函数 f(R, phi)。
#    f_xy: 2D numpy array，shape 为 (height, width)
#    r_max: 极坐标最大半径
#    grid_shape: 输出极坐标网格形状 (n_phi, n_r)

#    返回：
#        f_rphi: 极坐标下的函数值数组
#        R, Phi: 极坐标网格
#    """
#    height, width = f_xy.shape
#    n_phi, n_r = grid_shape

#    # 创建极坐标网格
#    r = np.linspace(0, r_max, n_r)
#    phi = np.linspace(0, 2 * np.pi, n_phi)
#    R, Phi = np.meshgrid(r, phi)

#    # 极坐标转直角坐标
#    X = R * np.cos(Phi)
#    Y = R * np.sin(Phi)

    # 映射到原始图像坐标（0 到 width/height）
#    x_coords = ((X + r_max) / (2 * r_max)) * (width - 1)
#    y_coords = ((Y + r_max) / (2 * r_max)) * (height - 1)

#    coords = np.array([y_coords.flatten(), x_coords.flatten()])

#    # 插值
#    f_rphi = map_coordinates(f_xy, coords, order=1, mode='nearest').reshape(grid_shape)

#    return f_rphi, R, Phi

def cartesian_to_polar(ARPDF_xy, XY, RPhi):

    X, Y = XY
    R, phi = RPhi

    # 原图大小
    H, W = ARPDF_xy.shape
    x_min, x_max = X.min(), X.max()
    y_min, y_max = Y.min(), Y.max()
    
    dx = (x_max - x_min) / (W - 1)
    dy = (y_max - y_min) / (H - 1)

    # 将极坐标转换为直角坐标
    x_sample = R * np.cos(phi)
    y_sample = R * np.sin(phi)

    # 转为 array 索引（注意：Y 是第一个轴）
    i_sample = (y_sample - y_min) / dy
    j_sample = (x_sample - x_min) / dx

    coords = np.vstack([i_sample.ravel(), j_sample.ravel()])

    # 插值
    sampled = map_coordinates(ARPDF_xy, coords, order=1, mode='constant', cval=0.0)
    return sampled.reshape(R.shape)

#def cartesian_extend_quadrants(ARPDF_quadrant):
#    """
#    将第一象限的 ARPDF 拓展到整个平面，假设左右上下镜像对称。

#    输入：
#    - ARPDF_quadrant: 第一象限图像 (H, W)

#    输出：
#    - ARPDF_full: 全平面图像 (2H, 2W)
#    """
#    # 上下翻转（mirror y）
#    bottom_half = np.flipud(ARPDF_quadrant)
    
#    # 左右翻转（mirror x）
#    left_quadrant = np.fliplr(ARPDF_quadrant)
#    bottom_left = np.fliplr(bottom_half)

#    # 拼接成完整图像
#    top = np.hstack([left_quadrant, ARPDF_quadrant])
#    bottom = np.hstack([bottom_left, bottom_half])

#    ARPDF_full = np.vstack([bottom, top])
#    return ARPDF_full



def polar_extend_2pi(ARPDF_rphi, R, phi, n_fold=4):
    """
    将 90° 范围的 ARPDF 在极坐标系下按对称性扩展为更大的角度范围。

    支持扩展倍数 n_fold = 2, 3, 4。

    参数:
        ARPDF_rphi: (Nr, Nphi) 原始极坐标图像
        R:         (Nr, Nphi) 或 (Nr, 1)
        phi:       (Nr, Nphi) 或 (1, Nphi)，角度必须覆盖 pi/2
        n_fold:    扩展倍数，仅支持 2, 3, 4

    返回:
        ARPDF_extended, R_extended, phi_extended
    """

    if n_fold not in [2, 3, 4]:
        raise ValueError("仅支持 n_fold = 2, 3, 4")

    # 处理 phi 为 2D
    if phi.ndim == 1:
        phi = np.tile(phi, (R.shape[0], 1))

    # 生成扩展序列
    ARPDF_list = []
    phi_list = []

    for i in range(n_fold):
        shift = i * (np.pi / 2)

        if i % 2 == 0:
            ARPDF_i = ARPDF_rphi.copy()
            phi_i = phi.copy()
        else:
            ARPDF_i = np.flip(ARPDF_rphi, axis=1)
            phi_i = np.flip(phi, axis=1)

        phi_i = phi_i + shift

        ARPDF_list.append(ARPDF_i)
        phi_list.append(phi_i)

    ARPDF_extended = np.concatenate(ARPDF_list, axis=1)
    phi_extended = np.concatenate(phi_list, axis=1)

    return ARPDF_extended, R, phi_extended




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