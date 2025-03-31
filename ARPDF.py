from typing import Dict, List, Optional, Tuple
import cupy as cp
import numpy as np
import abel
import MDAnalysis as mda
import MDAnalysis.analysis.distances as mda_dist
from cupyx.scipy.ndimage import gaussian_filter as gaussian_filter_cp
from scipy.ndimage import gaussian_filter as gaussian_filter_np
import matplotlib.pyplot as plt
from utils import generate_field_cuda, box_shift, generate_grids, calc_AFF, abel_inversion, to_cupy, to_numpy
from utils import ArrayType

def compute_all_atom_pairs(
        universe: mda.Universe, 
        cutoff: float = 10.0, 
        modified_atoms: List[int] = None, 
        polar_axis = (0, 0, 1),
        periodic = False
    ) -> Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]]:
    """
    Compute (r, theta) for all atom pairs within a cutoff distance and group them by atom types.

    Parameters:
        universe (mda.Universe): The MDAnalysis Universe object.
        cutoff (float): Cutoff distance for neighbor search.
        modified_atoms (List[int], optional): Atom indices to limit pair search.
        polar_axis (tuple): Axis to compute theta angle against.
        periodic (bool): if True, consider periodic boundary condition.

    Returns:
        atom_pairs (dict): Mapping from (atom_type_1, atom_type_2) -> (r_values, theta_values).
    """
    # Step 1: Select atoms
    if modified_atoms is not None:
        center_group = universe.atoms[modified_atoms]
        around_group = center_group + universe.select_atoms(f"around {cutoff} group center", center=center_group, periodic=periodic)
    else:
        center_group = universe.atoms
        around_group = universe.atoms

    # Step 2: Compute pairwise distance matrix
    box = universe.dimensions if periodic else None
    dist_box = np.array(
        mda_dist.distance_array(center_group.positions, around_group.positions, box),
        dtype=np.float32
    )

    # Step 3: Mask valid pairs (distance < cutoff and i < j)
    mask = (dist_box < cutoff) & np.triu(np.ones_like(dist_box, dtype=np.bool_), k=1)
    i_idx, j_idx = np.nonzero(mask)

    # Step 4: Compute r values and theta values
    r_vals = dist_box[mask]
    vectors = box_shift(np.array(around_group.positions)[j_idx] - np.array(center_group.positions)[i_idx], box)
    polar_axis = np.asarray(polar_axis, dtype=cp.float32)
    polar_axis /= np.linalg.norm(polar_axis)  # normalize
    theta_vals = np.arccos(np.clip(np.sum(vectors * polar_axis, axis=1) / np.linalg.norm(vectors, axis=1), -1.0, 1.0))

    # Step 5: Prepare atom type pairs
    all_atom_types = sorted(set(universe.atoms.types))
    atom_types_center = np.array(center_group.types, dtype="<U4")
    atom_types_around = np.array(around_group.types, dtype="<U4")
    atom_pair_types = np.stack([atom_types_center[i_idx], atom_types_around[j_idx]], axis=1)
    atom_pair_types.sort(axis=1)  # enforce type1 <= type2

    # Step 6: Organize into dictionary grouped by atom pair types
    atom_pairs = {}
    for i, type1 in enumerate(all_atom_types):
        for type2 in all_atom_types[i:]:
            pair_mask = np.all(atom_pair_types == [type1, type2], axis=1)
            atom_pairs[(type1, type2)] = (r_vals[pair_mask], theta_vals[pair_mask])

    return atom_pairs

def generate_field(X: ArrayType, Y: ArrayType, r_vals: ArrayType, theta_vals: ArrayType, delta: np.float32) -> ArrayType:
    """
    Wrapper for field generation.
    """
    xp = cp.get_array_module(X, Y, r_vals, theta_vals)
    field = generate_field_cuda(X, Y, r_vals, theta_vals, delta)
    return field if xp.__name__ == "cupy" else field.get()

def compute_fields(
    atom_pairs: Dict[Tuple[str, str], Tuple[ArrayType, ArrayType]],
    X: ArrayType,
    Y: ArrayType,
    verbose: bool = False
) -> Dict[Tuple[str, str], ArrayType]:
    """
    Compute fields for all atom pair types.

    Returns:
        fields: Dict of atom-pair-type -> computed field.
    """
    h = X[1, 1] - X[0, 0]

    fields = {}
    for atom_pair_type, (r_vals, theta_vals) in atom_pairs.items():
        fields[atom_pair_type] = generate_field(X, Y, r_vals, theta_vals, delta=h)
        if verbose:
            print(f"Computed field for {atom_pair_type}, {r_vals.shape[0]} atom pairs.")

    return fields

def get_diff_fields(
    fields1: Dict[Tuple[str, str], ArrayType],
    fields2: Dict[Tuple[str, str], ArrayType]
) -> Dict[Tuple[str, str], ArrayType]:
    """
    Compute field differences between two sets of fields.

    Returns:
        diff_fields: Dict of atom-pair-type -> field difference.
    """
    return {pair_type: fields2[pair_type] - fields1[pair_type] for pair_type in fields1}

def forward_transform(diff_fields: Dict[Tuple[str, str], ArrayType], X: ArrayType, Y: ArrayType) -> ArrayType:
    """
    Forward Fourier filtering, Atomic Form Factor weighting, Inverse FFT, Inverse Abel Transform to get ARPDF.

    Parameters:
        diff_fields     : Dictionary mapping (atom_type1, atom_type2) -> diff fields
        X, Y            : 2D real-space grids

    Returns:
        ARPDF : 2D numpy array of Angularly Resolved Pair Distribution Function
    """
    xp = cp.get_array_module(X, Y)
    N = X.shape[0]
    h = X[1, 1] - X[0, 0]  # grid spacing

    # Fourier grid
    kx = xp.fft.fftfreq(N, d=h)
    ky = xp.fft.fftfreq(N, d=h)
    kX, kY = xp.meshgrid(kx, ky)
    S = xp.sqrt(kX**2 + kY**2)

    # Atomic Form Factors in Fourier space
    types_map = np.unique(np.array(tuple(diff_fields.keys())))
    AFFs = {atom: calc_AFF(atom, S) for atom in types_map}

    # Total FFT after applying atomic form factors
    total_fft = xp.zeros_like(X, dtype=xp.complex64)
    for (a1, a2), diff_field in diff_fields.items():
        fft = xp.fft.fft2(diff_field)           # 2D FFT
        fft *= AFFs[a1] * AFFs[a2]      # Apply atom form factors
        total_fft += fft

    # Filter in Fourier space
    _filter = (1 - xp.exp(-(kX**2 / 0.3 + kY**2 / 0.1)))**3 * xp.exp(-0.08 * S**2)

    # Approximate atomic form factor normalization
    I_atom = 1266 * xp.exp(-S**2 / (2.15044389**2)) + 414 * xp.exp(-S**2 / (4.60625893**2)) # TODO: I_atom should be calculated from box

    # Inverse FFT to real space
    total_fft = total_fft * _filter / I_atom
    total_ifft = xp.fft.ifft2(total_fft).real

    # Inverse Abel transform to get ARPDF
    # Inverse_Abel_total = abel.Transform(cp.asnumpy(total_ifft), method='basex', direction='inverse', transform_options={"verbose": False}).transform
    Inverse_Abel_total = abel_inversion(total_ifft)

    # Smoothing & r-weighting
    sigma0 = 0.4
    _gaussian_filter = gaussian_filter_cp if xp.__name__ == "cupy" else gaussian_filter_np
    ARPDF = _gaussian_filter(Inverse_Abel_total, sigma=sigma0/h) * (X**2 + Y**2)

    return ARPDF



def show_fields(fields: Dict[Tuple[str, str], np.ndarray], rmax: float = 10.0, title_prefix="Field"):
    """
    Visualize multiple field maps.

    Parameters:
        fields : Dictionary of 2D fields to plot
        rmax   : Display range in units
        title_prefix : Text prefix for subplot titles
    """
    fig, axs = plt.subplots(1, len(fields), figsize=(5 * len(fields), 5))
    axs = axs if isinstance(axs, np.ndarray) else [axs]
    for ax, (pair, field) in zip(axs, fields.items()):
        ax.imshow(field, cmap='inferno', origin='lower', extent=(-rmax, rmax, -rmax, rmax))
        ax.set_title(f"{title_prefix} {pair[0]}-{pair[1]}")
    plt.show()


def show_ARPDF(ARPDF: np.ndarray, plot_rmax: float = 10.0, show_rmax: float = 8.0, max_intensity: float = 0.1):
    """
    Visualize the reconstructed ARPDF.

    Parameters:
        ARPDF : ARPDF array
        rmax  : Display range in real-space
        xy_range : x-y axis display range
        max_intensity : Colorbar clipping intensity
    """
    plt.imshow(ARPDF, cmap='bwr', origin='lower', extent=(-plot_rmax, plot_rmax, -plot_rmax, plot_rmax))
    plt.xlim([-show_rmax, show_rmax])
    plt.ylim([-show_rmax, show_rmax])
    plt.clim(-max_intensity, max_intensity)
    plt.colorbar(label="Reconstructed Intensity")
    plt.show()


def compute_ARPDF(
    u1: mda.Universe,
    u2: mda.Universe,
    cutoff: float = 10.0,
    N: int = 512,
    grids_XY: Optional[Tuple[ArrayType, ArrayType]] = None,
    modified_atoms: Optional[List[int]] = None,
    polar_axis = (0, 0, 1),
    periodic: bool = False,
    verbose: bool = False,
    use_cupy: bool = True
) -> np.ndarray:
    """
    Main pipeline: u1, u2 -> generate diff fields -> FFT -> AFF+filter -> Inverse FFT -> Inverse Abel -> ARPDF

    Parameters:
        u1, u2          : MDAnalysis Universe objects (before/after structure)
        cutoff          : cutoff radius (A)
        N               : grid size (NxN)
        grids_XY        : optional pre-generated grids
        modified_atoms  : optional list of atom indices to modify
        polar_axis      : polarization axis of the laser
        periodic        : if True, consider periodic boundary condition
        verbose         : if True, show intermediate plots

    Returns:
        ARPDF           : Angularly Resolved Pair Distribution Function (numpy array)
    """
    if grids_XY is None:
        X, Y = generate_grids(cutoff, N, use_cupy=use_cupy)
    else:
        X, Y = grids_XY
        N = X.shape[0]

    # Compute all atom pairs
    atom_pairs1 = compute_all_atom_pairs(u1, cutoff, modified_atoms, polar_axis, periodic)
    atom_pairs2 = compute_all_atom_pairs(u2, cutoff, modified_atoms, polar_axis, periodic)

    # Convert to cupy if needed
    if use_cupy:
        atom_pairs1 = to_cupy(atom_pairs1)
        atom_pairs2 = to_cupy(atom_pairs2)
    
    # Compute 2D projected fields for both structures
    fields1 = compute_fields(atom_pairs1, X, Y, verbose)
    fields2 = compute_fields(atom_pairs2, X, Y, verbose)

    # Difference fields
    diff_fields = get_diff_fields(fields1, fields2)

    if verbose:
        show_fields(to_numpy(diff_fields), rmax=cutoff, title_prefix="Diff Field")
        show_fields(to_numpy(fields1), rmax=cutoff, title_prefix="Original Field")

    # ARPDF computation
    ARPDF = to_numpy(forward_transform(diff_fields, X, Y))

    if verbose:
        show_ARPDF(ARPDF, plot_rmax=cutoff, show_rmax=8, max_intensity=0.1)

    return ARPDF

