from typing import Dict, List, Optional, Tuple
import cupy as cp
import numpy as np
import abel
import MDAnalysis as mda
import MDAnalysis.analysis.distances as mda_dist
from cupyx.scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from utils import generate_field_cuda, box_shift, generate_grids, AFF_map_cp as AFF_map, abel_inversion, cosine_similarity

def compute_all_atom_pairs(
        universe: mda.Universe, 
        cutoff: float = 10.0, 
        modified_atoms: List[int] = None, 
        polar_axis = (0, 0, 1),
        periodic = False
    ) -> Dict[Tuple[str, str], Tuple[cp.ndarray, cp.ndarray]]:
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
    dist_box = cp.array(
        mda_dist.distance_array(center_group.positions, around_group.positions, box),
        dtype=cp.float32
    )

    # Step 3: Mask valid pairs (distance < cutoff and i < j)
    mask = (dist_box < cutoff) & cp.triu(cp.ones_like(dist_box, dtype=cp.bool_), k=1)
    i_idx, j_idx = cp.nonzero(mask)

    # Step 4: Compute r values and theta values
    r_vals = dist_box[mask]
    vectors = box_shift(cp.array(around_group.positions)[j_idx] - cp.array(center_group.positions)[i_idx], box)
    polar_axis = cp.asarray(polar_axis, dtype=cp.float32)
    polar_axis /= cp.linalg.norm(polar_axis)  # normalize
    theta_vals = cp.arccos(cp.clip(cp.sum(vectors * polar_axis, axis=1) / cp.linalg.norm(vectors, axis=1), -1.0, 1.0))

    # Step 5: Prepare atom type pairs
    all_atom_types = sorted(set(universe.atoms.types))
    atom_types_center = np.array(center_group.types, dtype="<U4")
    atom_types_around = np.array(around_group.types, dtype="<U4")
    atom_pair_types = np.stack([atom_types_center[i_idx.get()], atom_types_around[j_idx.get()]], axis=1)
    atom_pair_types.sort(axis=1)  # enforce type1 <= type2

    # Step 6: Organize into dictionary grouped by atom pair types
    atom_pairs = {}
    for i, type1 in enumerate(all_atom_types):
        for type2 in all_atom_types[i:]:
            pair_mask = np.all(atom_pair_types == [type1, type2], axis=1)
            atom_pairs[(type1, type2)] = (r_vals[pair_mask], theta_vals[pair_mask])

    return atom_pairs

def generate_field(X: cp.ndarray, Y: cp.ndarray, r_vals: cp.ndarray, theta_vals: cp.ndarray, delta: cp.float32) -> cp.ndarray:
    """
    Wrapper for CUDA-based field generation.
    """
    return generate_field_cuda(X, Y, r_vals, theta_vals, delta)

def compute_fields(
    universe: mda.Universe,
    X: cp.ndarray,
    Y: cp.ndarray,
    cutoff: float = 10.0,
    modified_atoms: Optional[list] = None,
    polar_axis = (0, 0, 1),
    periodic = False,
    verbose: bool = False
) -> Tuple[Dict[Tuple[str, str], cp.ndarray], cp.ndarray, cp.ndarray]:
    """
    Compute fields for all atom pair types.

    Returns:
        fields: Dict of atom-pair-type -> computed field.
    """
    h = X[1, 1] - X[0, 0]

    atom_pairs = compute_all_atom_pairs(universe, cutoff, modified_atoms, polar_axis, periodic)

    fields = {}
    for atom_pair_type, (r_vals, theta_vals) in atom_pairs.items():
        fields[atom_pair_type] = generate_field(X, Y, r_vals, theta_vals, delta=h)
        if verbose:
            print(f"Computed field for {atom_pair_type}, {r_vals.shape[0]} atom pairs.")

    return fields

def get_diff_fields(
    fields1: Dict[Tuple[str, str], cp.ndarray],
    fields2: Dict[Tuple[str, str], cp.ndarray]
) -> Dict[Tuple[str, str], cp.ndarray]:
    """
    Compute field differences between two sets of fields.

    Returns:
        diff_fields: Dict of atom-pair-type -> field difference.
    """
    return {pair_type: fields2[pair_type] - fields1[pair_type] for pair_type in fields1}

def forward_transform(diff_fields: Dict[Tuple[str, str], cp.ndarray], X: cp.ndarray, Y: cp.ndarray) -> cp.ndarray:
    """
    Forward Fourier filtering, Atomic Form Factor weighting, Inverse FFT, Inverse Abel Transform to get ARPDF.

    Parameters:
        diff_fields     : Dictionary mapping (atom_type1, atom_type2) -> diff fields
        X, Y            : 2D real-space grids

    Returns:
        ARPDF : 2D numpy array of Angularly Resolved Pair Distribution Function
    """
    N = X.shape[0]
    h = X[1, 1] - X[0, 0]  # grid spacing

    # Fourier grid
    kx = cp.fft.fftfreq(N, d=h)
    ky = cp.fft.fftfreq(N, d=h)
    kX, kY = cp.meshgrid(kx, ky)
    S = cp.sqrt(kX**2 + kY**2)

    # Atomic Form Factors in Fourier space
    types_map = np.unique(np.array(tuple(diff_fields.keys())))
    AFF_vals = {atom: AFF_map[atom](S) for atom in types_map}

    # Total FFT after applying atomic form factors
    total_fft = cp.zeros_like(X, dtype=cp.complex64)
    for (a1, a2), diff_field in diff_fields.items():
        fft = cp.fft.fft2(diff_field)           # 2D FFT
        fft *= AFF_vals[a1] * AFF_vals[a2]      # Apply atom form factors
        total_fft += fft

    # Filter in Fourier space
    _filter = (1 - cp.exp(-(kX**2 / 0.3 + kY**2 / 0.1)))**3 * cp.exp(-0.08 * S**2)

    # Approximate atomic form factor normalization
    I_atom = 1266 * cp.exp(-S**2 / (2.15044389**2)) + 414 * cp.exp(-S**2 / (4.60625893**2)) # TODO: I_atom should be calculated from box

    # Inverse FFT to real space
    total_fft = total_fft * _filter / I_atom
    total_ifft = cp.fft.ifft2(total_fft).real

    # Inverse Abel transform to get ARPDF
    # Inverse_Abel_total = abel.Transform(cp.asnumpy(total_ifft), method='basex', direction='inverse', transform_options={"verbose": False}).transform
    Inverse_Abel_total = abel_inversion(total_ifft)

    # Smoothing & r-weighting
    sigma0 = 0.4
    ARPDF = gaussian_filter(Inverse_Abel_total, sigma=sigma0/h) * (X**2 + Y**2)

    return ARPDF



def show_fields(fields: Dict[Tuple[str, str], cp.ndarray], rmax: float = 10.0, title_prefix="Field"):
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
        ax.imshow(cp.asnumpy(field), cmap='inferno', origin='lower', extent=(-rmax, rmax, -rmax, rmax))
        ax.set_title(f"{title_prefix} {pair[0]}-{pair[1]}")
    plt.show()


def show_ARPDF(ARPDF: np.ndarray, rmax: float = 10.0, xy_range: Tuple[float, float] = (-8, 8), max_intensity: float = 0.1):
    """
    Visualize the reconstructed ARPDF.

    Parameters:
        ARPDF : ARPDF array
        rmax  : Display range in real-space
        xy_range : x-y axis display range
        max_intensity : Colorbar clipping intensity
    """
    plt.imshow(ARPDF, cmap='bwr', origin='lower', extent=(-rmax, rmax, -rmax, rmax))
    plt.xlim(xy_range)
    plt.ylim(xy_range)
    plt.clim(-max_intensity, max_intensity)
    plt.colorbar(label="Reconstructed Intensity")
    plt.show()


def compute_ARPDF(
    u1: mda.Universe,
    u2: mda.Universe,
    cutoff: float = 10.0,
    N: int = 512,
    grids_XY: Optional[Tuple[cp.ndarray, cp.ndarray]] = None,
    modified_atoms: Optional[List[int]] = None,
    polar_axis = (0, 0, 1),
    periodic: bool = False,
    verbose: bool = False
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
        X, Y = generate_grids(cutoff, N)
    else:
        X, Y = grids_XY

    # Compute 2D projected fields for both structures
    fields1 = compute_fields(u1, X, Y, cutoff, modified_atoms, polar_axis, periodic, verbose)
    fields2 = compute_fields(u2, X, Y, cutoff, modified_atoms, polar_axis, periodic, verbose)

    # Difference fields
    diff_fields = get_diff_fields(fields1, fields2)

    if verbose:
        show_fields(diff_fields, rmax=cutoff, title_prefix="Diff Field")
        show_fields(fields1, rmax=cutoff, title_prefix="Original Field")

    # ARPDF computation
    ARPDF = forward_transform(diff_fields, X, Y).get()

    if verbose:
        show_ARPDF(ARPDF, rmax=cutoff, xy_range=(-8, 8), max_intensity=0.1)

    return ARPDF

