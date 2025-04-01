from collections import Counter
from typing import Dict, List, Optional, Tuple
import cupy as cp
from matplotlib import pyplot as plt
import numpy as np
import MDAnalysis as mda
import MDAnalysis.analysis.distances as mda_dist
from cupyx.scipy.ndimage import gaussian_filter as gaussian_filter_cp
from scipy.ndimage import gaussian_filter as gaussian_filter_np
from utils import (
    generate_field_cuda, box_shift, generate_grids, calc_AFF, abel_inversion, 
    to_cupy, to_numpy, show_images, cosine_similarity
)
from utils import ArrayType

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
    polar_axis = np.asarray(polar_axis, dtype=cp.float32)
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

def generate_field(X: ArrayType, Y: ArrayType, r_vals: ArrayType, theta_vals: ArrayType, delta: np.float32) -> ArrayType:
    """
    Wrapper for field generation.
    """
    xp = cp.get_array_module(X, Y, r_vals, theta_vals)
    field = generate_field_cuda(X, Y, r_vals, theta_vals, delta)
    return field if xp.__name__ == "cupy" else field.get()

def compute_fields(
    atom_pairs: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]],
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
        fields[atom_pair_type] = generate_field(X, Y, r_vals, theta_vals, delta=2*h)
        if verbose:
            print(f"Computed field for {atom_pair_type[0]}-{atom_pair_type[1]}: {r_vals.shape[0]} atom pairs.")

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

def forward_transform(diff_fields: Dict[Tuple[str, str], ArrayType], X: ArrayType, Y: ArrayType, type_counts: Dict[str, int]) -> ArrayType:
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
    AFFs = {atom: calc_AFF(atom, S) for atom in type_counts}
    I_atom = sum([num_atom * AFFs[atom]**2 for atom, num_atom in type_counts.items()])

    # Total FFT after applying atomic form factors
    total_fft = xp.zeros_like(X, dtype=xp.complex64)
    for (a1, a2), diff_field in diff_fields.items():
        fft = xp.fft.fft2(diff_field)           # 2D FFT
        fft *= AFFs[a1] * AFFs[a2]      # Apply atom form factors
        total_fft += fft

    # Filter in Fourier space
    _filter = (1 - xp.exp(-(kX**2 / 0.3 + kY**2 / 0.1)))**3 * xp.exp(-0.08 * S**2)

    # Inverse FFT to real space
    total_fft = total_fft * _filter / I_atom
    total_ifft = xp.fft.ifft2(total_fft).real

    # Inverse Abel transform to get ARPDF
    # Inverse_Abel_total = abel.Transform(cp.asnumpy(total_ifft), method='basex', direction='inverse', transform_options={"verbose": False}).transform
    Inverse_Abel_total = abel_inversion(total_ifft) / h

    # Smoothing & r-weighting
    sigma0 = 0.4
    _gaussian_filter = gaussian_filter_cp if xp.__name__ == "cupy" else gaussian_filter_np
    ARPDF = _gaussian_filter(Inverse_Abel_total, sigma=sigma0/h) * (X**2 + Y**2)

    return ARPDF

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
    _print_func = print if verbose else lambda *args: None

    # Compute all atom pairs
    atom_pairs1, num_sel1 = compute_all_atom_pairs(u1, cutoff, modified_atoms, polar_axis, periodic)
    atom_pairs2, num_sel2 = compute_all_atom_pairs(u2, cutoff, modified_atoms, polar_axis, periodic)
    _print_func(f"Selected {num_sel1} atoms for universe 1, {num_sel2} atoms for universe 2.")

    # Convert to cupy if needed
    if use_cupy:
        atom_pairs1 = to_cupy(atom_pairs1)
        atom_pairs2 = to_cupy(atom_pairs2)
    
    # Compute 2D projected fields for both structures
    _print_func("Computing fields of universe 1...")
    fields1 = compute_fields(atom_pairs1, X, Y, verbose)
    _print_func("Computing fields of universe 2...")
    fields2 = compute_fields(atom_pairs2, X, Y, verbose)

    # Difference fields
    diff_fields = get_diff_fields(fields1, fields2)

    if verbose:
        show_images(to_numpy(fields1).items(), plot_range=cutoff, colorbar="all", cmap="inferno", 
                    title=lambda x: f"Field for {x[0]}-{x[1]}")
        diff_fields_np = to_numpy(diff_fields)
        c_range = np.array([np.abs(field).max() for field in diff_fields_np.values()])
        c_range = np.array([-c_range, c_range]).T
        show_images(diff_fields_np.items(), plot_range=cutoff, c_range=c_range, colorbar="all", cmap="bwr", 
                    title=lambda x: f"Diff Field for {x[0]}-{x[1]}")

    # ARPDF computation
    _print_func("Computing ARPDF...")
    ARPDF = to_numpy(forward_transform(diff_fields, X, Y, Counter(u1.atoms.types)))
    normalize_factor = num_sel1 / len(u1.atoms)
    ARPDF = ARPDF / normalize_factor * 100

    if verbose:
        show_images([("ARPDF", ARPDF)], plot_range=cutoff, show_range=8, cmap="bwr", c_range=0.5*ARPDF.max(), 
                    clabel="Reconstructed Intensity") #, interpolation='bicubic')
        plt.show()

    return ARPDF

def compare_ARPDF(ARPDF, ARPDF_exp, grids_XY, cos_sim = None, show_range = 8.0):
    if cos_sim is None:
        cos_sim = cosine_similarity(ARPDF, ARPDF_exp)
    X, Y = grids_XY
    h = X[1, 1] - X[0, 0]
    ARPDF /= np.linalg.norm(ARPDF) * h**2 + 1e-3
    ARPDF_exp /= np.linalg.norm(ARPDF_exp) * h**2 + 1e-3
    xmin, xmax = X.min(), X.max()
    ymin, ymax = Y.min(), Y.max()
    return show_images({f"ARPDF (Sim: {cos_sim:0.2f})": ARPDF, "ARPDF (Experimental)": ARPDF_exp}.items(), 
                      plot_range=[xmin, xmax, ymin, ymax], show_range=show_range, c_range=1.5,
                        cmap="bwr", colorbar="align")