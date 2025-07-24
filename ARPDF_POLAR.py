from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple
from matplotlib import pyplot as plt
import numpy as np
import MDAnalysis as mda
import MDAnalysis.analysis.distances as mda_dist
from scipy.ndimage import gaussian_filter as gaussian_filter_np
from utils import box_shift, generate_grids, calc_AFF, show_images, show_images_polar,compute_all_atom_pairs, get_crossection
from utils.core_functions import ArrayType, get_array_module, to_cupy, to_numpy, abel_inversion, generate_field_polar
from types import ModuleType
from scipy.special import i0
from utils.similarity import cosine_similarity

'''
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
'''

def get_atoms_pos(
        universe: mda.Universe, 
        modified_atoms: List[int], 
        cutoff: float = 10.0, 
        periodic = False
    ) -> Tuple[np.ndarray, np.ndarray, Dict[Tuple[str, str], np.ndarray]]:
    center_group = universe.atoms[modified_atoms]
    around_group = universe.select_atoms(f"around {cutoff} group center", center=center_group, periodic=periodic)
    center_pos = center_group.positions
    around_pos = around_group.positions
    if periodic:
        _center = center_pos[[0]]
        center_pos = _center + box_shift(center_pos - _center, box=universe.dimensions)
        around_pos = _center + box_shift(around_pos - _center, box=universe.dimensions)
    concat_group = center_group + around_group
    
    mask = np.triu(np.ones((len(center_group), len(concat_group)), dtype=np.bool_), k=1)
    i_idx, j_idx = np.nonzero(mask)

    all_atom_types = sorted(set(universe.atoms.types))
    atom_types_center = np.array(center_group.types, dtype="<U4")
    atom_types_concat = np.array(concat_group.types, dtype="<U4")
    atom_pair_types = np.stack([atom_types_center[i_idx], atom_types_concat[j_idx]], axis=1)
    atom_pair_types.sort(axis=1)
    ij_idx = np.stack([i_idx, j_idx], axis=1)

    atom_pairs = {}
    for i, type1 in enumerate(all_atom_types):
        for type2 in all_atom_types[i:]:
            pair_mask = np.all(atom_pair_types == [type1, type2], axis=1)
            atom_pairs[(type1, type2)] = ij_idx[pair_mask]
    return center_pos, around_pos, center_group.masses, atom_pairs

def compute_fields_polar(
    atom_pairs: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]],
    R: ArrayType,
    Phi: ArrayType,
    sigma0: Optional[float] = None,  
    verbose: bool = False
) -> Dict[Tuple[str, str], ArrayType]:
    """
    Compute fields for all atom pair types using precomputed polar coordinates.

    Parameters:
        atom_pairs: Dict mapping atom pair (e.g., ('O', 'H')) to (r_vals, theta_vals) arrays.
        R, Phi: Polar coordinate grid arrays (can be NumPy or CuPy).
        verbose: Whether to print progress info.

    Returns:
        Dict of atom-pair-type -> computed polar field.
    """
    
    xp = get_array_module(R)  # Determine NumPy or CuPy backend
    if sigma0 is None:
        h = R[1, 1] - R[0, 0]  # Assume uniform spacing in R
        sigma0 = 2 * h 

    fields = {}
    
    for atom_pair_type, (r_vals, theta_vals) in atom_pairs.items():
        r_vals = xp.array(r_vals)
        theta_vals = xp.array(theta_vals)
        field = generate_field_polar(R, Phi, r_vals, theta_vals, sigma0)
        fields[atom_pair_type] = field

        if verbose:
            print(f"Computed polar field for {atom_pair_type[0]}-{atom_pair_type[1]}: {r_vals.shape[0]} atom pairs.")

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

   

def compute_ARPDF_polar(
    u1: mda.Universe,
    u2: mda.Universe,
    N: int | None = 512,
    cutoff: float = 10.0,
    sigma0=0.4,
    #delta=None,
    grids_polar: Optional[Tuple[ArrayType, ArrayType]] = None,  # (R, Phi)
    modified_atoms: Optional[List[int]] = None,
    polar_axis=(0, 0, 1),
    periodic: bool = False,
    #filter_fourier: Optional[Callable[[ArrayType, ArrayType, ModuleType], ArrayType]] = None,
    verbose: bool = False,
    neg: bool = False
) -> ArrayType:
    """
    Main pipeline (polar version): 
    u1, u2 -> atom pair projections (r, θ) -> generate polar fields -> diff -> ARPDF.

    Parameters
    ----------
    u1, u2          : MDAnalysis Universe objects (before/after structure)
    cutoff          : cutoff radius (A)
    N               : grid size (Nr × Nθ)
    grids_polar     : optional pre-generated (R, Phi) meshgrid
    modified_atoms  : optional list of atom indices to modify
    polar_axis      : polarization axis of the laser
    periodic        : if True, consider periodic boundary condition
    verbose         : if True, show intermediate output

    Returns
    -------
    ARPDF           : Dict of angularly resolved difference fields in polar coordinates.
    """
    _print_func = print if verbose else lambda *args: None

    try:
        import cupy
        try:
            _ = cp.zeros((1,), dtype=cp.float32)  # 若CUDA不可用，此处可能报错
            has_cupy = True
            _print_func("Using cupy to compute ARPDF...")
        except Exception:
            has_cupy = False
            _print_func("Cannot use cupy, using numpy instead.")
    except ImportError:
        has_cupy = False
        _print_func("Cannot find cupy, using numpy instead.")

    # Handle polar grids
    if grids_polar is None:
        R, Phi = generate_grids(cutoff, N)  # You need to define this helper
        input_type = "numpy"
    else:
        R, Phi = grids_polar
        input_type = get_array_module(R).__name__
        N = R.shape[0]

    if has_cupy and input_type == "numpy":
        R, Phi = to_cupy(R), to_cupy(Phi)

    # Compute atom pairs in polar projection (returns (r_vals, theta_vals))
    atom_pairs1, num_sel1 = compute_all_atom_pairs(u1, cutoff, modified_atoms, polar_axis, periodic=periodic)
    atom_pairs2, num_sel2 = compute_all_atom_pairs(u2, cutoff, modified_atoms, polar_axis, periodic=periodic)

    if has_cupy:
        atom_pairs1 = to_cupy(atom_pairs1)
        atom_pairs2 = to_cupy(atom_pairs2)

    _print_func(f"Selected {num_sel1} atoms in u1, {num_sel2} atoms in u2.")

    # Compute projected fields in polar coordinates
    _print_func("Computing polar fields for universe 1...")
    fields1 = compute_fields_polar(atom_pairs1, R, Phi, sigma0, verbose)
    _print_func("Computing polar fields for universe 2...")
    fields2 = compute_fields_polar(atom_pairs2, R, Phi, sigma0, verbose)

    # Difference
    diff_fields = get_diff_fields(fields1, fields2)

    if verbose:
        diff_fields_np = to_numpy(diff_fields)
        show_images_polar(
            diff_fields_np.items(),
            r_range=(0, cutoff),                # 你的半径范围
            phi_range=(0, 0.5 * np.pi),         # 角度范围
            cmap="bwr",
            title=lambda x: f"Polar Diff Field for {x[0]}-{x[1]}",
        )

    # Final ARPDF = difference of fields (can include filtering, IFFT, Abel, etc. as needed)
    ARPDF = diff_fields

    if neg:
        for key in ARPDF:
            ARPDF[key][ARPDF[key] > 0] = 0

    # Weighted sum of ARPDF
    #weights = generate_pair_weights() 
    crossection = {atom: get_crossection(atom) for atom in Counter(u1.atoms.types)}

    total_ARPDF = None
    for key, field in ARPDF.items():
        A, B = key  # key 形如 ('C', 'Cl')
        #sorted_key = tuple(sorted(key))
        #sorted_key = tuple(sorted(key))
        #weight = weights.get(sorted_key, 1)
        weight = crossection[A]*crossection[B]
        weighted_field = field * weight
        total_ARPDF = weighted_field if total_ARPDF is None else total_ARPDF + weighted_field

    ARPDF["total"] = total_ARPDF

    if verbose:
        ARPDF_np = to_numpy(ARPDF) 
        show_images_polar(
            [("total", ARPDF_np["total"])],
            r_range=(0, cutoff),
            phi_range=(0, 0.5 * np.pi),
            cmap="bwr",
            title=lambda x: f"Weighted Polar Diff Field (Total)"
        )

    field = {k: v for k, v in ARPDF.items() if k != "total"}
    ARPDF = ARPDF["total"]
    return ARPDF if input_type == "cupy" else to_numpy(ARPDF), field if input_type == "cupy" else to_numpy(field)

    #return ARPDF if input_type == "cupy" else to_numpy(ARPDF)

def compare_ARPDF_polar(ARPDF, ARPDF_ref, grids_polar, sim_name="Polar Sim", sim_value=None, show_range=8.0, weight_cutoff=5.0):
    """
    Compare ARPDF and ARPDF_ref in polar coordinates.

    Parameters
    ----------
    ARPDF        : 2D array in polar (R, Phi)
    ARPDF_ref    : reference 2D array in polar (R, Phi)
    grids_polar  : tuple (R, Phi), both are 2D arrays from meshgrid
    sim_name     : name of similarity metric
    sim_value    : similarity value (optional, can be computed elsewhere)
    show_range   : radial range for plot (optional)
    weight_cutoff: normalization cutoff (optional)
    """
    if sim_value is None:
        sim_value = cosine_similarity(ARPDF, ARPDF_ref)  # or whatever metric you're using

    R_grid, Phi_grid = grids_polar
    ARPDF = ARPDF.copy()
    ARPDF_ref = ARPDF_ref.copy()

    # Normalize based on region within weight_cutoff
    mask = R_grid < (weight_cutoff + 0.5)
    ARPDF /= ARPDF[mask].max() + 1e-6
    ARPDF_ref /= ARPDF_ref[mask].max() + 1e-6

    # Make plot titles
    images = {
        f"ARPDF ({sim_name}: {sim_value:.2f})": ARPDF,
        "ARPDF (Reference)": ARPDF_ref
    }

    # Display the polar image using imshow (in polar domain)
    return show_images_polar(
        images.items(),
        r_range=(0, show_range),
        phi_range=(0, 0.5 * np.pi),
        cmap="bwr",
        colorbar="align"
    )
