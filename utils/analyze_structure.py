import numpy as np
import MDAnalysis as mda
import MDAnalysis.analysis.distances as mda_dist
from typing import Tuple, Dict, Optional, Callable, List
import os
from utils import load_structure_data, box_shift, rotation_matrix
from dataclasses import dataclass

@dataclass
class StructureAnalysisResult:
    dist_C_A_CL_A: float  # Distance between C and CL in first molecule
    dist_CL_A_CL_B: float  # Distance between two CL atoms
    dist_C_B_CL_B: float  # Distance between C and CL in second molecule
    dist_C_A_C_B: float  # Distance between two C atoms
    theta_CL_A_CL_B: float  # Angle between CL-CL vector and target axis (in degrees)

def select_ccl4_molecules(
    u: mda.Universe,
    cl_index: int,
    cutoff_distance: float = 5.0
) -> List[int]:
    """
    Select CCl4 molecules for analysis.
    
    Args:
        u (mda.Universe): MDAnalysis universe containing the structure
        cl_index (int): Index of the reference CL atom
        cutoff_distance (float): Distance cutoff for finding nearby CL atoms
        
    Returns:
        List[int]: Indices of selected atoms
    """
    # Get reference CL atom and its molecule
    CL_A = u.atoms[cl_index]
    resid_A = CL_A.resid
    mol_A = u.select_atoms(f"resid {resid_A}")
    
    # Find C atom in first molecule
    C_A = mol_A[np.nonzero(np.array(mol_A.types) == 'C')[0]][0]
    
    # Find nearest CL atom
    around_group = u.select_atoms(f"(around {cutoff_distance} group mol_A) and type CL", mol_A=mol_A)
    d = mda_dist.distance_array(CL_A.position, around_group.positions, u.dimensions)[0]
    CL_B = around_group[d.argmin()]
    
    # Get second molecule
    resid_B = CL_B.resid
    mol_B = u.select_atoms(f"resid {resid_B}")
    
    # Find C atom in second molecule
    C_B = mol_B[np.nonzero(np.array(mol_B.types) == 'C')[0]][0]
    
    # Construct ordered indices list with C atoms first, followed by CL atoms
    mol_A_indices = [C_A.index, CL_A.index] + [i for i in mol_A.indices if i not in [C_A.index, CL_A.index]]
    mol_B_indices = [C_B.index, CL_B.index] + [i for i in mol_B.indices if i not in [C_B.index, CL_B.index]]
    print(mol_A_indices)
    print(mol_B_indices)

    # Return combined ordered indices
    return mol_A_indices + mol_B_indices

def analyze_ccl4_structure(
    u: mda.Universe,
    cl_index: int,
    polar_axis: List[float] | None = None,
    output_dir: str = "output",
    cutoff_distance: float = 5.0,
    save_structure: bool = True
) -> StructureAnalysisResult:
    """
    Analyze molecular structure and extract key geometric parameters.
    
    Args:
        u (mda.Universe): MDAnalysis universe containing the structure
        cl_index (int): Index of the reference CL atom
        output_dir (str): Directory to save output files
        selector (Callable): Function to select molecules for analysis
        cutoff_distance (float): Distance cutoff for finding nearby CL atoms
        save_structure (bool): Whether to save the processed structure
        
    Returns:
        StructureAnalysisResult: Analysis results
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Select molecules using the provided selector
    selected_indices = select_ccl4_molecules(u, cl_index, cutoff_distance)
    
    # Get reference atoms
    mol_A = u.atoms[selected_indices[:5]]
    mol_B = u.atoms[selected_indices[5:]]
    C_A = mol_A[0]
    CL_A = mol_A[1]
    C_B = mol_B[0]
    CL_B = mol_B[1]
    print(C_A.index, CL_A.index, C_B.index, CL_B.index)
    
    # Calculate polar_axis (C to CL direction)
    if polar_axis is None:
        polar_axis = CL_A.position - C_A.position
        polar_axis /= np.linalg.norm(polar_axis)
    
    # Rotate and translate molecules
    target_axis = np.array([0, 0, 1])
    R = rotation_matrix(polar_axis, target_axis)
    _center = mol_A[0].position[None, :]
    mol_A.positions = box_shift(mol_A.positions - _center, u.dimensions) @ R.T
    mol_B.positions = box_shift(mol_B.positions - _center, u.dimensions) @ R.T
    
    # Save processed structure if requested
    if save_structure:
        (mol_A + mol_B).write(f"{output_dir}/selected_structure.gro")
    
    # Calculate geometric parameters
    dist_C_A_CL_A = mda_dist.distance_array(C_A.position, CL_A.position, u.dimensions)[0, 0]
    dist_CL_A_CL_B = mda_dist.distance_array(CL_A.position, CL_B.position, u.dimensions)[0, 0]
    dist_C_B_CL_B = mda_dist.distance_array(C_B.position, CL_B.position, u.dimensions)[0, 0]
    dist_C_A_C_B = mda_dist.distance_array(C_A.position, C_B.position, u.dimensions)[0, 0]
    
    vector_CL_A_CL_B = box_shift(CL_B.position - CL_A.position, u.dimensions)
    vector_CL_A_CL_B /= dist_CL_A_CL_B
    theta_CL_A_CL_B = np.arccos(np.dot(vector_CL_A_CL_B, target_axis))
    
    # Return results
    return StructureAnalysisResult(
        dist_C_A_CL_A=dist_C_A_CL_A,
        dist_CL_A_CL_B=dist_CL_A_CL_B,
        dist_C_B_CL_B=dist_C_B_CL_B,
        dist_C_A_C_B=dist_C_A_C_B,
        theta_CL_A_CL_B=np.degrees(theta_CL_A_CL_B)
    ) 