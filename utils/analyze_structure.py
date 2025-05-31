import numpy as np
import MDAnalysis as mda
import MDAnalysis.analysis.distances as mda_dist
from typing import Tuple, Dict, Optional, Callable, List
import os
from utils import load_structure_data, box_shift, rotation_matrix, copy_atom_group
from dataclasses import dataclass

@dataclass
class StructureAnalysisResult:
    dist_C_A_CL_A: float  # Distance between C and CL in first molecule
    dist_CL_A_CL_B: float  # Distance between two CL atoms
    dist_C_B_CL_B: float  # Distance between C and CL in second molecule
    dist_C_A_C_B: float  # Distance between two C atoms
    theta_CL_A_CL_B: float  # Angle between CL-CL vector and target axis (in degrees)
    umbrella_angle: float # Angle of CCl3 cylinder
    dists_CL_A_to_others: List[float] #Distance between CL_A and other CL

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

    # Return combined ordered indices
    return mol_A_indices + mol_B_indices


def select_nearest_ccl4_molecules(
        u: mda.Universe,
        cl_index: int,
        n_neighbors: int = 3,  # 新增参数
        cutoff_distance: float = 10.0,  # 给大一点保证能找到足够多
    ) -> List[List[int]]:
    """
    Select CCl4 molecules for analysis.

    Args:
        u (mda.Universe): MDAnalysis universe containing the structure
        cl_index (int): Index of the reference CL atom
        n_neighbors (int): Number of nearest neighbors to find
        cutoff_distance (float): Distance cutoff
    Returns:
        List[List[int]]: List of indices list for each neighbor
    """
    CL_A = u.atoms[cl_index]
    resid_A = CL_A.resid
    mol_A = u.select_atoms(f"resid {resid_A}")
    C_A = mol_A[np.nonzero(np.array(mol_A.types) == 'C')[0]][0]

    # Find nearby CL atoms
    around_group = u.select_atoms(f"(around {cutoff_distance} group mol_A) and type CL", mol_A=mol_A)
    d = mda_dist.distance_array(CL_A.position, around_group.positions, u.dimensions)[0]

    # 按距离排序，取最近的n_neighbors个
    nearest_indices = d.argsort()[:n_neighbors]
    CL_B_list = [around_group[i] for i in nearest_indices]

    result = []
    for CL_B in CL_B_list:
        resid_B = CL_B.resid
        mol_B = u.select_atoms(f"resid {resid_B}")
        C_B = mol_B[np.nonzero(np.array(mol_B.types) == 'C')[0]][0]

        mol_A_indices = [C_A.index, CL_A.index] + [i for i in mol_A.indices if i not in [C_A.index, CL_A.index]]
        mol_B_indices = [C_B.index, CL_B.index] + [i for i in mol_B.indices if i not in [C_B.index, CL_B.index]]

        # 一组组合：自己（mol_A）和一个邻居（mol_B）
        result.append(mol_A_indices + mol_B_indices)

    return result  # 注意，这里是 List[List[int]]



def rotate_ccl4_molecules(
        u: mda.Universe,
        ccl4_indices: List[int],
        selection: List[int] | None = None,
        polar_axis: np.ndarray | None = None
    ) -> mda.AtomGroup:
    """
    Rotate and translate CCl4 molecules to align the polar axis to z-axis.
    This will copy the atoms to a new universe and return the new atom group.

    Args:
        u (mda.Universe): MDAnalysis universe containing the structure
        ccl4_indices (List[int]): Indices of CCl4 molecules
        selection (List[int]): Indices of selected atoms
        polar_axis (np.ndarray): Polar axis for alignment
        
    Returns:
        mda.AtomGroup: Selected atoms after alignment
    """
    ccl4_mols = u.atoms[ccl4_indices]
    if selection is not None:
        ag_new = copy_atom_group(u.atoms[selection])
    else:
        ag_new = copy_atom_group(ccl4_mols)
    if polar_axis is None:
        polar_axis = ccl4_mols[1].position - ccl4_mols[0].position
        polar_axis /= np.linalg.norm(polar_axis)
    R = rotation_matrix(polar_axis, np.array([0, 0, 1]))
    CL_B_pos = ccl4_mols[6].position
    _center = ccl4_mols[0].position
    CL_B_pos = box_shift(CL_B_pos - _center, u.dimensions) @ R.T

    # Rotate around z-axis to align CL_B to xz plane
    theta = np.arctan2(CL_B_pos[1], CL_B_pos[0])
    R_z = np.array([
        [np.cos(theta), np.sin(theta), 0],
        [-np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    ag_new.positions = box_shift(ag_new.positions - _center, u.dimensions) @ (R_z @ R).T
    return ag_new
    

def analyze_ccl4_structure(
        u: mda.Universe,
        cl_index: int,
        polar_axis: List[float] | None = None,
        selected_indices: List[int] | None = None
    ) -> StructureAnalysisResult:
    """
    Analyze molecular structure and extract key geometric parameters.
    
    Args:
        u (mda.Universe): MDAnalysis universe containing the structure
        cl_index (int): Index of the reference CL atom
        polar_axis (List[float]): Polar axis for alignment.
        selected_indices (List[int]): Indices of selected atoms.
        
    Returns:
        StructureAnalysisResult: Analysis results
    """
    # Select molecules using the provided selector
    if selected_indices is None:
        selected_indices = select_ccl4_molecules(u, cl_index, 5.0)
    
    # Get reference atoms
    ccl4_mols = u.atoms[selected_indices].positions
    ccl4_mols = box_shift(ccl4_mols - ccl4_mols[0, :], u.dimensions)
    mol_A = ccl4_mols[:5]
    mol_B = ccl4_mols[5:]
    C_A = mol_A[0]
    CL_A = mol_A[1]
    C_B = mol_B[0]
    CL_B = mol_B[1]
    
    if polar_axis is None:
        polar_axis = ccl4_mols[1] - ccl4_mols[0]
        polar_axis /= np.linalg.norm(polar_axis)
    
    calc_dist = lambda a, b: np.linalg.norm(a - b, axis=-1)

    # Calculate geometric parameters
    dist_C_A_CL_A = calc_dist(C_A, CL_A)
    dist_CL_A_CL_B = calc_dist(CL_A, CL_B)
    dist_C_B_CL_B = calc_dist(C_B, CL_B)
    dist_C_A_C_B = calc_dist(C_A, C_B)
    
    vector_CL_A_CL_B = (CL_B - CL_A) / (dist_CL_A_CL_B + 1e-10)
    theta_CL_A_CL_B = np.arccos(np.dot(vector_CL_A_CL_B, polar_axis))

    # 取出除了参考Cl之外的其他3个Cl
    other_CLs = mol_A[2:]  # shape (3, 3)

    # 计算Cl3平面的法向量
    v1 = other_CLs[1] - other_CLs[0]
    v2 = other_CLs[2] - other_CLs[0]
    normal_vector = np.cross(v1, v2)
    normal_vector /= np.linalg.norm(normal_vector)  # 单位化

    # 选其中一个 Cl（例如 other_CLs[0]）作为参考 Cl
    selected_cl = other_CLs[0]
    vec_C_to_CL = selected_cl - C_A
    vec_C_to_CL /= np.linalg.norm(vec_C_to_CL)  # 单位化

    # 计算夹角
    cos_angle = np.clip(np.dot(vec_C_to_CL, normal_vector), -1.0, 1.0)
    umbrella_angle = np.arccos(cos_angle)  # radians
    umbrella_angle = np.degrees(umbrella_angle)  # degrees

    dists_CL_A_to_others = [calc_dist(CL_A, cl) for cl in other_CLs]
    #dist_CL_A_CL_other = np.mean(dists_CL_A_to_others)

    
    # Return results
    return StructureAnalysisResult(
        dist_C_A_CL_A=dist_C_A_CL_A,
        dist_CL_A_CL_B=dist_CL_A_CL_B,
        dist_C_B_CL_B=dist_C_B_CL_B,
        dist_C_A_C_B=dist_C_A_C_B,
        theta_CL_A_CL_B=np.degrees(theta_CL_A_CL_B),
        umbrella_angle=umbrella_angle,
        dists_CL_A_to_others=dists_CL_A_to_others
    )

