from typing import List
from utils import compute_axis_direction, adjust_ccl, adjust_ccl3_structure

def select_cl_atoms(universe):
    """Select Cl atoms as center molecules for analysis
    
    Args:
        universe (mda.Universe): MDAnalysis universe containing the structure
        
    Returns:
        list: List of indices of selected Cl atoms
    """
    cl_atoms = universe.atoms.select_atoms("name Cl")
    return list(cl_atoms.indices)

class CCL4Modifier_CL:
    """Handle molecular structure modifications for CCl4"""
    
    def __init__(self, universe, stretch_distances: List[float] = None, periodic: bool = None):
        """Initialize the structure modifier
        
        Args:
            universe (mda.Universe): MDAnalysis universe containing the structure
            stretch_distances (list, optional): List of distances to stretch bonds
            periodic (bool, optional): Whether to use periodic boundary conditions
        """
        self.universe = universe
        self.box = universe.dimensions if periodic else None
        self.stretch_distances = stretch_distances or [round(1.0 + 0.1 * i, 1) for i in range(15)]
    
    def generate_modified_structures(self, molecule):
        """
        Generate modified structures for different stretch distances.
        Only modify `CL` atom.
        
        Args:
            molecule (int): Index of the target Cl atom to modify
            
        Returns:
            list: List of tuples containing:
                - polar_axis (ndarray): Direction vector of the C-Cl bond
                - u2 (Universe): Modified MDAnalysis universe 
                - modified_atoms (list): List of modified atom indices
                
        Raises:
            ValueError: If the molecule does not have enough Cl atoms
        """
        results = []
        
        for distance in self.stretch_distances:
            # Create a copy of the universe to modify
            u2 = self.universe.copy()
            
            # Get the target Cl atom and its molecule number
            target_cl = u2.atoms.select_atoms(f"name Cl and index {molecule}")[0]
            mol_number = target_cl.resid
            
            # Select all atoms in the molecule and find the C atom
            molecule_atoms = u2.select_atoms(f"resid {mol_number}")
            target_c = molecule_atoms.select_atoms("name C")[0]
            other_cls = molecule_atoms.select_atoms(f"name Cl and not (index {molecule})")
            
            # Check if molecule has enough Cl atoms
            if len(other_cls) < 3:
                raise ValueError(f"分子 {mol_number} 中 Cl 原子数量不足，无法调整 CCl₃ 结构！")
            
            # Compute the direction of the C-Cl bond
            polar_axis = compute_axis_direction(target_c, target_cl, box=self.box)
            modified_atoms = []
            
            # Adjust the C-Cl bond length
            adjust_ccl(
                target_c, target_cl, 
                stretch_distance=distance, 
                modified_atoms=modified_atoms, 
                box=self.box
            )
            
            results.append((polar_axis, u2, modified_atoms))
        
        return results

class CCL4Modifier_C_CL:
    """Handle molecular structure modifications for CCl4"""
    
    def __init__(self, universe, stretch_distances: List[float] = None, periodic: bool = None):
        """Initialize the structure modifier
        
        Args:
            universe (mda.Universe): MDAnalysis universe containing the structure
            stretch_distances (list, optional): List of distances to stretch bonds
            periodic (bool, optional): Whether to use periodic boundary conditions
        """
        self.universe = universe
        self.box = universe.dimensions if periodic else None
        self.stretch_distances = stretch_distances or [round(1.0 + 0.1 * i, 1) for i in range(15)]
    
    def generate_modified_structures(self, molecule):
        """
        Generate modified structures for different stretch distances.
        Only modify `C` and `CL` atoms.
        
        Args:
            molecule (int): Index of the target Cl atom to modify
            
        Returns:
            list: List of tuples containing:
                - polar_axis (ndarray): Direction vector of the C-Cl bond
                - u2 (Universe): Modified MDAnalysis universe 
                - modified_atoms (list): List of modified atom indices
                
        Raises:
            ValueError: If the molecule does not have enough Cl atoms
        """
        results = []
        
        for distance in self.stretch_distances:
            # Create a copy of the universe to modify
            u2 = self.universe.copy()
            
            # Get the target Cl atom and its molecule number
            target_cl = u2.atoms.select_atoms(f"name Cl and index {molecule}")[0]
            mol_number = target_cl.resid
            
            # Select all atoms in the molecule and find the C atom
            molecule_atoms = u2.select_atoms(f"resid {mol_number}")
            target_c = molecule_atoms.select_atoms("name C")[0]
            other_cls = molecule_atoms.select_atoms(f"name Cl and not (index {molecule})")
            
            # Check if molecule has enough Cl atoms
            if len(other_cls) < 3:
                raise ValueError(f"分子 {mol_number} 中 Cl 原子数量不足，无法调整 CCl₃ 结构！")
            
            # Compute the direction of the C-Cl bond
            polar_axis = compute_axis_direction(target_c, target_cl, box=self.box)
            modified_atoms = []
            
            # Adjust the C-Cl bond length
            adjust_ccl3_structure(
                target_c, target_cl, other_cls,
                stretch_distance=distance, 
                modified_atoms=modified_atoms, 
                box=self.box
            )
            
            results.append((polar_axis, u2, modified_atoms))
        
        return results
