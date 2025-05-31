import numpy as np
from typing import List
from utils import compute_axis_direction, adjust_ccl, rotate_CH3CN_along_CN, adjust_ccn

def select_n_atoms(universe):
    """Select Cl atoms as center molecules for analysis
    
    Args:
        universe (mda.Universe): MDAnalysis universe containing the structure
        
    Returns:
        list: List of indices of selected Cl atoms
    """
    n_atoms = universe.atoms.select_atoms("name N")
    return list(n_atoms.indices)

class CH3CNModifier:
    """Handle molecular structure modifications for CH₃CN"""
    
    def __init__(self, universe, stretch_distances: List[float] = None, periodic: bool = None):
        """
        Args:
            universe (mda.Universe): MDAnalysis universe
            stretch_distances (list, optional): Distances to stretch the C≡N bond
            periodic (bool, optional): Whether to consider PBC
        """
        self.universe = universe
        self.box = universe.dimensions if periodic else None
        self.stretch_distances = stretch_distances or [round(1.0 + 0.05 * i, 2) for i in range(20)]
    
    def generate_modified_structures(self, n_atom_index: int):
        """
        Generate modified structures for each stretch distance of the C≡N bond.
        
        Args:
            n_atom_index (int): Index of the target N atom
            
        Returns:
            list of (polar_axis, universe, modified_atom_indices)
        """
        results = []
        
        for distance in self.stretch_distances:
            u2 = self.universe.copy()
            target_n = u2.atoms[n_atom_index]
            mol_number = target_n.resid
            
            # 找到该分子的所有原子
            molecule_atoms = u2.select_atoms(f"resid {mol_number}")
            c_atoms = molecule_atoms.select_atoms("name C")  # 应该有两个 C
            if len(c_atoms) < 2:
                raise ValueError(f"分子 {mol_number} 中 C 原子数量不足，无法确定 C≡N 键")
            
            # 近 N 的那个 C 原子
            target_c = min(c_atoms, key=lambda atom: np.linalg.norm(atom.position - target_n.position))
            
            polar_axis = compute_axis_direction(target_c, target_n, box=self.box)
            modified_atoms = []
            
            # 调整 C–N 键距离
            adjust_ccl(
                target_c, target_n,
                stretch_distance=distance,
                modified_atoms=modified_atoms,
                box=self.box
            )
            
            results.append((polar_axis, u2, modified_atoms))
        
        return results
    
class CH3CNBondStretcher:
    """Modify CH3CN structures by stretching the C–C bond only."""

    def __init__(self, universe, stretch_distances=None, periodic=None):
        """
        Args:
            universe (MDAnalysis.Universe): The universe object
            stretch_distances (list of float, optional): C–C distances to stretch to
            periodic (bool, optional): Use periodic boundary conditions
        """
        self.universe = universe
        self.stretch_distances = stretch_distances or [round(1.2 + 0.1 * i, 1) for i in range(15)]
        self.box = universe.dimensions if periodic else None

    def stretch(self, molecule_index):
        """
        Stretch the C–C bond in CH3CN

        Args:
            molecule_index (int): Index of CH3-side carbon atom

        Returns:
            list of (polar_axis, modified_universe, modified_atom_indices)
        """
        results = []

        for distance in self.stretch_distances:
            u2 = self.universe.copy()

            c1 = u2.atoms.select_atoms(f"name C and index {molecule_index}")[0]
            mol_number = c1.resid
            mol_atoms = u2.select_atoms(f"resid {mol_number}")

            other_c = mol_atoms.select_atoms(f"name C and not index {molecule_index}")[0]

            modified_atoms = []
            adjust_ccl(c1, other_c, distance, modified_atoms, box=self.box)

            polar_axis = compute_axis_direction(c1, other_c, box=self.box)

            results.append((polar_axis, u2, modified_atoms))

        return results
    
class CH3CNAngleSampler:
    """Apply specified (phi, theta) rotations to CH3CN molecules."""

    def __init__(self, universe, angle_samples=None, periodic=None):
        """
        Args:
            universe (MDAnalysis.Universe): 原始 Universe 对象。
            angle_samples (list of tuple, optional): 列表，每项为 (phi, theta)，单位为度。
            periodic (bool, optional): 是否使用周期性边界条件。
        """
        self.universe = universe
        self.angle_samples = angle_samples or [(phi, theta) for phi in range(0, 360, 60) for theta in range(0, 360, 60)]
        self.box = universe.dimensions if periodic else None

    def generate_modified_structures(self, molecule_index):
        """
        对指定的 CH3CN 分子执行采样旋转。

        Args:
            molecule_index (int): N 原子的编号（index）

        Returns:
            list of (polar_axis, modified_universe, modified_atom_indices)
        """
        results = []

        for phi, theta in self.angle_samples:
            u2 = self.universe.copy()

            n_atom = u2.atoms.select_atoms(f"name N and index {molecule_index}")[0]
            mol_number = n_atom.resid
            mol_atoms = u2.select_atoms(f"resid {mol_number}")

            c_atom = mol_atoms.select_atoms(f"name C and around 1.6 index {molecule_index}")[0]

            modified_atoms = []
            rotate_CH3CN_along_CN(
                mol_atoms,
                phi_deg=phi,
                theta_deg=theta,
                modified_atoms=modified_atoms,
                box=self.box,
            )

            # 注意：我们保留原始极轴，不根据结构变化更新
            polar_axis = compute_axis_direction(c_atom, n_atom, box=self.box)

            results.append((polar_axis, u2, modified_atoms))

        return results
    

class CH3CNAngle_LengthSampler:
    """Apply specified (phi, theta) rotations to CH3CN molecules and stretch C–C bond."""

    def __init__(self, universe, angle_samples=None, stretch_range=None, periodic=None):
        """
        Args:
            universe (MDAnalysis.Universe): 原始 Universe 对象。
            angle_samples (list of tuple, optional): 每项为 (phi, theta)，单位为度。
            stretch_range (tuple, optional): 伸长C–C键的区间 (start, stop, step)，单位为 Å。
            periodic (bool, optional): 是否使用周期性边界条件。
        """
        self.universe = universe
        self.angle_samples = angle_samples or [(phi, theta) for phi in range(0, 360, 60) for theta in range(0, 360, 60)]
        self.stretch_range = stretch_range or [round(0.5 + 0.1 * i, 1) for i in range(10)]
        self.box = universe.dimensions if periodic else None

    def generate_modified_structures(self, molecule_index):
        """
        对指定的 CH3CN 分子执行采样旋转与 C–C 键伸长。

        Args:
            molecule_index (int): N 原子的编号（index）

        Returns:
            list of (polar_axis, modified_universe, modified_atom_indices)
        """
        results = []

        for phi, theta in self.angle_samples:
            for distance in self.stretch_range:
                u2 = self.universe.copy()

                n_atom = u2.atoms.select_atoms(f"name N and index {molecule_index}")[0]
                mol_number = n_atom.resid
                mol_atoms = u2.select_atoms(f"resid {mol_number}")

                # 找出连接关系: CN 基团的 C 原子（靠近 N）
                cn_c_atom = mol_atoms.select_atoms(f"name C and around 1.6 index {n_atom.index}")[0]

                # CH3 那一侧的 C 原子：远离 N 的 C
                c_atoms = mol_atoms.select_atoms("name C")
                ch3_c_atom = [c for c in c_atoms if c.index != cn_c_atom.index][0]

                polar_axis = compute_axis_direction(cn_c_atom, n_atom, box=self.box)

                modified_atoms = []

                # 执行旋转
                rotate_CH3CN_along_CN(
                    mol_atoms,
                    phi_deg=phi,
                    theta_deg=theta,
                    modified_atoms=modified_atoms,
                    box=self.box,
                )

                # 拉伸 CH3-C–C≡N 中的 C–C
                adjust_ccn(
                    ch3_c_atom=ch3_c_atom,
                    cn_c_atom=cn_c_atom,
                    n_atom=n_atom,
                    stretch_distance=distance,
                    modified_atoms=modified_atoms,
                    box=self.box
                )

                modified_atoms = sorted(set(modified_atoms))
                results.append((polar_axis, u2, modified_atoms))

        return results






