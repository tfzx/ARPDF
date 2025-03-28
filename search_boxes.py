import cupy as cp
import numpy as np
import MDAnalysis as MDA
from ARPDF import compute_ARPDF
from utils import generate_grids, cosine_similarity
from utils import compute_axis_direction, adjust_ccl3_structure


def search_structure(universe, ARPDF_exp, cutoff=10.0, N=512):

    def sample_center_molecules():
        """ Return a list of atoms indices of molecules """
        cl_atoms = universe.atoms.select_atoms("name Cl")
        return list(cl_atoms.indices)
    
    def generate_u2(molecule):
        """ Return List[(polar_axis, u2)] """
        u2 = universe.copy()
    
        target_cl = u2.atoms.select_atoms(f"name Cl and index {molecule}")
        if len(target_cl) == 0:
            raise ValueError(f"未找到编号为 {molecule} 的 Cl 原子！")

        target_cl = target_cl[0]
        mol_number = target_cl.resid  # 获取分子编号 (Residue ID)

        molecule_atoms = u2.select_atoms(f"resid {mol_number}")
        target_c = molecule_atoms.select_atoms("name C")[0]  # 取第一个 C 原子
        other_cls = molecule_atoms.select_atoms(f"name Cl and index != {molecule}")  # 其他 Cl 原子

        if len(other_cls) < 3:
            raise ValueError(f"分子 {mol_number} 中 Cl 原子数量不足，无法调整 CCl₃ 结构！")

        # 计算 C->Cl 方向
        polar_axis = compute_axis_direction(target_c, target_cl)

        # 调整 CCl₃ 结构
        modified_atoms = []
        adjust_ccl3_structure(target_c, target_cl, other_cls, stretch_distance=0.2, modified_atoms=modified_atoms)

        return [(polar_axis, u2, modified_atoms)]

    X, Y = generate_grids(cutoff, N)
    molecule_list = sample_center_molecules()
    results = {}
    for molecule in molecule_list:
        # TODO: parallelize this loop
        best_similarity = -1.0
        best_polar_axis = None
        best_u2 = None
        best_ARPDF = None
        for polar_axis, u2, modified_atoms in generate_u2(molecule):
            ARPDF = compute_ARPDF(universe, u2, cutoff, N, grids_XY=(X, Y), modified_atoms=modified_atoms, 
                                                polar_axis=polar_axis, verbose=False)
            similarity = cosine_similarity(ARPDF, ARPDF_exp)
            if similarity > best_similarity:
                best_polar_axis = polar_axis
                best_similarity = similarity
                best_u2 = u2
                best_ARPDF = ARPDF
        results[tuple(molecule)] = (best_polar_axis, best_u2, best_ARPDF, best_similarity)
    return results

def workflow_demo():
    def get_box_iter():
        """Run and sample boxes from the MD simulation."""
        pass
    def get_universe(box):
        pass
    def load_ARPDF_exp():
        pass
    def dump_results(results):
        pass
    ARPDF_exp = load_ARPDF_exp()
    for box in get_box_iter():
        universe = get_universe(box)
        results = search_structure(universe, ARPDF_exp)
        dump_results(results)
