import cupy as cp
import numpy as np
import MDAnalysis as mda
from ARPDF import compute_ARPDF, show_ARPDF
from utils import generate_grids, cosine_similarity, preprocess_ARPDF, to_cupy
from utils import compute_axis_direction, adjust_ccl3_structure


def search_structure(universe, grids_XY, ARPDF_exp, cutoff=10.0, N=512):

    def sample_center_molecules():
        """ Return a list of atoms indices of molecules """
        cl_atoms = universe.atoms.select_atoms("name Cl")
        return list(cl_atoms.indices)
    
    def generate_u2(molecule, periodic=None):
        """ Return List[(polar_axis, u2)] """

        box = universe.dimensions if periodic else None

        u2 = universe.copy()
    
        target_cl = u2.atoms.select_atoms(f"name Cl and index {molecule}")
        if len(target_cl) == 0:
            raise ValueError(f"未找到编号为 {molecule} 的 Cl 原子！")

        target_cl = target_cl[0]
        mol_number = target_cl.resid  # 获取分子编号 (Residue ID)

        molecule_atoms = u2.select_atoms(f"resid {mol_number}")
        target_c = molecule_atoms.select_atoms("name C")[0]  # 取第一个 C 原子
        other_cls = molecule_atoms.select_atoms(f"name Cl and not (index {molecule})")  # 其他 Cl 原子

        if len(other_cls) < 3:
            raise ValueError(f"分子 {mol_number} 中 Cl 原子数量不足，无法调整 CCl₃ 结构！")

        # 计算 C->Cl 方向
        polar_axis = compute_axis_direction(target_c, target_cl, box=box)

        # 调整 CCl₃ 结构
        modified_atoms = []
        adjust_ccl3_structure(target_c, target_cl, other_cls, stretch_distance=0.2, modified_atoms=modified_atoms, box=box)

        return [(polar_axis, u2, modified_atoms)]

    molecule_list = sample_center_molecules()
    grids_XY = to_cupy(grids_XY)
    results = {}
    for molecule in molecule_list:
        # TODO: parallelize this loop
        best_similarity = -1.0
        best_polar_axis = None
        best_u2 = None
        best_ARPDF = None
        for polar_axis, u2, modified_atoms in generate_u2(molecule):
            ARPDF = compute_ARPDF(universe, u2, cutoff, N, grids_XY, modified_atoms=modified_atoms, 
                                                polar_axis=polar_axis, verbose=False)
            similarity = cosine_similarity(ARPDF, ARPDF_exp)
            if similarity > best_similarity:
                best_polar_axis = polar_axis
                best_similarity = similarity
                best_u2 = u2
                best_ARPDF = ARPDF
        results[molecule] = (best_polar_axis, best_u2, best_ARPDF, best_similarity)
    return results

def workflow_demo():
    def get_box_iter():
        """Run and sample boxes from the MD simulation."""
        return [mda.Universe('data/CCl4/CCl4.gro')]
    def get_universe(box):
        return box
    def load_ARPDF_exp(file_name):
        ARPDF_exp_raw = np.load(file_name)
        ori_range = 9.924650203173275
        X, Y, ARPDF_exp = preprocess_ARPDF(ARPDF_exp_raw, ori_range, rmax=9.0)
        return X, Y, ARPDF_exp
    def dump_results(results):
        best_mol = max(results, key=lambda x: results[x][3])
        # for molecule, (polar_axis, u2, ARPDF, similarity) in results.items():
        polar_axis, u2, ARPDF, similarity = results[best_mol]
        print(f"Molecule {best_mol}: Similarity = {similarity}")
        print(f"  Polar axis: {polar_axis}")
        show_ARPDF(ARPDF, plot_rmax=9.0, show_rmax=8.0, max_intensity=0.1)
        pass
    X, Y, ARPDF_exp = load_ARPDF_exp("data/CCl4/ARPDF_exp.npy")
    for box in get_box_iter():
        universe = get_universe(box)
        results = search_structure(universe, (X, Y), ARPDF_exp)
        dump_results(results)

if __name__ == "__main__":
    workflow_demo()