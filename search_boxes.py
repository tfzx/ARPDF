import os
import pickle
import cupy as cp
from matplotlib import pyplot as plt
import numpy as np
import MDAnalysis as mda
import json
from ARPDF import compute_ARPDF, compare_ARPDF
from utils import preprocess_ARPDF, box_shift
from utils.core_functions import cosine_similarity, to_cupy, get_circular_weight, weighted_similarity, oneD_similarity, angular_average_similarity 
from utils import compute_axis_direction, adjust_ccl3_structure, adjust_ccl, clean_gro_box

def search_structure(universe, grids_XY, ARPDF_exp, filter_fourier=None, cutoff=10.0, sigma0=0.2, metric='cosine', weight_cutoff=4.0, neg=False):

    def sample_center_molecules():
        """ Return a list of atoms indices of molecules """
        cl_atoms = universe.atoms.select_atoms("name Cl")
        # cl_atoms = universe.atoms.select_atoms("name =~ 'Cl.*'")
        return list(cl_atoms.indices)
    
    

    def generate_u2(molecule, stretch_distances, periodic=None):
        """ 
        Return List[(polar_axis, u2, modified_atoms)] for different stretch_distances.
        """

        results = []
        box = universe.dimensions if periodic else None

        for distance in stretch_distances:
            u2 = universe.copy()

            target_cl = u2.atoms.select_atoms(f"name Cl and index {molecule}")
            # target_cl = u2.atoms.select_atoms(f"name =~ 'Cl.*' and index {molecule}")
            if len(target_cl) == 0:
                raise ValueError(f"未找到编号为 {molecule} 的 Cl 原子！")

            target_cl = target_cl[0]
            mol_number = target_cl.resid  # 获取分子编号 (Residue ID)

            molecule_atoms = u2.select_atoms(f"resid {mol_number}")
            target_c = molecule_atoms.select_atoms("name C")[0]  # 取第一个 C 原子
            other_cls = molecule_atoms.select_atoms(f"name Cl and not (index {molecule})")  # 其他 Cl 原子

            # target_c = molecule_atoms.select_atoms("name =~ 'C.*'")[0]  # 第一个 C 原子
            # other_cls = molecule_atoms.select_atoms(f"name =~ 'Cl.*' and not index {molecule}")


            if len(other_cls) < 3:
                raise ValueError(f"分子 {mol_number} 中 Cl 原子数量不足，无法调整 CCl₃ 结构！")

            # 计算 C->Cl 方向
            polar_axis = compute_axis_direction(target_c, target_cl, box=box)

            # 调整 CCl 结构
            modified_atoms = []
            '''
            adjust_ccl3_structure(
                target_c, target_cl, other_cls, 
                stretch_distance=distance, 
                modified_atoms=modified_atoms, 
                box=box
            )
            '''
            
            adjust_ccl(
                target_c, target_cl, 
                stretch_distance=distance, 
                modified_atoms=modified_atoms, 
                box=box
            )
            

            results.append((polar_axis, u2, modified_atoms))

        return results



    X, Y, ARPDF_exp = to_cupy(*grids_XY, ARPDF_exp)
    metric_func = {
        'cosine': lambda x, y: cosine_similarity(x, y, cos_weight), 
        'circle': lambda x, y: cp.vdot(r_weight, weighted_similarity(circular_weights, x, y)),
        '1D': lambda x, y: oneD_similarity(x, y, axis=0, weight=axis_weight),
        '1D_average': lambda x, y: angular_average_similarity(x, y, weight=average_weight)
    }[metric]

    molecule_list = sample_center_molecules()
    R = cp.sqrt(X**2 + Y**2)
    cos_weight = cp.exp(-cp.maximum(R - weight_cutoff, 0)**2 / (2 * (0.5)**2))/(1 + cp.exp(-10 * (R - 1)))
    r0_arr = cp.linspace(0, 8, 40)
    dr0 = r0_arr[1] - r0_arr[0]
    circular_weights = get_circular_weight(R, r0_arr, sigma=dr0/6.0)
    r_weight = cp.exp(-cp.maximum(r0_arr - weight_cutoff, 0)**2 / (2 * (0.5)**2))/ (1 + cp.exp(-10 * (r0_arr - 1)))
    r_weight /= r_weight.sum()
    
    axis_weight = cp.exp(-cp.maximum(R - weight_cutoff, 0)**2 / (2 * (0.5)**2))/(1 + cp.exp(-10 * (R - 1)))
    axis_weight /= axis_weight.sum()

    average_weight = cp.exp(-cp.maximum(R - weight_cutoff, 0)**2 / (2 * (0.5)**2))/ (1 + cp.exp(-10 * (R - 1)))
    average_weight /= average_weight.sum()

    results = {}

    stretch_values = [round(1.0 + 0.1 * i, 1) for i in range(15)]


    for molecule in molecule_list:
        # TODO: parallelize this loop
        best_similarity = -np.inf
        best_polar_axis = None
        best_u2 = None
        best_ARPDF = None
        best_modified_atoms = None
        for polar_axis, u2, modified_atoms in generate_u2(molecule,stretch_distances=stretch_values):
            ARPDF = compute_ARPDF(universe, u2, 256, cutoff, sigma0, (X, Y), modified_atoms=modified_atoms, 
                                    polar_axis=polar_axis, periodic=True, filter_fourier=filter_fourier, verbose=False, neg=neg)
            # similarity = cp.vdot(r_weight, Similarity(circular_weights, ARPDF, ARPDF_exp)).get()
            similarity = metric_func(ARPDF, ARPDF_exp).get()
            if similarity > best_similarity:
                best_polar_axis = polar_axis
                best_similarity = similarity
                best_u2 = u2
                best_ARPDF = ARPDF.get()
                best_modified_atoms = modified_atoms
        results[molecule] = (best_polar_axis, best_u2, best_ARPDF, best_similarity, best_modified_atoms)
    return results

def workflow_demo(X, Y, ARPDF_ref, filter_fourier=None, sigma0=0.2, exp_name: str="exp", metric: str="1D_average", weight_cutoff=5.0, neg=False):
    def get_box_iter():
        """Run and sample boxes from the MD simulation."""
        clean_gro_box('data/CCl4/CCl4.gro','data/CCl4/CCl4_clean.gro')
        return [mda.Universe('data/CCl4/CCl4_clean.gro')]
    def get_universe(box):
        return box
    def dump_results(results):
        best_mol = max(results, key=lambda x: results[x][3])
        worst_mol = min(results, key=lambda x: results[x][3])

        polar_axis, u2, ARPDF, similarity, modified_atoms = results[best_mol]
        modified_atoms = [int(x) for x in modified_atoms]
        print(f"Molecule {best_mol}: Similarity = {similarity}")
        print(f"Polar axis: {polar_axis}")
        print(f"Modified atoms: {modified_atoms}")
        fig = compare_ARPDF(ARPDF, ARPDF_ref, (X, Y), cos_sim=similarity, show_range=8.0)
        fig.savefig(f"{out_dir}/CCl4_best_init.png")
        # plt.show()
        universe.atoms.write(f"{out_dir}/CCl4.gro")
        u2.atoms.write(f"{out_dir}/CCl4_best_init.gro")
        center_group = u2.atoms[modified_atoms]
        selected_group = center_group + u2.select_atoms("around 6 group center", center = center_group)
        _center = center_group.positions[0:1]
        center_group.positions = _center + box_shift(center_group.positions - _center, box=u2.dimensions)
        selected_group.positions = _center + box_shift(selected_group.positions - _center, box=u2.dimensions)
        selected_group.write(f"{out_dir}/CCl4_selected.gro")
        with open(f"{out_dir}/metadata.json", "w") as f:
            json.dump({
                "name": "CCl4",
                "structure_info": {
                    "u1_name": "CCl4.gro",
                    "u2_name": "CCl4_best_init.gro",
                    "polar_axis": [float(x) for x in polar_axis],
                    "modified_atoms": modified_atoms
                }
            }, f, indent=4)

        polar_axis_w, u2_w, ARPDF_w, similarity_w, modified_atoms_w = results[worst_mol]
        modified_atoms_w = [int(x) for x in modified_atoms_w]
        print(f"[WORST] Molecule {worst_mol}: Similarity = {similarity_w}")
        fig_worst = compare_ARPDF(ARPDF_w, ARPDF_ref, (X, Y), cos_sim=similarity_w, show_range=8.0)
        fig_worst.savefig(f"{out_dir}/CCl4_worst_init.png")
        u2_w.atoms.write(f"{out_dir}/CCl4_worst_init.gro")

        center_group_w = u2_w.atoms[modified_atoms_w]
        selected_group_w = center_group_w + u2_w.select_atoms("around 6 group center", center=center_group_w)
        _center_w = center_group_w.positions[0:1]
        center_group_w.positions = _center_w + box_shift(center_group_w.positions - _center_w, box=u2_w.dimensions)
        selected_group_w.positions = _center_w + box_shift(selected_group_w.positions - _center_w, box=u2_w.dimensions)
        selected_group_w.write(f"{out_dir}/CCl4_worst_selected.gro")

        return
    
    out_dir = f"tmp/{exp_name}"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    # X, Y, ARPDF_ref = load_ARPDF_exp("data/CCl4/ARPDF_exp.npy")
    for box in get_box_iter():
        universe = get_universe(box)
        results = search_structure(
            universe, (X, Y), ARPDF_ref, filter_fourier=filter_fourier, sigma0=sigma0, metric=metric, weight_cutoff=weight_cutoff, neg=neg)
        with open(f"{out_dir}/results.pkl", "wb") as f:
            pickle.dump(results, f)
        dump_results(results)
