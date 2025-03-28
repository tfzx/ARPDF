import cupy as cp
import numpy as np
from ARPDF import compute_ARPDF
from utils import generate_grids, cosine_similarity


def search_structure(universe, ARPDF_exp, cutoff=10.0, N=512):

    def sample_center_molecules():
        """ Return a list of atoms indices of molecules """
        pass
    def generate_u2(molecule):
        """ Return List[(polar_axis, u2)] """
        pass

    X, Y = generate_grids(cutoff, N)
    molecule_list = sample_center_molecules()
    results = {}
    for molecule in molecule_list:
        # TODO: parallelize this loop
        best_similarity = -1.0
        best_polar_axis = None
        best_u2 = None
        best_ARPDF = None
        for polar_axis, u2 in generate_u2(molecule):
            ARPDF = compute_ARPDF(universe, u2, cutoff, N, grids_XY=(X, Y), modified_atoms=molecule, 
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
