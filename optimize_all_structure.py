from collections import Counter
import os
import pickle
from typing import List
import numpy as np
import MDAnalysis as mda
import torch
from tqdm import tqdm
from optimize_ARPDF import ARPDFOptimizer
from search_boxes import SearchResult
from utils import load_structure_data, generate_grids, update_metadata, select_nbr_mols
import json

def optimize_all_structures(exp_dir: str, output_dir: str = "optimize"):
    """
    Optimize all structures in the search results and save the results to the output directory.

    Args:
        exp_dir (str): Directory containing the search results (results.pkl).
        output_dir (str): Subdirectory to save the optimized results (default: "optimize").
    """
    # Load the search results
    with open(os.path.join(exp_dir, "results.pkl"), "rb") as f:
        results: List[SearchResult] = pickle.load(f)

    # Create the output directory if it doesn't exist
    output_path = os.path.join(exp_dir, output_dir)
    os.makedirs(output_path, exist_ok=True)

    # Load reference structure and metadata
    reference_dir = "data/CCl4"
    u1_ref, u2_ref, modified_atoms_ref, polar_axis_ref = load_structure_data(reference_dir)
    ARPDF_ref = np.load(os.path.join(exp_dir, "ARPDF_ref.npy"))
    with open(os.path.join(exp_dir, "metadata.json"), "r") as f:
        root_metadata = json.load(f)
    xy_range = root_metadata["search_info"]["parameters"]["grids_range"]
    N, M = root_metadata["search_info"]["parameters"]["grids_shape"]
    X, Y = generate_grids(xy_range, N, M)
    cutoff = 10.0
    sigma0 = root_metadata["search_info"]["parameters"]["sigma0"]
    weight_cutoff = root_metadata["search_info"]["parameters"]["weight_cutoff"]
    device = 'cuda'

    # Initialize the optimizer
    optimizer = ARPDFOptimizer(
        X, Y,
        ARPDF_exp=ARPDF_ref,
        type_counts=Counter(u1_ref.atoms.types),
        cutoff=cutoff,
        sigma0=sigma0,
        weight_cutoff=weight_cutoff,
        lr=0.01,
        gamma=0.995,
        f_lb=-0.9, 
        s=0.0, 
        beta=0.0, 
        epochs=500,
        loss_name="cosine",
        device=device
    )
    total_structures = len(results)

    # Iterate over each search result
    for i, result in tqdm(enumerate(results), desc="Optimizing all structures", total=total_structures, position=1):

        # Create a subdirectory for this structure
        struct_dir = os.path.join(output_path, f"structure_{i}")
        os.makedirs(struct_dir, exist_ok=True)
        
        # Save the original structure
        result.modified_universe.atoms.write(os.path.join(struct_dir, f"structure_{i}.gro"))

        # Save the metadata
        update_metadata(struct_dir, {
            "name": root_metadata["name"],
            "structure_info": {
                "u1_name": "../" + root_metadata["structure_info"]["u1_name"],
                "u2_name": f"structure_{i}.gro",
                "polar_axis": result.polar_axis,
                "modified_atoms": result.modified_atoms,
                "similarity": result.similarity,
                "molecule": result.molecule
            }
        })

        # Set the system for optimization
        modified_atoms = select_nbr_mols(u1_ref, result.modified_atoms, nbr_distance=None, periodic=True)
        optimizer.set_system(
            out_dir=struct_dir,
            u1=u1_ref,
            u2=result.modified_universe,
            modified_atoms=modified_atoms,
            polar_axis=result.polar_axis
        )

        # Run optimization
        optimizer.optimize(verbose=True, log_step=5, print_step=50, leave=False, prefix=f"Optimizing structure {i + 1}/{total_structures}")
        tqdm.write(f"Structure {i + 1}/{total_structures} finished.")

    print(f"Optimization completed. Results saved to {output_path}.")

# Example usage
if __name__ == "__main__":
    exp_dir = "tmp/exp_opt_1D_avg"  # Adjust based on your directory
    optimize_all_structures(exp_dir)
