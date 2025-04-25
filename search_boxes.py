import os
import pickle
import cupy as cp
import numpy as np
import MDAnalysis as mda
import json
from ARPDF import compute_ARPDF, compare_ARPDF
from utils import box_shift
from utils.core_functions import cosine_similarity, to_cupy, get_circular_weight, weighted_similarity, oneD_similarity, angular_average_similarity 
from utils import clean_gro_box
from ccl4_modifier import StructureModifier, select_center_molecules
from typing import Callable, List, Tuple, Optional, Any
from dataclasses import dataclass

@dataclass
class SearchResult:
    """Data class to store search results for a single molecule
    
    Attributes:
        polar_axis (ndarray): Direction vector of the modified bond
        modified_universe (mda.Universe): Modified structure
        ARPDF (ndarray): Calculated ARPDF for the modified structure
        similarity (float): Similarity score between ARPDF and experimental data
        modified_atoms (List[int]): Indices of modified atoms
    """
    polar_axis: np.ndarray
    modified_universe: mda.Universe
    ARPDF: np.ndarray
    similarity: float
    modified_atoms: List[int]

class SimilarityCalculator:
    """Handle similarity calculations between ARPDFs"""
    
    def __init__(self, X, Y, weight_cutoff=5.0, metric='cosine'):
        self.X, self.Y = X, Y
        self.R = cp.sqrt(X**2 + Y**2)
        self.weight_cutoff = weight_cutoff
        self._initialize_weights()
        self._metric = self._get_metric_function(metric)
    
    def _initialize_weights(self):
        """Initialize various weights for similarity calculations"""
        # Cosine weight
        self.cos_weight = cp.exp(-cp.maximum(self.R - self.weight_cutoff, 0)**2 / (2 * (0.5)**2))/(1 + cp.exp(-10 * (self.R - 1)))
        
        # Circular weights
        self.r0_arr = cp.linspace(0, 8, 40)
        dr0 = self.r0_arr[1] - self.r0_arr[0]
        self.circular_weights = get_circular_weight(self.R, self.r0_arr, sigma=dr0/6.0)
        self.r_weight = cp.exp(-cp.maximum(self.r0_arr - self.weight_cutoff, 0)**2 / (2 * (0.5)**2))/ (1 + cp.exp(-10 * (self.r0_arr - 1)))
        self.r_weight /= self.r_weight.sum()
        
        # Axis weights
        self.axis_weight = cp.exp(-cp.maximum(self.R - self.weight_cutoff, 0)**2 / (2 * (0.5)**2))/(1 + cp.exp(-10 * (self.R - 1)))
        self.axis_weight /= self.axis_weight.sum()
        
        # Average weights
        self.average_weight = cp.exp(-cp.maximum(self.R - self.weight_cutoff, 0)**2 / (2 * (0.5)**2))/ (1 + cp.exp(-10 * (self.R - 1)))
        self.average_weight /= self.average_weight.sum()
    
    def calc_similarity(self, ARPDF, ARPDF_exp):
        """Calculate similarity between two ARPDFs"""
        return self._metric(ARPDF, ARPDF_exp)
    
    def _get_metric_function(self, metric):
        """Get the appropriate metric function"""
        metric_funcs = {
            'cosine': lambda x, y: cosine_similarity(x, y, self.cos_weight),
            'circle': lambda x, y: cp.vdot(self.r_weight, weighted_similarity(self.circular_weights, x, y)),
            '1D': lambda x, y: oneD_similarity(x, y, axis=0, weight=self.axis_weight),
            '1D_average': lambda x, y: angular_average_similarity(x, y, weight=self.average_weight)
        }
        return metric_funcs[metric]

class StructureSearcher:
    """Handle structure search and result management"""
    
    def __init__(self, 
                 output_dir: str,
                 universe: mda.Universe,
                 grids_XY: Tuple[np.ndarray, np.ndarray],
                 ARPDF_exp: np.ndarray,
                 molecule_selector: Callable[[mda.Universe], List[int]],
                 structure_modifier: Any,
                 filter_fourier: Optional[Callable] = None,
                 cutoff: float = 10.0,
                 sigma0: float = 0.2,
                 metric: str = 'cosine',
                 weight_cutoff: float = 4.0,
                 neg: bool = False):
        """Initialize the structure searcher
        
        Args:
            output_dir (str): Directory to save results
            universe (mda.Universe): Initial structure
            grids_XY (tuple): Grid coordinates (X, Y)
            ARPDF_exp (ndarray): Experimental ARPDF data
            molecule_selector (callable): Function to select molecules for analysis
            structure_modifier: Object that implements generate_modified_structures method
            filter_fourier (callable, optional): Fourier filter function
            cutoff (float, optional): Distance cutoff for ARPDF calculation
            sigma0 (float, optional): Gaussian width for ARPDF calculation
            metric (str, optional): Similarity metric to use
            weight_cutoff (float, optional): Weight cutoff for similarity calculation
            neg (bool, optional): Whether to use negative ARPDF
        """
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        self.universe = universe
        self.ARPDF_exp = ARPDF_exp
        self.filter_fourier = filter_fourier
        self.cutoff = cutoff
        self.sigma0 = sigma0
        self.neg = neg
        
        # Store selection and modification functions
        self.molecule_selector = molecule_selector
        self.structure_modifier = structure_modifier
        
        # Initialize components
        X, Y, self.ARPDF_exp = to_cupy(*grids_XY, ARPDF_exp)
        self.grids_XY = X, Y
        self.similarity_calc = SimilarityCalculator(X, Y, weight_cutoff, metric)
        
        # Initialize results
        self.results = {}
    
    def search(self):
        """Search for optimal molecular structures
        
        Returns:
            dict: Dictionary of results for each molecule
        """
        # Get molecule list
        molecule_list = self.molecule_selector(self.universe)
        
        # Search for each molecule
        for molecule in molecule_list:
            best_similarity = -np.inf
            best_result = None
            
            for polar_axis, u2, modified_atoms in self.structure_modifier.generate_modified_structures(molecule):
                ARPDF = compute_ARPDF(
                    self.universe, u2, 256, self.cutoff, self.sigma0, self.grids_XY,
                modified_atoms=modified_atoms, 
                    polar_axis=polar_axis,
                    periodic=True,
                    filter_fourier=self.filter_fourier,
                    verbose=False,
                    neg=self.neg
                )
                
                similarity = self.similarity_calc.calc_similarity(ARPDF, self.ARPDF_exp).get()
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_result = SearchResult(
                        polar_axis=[float(x) for x in polar_axis],
                        modified_universe=u2,
                        ARPDF=ARPDF.get(),
                        similarity=similarity,
                        modified_atoms=[int(x) for x in modified_atoms]
                    )
            
            self.results[molecule] = best_result
        
        return self.results
    
    def save_results(self, X, Y, ARPDF_ref):
        """Save analysis results
        
        Args:
            X (ndarray): X grid coordinates
            Y (ndarray): Y grid coordinates
            ARPDF_ref (ndarray): Reference ARPDF data
        """
        best_mol = max(self.results, key=lambda x: self.results[x].similarity)
        worst_mol = min(self.results, key=lambda x: self.results[x].similarity)
        
        self._save_best_results(self.results[best_mol], X, Y, ARPDF_ref)
        self._save_worst_results(self.results[worst_mol], X, Y, ARPDF_ref)
        
        # Save raw results
        with open(f"{self.output_dir}/results.pkl", "wb") as f:
            pickle.dump(self.results, f)
    
    def _save_best_results(self, result: SearchResult, X, Y, ARPDF_ref):
        """Save best results"""
        modified_atoms = result.modified_atoms
        
        # Save metadata
        with open(f"{self.output_dir}/metadata.json", "w") as f:
            json.dump({
                "name": "CCl4",
                "structure_info": {
                    "u1_name": "CCl4.gro",
                    "u2_name": "CCl4_best_init.gro",
                    "polar_axis": result.polar_axis,
                    "modified_atoms": modified_atoms
                }
            }, f, indent=4)
        
        # Save structures
        self.universe.atoms.write(f"{self.output_dir}/CCl4.gro")
        result.modified_universe.atoms.write(f"{self.output_dir}/CCl4_best_init.gro")
        
        # Save selected atoms
        center_group = result.modified_universe.atoms[modified_atoms]
        selected_group = center_group + result.modified_universe.select_atoms("around 6 group center", center=center_group)
        _center = center_group.positions[0:1]
        center_group.positions = _center + box_shift(center_group.positions - _center, box=result.modified_universe.dimensions)
        selected_group.positions = _center + box_shift(selected_group.positions - _center, box=result.modified_universe.dimensions)
        selected_group.write(f"{self.output_dir}/CCl4_selected.gro")
        
        # Save visualization
        fig = compare_ARPDF(result.ARPDF, ARPDF_ref, (X, Y), cos_sim=result.similarity, show_range=8.0)
        fig.savefig(f"{self.output_dir}/CCl4_best_init.png")
    
    def _save_worst_results(self, result: SearchResult, X, Y, ARPDF_ref):
        """Save worst results"""
        modified_atoms = result.modified_atoms
        
        # Save structures
        result.modified_universe.atoms.write(f"{self.output_dir}/CCl4_worst_init.gro")
        
        # Save selected atoms
        center_group = result.modified_universe.atoms[modified_atoms]
        selected_group = center_group + result.modified_universe.select_atoms("around 6 group center", center=center_group)
        _center = center_group.positions[0:1]
        center_group.positions = _center + box_shift(center_group.positions - _center, box=result.modified_universe.dimensions)
        selected_group.positions = _center + box_shift(selected_group.positions - _center, box=result.modified_universe.dimensions)
        selected_group.write(f"{self.output_dir}/CCl4_worst_selected.gro")
        
        # Save visualization
        fig = compare_ARPDF(result.ARPDF, ARPDF_ref, (X, Y), cos_sim=result.similarity, show_range=8.0)
        fig.savefig(f"{self.output_dir}/CCl4_worst_init.png")

def workflow_demo(
        X, Y, ARPDF_ref, 
        filter_fourier=None, 
        sigma0=0.2, 
        exp_name: str="exp", 
        metric: str="1D_average", 
        weight_cutoff=5.0, 
        neg=False,
        stretch_distances: List[float] = None
    ):
    """Demo workflow for structure search and analysis"""
    # Clean and load structure
    clean_gro_box('data/CCl4/CCl4.gro', 'data/CCl4/CCl4_clean.gro')
    universe = mda.Universe('data/CCl4/CCl4_clean.gro')
    
    # Initialize searcher with CCl4-specific components
    searcher = StructureSearcher(
        output_dir=f"tmp/{exp_name}",
        universe=universe,
        grids_XY=(X, Y),
        ARPDF_exp=ARPDF_ref,
        molecule_selector=select_center_molecules,
        structure_modifier=StructureModifier(universe, stretch_distances, periodic=True),
        filter_fourier=filter_fourier,
        sigma0=sigma0,
        metric=metric,
        weight_cutoff=weight_cutoff,
        neg=neg
    )
    
    # Search and save results
    searcher.search()
    searcher.save_results(X, Y, ARPDF_ref)
