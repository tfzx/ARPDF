import os
import pickle
import cupy as cp
import numpy as np
import MDAnalysis as mda
from ARPDF import compute_ARPDF, compare_ARPDF
from ARPDF_POLAR import compute_ARPDF_polar, compare_ARPDF_polar
from utils import select_nbr_mols, clean_gro_box, rotate_ccl4_molecules, select_ccl4_molecules, update_metadata
from utils.similarity import cosine_similarity, get_angular_filters, get_gaussian_filters, angular_similarity, strength_similarity, oneD_similarity, angular_average_similarity, polar_angular_similarity
from utils.core_functions import to_cupy
from ccl4_modifier import CCL4Modifier_CL, select_cl_atoms
from typing import Callable, List, Tuple, Optional, Protocol
from dataclasses import dataclass

@dataclass
class SearchResult:
    """Data class to store search results for a single molecule
    
    Attributes:
        molecule (int): Index of the target molecule
        polar_axis (ndarray): Direction vector of the modified bond
        modified_universe (mda.Universe): Modified structure
        ARPDF (ndarray): Calculated ARPDF for the modified structure
        similarity (float): Similarity score between ARPDF and experimental data
        modified_atoms (List[int]): Indices of modified atoms
    """
    molecule: int
    polar_axis: List[float]
    modified_universe: mda.Universe
    ARPDF: np.ndarray
    similarity: float
    modified_atoms: List[int]

class StrucModProtocol(Protocol):
    """Handle molecular structure modifications"""
    def generate_modified_structures(self, molecule: int) -> List[Tuple[List[float], mda.Universe, List[int]]]:
        """Generate modified structures
        Args:
            molecule (int): Index of the target atom or molecule
            
        Returns:
            List[Tuple[List[float], mda.Universe, List[int]]]: List of tuples containing:
                - polar_axis (List[float]): Polarization axis
                - u2 (Universe): Modified MDAnalysis universe 
                - modified_atoms (List[int]): List of modified atom indices
        """
    


def save_ccl4_result(result: SearchResult, file_name: str, nbr_distance: float = 5.0):
    """Save the result of CCl4 structure"""
    ccl4_indices = select_ccl4_molecules(result.modified_universe, result.molecule, cutoff_distance=nbr_distance)
    nbr_indices = select_nbr_mols(result.modified_universe, result.modified_atoms, nbr_distance=nbr_distance)
    nbr_group = rotate_ccl4_molecules(result.modified_universe, ccl4_indices, nbr_indices, result.polar_axis)
    nbr_group.write(file_name)


class SimilarityCalculator:
    """Handle similarity calculations between ARPDFs"""
    
    def __init__(self, X, Y, weight_cutoff=5.0, metric='cosine'):
        self.X, self.Y = to_cupy(X, Y)
        self.R = cp.sqrt(self.X**2 + self.Y**2)
        self.weight_cutoff = weight_cutoff
        self._initialize_weights()
        self._metric = self._get_metric_function(metric)
    
    def _initialize_weights(self):
        """Initialize various weights for similarity calculations"""
        # Cosine weight
        self.cos_weight = cp.exp(-cp.maximum(self.R - self.weight_cutoff, 0)**2 / (2 * (0.5)**2))/(1 + cp.exp(-10 * (self.R - 1)))
        
        # Cosine_r weight
        self.cos_r_weight = cp.exp(-cp.maximum(self.R - self.weight_cutoff, 0)**2 / (2 * (0.5)**2))/((1 + cp.exp(-20 * (self.R - 1)))*(self.R + 1e-8))
        # Circular weights
        self.r0_arr = cp.linspace(0, 8, 40)
        dr0 = self.r0_arr[1] - self.r0_arr[0]
        self.angular_filters = get_angular_filters(self.R, self.r0_arr, sigma=dr0/6.0)
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
            'cosine_r': lambda x, y: cosine_similarity(x, y, self.cos_r_weight),
            '1D': lambda x, y: oneD_similarity(x, y, axis=0, weight=self.axis_weight),
            '1D_average': lambda x, y: angular_average_similarity(x, y, weight=self.average_weight),
            'angular': lambda x, y: angular_similarity(x, y, self.angular_filters, self.r_weight),
            'angular_scale': lambda x, y: angular_similarity(x, y, self.angular_filters, self.r_weight) * strength_similarity(x, y, self.angular_filters, self.r_weight)
        }
        return metric_funcs[metric]


class PolarSimilarityCalculator:
    """Handle similarity calculations between polar ARPDFs"""
    
    def __init__(self, R, Theta, weight_cutoff=5.0, metric='cosine'):
        self.R, self.Theta = to_cupy(R, Theta)
        self.weight_cutoff = weight_cutoff
        self._initialize_weights()
        self._metric = self._get_metric_function(metric)
    
    def _initialize_weights(self):
        """Initialize various weights for polar similarity calculations"""
        # Circular weights
        self.r0_arr = cp.linspace(0, 8, 40)
        dr0 = self.r0_arr[1] - self.r0_arr[0]
        self.gaussian_filter = get_gaussian_filters(self.R, self.r0_arr, sigma=dr0/6.0)
        self.r_weight = cp.exp(-cp.maximum(self.r0_arr - self.weight_cutoff, 0)**2 / (2 * (0.5)**2))/ (1 + cp.exp(-10 * (self.r0_arr - 1)))
        self.r_weight /= self.r_weight.sum()
        
    
    def calc_similarity(self, ARPDF, ARPDF_exp):
        """Calculate similarity between two ARPDFs"""
        return self._metric(ARPDF, ARPDF_exp)
    
    def _get_metric_function(self, metric):
        """Get the appropriate metric function"""
        metric_funcs = {
            'angular': lambda x, y: polar_angular_similarity(x, y, self.gaussian_filter, self.r_weight),
            'angular_scale': lambda x, y: polar_angular_similarity(x, y, self.gaussian_filter, self.r_weight) * strength_similarity(x, y, self.gaussian_filter, self.r_weight)
        }
        return metric_funcs[metric]


class StructureSearcher:
    """Handle structure search and result management"""
    
    def __init__(self, 
                 output_dir: str,
                 universe: mda.Universe,
                 grids_XY: Tuple[np.ndarray, np.ndarray],
                 ARPDF_ref: np.ndarray,
                 molecule_selector: Callable[[mda.Universe], List[int]],
                 structure_modifier: StrucModProtocol,
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
            ARPDF_ref (ndarray): Reference ARPDF data
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
        self.X, self.Y = grids_XY
        self.ARPDF_ref = ARPDF_ref
        self.filter_fourier = filter_fourier
        self.cutoff = cutoff
        self.sigma0 = sigma0
        self.neg = neg
        
        # Store selection and modification functions
        self.molecule_selector = molecule_selector
        self.structure_modifier = structure_modifier
        
        # Initialize components
        self.weight_cutoff = weight_cutoff
        self.metric = metric
        self.similarity_calc = SimilarityCalculator(self.X, self.Y, weight_cutoff, metric)
        
        # Initialize results
        self.results: List[SearchResult] = []
    
    def search(self):
        """Search for optimal molecular structures
        
        Returns:
            list: List of SearchResult objects for each molecule
        """
        grids_XY = to_cupy(self.X, self.Y)
        ARPDF_ref = to_cupy(self.ARPDF_ref)
        # Get molecule list
        molecule_list = self.molecule_selector(self.universe)
        
        # Search for each molecule
        for molecule in molecule_list:
            best_similarity = -np.inf
            best_result = None
            
            for polar_axis, u2, modified_atoms in self.structure_modifier.generate_modified_structures(molecule):
                ARPDF = compute_ARPDF(
                    self.universe, u2, 256, self.cutoff, self.sigma0, grids_XY,
                    modified_atoms=modified_atoms, 
                    polar_axis=polar_axis,
                    periodic=True,
                    filter_fourier=self.filter_fourier,
                    verbose=False,
                    neg=self.neg
                )
                
                similarity = self.similarity_calc.calc_similarity(ARPDF, ARPDF_ref).get()
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_result = SearchResult(
                        molecule=int(molecule),
                        polar_axis=[float(x) for x in polar_axis],
                        modified_universe=u2,
                        ARPDF=ARPDF.get(),
                        similarity=float(similarity),
                        modified_atoms=[int(x) for x in modified_atoms]
                    )
            
            if best_result is not None:
                self.results.append(best_result)
        
        return self.results
    
    def save_results(self) -> None:
        """Save analysis results"""
        X, Y, ARPDF_ref = self.X, self.Y, self.ARPDF_ref
        
        if not self.results:
            return
            
        # get best and worst results
        best_result = max(self.results, key=lambda x: x.similarity)
        worst_result = min(self.results, key=lambda x: x.similarity)

        # Save structures
        self.universe.atoms.write(f"{self.output_dir}/CCl4.gro")
        best_result.modified_universe.atoms.write(f"{self.output_dir}/CCl4_best_init.gro")

        # Save best and worst results
        self._save_single_result(best_result, prefix="best_init")
        self._save_single_result(worst_result, prefix="worst_init")

        # save ARPDF_ref
        np.save(f"{self.output_dir}/ARPDF_ref.npy", ARPDF_ref)

        # Save raw results
        with open(f"{self.output_dir}/results.pkl", "wb") as f:
            pickle.dump(self.results, f)

        # Prepare metadata with search parameters
        grids_range = (X.min(), X.max(), Y.min(), Y.max())
        grids_range = list(float(x) for x in grids_range)

        # Save metadata
        update_metadata(self.output_dir, {
            "name": "CCl4",
            "structure_info": {
                "u1_name": "CCl4.gro",
                "u2_name": "CCl4_best_init.gro",
                "polar_axis": best_result.polar_axis,
                "modified_atoms": best_result.modified_atoms
            },
            "search_info": {
                "parameters": {
                    "grids_range": grids_range,
                    "grids_shape": list(self.X.shape),
                    "molecule_selector": self.molecule_selector.__name__,
                    "structure_modifier": self.structure_modifier.__class__.__name__,
                    "filter_fourier": str(self.filter_fourier),
                    "cutoff": self.cutoff,
                    "weight_cutoff": self.weight_cutoff,
                    "metric": self.metric,
                    "sigma0": self.sigma0,
                    "neg": self.neg,
                },
                "best_result": {
                    "file_name": "CCl4_best_init.gro",
                    "similarity": best_result.similarity,
                    "polar_axis": best_result.polar_axis,
                    "modified_atoms": best_result.modified_atoms,
                    "molecule": best_result.molecule
                },
                "worst_result": {
                    "file_name": "CCl4_worst_init.gro",
                    "similarity": worst_result.similarity,
                    "polar_axis": worst_result.polar_axis,
                    "modified_atoms": worst_result.modified_atoms,
                    "molecule": worst_result.molecule
                }
            },
        })

    def _save_single_result(self, result: SearchResult, prefix: str) -> None:
        """
        Save a single result (best or worst) to files.

        Args:
            result (SearchResult): The result to save.
            prefix (str, optional): Prefix for the result.
        """
        X, Y, ARPDF_ref = self.X, self.Y, self.ARPDF_ref
        save_ccl4_result(result, f"{self.output_dir}/{prefix}_selected.gro", nbr_distance=5.0)
        # Save visualization
        _metric_name = self.metric.replace("_", " ").title()
        fig = compare_ARPDF(
            result.ARPDF, ARPDF_ref, (X, Y), 
            sim_name=f"{_metric_name} Sim", sim_value=result.similarity, 
            show_range=8.0, weight_cutoff=self.weight_cutoff
        )
        fig.savefig(f"{self.output_dir}/{prefix}.png")


class PolarStructureSearcher:
    """Structure searcher using polar coordinates and polar similarity"""

    def __init__(self, 
                 output_dir: str,
                 universe: mda.Universe,
                 grids_polar: Tuple[np.ndarray, np.ndarray],  # (R, Theta)
                 ARPDF_ref: np.ndarray,
                 molecule_selector: Callable[[mda.Universe], List[int]],
                 structure_modifier: StrucModProtocol,
                 filter_fourier: Optional[Callable] = None,
                 cutoff: float = 10.0,
                 weight_cutoff: float = 4.0,
                 delta: float = 0.2,
                 metric: str = 'angular_scale',
                 sigma_similarity: float = 0.3,
                 neg: bool = False):
        """
        Args:
            grids_polar: Tuple of (R, Theta) meshgrid
            ARPDF_ref: Polar ARPDF to compare against (shape [Nr, Nθ])
            r_vals: 1D radial coordinate array
            sigma_similarity: Gaussian width for similarity weighting
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.universe = universe
        self.R_polar, self.Theta_polar = to_cupy(grids_polar)
        self.ARPDF_ref = to_cupy(ARPDF_ref)
        self.weight_cutoff = weight_cutoff

        self.metric = metric
        self.similarity_calc = PolarSimilarityCalculator(self.R_polar, self.Theta_polar, weight_cutoff, metric)
        

        #self.r0_arr = cp.linspace(0, 8, 40)
        #dr0 = self.r0_arr[1] - self.r0_arr[0]
        #self.gaussian_filters = get_gaussian_filters(self.R_polar, self.r0_arr, sigma=dr0/6.0)
        #self.r_weight = cp.exp(-cp.maximum(self.r0_arr - self.weight_cutoff, 0)**2 / (2 * (0.5)**2))/ (1 + cp.exp(-10 * (self.r0_arr - 1)))
        #self.r_weight /= self.r_weight.sum()

        self.filter_fourier = filter_fourier
        self.cutoff = cutoff
        self.delta = delta
        self.neg = neg
        self.sigma_similarity = sigma_similarity

        self.molecule_selector = molecule_selector
        self.structure_modifier = structure_modifier
        self.results: List[SearchResult] = []

    def search(self):
        """Perform structure search"""
        molecule_list = self.molecule_selector(self.universe)

        ARPDF_ref = to_cupy(self.ARPDF_ref)
        
        for molecule in molecule_list:
            best_similarity = -np.inf
            best_result = None
            
            for polar_axis, u2, modified_atoms in self.structure_modifier.generate_modified_structures(molecule):
                ARPDF = compute_ARPDF_polar(
                    u1=self.universe,
                    u2=u2,
                    N=512,
                    cutoff=self.cutoff,
                    sigma0=0.0,  # not used
                    delta=self.delta,
                    grids_polar=(self.R_polar, self.Theta_polar),
                    modified_atoms=modified_atoms,
                    polar_axis=polar_axis,
                    periodic=True,
                    filter_fourier=self.filter_fourier,
                    verbose=False,
                    neg=self.neg
                )

                similarity = self.similarity_calc.calc_similarity(ARPDF['total'], ARPDF_ref).get()

                '''
                similarity = polar_angular_similarity(
                    ARPDF["total"], 
                    self.ARPDF_ref, 
                    self.gaussian_filters, 
                    self.r_weight
                ).get() * strength_similarity(ARPDF["total"], 
                    self.ARPDF_ref, 
                    self.gaussian_filters, 
                    self.r_weight).get()
                '''
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_result = SearchResult(
                        molecule=int(molecule),
                        polar_axis=[float(x) for x in polar_axis],
                        modified_universe=u2,
                        ARPDF=ARPDF["total"].get(),
                        similarity=float(similarity),
                        modified_atoms=[int(x) for x in modified_atoms]
                    )
            
            if best_result is not None:
                self.results.append(best_result)

        return self.results

    def save_results(self):
        """Save best and worst results to output_dir"""
        if not self.results:
            return
        
        best_result = max(self.results, key=lambda x: x.similarity)
        worst_result = min(self.results, key=lambda x: x.similarity)

        self.universe.atoms.write(f"{self.output_dir}/CCl4.gro")
        best_result.modified_universe.atoms.write(f"{self.output_dir}/CCl4_best_init.gro")

        self._save_single_result(best_result, "best_init")
        self._save_single_result(worst_result, "worst_init")

        np.save(f"{self.output_dir}/ARPDF_ref.npy", self.ARPDF_ref.get())

        with open(f"{self.output_dir}/results.pkl", "wb") as f:
            pickle.dump(self.results, f)

        update_metadata(self.output_dir, {
            "name": "CCl4",
            "structure_info": {
                "u1_name": "CCl4.gro",
                "u2_name": "CCl4_best_init.gro",
                "polar_axis": best_result.polar_axis,
                "modified_atoms": best_result.modified_atoms
            },
            "search_info": {
                "parameters": {
                    "grids_shape": list(self.R_polar.shape),
                    "cutoff": self.cutoff,
                    "delta": self.delta,
                    "sigma_similarity": self.sigma_similarity,
                    "neg": self.neg,
                },
                "best_result": {
                    "file_name": "CCl4_best_init.gro",
                    "similarity": best_result.similarity,
                    "polar_axis": best_result.polar_axis,
                    "modified_atoms": best_result.modified_atoms,
                    "molecule": best_result.molecule
                },
                "worst_result": {
                    "file_name": "CCl4_worst_init.gro",
                    "similarity": worst_result.similarity,
                    "polar_axis": worst_result.polar_axis,
                    "modified_atoms": worst_result.modified_atoms,
                    "molecule": worst_result.molecule
                }
            },
        })

    def _save_single_result(self, result: SearchResult, prefix: str):
        save_ccl4_result(result, f"{self.output_dir}/{prefix}_selected.gro", nbr_distance=5.0)
        fig = compare_ARPDF_polar(
            result.ARPDF, 
            self.ARPDF_ref.get(), 
            (self.R_polar.get(), self.Theta_polar.get()),
            sim_name="Polar Gaussian Sim", 
            sim_value=result.similarity,
            show_range=8.0,
            weight_cutoff=self.cutoff
        )
        fig.savefig(f"{self.output_dir}/{prefix}.png")



def workflow_demo(
        X, Y, ARPDF_ref, 
        filter_fourier=None, 
        sigma0=0.2, 
        exp_name: str="exp", 
        metric: str="1D_average", 
        weight_cutoff=5.0, 
        stretch_distances: List[float] = None,
        neg=False
    ):
    """Demo workflow for structure search and analysis"""
    # Clean and load structure
    clean_gro_box('data/CCl4/CCl4.gro', 'data/CCl4/CCl4_clean.gro')
    universe = mda.Universe('data/CCl4/CCl4_clean.gro')
    if neg:
        ARPDF_ref = ARPDF_ref.copy()
        ARPDF_ref[ARPDF_ref > 0] = 0
    
    # Initialize searcher with CCl4-specific components
    searcher = StructureSearcher(
        output_dir=f"tmp/{exp_name}",
        universe=universe,
        grids_XY=(X, Y),
        ARPDF_ref=ARPDF_ref,
        molecule_selector=select_cl_atoms,
        structure_modifier=CCL4Modifier_CL(universe, stretch_distances, periodic=True),
        filter_fourier=filter_fourier,
        sigma0=sigma0,
        metric=metric,
        weight_cutoff=weight_cutoff,
        neg=neg
    )
    
    # Search and save results
    searcher.search()
    searcher.save_results()


def polar_workflow_demo(
        R: np.ndarray,
        Theta: np.ndarray,
        ARPDF_ref: np.ndarray,
        filter_fourier=None,
        exp_name: str = "exp_polar",
        sigma_similarity: float = 0.3,
        delta: float = 0.2,
        cutoff: float = 10.0,
        metric: str = 'angular_scale',
        weight_cutoff: float = 5.0,
        stretch_distances: List[float] = None,
        neg: bool = False
    ):
    """
    Workflow demo for polar-coordinate-based structure search using PolarStructureSearcher.
    
    Args:
        R: 2D radial meshgrid (shape [Nr, Nθ])
        Theta: 2D angular meshgrid (shape [Nr, Nθ])
        ARPDF_ref: Reference polar ARPDF (shape [Nr, Nθ])
        r_vals: 1D radial grid corresponding to R
        filter_fourier: Optional filtering function for ARPDF
        exp_name: Experiment name for output directory
        sigma_similarity: Gaussian similarity width
        delta: Angular grid step size (rad)
        cutoff: Cutoff for RDF and weighting
        stretch_distances: Distances to stretch in CCl4Modifier_CL
        neg: Whether to use negative ARPDF values only
    """
    # Clean and load structure
    clean_gro_box('data/CCl4/CCl4.gro', 'data/CCl4/CCl4_clean.gro')
    universe = mda.Universe('data/CCl4/CCl4_clean.gro')

    # Optional preprocessing
    if neg:
        ARPDF_ref = ARPDF_ref.copy()
        ARPDF_ref[ARPDF_ref > 0] = 0

    # Initialize polar structure searcher
    searcher = PolarStructureSearcher(
        output_dir=f"tmp/{exp_name}",
        universe=universe,
        grids_polar=(R, Theta),
        ARPDF_ref=ARPDF_ref,
        molecule_selector=select_cl_atoms,
        structure_modifier=CCL4Modifier_CL(universe, stretch_distances, periodic=True),
        filter_fourier=filter_fourier,
        cutoff=cutoff,
        metric=metric,
        delta=delta,
        weight_cutoff=weight_cutoff,
        sigma_similarity=sigma_similarity,
        neg=neg
    )

    # Run the search
    searcher.search()

    # Save results
    searcher.save_results()

