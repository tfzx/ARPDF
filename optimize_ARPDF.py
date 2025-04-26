import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import MDAnalysis as mda
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import matplotlib.pyplot as plt
from ARPDF import compare_ARPDF, get_atoms_pos
import utils
from utils import calc_AFF, update_metadata
from utils.core_functions import get_circular_weight, weighted_similarity, cosine_similarity
from utils.core_functions_torch import generate_gaussian_kernel, gaussian_filter, get_abel_trans_mat, generate_field, toTensor, GND
from tqdm import tqdm


class ARPDFModel:
    def __init__(self, X, Y, type_counts: Dict[str, int], filter_fourier = None, cutoff = 10.0, sigma0 = 0.4, field_batch_size=256):
        super(ARPDFModel, self).__init__()
        self.X = toTensor(X).float().contiguous()
        self.Y = toTensor(Y).float().contiguous()
        self.R = torch.sqrt(self.X**2 + self.Y**2)
        self.hx = (self.X[1, 1] - self.X[0, 0]).item()
        self.hy = (self.Y[1, 1] - self.Y[0, 0]).item()
        self.h = max(self.hx, self.hy)
        self.cutoff = cutoff
        self.sigma0 = sigma0
        self.field_batch_size = field_batch_size
        self.prepare_transform(type_counts, filter_fourier)

    def set_system(self, around_pos, atom_pairs: Dict[str, Any], polar_axis):
        device = self.X.device
        self.around_pos = toTensor(around_pos, device=device).float()
        self.all_pair_types: List[Tuple[str, str]] = list(atom_pairs.keys())
        for (atom_type1, atom_type2), ij_idx in atom_pairs.items():
            setattr(self, f"atom_pairs_{atom_type1}_{atom_type2}", toTensor(ij_idx, dtype=torch.int64, device=device))
        polar_axis = torch.tensor(polar_axis, dtype=torch.float32, device=device)
        polar_axis /= torch.linalg.norm(polar_axis)
        self.polar_axis = polar_axis
        
    def prepare_transform(self, type_counts: Dict[str, int], filter_fourier = None):
        N = self.X.shape[0]

        # Fourier grids
        kx = np.fft.fftfreq(N, d=self.hx)
        ky = np.fft.fftfreq(N, d=self.hy)
        kX, kY = np.meshgrid(kx, ky)
        kX, kY = toTensor(kX).float(), toTensor(kY).float()
        S = torch.sqrt(kX**2 + kY**2)

        # Calculate AFFs
        for atom_type in type_counts:
            setattr(self, f"AFF_{atom_type}", calc_AFF(atom_type, S))
        # Filter in Fourier space
        if filter_fourier is None:
            # filter_ = (1 - torch.exp(-(kX**2 / 0.3 + kY**2 / 0.1)))**3 * torch.exp(-0.08 * S**2)
            filter_ = 1.0
        else:
            filter_ = filter_fourier(kX, kY, torch)
        # Atomic form factor normalization
        I_atom = sum([num_atom * self.AFFs(atom)**2 for atom, num_atom in type_counts.items()]) / sum(type_counts.values())
        self.factor = filter_ / I_atom

        # Abel transform matrix
        self.abel_trans_mat = get_abel_trans_mat(N)

        # Gaussian kernel
        self.gaussian_kernel = generate_gaussian_kernel((self.sigma0/self.hx, self.sigma0/self.hy))

    def to(self, *args, **kwargs):
        self.__dict__.update(toTensor(self.__dict__, *args, **kwargs))
        return self
    
    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def abel_inversion(self, image: torch.Tensor):
        return image @ self.abel_trans_mat

    def atom_pairs(self, atom_type1, atom_type2):
        return getattr(self, f"atom_pairs_{atom_type1}_{atom_type2}")

    def AFFs(self, atom_type):
        return getattr(self, f"AFF_{atom_type}")
    
    def forward(self, center_pos: torch.Tensor):
        all_pos = torch.cat([center_pos, self.around_pos], dim=0)

        total_fft = torch.zeros_like(self.X, dtype=torch.complex64)
        for atom_type1, atom_type2 in self.all_pair_types:
            ij_idx = self.atom_pairs(atom_type1, atom_type2)
            vectors = all_pos[ij_idx[:, 1]] - all_pos[ij_idx[:, 0]]
            r_vals = torch.linalg.norm(vectors, dim=1)
            cos_theta_vals = (vectors @ self.polar_axis) / r_vals
            mask = r_vals <= self.cutoff
            field = generate_field(self.R, self.Y, r_vals[mask], cos_theta_vals[mask], delta=4*self.h, batch_size=self.field_batch_size)
            fft = torch.fft.fft2(field)           # 2D FFT
            fft *= self.AFFs(atom_type1) * self.AFFs(atom_type2)      # Apply atom form factors
            total_fft += fft
        
        total_fft *= self.factor
        total_ifft = torch.fft.ifft2(total_fft).real
        Inverse_Abel_total = self.abel_inversion(total_ifft)
        image = gaussian_filter(self.gaussian_kernel, Inverse_Abel_total) * self.R**2

        return image

class ARPDFOptimizer:
    def __init__(
            self, 
            X, Y,
            ARPDF_exp, 
            type_counts: Dict[str, int],
            filter_fourier = None,
            cutoff=10.0,
            sigma0=0.4,
            weight_cutoff=6.0,
            lr=0.01, 
            gamma=0.995,
            f_lb=0.0, 
            s=0.1, 
            beta=0.1, 
            epochs=1000,
            loss_name="circular",
            device="cpu",
        ):
        self.X = toTensor(X, device=device).float().contiguous()
        self.Y = toTensor(Y, device=device).float().contiguous()
        self.ARPDF_exp = toTensor(ARPDF_exp, device=device).float().contiguous()
        self.h = X[1, 1] - X[0, 0]
        self.type_counts = type_counts
        self.model = ARPDFModel(X, Y, type_counts, filter_fourier, cutoff, sigma0, field_batch_size=128).to(device=device)
        self.cutoff = cutoff
        self.sigma0 = sigma0
        self.f_lb = f_lb
        self.lr = lr
        self.gamma = gamma
        self.s = s
        self.beta = beta
        self.epochs = epochs
        self.loss_name = loss_name
        self.device = device
        self._loss_func = self._get_loss_func(loss_name)
        self._prepare_weights(weight_cutoff=weight_cutoff)

    def set_system(
            self, 
            out_dir: Optional[str], 
            u1: Optional[mda.Universe] = None, 
            u2: Optional[mda.Universe] = None, 
            modified_atoms: Optional[List[int]] = None, 
            polar_axis = None
        ):
        if any(x is None for x in (u1, modified_atoms, polar_axis)):
            u1, u2, modified_atoms, polar_axis = utils.load_structure_data(out_dir)
        self.out_dir = out_dir if out_dir else "tmp"
        self.u1 = u1
        self.modified_atoms = modified_atoms
        self.polar_axis = polar_axis
        center_pos1, around_pos1, center_masses, atom_pairs = get_atoms_pos(u1, modified_atoms, cutoff=self.cutoff + 2.0, periodic=True)
        if u2 is not None:
            center_pos2, _, _, _ = get_atoms_pos(u2, modified_atoms, cutoff=self.cutoff + 2.0, periodic=True)
        else:
            center_pos2 = np.copy(center_pos1)
        self.center_pos1 = toTensor(center_pos1, device=self.device).float()
        self.center_pos2_init = toTensor(center_pos2, device=self.device).float()
        self.center_masses = toTensor(center_masses, device=self.device).float()
        self.model.set_system(around_pos1, atom_pairs, polar_axis)
        with torch.no_grad():
            self.image1 = self.model(self.center_pos1)
        
        self.num_atoms = center_pos1.shape[0]
        self._params = torch.zeros((self.num_atoms, 3), dtype=torch.float32, device=self.device, requires_grad=True)
        self.optimizer = optim.Adam([self._params], lr=self.lr)
        self.noise_scheduler = GND(self.optimizer, s=self.s, f_lb=self.f_lb)
        self.lr_scheduler = ExponentialLR(self.optimizer, gamma=self.gamma)

    def to(self, *args, **kwargs):
        self.__dict__.update(toTensor(self.__dict__, *args, **kwargs))
        self.model.to(*args, **kwargs)
        return self
    
    def _prepare_weights(self, weight_cutoff=6.0):
        R = self.model.R
        sigma_R = 1 / 3
        self.cosine_weight = torch.exp(-torch.clip(R - weight_cutoff, 0.0)**2 / (2 * (0.5)**2))/(1 + torch.exp(-10 * (R - 1)))
        # self.cosine_weight = torch.exp(-torch.clip(R - weight_cutoff, 0.0)**2 / (2 * sigma_R**2))
        self.cosine_weight /= self.cosine_weight.sum()
        r0_arr = torch.linspace(0, 8, 40)
        dr0 = (r0_arr[1] - r0_arr[0]).item()
        self.circular_weights = toTensor(get_circular_weight(R.cpu().numpy(), r0_arr.numpy(), sigma=dr0/6.0), device=self.device).float()
        r_weight = torch.exp(-torch.clip(r0_arr - weight_cutoff, 0)**2 / (2 * (sigma_R)**2)).to(device=self.device, dtype=torch.float32)
        r_weight /= r_weight.sum()
        self.r_weight = r_weight


    def _get_delta_pos(self):
        return self._params

    def _get_loss_func(self, loss_name: str) -> Callable[[torch.Tensor], torch.Tensor]:
        loss_func_map = {
            "cosine": self._loss_cosine,
            "circular": self._loss_circular
        }
        return loss_func_map[loss_name.strip().lower()]

    def _loss_cosine(self, ARPDF_pred):
        return -cosine_similarity(ARPDF_pred, self.ARPDF_exp, self.cosine_weight)

    def _loss_circular(self, ARPDF_pred):
        return -torch.vdot(self.r_weight, weighted_similarity(self.circular_weights, ARPDF_pred, self.ARPDF_exp))

    def _normalization(self, delta_pos):
        return torch.sum(delta_pos**2, dim=1).mean()

    def optimize(self, verbose=True, log_step=5, print_step=50, leave=True, prefix="Optimizing Atoms"):
        traj = np.zeros((self.epochs + 1, self.num_atoms, 3), dtype=np.float32)
        log = {
            "epoch": [],
            "lr": [],
            "loss": [],
            "cos_sim": [],
        }

        for epoch in tqdm(range(self.epochs), desc=prefix, disable=not verbose, leave=leave, position=0):
            self.optimizer.zero_grad()

            delta_pos = self._get_delta_pos()
            center_pos2 = self.center_pos2_init + delta_pos
            traj[epoch] = center_pos2.detach().cpu().numpy()

            ARPDF_pred = self.model(center_pos2) - self.image1
            loss = self._loss_func(ARPDF_pred) + self.beta * self._normalization(delta_pos)
            loss.backward()
            self.optimizer.step()

            if self.s > 0:
                with torch.no_grad():
                    new_delta = self._get_delta_pos()
                    center_pos2_tmp = self.center_pos2_init + new_delta
                    ARPDF_tmp = self.model(center_pos2_tmp) - self.image1
                    loss_tmp = self._loss_func(ARPDF_tmp)
                    self.noise_scheduler.step(loss_tmp)

            self.lr_scheduler.step()

            if epoch % log_step == 0:
                lr = self.lr_scheduler.get_last_lr()[0]
                cos_sim = F.cosine_similarity(ARPDF_pred.flatten(), self.ARPDF_exp.flatten(), dim=0)
                if verbose and epoch % print_step == 0:
                    tqdm.write(f"Epoch {epoch}: Loss={loss.item():.6f}, CosSim={cos_sim.item():.6f}, LR={lr:.6f}")
                log["epoch"].append(epoch)
                log["lr"].append(lr)
                log["loss"].append(loss.item())
                log["cos_sim"].append(cos_sim.item())

        self.center_pos2_final = (self.center_pos2_init + self._get_delta_pos()).detach()
        traj[-1] = self.center_pos2_final.detach().cpu().numpy()
        self.dump_results(traj, log)
    
    def dump_results(self, traj, log):
        ARPDF_optimized = self.model(self.center_pos2_final) - self.image1
        fig = compare_ARPDF(ARPDF_optimized.cpu().numpy(), self.ARPDF_exp.cpu().numpy(), grids_XY=(self.X.cpu().numpy(), self.Y.cpu().numpy()), show_range=8.0)
        fig.savefig(os.path.join(self.out_dir, 'CCl4_optimized.png'))
        np.save(os.path.join(self.out_dir, 'traj.npy'), traj)
        df = pd.DataFrame(log)
        df.reindex(columns=["epoch", "lr", "loss", "cos_sim"])
        df.to_csv(os.path.join(self.out_dir, 'log.txt'), index=False)
        u2_opt = self.u1.copy()
        u2_opt.atoms[self.modified_atoms].positions = self.center_pos2_final.cpu().numpy()
        u2_opt.atoms.write(os.path.join(self.out_dir, 'CCl4_optimized.gro'))
        update_metadata(self.out_dir, {
            "optimization_info": {
                "u2_opt_name": "CCl4_optimized.gro",
                "log_name": "log.txt",
                "traj_name": "traj.npy",
                "ARPDF_size": self.ARPDF_exp.shape,
                "type_counts": self.type_counts,
                "hyperparameters": {
                    "cutoff": self.cutoff,
                    "sigma0": self.sigma0,
                    "weight_cutoff": self.cutoff,
                    "lr": self.lr,
                    "gamma": self.gamma,
                    "f_lb": self.f_lb,
                    "s": self.s,
                    "beta": self.beta,
                    "epochs": self.epochs,
                    "loss_name": self.loss_name,
                    "device": self.device
                },
            }
        })
