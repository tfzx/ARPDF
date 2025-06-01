import numpy as np
import matplotlib.pyplot as plt
from ARPDF import compute_ARPDF, compare_ARPDF
from utils import (
    load_exp_data, 
    load_structure_data
)
from search_boxes import SimilarityCalculator

# 数据目录
data_dir = "data/CCl4"

# 加载实验数据
X_exp, Y_exp, ARPDF_exp = load_exp_data(
    data_dir=data_dir,
    rmax=10.0,  # 最大半径
    new_grid_size=None,  # 不进行重采样
    max_intensity=1.0  # 归一化最大强度
)

# 加载结构数据
u1_ref, u2_ref, modified_atoms, polar_axis = load_structure_data(data_dir)
polar_axis = np.array(polar_axis)
filter_fourier = lambda kX, kY, xp: (1 - xp.exp(-(kX**2 / 0.3 + kY**2 / 0.1)))**3 * xp.exp(-0.08 * (kX**2 + kY**2))

ARPDF_ref = compute_ARPDF(
    u1=u1_ref,
    u2=u2_ref,
    N=None,  # 不指定N，将使用grids_XY的大小
    cutoff=10.0,
    sigma0=0.4,
    grids_XY=(X_exp, Y_exp),  # 使用实验数据的网格
    modified_atoms=modified_atoms,
    polar_axis=polar_axis,
    periodic=True,
    filter_fourier=filter_fourier,
    verbose=False
)

C_index, Cl_index = 515, 519
sim_calculator = SimilarityCalculator(X_exp, Y_exp, weight_cutoff=5.0, metric='angular_scale')

true_delta_C = np.vdot(u2_ref.atoms[C_index].position - u1_ref.atoms[C_index].position, polar_axis)
true_delta_Cl = np.vdot(u2_ref.atoms[Cl_index].position - u1_ref.atoms[Cl_index].position, polar_axis)
print(f"true_delta_C: {true_delta_C:.2f}, true_delta_Cl: {true_delta_Cl:.2f}")

def move_atoms(u, polar_axis, delta_C, delta_Cl):
    u.atoms[C_index].position += polar_axis * delta_C
    u.atoms[Cl_index].position += polar_axis * delta_Cl
    return u

def get_similarity(delta_C, delta_Cl):
    # u2 = u2_ref.copy()
    # u2.atoms[[C_index, Cl_index]].positions = u1_ref.atoms[[C_index, Cl_index]].positions
    # u2 = move_atoms(u2, polar_axis, delta_C, delta_Cl)
    u2 = move_atoms(u1_ref.copy(), polar_axis, delta_C, delta_Cl)
    # u2.atoms[modified_atoms[-1]].position = u2_ref.atoms[modified_atoms[-1]].position
    # 使用实验数据的网格来计算ARPDF
    ARPDF_calc = compute_ARPDF(
        u1=u1_ref,
        u2=u2,
        N=None,  # 不指定N，将使用grids_XY的大小
        cutoff=10.0,
        sigma0=0.4,
        grids_XY=(X_exp, Y_exp),  # 使用实验数据的网格
        modified_atoms=[C_index, Cl_index],
        polar_axis=polar_axis,
        periodic=True,
        filter_fourier=filter_fourier,
        verbose=False
    )
    # 计算相似度
    similarity = sim_calculator.calc_similarity(ARPDF_calc, ARPDF_ref)
    return similarity

delta_C_range = np.linspace(-1.5, 0.5, 50)
delta_Cl_range = np.linspace(-1.0, 2.0, 100)
# delta_C_range = np.linspace(-0.5, 0.5, 50)
# delta_Cl_range = np.linspace(-1.0, 1.0, 100)

similarity_matrix = np.zeros((len(delta_C_range), len(delta_Cl_range)))

for i, delta_C in enumerate(delta_C_range):
    for j, delta_Cl in enumerate(delta_Cl_range):
        similarity_matrix[i, j] = get_similarity(delta_C, delta_Cl)

plt.figure(figsize=(10, 8))
plt.contourf(delta_C_range, delta_Cl_range, similarity_matrix.T, levels=20, cmap='viridis')
plt.colorbar(label='Similarity')
plt.scatter(true_delta_C, true_delta_Cl, color='red', marker='*', s=100, label='Current')
plt.legend()

plt.xlabel(r'$\Delta C$')
plt.ylabel(r'$\Delta Cl$')
plt.title('Similarity Landscape')
plt.show()



