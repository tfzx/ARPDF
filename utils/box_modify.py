import numpy as np
import MDAnalysis as mda
import re
from utils import box_shift

'''
def apply_PBC(vec, box):
    """
    应用周期性边界条件，确保向量在盒子内。
    """
    for i in range(3):
        if vec[i] > 0.5 * box[i]:
            vec[i] -= box[i]
        elif vec[i] < -0.5 * box[i]:
            vec[i] += box[i]
    return vec
'''

def compute_axis_direction(c_atom, cl_atom, box=None):
    """
    计算 C->Cl 方向的单位向量，并应用周期性边界条件（如果提供了盒子信息）。
    """
    # 计算原子之间的向量
    vec = cl_atom.position - c_atom.position

    # 如果提供了盒子信息（即考虑周期性边界条件），则应用PBC
    if box is not None:
        vec = box_shift(vec, box)

    # 计算向量的模长
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("C 和 Cl 位置重合，无法计算方向")

    # 返回单位向量
    return vec / norm

def adjust_ccl3_structure(c_atom, cl_target, other_cls, stretch_distance=0.2, modified_atoms=None, box=None):
    """
    调整 C 位置使其位于 CCl₃ 平面中心，并沿 C-Cl 方向伸长 Cl 位置。

    参数：
    - c_atom: C 原子 (MDAnalysis Atom 对象)
    - cl_target: 目标 Cl 原子 (MDAnalysis Atom 对象)
    - other_cls: 其他 Cl 原子列表 (MDAnalysis Atom 对象列表)
    - stretch_distance: Cl 伸长的距离 (默认 0.2)
    - modified_atoms: 记录被修改原子的编号列表 (可选)
    - box: 盒子的尺寸信息 (如果存在周期性边界条件)

    返回：
    - new_c_pos: 调整后的 C 位置
    - new_cl_pos: 调整后的 Cl 位置
    """
    # 计算 CCl₃ 平面中心，考虑周期性边界条件
    cl_positions = np.array([cl.position for cl in other_cls])

    if box is not None:
        # 应用PBC校正所有Cl原子与C原子之间的位置
        for i in range(len(cl_positions)):
            cl_positions[i] = box_shift(cl_positions[i] - c_atom.position, box) + c_atom.position

    new_c_pos = np.mean(cl_positions, axis=0)

    # 计算 C->Cl 方向，并考虑周期性边界条件
    cl_vector = cl_target.position - new_c_pos
    
    # 应用周期性边界条件
    if box is not None:
        cl_vector = box_shift(cl_vector, box)
    
    cl_direction = cl_vector / np.linalg.norm(cl_vector)
    
    # 计算伸长后的 Cl 位置
    new_cl_pos = new_c_pos + cl_direction * (np.linalg.norm(cl_vector) + stretch_distance)


    # 记录修改的原子编号
    if modified_atoms is not None:
        modified_atoms.append(c_atom.index)
        modified_atoms.append(cl_target.index)

    # 更新原子位置
    c_atom.position = new_c_pos
    cl_target.position = new_cl_pos

def adjust_ccl(c_atom, cl_target, stretch_distance=0.2, modified_atoms=None, box=None):
    """
    沿 C-Cl 方向伸长 Cl 位置。

    参数：
    - c_atom: C 原子 (MDAnalysis Atom 对象)
    - cl_target: 目标 Cl 原子 (MDAnalysis Atom 对象)
    - stretch_distance: Cl 伸长的距离 (默认 0.2)
    - modified_atoms: 记录被修改原子的编号列表 (可选)
    - box: 盒子的尺寸信息 (如果存在周期性边界条件)

    返回：
    - new_cl_pos: 调整后的 Cl 位置
    """
  
    # 计算 C->Cl 方向，并考虑周期性边界条件
    cl_vector = cl_target.position - c_atom.position
    
    # 应用周期性边界条件
    if box is not None:
        cl_vector = box_shift(cl_vector, box)
    
    cl_direction = cl_vector / np.linalg.norm(cl_vector)
    
    # 计算伸长后的 Cl 位置
    new_cl_pos =  c_atom.position + cl_direction * (np.linalg.norm(cl_vector) + stretch_distance)


    # 记录修改的原子编号
    if modified_atoms is not None:
        modified_atoms.append(cl_target.index)
        #modified_atoms += list(other_cls.indices)

    # 更新原子位置
    cl_target.position = new_cl_pos

def rotate_CH3CN_along_CN(mol_atoms, phi_deg=0, theta_deg=0, modified_atoms=None, box=None):
    """
    将乙腈分子绕 CN 方向进行旋转，考虑周期性边界条件。
    
    参数：
    - mol_atoms: 乙腈分子的原子集合 (MDAnalysis AtomGroup)
    - phi_deg: 绕 CN 轴的旋转角度（单位：度，右手法则）
    - theta_deg: 绕垂直于 CN 的轴的旋转角度（单位：度）
    - modified_atoms: 若提供，将记录所有被修改坐标的原子编号（index）
    - box: 周期性边界条件的盒子尺寸 (None 表示不考虑 PBC)
    """
    # 选出C与N原子
    c_atom = mol_atoms.select_atoms("name C")[0]
    n_atom = mol_atoms.select_atoms("name N")[0]
    
    # 几何中心为旋转中心
    center = mol_atoms.center_of_geometry()

    # 构造 CN 方向向量（单位化），考虑 PBC
    cn_vec = n_atom.position - c_atom.position
    if box is not None:
        cn_vec = box_shift(cn_vec, box)
    z_axis = cn_vec / np.linalg.norm(cn_vec)  # 新的 Z 轴

    # 构造垂直于 z 的其他轴：新 x、y
    arbitrary = np.array([1.0, 0.0, 0.0]) if abs(np.dot(z_axis, [1, 0, 0])) < 0.99 else np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(arbitrary, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    # 坐标系变换矩阵：世界 -> 分子局部坐标系
    R_world_to_local = np.stack([x_axis, y_axis, z_axis], axis=1)  # shape (3,3)

    # 构造绕 z（CN方向）的旋转
    phi = np.radians(phi_deg)
    Rz = np.array([
        [np.cos(phi), -np.sin(phi), 0],
        [np.sin(phi),  np.cos(phi), 0],
        [0, 0, 1]
    ])

    # 构造绕 x 的旋转（theta）
    theta = np.radians(theta_deg)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta),  np.cos(theta)]
    ])

    # 总旋转（在局部坐标系下）
    R_local = Rx @ Rz

    # 变换回世界坐标系
    R_total = R_world_to_local @ R_local @ R_world_to_local.T

    # 执行旋转（原子绕几何中心）
    for atom in mol_atoms:
        shift = atom.position - center
        new_pos = R_total @ shift + center
        atom.position = new_pos
        if modified_atoms is not None:
            modified_atoms.append(atom.index)

def adjust_ccn(ch3_c_atom, cn_c_atom, n_atom, stretch_distance=0.2, modified_atoms=None, box=None):
    """
    沿 CH3–C 到 C≡N 的方向，整体平移 CN 基团，使得 C–C 距离增加 stretch_distance。

    参数：
    - ch3_c_atom: CH3 端的 C 原子 (MDAnalysis Atom)
    - cn_c_atom: CN 基团的 C 原子 (MDAnalysis Atom)
    - n_atom: N 原子 (MDAnalysis Atom)
    - stretch_distance: C–C 延长的距离（单位 Å）
    - modified_atoms: 可选，用于记录被修改的原子索引列表
    - box: 周期性边界条件盒子维度 (可选)

    返回：
    - None（原地修改 cn_c_atom 和 n_atom 的位置）
    """
    # 计算 C–C 方向向量
    cc_vector = cn_c_atom.position - ch3_c_atom.position

    if box is not None:
        cc_vector = box_shift(cc_vector, box)

    cc_unit = cc_vector / np.linalg.norm(cc_vector)

    # 整体平移 CN 基团
    cn_c_atom.position += cc_unit * stretch_distance
    n_atom.position     += cc_unit * stretch_distance

    # 记录被修改的原子
    if modified_atoms is not None:
        modified_atoms.extend([cn_c_atom.index, n_atom.index])


import MDAnalysis as mda
import re

def clean_gro_box(input_gro, output_gro, rename_mapping=None):
    """
    读取一个 .gro 文件，并根据给定的 rename_mapping 规则将原子名称标准化，
    最后将结果写入一个新的 .gro 文件。

    参数:
        input_gro (str): 输入的 .gro 文件路径。
        output_gro (str): 输出的 .gro 文件路径。
        rename_mapping (dict): 可选字典，key 为正则表达式模式，value 为对应的标准名称，
                               默认规则为:
                                   {r"^Cl": "Cl", r"^C": "C"}
                               如果一个原子名匹配多个规则，只按第一个匹配的规则处理。
    """
    # 如果未提供重命名映射，则使用默认规则
    if rename_mapping is None:
        rename_mapping = {
            r"^Cl": "Cl",  # 以 "Cl" 开头的统一命名为 "Cl"
            r"^C": "C",    # 以 "C" 开头的统一命名为 "C"
        }
    
    # 读取 gro 文件（gro 文件包含拓扑和坐标）
    u = mda.Universe(input_gro)
    atoms = u.atoms  # 全部原子
    
    # 遍历每个原子，根据映射规则重命名
    for atom in atoms:
        original_name = atom.name.strip()
        new_name = original_name  # 默认保持原名
        for pattern, standard_name in rename_mapping.items():
            if re.match(pattern, original_name):
                new_name = standard_name
                break  # 按第一条匹配规则处理
        atom.name = new_name

    # 利用 MDAnalysis Writer 写出新的 gro 文件，注意原子顺序及坐标信息不变
    with mda.Writer(output_gro, atoms.n_atoms) as writer:
        writer.write(atoms)
    print(f"已生成过滤并重命名后的文件: {output_gro}")
