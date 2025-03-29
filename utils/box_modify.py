import numpy as np
import MDAnalysis as mda

def compute_axis_direction(c_atom, cl_atom):
    """
    计算 C->Cl 方向的单位向量
    """
    vec = cl_atom.position - c_atom.position
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("C 和 Cl 位置重合，无法计算方向")
    return vec / norm

def adjust_ccl3_structure(c_atom, cl_target, other_cls, stretch_distance=0.2, modified_atoms=None):
    """
    调整 C 位置使其位于 CCl₃ 平面中心，并沿 C-Cl 方向伸长 Cl 位置。

    参数：
    - c_atom: C 原子 (MDAnalysis Atom 对象)
    - cl_target: 目标 Cl 原子 (MDAnalysis Atom 对象)
    - other_cls: 其他 Cl 原子列表 (MDAnalysis Atom 对象列表)
    - stretch_distance: Cl 伸长的距离 (默认 0.2)
    - modified_atoms: 记录被修改原子的编号列表 (可选)

    返回：
    - new_c_pos: 调整后的 C 位置
    - new_cl_pos: 调整后的 Cl 位置
    """
    # 计算 CCl₃ 平面中心
    cl_positions = np.array([cl.position for cl in other_cls])
    new_c_pos = np.mean(cl_positions, axis=0)

    # 计算 C->Cl 方向并伸长 Cl 位置
    cl_vector = cl_target.position - new_c_pos
    cl_direction = cl_vector / np.linalg.norm(cl_vector)
    new_cl_pos = new_c_pos + cl_direction * (np.linalg.norm(cl_vector) + stretch_distance)

    # 记录修改的原子编号
    if modified_atoms is not None:
        modified_atoms.append(c_atom.index)
        modified_atoms.append(cl_target.index)

    # 更新原子位置
    c_atom.position = new_c_pos
    cl_target.position = new_cl_pos