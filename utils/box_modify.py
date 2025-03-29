import numpy as np
import MDAnalysis as mda


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


def compute_axis_direction(c_atom, cl_atom, box=None):
    """
    计算 C->Cl 方向的单位向量，并应用周期性边界条件（如果提供了盒子信息）。
    """
    # 计算原子之间的向量
    vec = cl_atom.position - c_atom.position

    # 如果提供了盒子信息（即考虑周期性边界条件），则应用PBC
    if box is not None:
        vec = apply_PBC(vec, box)

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
            cl_positions[i] = apply_PBC(cl_positions[i] - c_atom.position, box) + c_atom.position

    new_c_pos = np.mean(cl_positions, axis=0)

    # 计算 C->Cl 方向，并考虑周期性边界条件
    cl_vector = cl_target.position - new_c_pos
    
    # 应用周期性边界条件
    if box is not None:
        cl_vector = apply_PBC(cl_vector, box)
    
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
