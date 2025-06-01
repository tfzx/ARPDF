import numpy as np
import MDAnalysis as mda
from MDAnalysis.coordinates import GRO

def remove_hydrogens(input_file, output_file):
    u = mda.Universe(input_file)
    non_h_atoms = u.select_atoms("not name H*")
    with mda.Writer(output_file, n_atoms=len(non_h_atoms)) as w:
        w.write(non_h_atoms)
    return output_file

def stretch_ccn_group(u, resid=241, stretch_amount=1.0):
    """
    沿 CH3–C–N 分子的 C–C≡N 方向将整个 CN 基团（C≡N）平移，远离 CH3端碳，实现整体伸长。

    Args:
        u (MDAnalysis.Universe): 要修改的 Universe。
        resid (int): 分子的 resid 编号。
        stretch_amount (float): C–C 键的伸长量（Å）。
    """
    mol = u.select_atoms(f"resid {resid}")
    c_atoms = mol.select_atoms("name C")
    n_atoms = mol.select_atoms("name N")
    
    if len(c_atoms) < 2 or len(n_atoms) < 1:
        raise ValueError("该分子中原子数量不足以构成 C–C≡N 键")

    # 通常第一个 C 是 CH3 端的，第二个是 C≡N 中的碳
    ch3_c = c_atoms[0]
    cn_c = c_atoms[1]
    n_atom = n_atoms[0]

    # 计算 C≡N 基团方向向量（从 CH3 端的 C 指向 CN 端的 C）
    direction = cn_c.position - ch3_c.position
    unit_direction = direction / np.linalg.norm(direction)

    # 平移 C≡N 中的两个原子
    cn_c.position += unit_direction * stretch_amount
    n_atom.position += unit_direction * stretch_amount

    print("📍 C≡N 基团已整体平移。被修改的原子编号：")
    print(f"  C (id={cn_c.id}), N (id={n_atom.id})")

    return u

def rotate_molecule(u, resid=241, phi_deg=45, theta_deg=45):
    mol = u.select_atoms(f"resid {resid}")
    center = mol.center_of_geometry()

    phi = np.radians(phi_deg)
    theta = np.radians(theta_deg)

    Rz = np.array([
        [np.cos(phi), -np.sin(phi), 0],
        [np.sin(phi),  np.cos(phi), 0],
        [0, 0, 1]
    ])

    Ry = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

    R = Ry @ Rz

    print("📍 旋转操作中被修改坐标的原子编号：")
    for atom in mol:
        print(f"  Atom ID: {atom.id}")
        shifted = atom.position - center
        atom.position = R @ shifted + center

    return u

def main():
    input_file = "CH3CN.gro"
    no_h_file = "CH3CN_no_H.gro"
    output_file = "CH3CN_no_H_modified.gro"

    print("🔹 移除氢原子...")
    remove_hydrogens(input_file, no_h_file)

    print("🔹 加载无氢结构...")
    u = mda.Universe(no_h_file)

    # 输出 polar_axis
    mol = u.select_atoms("resid 241")
    c_atoms = mol.select_atoms("name C")
    n_atoms = mol.select_atoms("name N")
    if len(c_atoms) >= 2 and len(n_atoms) >= 1:
        c2 = c_atoms[1]
        n = n_atoms[0]
        cn_vector = n.position - c2.position
        cn_unit = cn_vector / np.linalg.norm(cn_vector)
        print(f"📌 polar_axis (C–N 键方向单位向量): {cn_unit}")
    else:
        print("⚠️ 找不到 C–N，无法输出 polar_axis")

    print("🔹 C–C 键伸长 1 Å...")
    u = stretch_ccn_group(u, resid=241, stretch_amount=1.0)
    
    print("🔹 旋转分子 (phi=45°, theta=45°)...")
    u = rotate_molecule(u, resid=241, phi_deg=45, theta_deg=45)

    print("🔹 保存修改后的结构...")
    with mda.Writer(output_file, n_atoms=len(u.atoms)) as w:
        w.write(u.atoms)

    print("✅ 完成：文件保存为", output_file)

if __name__ == "__main__":
    main()
