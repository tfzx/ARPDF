# weights.py
from itertools import combinations_with_replacement

# 每种原子的单体权重
ATOM_WEIGHTS = {
    "C": 1.0,
    "Cl": 2.0,
    "N": 1.5,
    # 可以继续添加其他原子
}

def generate_pair_weights():
    """生成二元组组合权重字典，(A, B): weight_A * weight_B"""
    weights = {}
    atoms = list(ATOM_WEIGHTS.keys())
    for a1, a2 in combinations_with_replacement(atoms, 2):
        w = ATOM_WEIGHTS[a1] * ATOM_WEIGHTS[a2]
        weights[(a1, a2)] = w
        if a1 != a2:
            weights[(a2, a1)] = w  # 保证 (B, A) 也能匹配
    return weights
