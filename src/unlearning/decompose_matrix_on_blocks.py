import torch
from typing import List, Literal, Optional


def build_A_blocks(
    m: int,
    r_list: List[int],
    blocks_mode: Literal["qr", "perm_identity", "identity"] = "qr",
    device: str = "cpu",
) -> List[torch.Tensor]:
    """
    Build mutually orthogonal subspaces A_1,...,A_k with orthonormal columns.

    Args:
        m: ambient dimension (number of rows)
        r_list: list of block sizes r_i; R = sum(r_list) must satisfy R <= m
        mode:
            - "qr": random orthonormal basis via QR of Gaussian
            - "perm_identity": random permutation of identity columns (coordinate subspaces)
            - "identity": consecutive identity columns (coordinate subspaces)
        device: torch device
        seed: optional RNG seed for reproducibility (used for "qr" and "perm_identity")

    Returns:
        List of tensors [A_1, ..., A_k], where A_i ∈ R^{m × r_i}, with
        A_i^T A_i = I_{r_i} and A_i^T A_j = 0 for i ≠ j.
    """
    R = sum(r_list)
    assert (
        R == m
    ), f"Need m >= sum(r_i) to fit all orthogonal subspaces, {r_list} != {m}"

    if blocks_mode == "qr":
        return random_orthonormal_subspaces(m, r_list, device)

    elif blocks_mode == "perm_identity":
        return random_identity_permutation(m, r_list, device)

    elif blocks_mode == "identity":
        return build_identity_A(m, r_list, device)
    else:
        raise ValueError(
            f"Unknown mode='{blocks_mode}'. Use 'qr', 'perm_identity', or 'identity'."
        )


def build_identity_A(m: int, r_list, device="cpu"):
    # Consecutive coordinate subspaces from identity
    I = torch.eye(m, device=device)  # (m, m)
    A_blocks, s = [], 0
    for r in r_list:
        A_blocks.append(I[:, s : s + r])
        s += r
    return A_blocks


def random_identity_permutation(m: int, r_list, device="cpu"):
    # Random permutation of identity columns -> permutation matrix columns

    cols = torch.randperm(m, device=device)
    I = torch.eye(m, device=device)  # (m, m)
    P = I[:, cols]  # (m, m), columns are permuted basis vectors
    A_blocks, s = [], 0
    for r in r_list:
        A_blocks.append(P[:, s : s + r])
        s += r
    return A_blocks


def random_orthonormal_subspaces(m: int, r_list, device="cpu"):
    """
    Generate A_1,...,A_k with mutually orthogonal subspaces.
    Args:
        m (int): dimension of ambient space
        r_list (list[int]): sizes of subspaces
    Returns:
        A_blocks (list[Tensor]): list of A_i, with orthogonal column spaces
    """
    # Random Gaussian, then QR for orthonormal basis
    Q, _ = torch.linalg.qr(torch.randn(m, m), mode="reduced")

    A_blocks = []
    start = 0
    for r in r_list:
        A_blocks.append(Q[:, start : start + r].to(device))
        start += r
    return A_blocks


def solve_B_blocks(A_blocks: List[torch.Tensor], W: torch.Tensor) -> List[torch.Tensor]:
    """
    Given subspace bases A_i (m x r_i) and target matrix W (m x n),
    compute coefficient blocks B_i (r_i x n).
    """
    # Basic checks
    assert len(A_blocks) > 0, "A_blocks must be non-empty"
    m = A_blocks[0].shape[0]
    assert all(
        A.shape[0] == m for A in A_blocks
    ), "All A_i must have the same row dimension m"
    assert (
        W.shape[0] == m
    ), f"W must have the same row dimension m as A_i, but {W.shape} != {m}"

    # Fast path: projection onto each orthonormal subspace
    B_blocks = [A.T @ W for A in A_blocks]  # (r_i, n)
    return B_blocks
