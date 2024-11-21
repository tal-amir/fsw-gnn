import torch
import cvxpy as cp

def dsmetric(A1, V1, A2, V2, l = 1):
    [n, d] = V1.shape
    [n2, d2] = V2.shape
    
    assert n == n2
    assert d == d2

    # Compute pairwise differences
    diff = V1[:, None, :] - V2[None, :, :]  # Shape (n, n2, d)

    # Compute L2 norms of the differences
    norms = torch.norm(diff, dim=2)  # Shape (n1, n2)

    S = cp.Variable((n, n))

    # Constraints for doubly stochastic matrix
    constraints = [
        S >= 0,  # Non-negativity
        cp.sum(S, axis=1) == 1,  # Rows sum to 1
        cp.sum(S, axis=0) == 1   # Columns sum to 1
        ]

    structure_term = cp.norm(A1@S - S@A2)
    feature_term = torch.sum(S * norms)
    
    objective = cp.Minimize(l*structure_term + feature_term)
    problem = cp.Problem(objective, constraints)

    problem.solve()
    optimal_value = problem.value

    return optimal_value

