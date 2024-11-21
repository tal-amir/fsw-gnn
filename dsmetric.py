import torch
import cvxpy as cp
import numpy as np

def dsmetric(A1, V1, A2, V2, l=1):
    [n, d] = V1.shape
    [n2, d2] = V2.shape
    
    assert n == n2
    assert d == d2

    # Convert PyTorch tensors to NumPy arrays for CVXPY
    A1_np = A1.detach().numpy()
    V1_np = V1.detach().numpy()
    A2_np = A2.detach().numpy()
    V2_np = V2.detach().numpy()

    # Compute pairwise differences in NumPy
    diff = V1_np[:, None, :] - V2_np[None, :, :]  # Shape (n, n, d)
    norms = np.linalg.norm(diff, axis=2)  # Shape (n, n)

    # Define CVXPY variable for S
    S = cp.Variable((n, n))

    # Constraints for doubly stochastic matrix
    constraints = [
        S >= 0,  # Non-negativity
        cp.sum(S, axis=1) == 1,  # Rows sum to 1
        cp.sum(S, axis=0) == 1   # Columns sum to 1
    ]

    # Objective terms
    structure_term = cp.norm(A1_np @ S - S @ A2_np, "fro")
    feature_term = cp.sum(cp.multiply(S, norms))

    # Define and solve the optimization problem
    objective = cp.Minimize(l * structure_term + feature_term)
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Retrieve the optimal value and convert back to PyTorch tensors if needed
    optimal_value = problem.value
    S_optimized = torch.tensor(S.value, dtype=torch.float32)  # Convert S to PyTorch tensor

    return optimal_value, S_optimized
