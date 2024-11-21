import torch
import cvxpy as cp
import numpy as np

def dsmetric(A1, V1, A2, V2, lambda_features=1, use_squared_dists=False, return_S=False):
    """
    Compute the doubly stochastic metric between two vertex-featured graphs.
    
    Args:
        A1 (torch.Tensor): Adjacency matrix of graph 1 (n x n).
        V1 (torch.Tensor): Vertex-feature matrix of graph 1 (n x d).
        A2 (torch.Tensor): Adjacency matrix of graph 2 (n x n).
        V2 (torch.Tensor): Vertex-feature matrix of graph 2 (n x d).
        lambda_features (float): Weight of the vertex-feature term in the objective.
        use_squared_dists (bool): Whether to use squared distances in the feature term.
        return_S (bool): Whether to return the optimal doubly-stochastic matrix S in addition to the DS metric.
    
    Returns:
        float: Optimal value of the objective.
        torch.Tensor: Optimized doubly stochastic matrix (n x n). (optional)
    """
    [n, d] = V1.shape
    [n2, d2] = V2.shape
    
    assert n == n2, "Graph sizes (number of nodes) must match."
    assert d == d2, "Feature dimensions must match."

    # Convert PyTorch tensors to NumPy arrays for CVXPY
    A1_np = A1.detach().numpy()
    V1_np = V1.detach().numpy()
    A2_np = A2.detach().numpy()
    V2_np = V2.detach().numpy()

    # Compute pairwise distances
    diff = V1_np[:, None, :] - V2_np[None, :, :]  # Shape (n, n, d)
    dists = np.linalg.norm(diff, axis=2)  # Pairwise L2 norms, shape (n, n)

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
    if use_squared_dists:
        feature_term = cp.sqrt(cp.sum(cp.multiply(S, dists**2)))
    else:
        feature_term = cp.sum(cp.multiply(S, dists))

    # Define and solve the optimization problem
    objective = cp.Minimize(structure_term + lambda_features * feature_term)
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Ensure solver succeeded
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"Solver failed with status: {problem.status}")

    # Retrieve the optimal value and convert back to PyTorch tensors if needed
    optimal_value = problem.value
    S_optimized = torch.tensor(S.value, dtype=torch.float32).to(A1.device)  # Convert S to PyTorch tensor

    if return_S:
        return optimal_value, S_optimized
    else:
        return optimal_value
