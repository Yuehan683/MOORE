import torch
import torch.nn as nn

class OrthogonalLayer1D_SVD(nn.Module):
    """
    SVD-based orthogonalization.

    Input:  x [n_models, n_samples, dim]
    Output: y [n_models, n_samples, dim]

    For each sample b:
      A_b in R^{n_models x dim}
      SVD: A_b = U_b S_b Vh_b
      Rows of Vh_b are orthonormal vectors in R^{dim}.
    """
    def __init__(self, full_matrices: bool = False):
        super().__init__()
        self.full_matrices = full_matrices

    def forward(self, x):
        # x: [k, B, d] -> A: [B, k, d]
        A = torch.transpose(x, 0, 1)
        _, _, Vh = torch.linalg.svd(A, full_matrices=self.full_matrices)  # Vh: [B, k, d]
        return torch.transpose(Vh, 0, 1)  # -> [k, B, d]