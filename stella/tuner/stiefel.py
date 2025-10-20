import torch
from torch import Tensor


def symm(A):
    return 0.5 * (A + A.mT)


@torch.jit.script
def euclidean2riemannian(x: Tensor, grad: Tensor) -> Tensor:
    return grad - x @ (grad.mT @ x)


@torch.jit.script
def tangent_project(x: Tensor, grad: Tensor) -> Tensor:
    return grad - x @ symm(x.mT @ grad)


@torch.jit.script
def exp_map(X: Tensor, grad: Tensor) -> Tensor:
    xTgrad = X.mT @ grad
    Q, R = torch.linalg.qr(grad - X @ xTgrad)
    Z = torch.zeros_like(R)
    Id = torch.eye(Z.shape[-2], device=Z.device)[None].expand(Z.shape[0], -1, -1)
    top_row = torch.cat([xTgrad, -R.mT], dim=-1)
    bottom_row = torch.cat([R, Z], dim=-1)
    matrix = torch.cat([top_row, bottom_row], dim=-2)
    exp_mat = torch.linalg.matrix_exp(matrix)
    IZ = torch.cat([Id, Z], dim=-2)
    MN = exp_mat @ IZ
    XQ = torch.cat([X, Q], dim=-1)
    out = XQ @ MN
    return out


def polar_uf(m: Tensor) -> Tensor:
    U, _, Vt = torch.linalg.svd(m, full_matrices=False, driver='gesvda')
    uf = U @ Vt
    return uf


@torch.jit.script
def polar_retraction(X: Tensor, grad: Tensor) -> Tensor:
    return polar_uf(X + grad)
