"""
Metric Tensor: G = L @ L^T (Cholesky SPD).

Tensor metrico aprendivel que garante positividade definida
durante todo o treinamento via decomposicao de Cholesky.

Formula:
    L = lower_triangular(parametros) + eps * I
    G = L @ L^T  (simetrica, positiva definida)

Propriedades:
    - SPD garantida (nao precisa de clipping)
    - Diferenciavel (backprop funciona)
    - Condition number controlavel via regularizacao
"""

import torch
import torch.nn as nn
import math


class LearnableMetricTensor(nn.Module):
    """Tensor metrico SPD aprendivel via Cholesky.

    Garante G = L @ L^T + eps * I (sempre SPD).
    Permite distancias de Mahalanobis diferenciaveisno manifold.

    Args:
        dim: Dimensao do manifold (5 para DRM epistemico)
        eps: Regularizacao diagonal minima para estabilidade
    """

    def __init__(self, dim: int = 5, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps

        # Parametros da triangular inferior L
        # Numero de elementos: dim * (dim + 1) / 2
        n_params = dim * (dim + 1) // 2
        self.L_params = nn.Parameter(torch.zeros(n_params))

        # Inicializa como identidade (L = I -> G = I)
        self._init_as_identity()

    def _init_as_identity(self) -> None:
        """Inicializa L para que G = I (identidade)."""
        with torch.no_grad():
            self.L_params.zero_()
            # Diagonal de L = 1.0 -> G = I
            idx = 0
            for i in range(self.dim):
                # Pula os off-diagonal
                idx += i
                self.L_params[idx] = 1.0
                idx += 1

    def _build_L(self) -> torch.Tensor:
        """Constroi matriz triangular inferior L a partir dos parametros.

        Returns:
            L: [dim, dim] triangular inferior
        """
        L = torch.zeros(self.dim, self.dim, device=self.L_params.device,
                         dtype=self.L_params.dtype)
        idx = 0
        for i in range(self.dim):
            for j in range(i + 1):
                L[i, j] = self.L_params[idx]
                idx += 1

        # Garante diagonal positiva via softplus
        diag_mask = torch.eye(self.dim, device=L.device, dtype=L.dtype).bool()
        L = torch.where(diag_mask, F.softplus(L), L)
        return L

    def forward(self) -> torch.Tensor:
        """Computa tensor metrico G = L @ L^T + eps * I.

        Returns:
            G: [dim, dim] simetrica positiva definida
        """
        L = self._build_L()
        G = L @ L.t() + self.eps * torch.eye(
            self.dim, device=L.device, dtype=L.dtype
        )
        return G

    def condition_number(self) -> torch.Tensor:
        """Computa condition number de G (para regularizacao).

        Returns:
            kappa: escalar (ratio max/min eigenvalue)
        """
        G = self.forward()
        eigenvalues = torch.linalg.eigvalsh(G)
        return eigenvalues[-1] / (eigenvalues[0] + 1e-10)

    def log_det(self) -> torch.Tensor:
        """Log-determinante de G (util para probabilidades).

        Returns:
            log_det: escalar
        """
        G = self.forward()
        return torch.logdet(G)

    def inverse(self) -> torch.Tensor:
        """Inversa de G via Cholesky solve (numericamente estavel).

        Returns:
            G_inv: [dim, dim]
        """
        G = self.forward()
        L_chol = torch.linalg.cholesky(G)
        I = torch.eye(self.dim, device=G.device, dtype=G.dtype)
        return torch.cholesky_solve(I, L_chol)


# Necessario importar F para softplus
import torch.nn.functional as F
