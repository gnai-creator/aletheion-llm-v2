"""
Metric Tensor: G constante e G(x) dependente de posicao.

LearnableMetricTensor: G = L @ L^T (constante, espaco plano).
MetricNet: G(x) via MLP + Cholesky (campo tensorial, curvatura real).

Quando G varia com posicao, os simbolos de Christoffel sao nao-zero
e o tensor de Riemann e nao-trivial -- curvatura Riemanniana real.

Propriedades:
    - SPD garantida em ambos os casos (Cholesky)
    - MetricNet inicializa proximo de I (zero init no ultimo layer)
    - Tanh para suavidade do campo (C1 minimo)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

# Pontos e pesos de Gauss-Legendre para n=5 (pre-computados)
_GL5_POINTS = [
    0.04691007703066800,
    0.23076534494715845,
    0.50000000000000000,
    0.76923465505284155,
    0.95308992296933200,
]
_GL5_WEIGHTS = [
    0.11846344252809454,
    0.23931433524968324,
    0.28444444444444444,
    0.23931433524968324,
    0.11846344252809454,
]


class LearnableMetricTensor(nn.Module):
    """Tensor metrico SPD aprendivel via Cholesky (constante).

    Garante G = L @ L^T + eps * I (sempre SPD).
    Define espaco plano com metrica obliqua.

    Args:
        dim: Dimensao do manifold (5 para DRM epistemico)
        eps: Regularizacao diagonal minima para estabilidade
    """

    def __init__(self, dim: int = 5, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps

        n_params = dim * (dim + 1) // 2
        self.L_params = nn.Parameter(torch.zeros(n_params))
        self._init_as_identity()

    def _init_as_identity(self) -> None:
        """Inicializa L para que G = I (identidade)."""
        with torch.no_grad():
            self.L_params.zero_()
            idx = 0
            for i in range(self.dim):
                idx += i
                self.L_params[idx] = 1.0
                idx += 1

    def _build_L(self) -> torch.Tensor:
        """Constroi matriz triangular inferior L.

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
        """Computa condition number de G.

        Returns:
            kappa: escalar (ratio max/min eigenvalue)
        """
        G = self.forward()
        eigenvalues = torch.linalg.eigvalsh(G)
        return eigenvalues[-1] / (eigenvalues[0] + 1e-10)

    def log_det(self) -> torch.Tensor:
        """Log-determinante de G.

        Returns:
            log_det: escalar
        """
        G = self.forward()
        return torch.logdet(G)

    def inverse(self) -> torch.Tensor:
        """Inversa de G via Cholesky solve.

        Returns:
            G_inv: [dim, dim]
        """
        G = self.forward()
        L_chol = torch.linalg.cholesky(G)
        I = torch.eye(self.dim, device=G.device, dtype=G.dtype)
        return torch.cholesky_solve(I, L_chol)


class MetricNet(nn.Module):
    """Campo tensorial G(x): coords -> tensor metrico SPD por ponto.

    G(x) varia com a posicao no manifold, produzindo curvatura
    Riemanniana real (Christoffel nao-zero, Riemann nao-trivial).

    Opcionalmente aceita gravity_field (campo de feedback acumulado)
    como input adicional. Com gravity_field=None ou zeros, o
    comportamento e identico ao real_geodesic.

    Usa MLP com Tanh (suavidade C1) e Cholesky para garantia SPD.
    Inicializa proximo de identidade (zero init no ultimo layer).

    Args:
        dim: Dimensao do manifold (5)
        hidden_dim: Dimensao oculta da MLP
        eps: Regularizacao diagonal minima
        n_quad: Pontos de quadratura Gauss-Legendre para integral de linha
        gravity_dim: Dimensao do campo gravitacional (0 = desabilitado)
    """

    def __init__(
        self,
        dim: int = 5,
        hidden_dim: int = 32,
        eps: float = 1e-6,
        n_quad: int = 5,
        gravity_dim: int = 0,
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.n_quad = n_quad
        self.n_chol = dim * (dim + 1) // 2  # 15 para dim=5
        self.gravity_dim = gravity_dim

        input_dim = dim + gravity_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.n_chol),
        )

        # Zero init no ultimo layer -> G(x) ~ I no inicio
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        # Indices para construir L triangular inferior (pre-computados)
        tril_idx = torch.tril_indices(dim, dim)
        self.register_buffer("tril_row", tril_idx[0])
        self.register_buffer("tril_col", tril_idx[1])

        diag_idx = torch.arange(dim)
        self.register_buffer("diag_idx", diag_idx)

        # Pontos e pesos de Gauss-Legendre
        if n_quad == 5:
            gl_t = torch.tensor(_GL5_POINTS, dtype=torch.float32)
            gl_w = torch.tensor(_GL5_WEIGHTS, dtype=torch.float32)
        else:
            gl_t, gl_w = self._compute_gauss_legendre(n_quad)
        self.register_buffer("gl_points", gl_t)
        self.register_buffer("gl_weights", gl_w)

    @staticmethod
    def _compute_gauss_legendre(n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computa pontos e pesos de Gauss-Legendre em [0, 1].

        Args:
            n: Numero de pontos de quadratura

        Returns:
            (points, weights) em [0, 1]
        """
        import numpy as np
        points, weights = np.polynomial.legendre.leggauss(n)
        # Transforma de [-1, 1] para [0, 1]
        points_01 = (points + 1.0) / 2.0
        weights_01 = weights / 2.0
        return (
            torch.tensor(points_01, dtype=torch.float32),
            torch.tensor(weights_01, dtype=torch.float32),
        )

    def _build_G_batched(self, raw: torch.Tensor) -> torch.Tensor:
        """Constroi matrizes G SPD a partir de parametros Cholesky.

        Args:
            raw: [..., n_chol] parametros Cholesky (qualquer batch shape)

        Returns:
            G: [..., dim, dim] matrizes SPD
        """
        batch_shape = raw.shape[:-1]
        device = raw.device
        dtype = raw.dtype

        # Constroi L triangular inferior
        L = torch.zeros(*batch_shape, self.dim, self.dim,
                         device=device, dtype=dtype)
        L[..., self.tril_row, self.tril_col] = raw

        # Diagonal positiva via softplus + offset minimo
        L[..., self.diag_idx, self.diag_idx] = (
            F.softplus(L[..., self.diag_idx, self.diag_idx]) + 1e-3
        )

        # G = L @ L^T (SPD garantido)
        G = torch.matmul(L, L.transpose(-1, -2))
        return G

    def forward(
        self,
        coords: torch.Tensor,
        gravity_field: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computa G(x) para cada ponto.

        Args:
            coords: [..., dim] coordenadas no manifold
            gravity_field: [..., gravity_dim] campo gravitacional (opcional).
                Com None ou zeros, identico ao real_geodesic.

        Returns:
            G: [..., dim, dim] tensor metrico SPD por ponto
        """
        if self.gravity_dim > 0:
            if gravity_field is None:
                gravity_field = torch.zeros(
                    *coords.shape[:-1], self.gravity_dim,
                    device=coords.device, dtype=coords.dtype,
                )
            net_input = torch.cat([coords, gravity_field], dim=-1)
        else:
            net_input = coords

        raw = self.net(net_input)  # [..., n_chol]

        # NaN guard
        if torch.isnan(raw).any():
            batch_shape = coords.shape[:-1]
            return torch.eye(
                self.dim, device=coords.device, dtype=coords.dtype
            ).expand(*batch_shape, -1, -1).clone()

        return self._build_G_batched(raw)

    def line_integral_distance(
        self,
        p: torch.Tensor,
        q: torch.Tensor,
        gravity_field: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Distancia via integral de linha com quadratura Gauss-Legendre.

        Computa o comprimento da linha reta p->q sob metrica G(x):
            d(p,q) = integral_0^1 sqrt(delta^T G(gamma(t)) delta) dt

        onde gamma(t) = p + t*(q - p) (caminho reto).

        Args:
            p: [B, T, dim] pontos de partida
            q: [dim] ou [B, T, dim] pontos de chegada
            gravity_field: [B, T, gravity_dim] campo gravitacional (opcional)

        Returns:
            distance: [B, T, 1] comprimento da integral de linha
        """
        if q.dim() == 1:
            q = q.unsqueeze(0).unsqueeze(0).expand_as(p)

        delta = q - p  # [B, T, D]
        total = torch.zeros(
            p.shape[0], p.shape[1], 1,
            device=p.device, dtype=p.dtype,
        )

        for i in range(self.n_quad):
            t = self.gl_points[i]
            w = self.gl_weights[i]

            x_t = p + t * delta  # [B, T, D]
            G_t = self.forward(x_t, gravity_field=gravity_field)  # [B, T, D, D]

            Gd = torch.matmul(delta.unsqueeze(-2), G_t).squeeze(-2)  # [B, T, D]
            integrand = (Gd * delta).sum(dim=-1, keepdim=True)  # [B, T, 1]

            total = total + w * torch.sqrt(integrand.clamp(min=1e-8))

        return total

    def forward_constant(self) -> torch.Tensor:
        """Retorna G avaliado em x=0.5 (centro do manifold), gravity=0.

        Util para backward compatibility e diagnostico.

        Returns:
            G: [dim, dim]
        """
        center = torch.full(
            (self.dim,), 0.5,
            device=self.net[0].weight.device,
            dtype=self.net[0].weight.dtype,
        )
        return self.forward(center, gravity_field=None)
