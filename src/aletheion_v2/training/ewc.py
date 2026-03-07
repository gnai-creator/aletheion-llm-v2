"""
Elastic Weight Consolidation (EWC) para treinamento continuo.

Baseado em Kirkpatrick et al. (2017) - "Overcoming catastrophic forgetting
in neural networks" + extensoes do survey Wang et al. (2302.00487).

Computa a Fisher Information Matrix (FIM) diagonal apos cada fase de
treinamento e adiciona penalidade quadratica que impede parametros
importantes de mudar muito.

L_ewc = (lambda_ewc / 2) * sum_i F_i * (theta_i - theta_star_i)^2

Onde:
- F_i: Fisher diagonal para parametro i (importancia)
- theta_star_i: parametro otimo da fase anterior
- theta_i: parametro atual
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, List
from contextlib import nullcontext


class EWCRegularizer(nn.Module):
    """Regularizador EWC com Fisher Information Matrix diagonal.

    Suporta multiplas fases (online EWC) com decay exponencial
    das Fisher de fases anteriores.

    Args:
        lambda_ewc: peso da regularizacao EWC
        online: se True, acumula Fisher de multiplas fases
        gamma: fator de decay para fases anteriores (online EWC)
    """

    def __init__(
        self,
        lambda_ewc: float = 100.0,
        online: bool = True,
        gamma: float = 0.9,
    ):
        super().__init__()
        self.lambda_ewc = lambda_ewc
        self.online = online
        self.gamma = gamma

        # Armazena Fisher e parametros de referencia
        self._fisher: Dict[str, torch.Tensor] = {}
        self._reference_params: Dict[str, torch.Tensor] = {}
        self._num_phases = 0

    @property
    def has_fisher(self) -> bool:
        """Retorna True se Fisher ja foi computada."""
        return len(self._fisher) > 0

    @torch.no_grad()
    def compute_fisher(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        num_samples: int = 256,
        device: str = "cpu",
        amp_context: Optional[object] = None,
    ) -> None:
        """Computa Fisher Information Matrix diagonal via amostras.

        Usa o gradiente da log-verossimilhanca empirica para estimar
        a diagonal da FIM.

        Args:
            model: modelo treinado
            dataloader: dados da fase atual
            num_samples: numero de amostras para estimar Fisher
            device: device de computacao
            amp_context: contexto de mixed precision (opcional)
        """
        model.eval()
        fisher_accum = {}

        # Inicializa acumuladores
        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher_accum[name] = torch.zeros_like(param)

        n_samples = 0
        ctx = amp_context if amp_context is not None else nullcontext()

        for batch in dataloader:
            if n_samples >= num_samples:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            model.zero_grad()

            with torch.enable_grad():
                with ctx:
                    output = model(input_ids, return_tomography=False)
                    # Log-verossimilhanca: negativo da CE
                    B, T, V = output.logits.shape
                    ce = nn.functional.cross_entropy(
                        output.logits.view(-1, V),
                        labels.view(-1),
                        reduction="mean",
                    )

                ce.backward()

                # Acumula quadrado dos gradientes (Fisher diagonal)
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        fisher_accum[name] += param.grad.data ** 2

            n_samples += input_ids.shape[0]

        # Normaliza pelo numero de amostras
        n_samples = max(n_samples, 1)
        for name in fisher_accum:
            fisher_accum[name] /= n_samples

        # Online EWC: decay Fisher anterior + adiciona nova
        if self.online and self._fisher:
            for name in fisher_accum:
                if name in self._fisher:
                    self._fisher[name] = (
                        self.gamma * self._fisher[name]
                        + fisher_accum[name]
                    )
                else:
                    self._fisher[name] = fisher_accum[name]
        else:
            self._fisher = fisher_accum

        # Salva parametros de referencia (copia)
        self._reference_params = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        self._num_phases += 1
        model.train()

    def forward(self, model: nn.Module) -> torch.Tensor:
        """Computa loss EWC.

        L_ewc = (lambda / 2) * sum_i F_i * (theta_i - theta*_i)^2

        Args:
            model: modelo atual

        Returns:
            loss EWC escalar
        """
        if not self.has_fisher:
            return torch.tensor(0.0)

        loss = torch.tensor(0.0, device=next(model.parameters()).device)

        for name, param in model.named_parameters():
            if name in self._fisher and name in self._reference_params:
                fisher = self._fisher[name].to(param.device)
                ref = self._reference_params[name].to(param.device)
                loss = loss + (fisher * (param - ref) ** 2).sum()

        return (self.lambda_ewc / 2.0) * loss

    def state_dict_ewc(self) -> Dict:
        """Retorna estado do EWC para checkpoint."""
        return {
            "fisher": {k: v.cpu() for k, v in self._fisher.items()},
            "reference_params": {
                k: v.cpu() for k, v in self._reference_params.items()
            },
            "num_phases": self._num_phases,
        }

    def load_state_dict_ewc(self, state: Dict) -> None:
        """Carrega estado do EWC de checkpoint."""
        self._fisher = state.get("fisher", {})
        self._reference_params = state.get("reference_params", {})
        self._num_phases = state.get("num_phases", 0)

    def get_importance_stats(self) -> Dict[str, float]:
        """Retorna estatisticas de importancia dos parametros.

        Util para debug e monitoramento.
        """
        if not self.has_fisher:
            return {}

        stats = {}
        total_importance = 0.0
        total_params = 0

        for name, fisher in self._fisher.items():
            mean_f = fisher.mean().item()
            max_f = fisher.max().item()
            stats[f"fisher/{name}/mean"] = mean_f
            stats[f"fisher/{name}/max"] = max_f
            total_importance += fisher.sum().item()
            total_params += fisher.numel()

        stats["fisher/total_importance"] = total_importance
        stats["fisher/avg_importance"] = total_importance / max(total_params, 1)
        stats["fisher/num_phases"] = self._num_phases
        return stats
