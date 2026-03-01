"""
Generator: Geracao de texto com tomografia epistemica.

Gera tokens auto-regressivamente, coletando EpistemicTomography
por token. Suporta top-k, top-p e temperatura adaptativa.

Cada token gerado produz:
- q1, q2, confidence, drm_coords, phi, vi_direction, etc.
"""

import torch
import torch.nn.functional as F
from typing import Optional, List
from dataclasses import dataclass, field

from aletheion_v2.core.model import AletheionV2Model
from aletheion_v2.core.output import EpistemicTomography
from aletheion_v2.mpc.navigator import ManifoldNavigator
from aletheion_v2.mpc.transition_model import TransitionModel


@dataclass
class GenerationResult:
    """Resultado completo de geracao."""
    token_ids: List[int]                    # Tokens gerados
    tomography_per_token: List[dict]        # Tomografia por token
    total_tokens: int
    avg_confidence: float
    avg_phi: float
    navigation_plans: List[dict] = field(default_factory=list)


class Generator:
    """Gerador auto-regressivo com tomografia.

    Args:
        model: AletheionV2Model treinado
        device: Device
        max_new_tokens: Maximo de tokens a gerar
        temperature: Temperatura base
        top_k: Top-k filtering (0 = desabilitado)
        top_p: Top-p (nucleus) filtering (1.0 = desabilitado)
        use_mpc: Se True, usa MPC navigator
    """

    def __init__(
        self,
        model: AletheionV2Model,
        device: str = "cpu",
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        use_mpc: bool = False,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

        # MPC navigator (opcional)
        self.navigator = None
        if use_mpc:
            transition = TransitionModel()
            self.navigator = ManifoldNavigator(transition)

    def _top_k_top_p_filter(
        self, logits: torch.Tensor
    ) -> torch.Tensor:
        """Aplica top-k e top-p filtering.

        Args:
            logits: [B, V]

        Returns:
            filtered_logits: [B, V]
        """
        # Top-k
        if self.top_k > 0:
            k = min(self.top_k, logits.shape[-1])
            vals, _ = logits.topk(k)
            threshold = vals[:, -1].unsqueeze(-1)
            logits = logits.masked_fill(logits < threshold, float("-inf"))

        # Top-p (nucleus)
        if self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                F.softmax(sorted_logits, dim=-1), dim=-1
            )
            # Remove tokens com cumulative > top_p
            sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= self.top_p
            sorted_logits[sorted_mask] = float("-inf")
            # Reordena
            logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

        return logits

    def _extract_token_tomography(
        self, tomo: EpistemicTomography, pos: int
    ) -> dict:
        """Extrai tomografia de um token especifico.

        Args:
            tomo: EpistemicTomography completa
            pos: Posicao do token (ultimo = -1)

        Returns:
            dict com valores escalares
        """
        return {
            "q1": tomo.q1[0, pos, 0].item(),
            "q2": tomo.q2[0, pos, 0].item(),
            "confidence": tomo.confidence[0, pos, 0].item(),
            "drm_coords": tomo.drm_coords[0, pos].tolist(),
            "directional_dim": tomo.directional_dim[0, pos, 0].item(),
            "metric_distance": tomo.metric_distance[0, pos, 0].item(),
            "phi_components": tomo.phi_components[0, pos].tolist(),
            "phi_total": tomo.phi_total[0, pos, 0].item(),
            "vi_direction": tomo.vi_direction[0, pos].tolist(),
            "vi_severity": tomo.vi_severity[0, pos, 0].item(),
            "temperature": tomo.temperature[0, pos, 0].item(),
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: Optional[int] = None,
    ) -> GenerationResult:
        """Gera tokens com tomografia.

        Args:
            input_ids: [1, T] prompt token ids
            max_new_tokens: Override de max tokens

        Returns:
            GenerationResult completo
        """
        max_tokens = max_new_tokens or self.max_new_tokens
        input_ids = input_ids.to(self.device)

        generated_ids = []
        tomography_list = []
        nav_plans = []
        total_confidence = 0.0
        total_phi = 0.0

        for step in range(max_tokens):
            # Forward
            output = self.model(input_ids, return_tomography=True)

            # Logits do ultimo token
            logits = output.logits[:, -1, :]  # [1, V]

            # Temperatura adaptativa
            temp = self.temperature
            if output.tomography is not None:
                ep_temp = output.tomography.temperature[0, -1, 0].item()
                temp = temp * max(ep_temp, 0.1)

            # Filtering + sampling
            logits = logits / max(temp, 1e-8)
            logits = self._top_k_top_p_filter(logits)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [1, 1]

            # Coleta tomografia
            if output.tomography is not None:
                tomo_dict = self._extract_token_tomography(
                    output.tomography, -1
                )
                tomography_list.append(tomo_dict)
                total_confidence += tomo_dict["confidence"]
                total_phi += tomo_dict["phi_total"]

                # MPC navigation
                if self.navigator is not None:
                    phi_comp = output.tomography.phi_components[0, -1]
                    plan = self.navigator.plan(phi_comp)
                    nav_plans.append({
                        "step": step,
                        "actions": plan.actions,
                        "predicted_phi": plan.predicted_phi,
                        "mode": plan.mode,
                    })

            # Append token
            token_id = next_token[0, 0].item()
            generated_ids.append(token_id)

            # Atualiza input (append)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Trunca se exceder max_seq_len
            max_seq = self.model.config.max_seq_len
            if input_ids.shape[1] > max_seq:
                input_ids = input_ids[:, -max_seq:]

        n = max(len(tomography_list), 1)
        return GenerationResult(
            token_ids=generated_ids,
            tomography_per_token=tomography_list,
            total_tokens=len(generated_ids),
            avg_confidence=total_confidence / n,
            avg_phi=total_phi / n,
            navigation_plans=nav_plans,
        )
