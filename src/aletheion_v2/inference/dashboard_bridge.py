"""
Dashboard Bridge: Converte EpistemicTomography para formato ATIC dashboard.

Produz JSON compativel com os endpoints do dashboard ATIC:
- GET /v1/dashboard/drm -> Manifold3DState
- GET /v1/dashboard/vi -> VIState
- GET /v1/dashboard/navigator -> NavigationState
"""

import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

from aletheion_v2.core.output import EpistemicTomography
from aletheion_v2.inference.generator import GenerationResult


@dataclass
class Manifold3DState:
    """Estado 3D do manifold para visualizacao."""
    points: List[List[float]]     # Coordenadas 5D de cada token
    confidence: List[float]       # Confianca por token
    distance: List[float]         # Distancia ao truth por token
    dim_D: List[float]            # Dimensionalidade direcional
    anchors: List[List[float]]    # Pontos ancora


@dataclass
class VIState:
    """Estado do VI para dashboard."""
    phi_total: float
    phi_components: List[float]   # [dim, disp, ent, conf]
    severity: float
    direction: List[float]        # [5] direcao de correcao
    mode: str                     # "healthy" | "warning" | "critical"


@dataclass
class DashboardSnapshot:
    """Snapshot completo para dashboard ATIC."""
    manifold: Manifold3DState
    vi: VIState
    epistemic: Dict[str, float]   # q1, q2, confidence medios
    generation: Dict[str, Any]    # Metricas de geracao


class DashboardBridge:
    """Converte outputs do AletheionV2 para formato ATIC dashboard.

    Uso:
        bridge = DashboardBridge()
        snapshot = bridge.from_generation_result(result)
        json_str = bridge.to_json(snapshot)
    """

    # Anchors fixos (mesmos do ManifoldEmbedding)
    ANCHORS = [
        [0.1, 0.1, 0.5, 0.9, 0.9],  # truth
        [0.3, 0.9, 0.5, 0.5, 0.2],  # ignorance
        [0.9, 0.3, 0.5, 0.5, 0.3],  # noise
        [0.5, 0.5, 0.9, 0.5, 0.5],  # complex
        [0.3, 0.3, 0.5, 0.1, 0.4],  # stale
        [0.2, 0.2, 0.3, 0.8, 0.8],  # ideal
    ]

    def from_generation_result(
        self, result: GenerationResult
    ) -> DashboardSnapshot:
        """Converte GenerationResult para DashboardSnapshot.

        Args:
            result: GenerationResult do Generator

        Returns:
            DashboardSnapshot compativel com ATIC
        """
        tomo_list = result.tomography_per_token

        if not tomo_list:
            return self._empty_snapshot()

        # Manifold state
        points = [t["drm_coords"] for t in tomo_list]
        confidences = [t["confidence"] for t in tomo_list]
        distances = [t["metric_distance"] for t in tomo_list]
        dim_ds = [t["directional_dim"] for t in tomo_list]

        manifold = Manifold3DState(
            points=points,
            confidence=confidences,
            distance=distances,
            dim_D=dim_ds,
            anchors=self.ANCHORS,
        )

        # VI state
        last = tomo_list[-1]
        phi_total = last["phi_total"]
        mode = "healthy" if phi_total > 0.5 else ("warning" if phi_total > 0.3 else "critical")

        vi = VIState(
            phi_total=phi_total,
            phi_components=last["phi_components"],
            severity=last["vi_severity"],
            direction=last["vi_direction"],
            mode=mode,
        )

        # Epistemic medias
        n = len(tomo_list)
        epistemic = {
            "avg_q1": sum(t["q1"] for t in tomo_list) / n,
            "avg_q2": sum(t["q2"] for t in tomo_list) / n,
            "avg_confidence": result.avg_confidence,
            "avg_phi": result.avg_phi,
        }

        # Generation info
        generation = {
            "total_tokens": result.total_tokens,
            "navigation_plans": len(result.navigation_plans),
        }

        return DashboardSnapshot(
            manifold=manifold,
            vi=vi,
            epistemic=epistemic,
            generation=generation,
        )

    def to_json(self, snapshot: DashboardSnapshot) -> str:
        """Serializa DashboardSnapshot para JSON.

        Args:
            snapshot: DashboardSnapshot

        Returns:
            JSON string
        """
        data = {
            "manifold": asdict(snapshot.manifold),
            "vi": asdict(snapshot.vi),
            "epistemic": snapshot.epistemic,
            "generation": snapshot.generation,
        }
        return json.dumps(data, indent=2)

    def to_atic_endpoints(
        self, snapshot: DashboardSnapshot
    ) -> Dict[str, Dict]:
        """Retorna dicts compativiveis com endpoints ATIC.

        Returns:
            Dict com chaves 'drm', 'vi', 'epistemic'
        """
        return {
            "drm": asdict(snapshot.manifold),
            "vi": asdict(snapshot.vi),
            "epistemic": snapshot.epistemic,
        }

    def _empty_snapshot(self) -> DashboardSnapshot:
        """Retorna snapshot vazio."""
        return DashboardSnapshot(
            manifold=Manifold3DState(
                points=[], confidence=[], distance=[],
                dim_D=[], anchors=self.ANCHORS,
            ),
            vi=VIState(
                phi_total=0.0, phi_components=[0, 0, 0, 0],
                severity=0.0, direction=[0, 0, 0, 0, 0], mode="critical",
            ),
            epistemic={"avg_q1": 0, "avg_q2": 0, "avg_confidence": 0, "avg_phi": 0},
            generation={"total_tokens": 0, "navigation_plans": 0},
        )
