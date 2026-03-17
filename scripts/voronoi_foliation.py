"""
Riemannian Voronoi Tessellation e deteccao de foliacao.

Pipeline:
    1. Carrega vetores epistemicos 5D extraidos
    2. Riemannian Voronoi tessellation (MetricNet G(x) ou identidade)
    3. LTSA por celula (PCA local, espectro de eigenvalues)
    4. Coerencia tangencial entre celulas vizinhas
    5. Reeb graph via level sets (com logit pre-conditioning)
    6. Persistent homology (T^2 validation: H1=Z^2, H2=Z)
    7. Null models (shuffled, uniform, backbone)
    8. Score de foliacao F

Uso:
    .venv/bin/python scripts/voronoi_foliation.py \
        --vectors eval_results/foliation/finetuned_vectors.npy \
        --checkpoint checkpoints/350m_epistemic_finetune_v2/final.pt \
        --output-dir eval_results/foliation \
        --n-seeds 30
"""

import os
import sys
import json
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Funcoes de analise separadas (foliation_analysis.py)
from foliation_analysis import (
    ltsa_per_cell,
    tangent_coherence,
    compute_reeb_graph,
    compute_persistent_homology,
    run_null_model,
    stability_test,
    compute_foliation_score,
)

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configura logging padrao."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


# Anchor seeds (mesmos de manifold_embedding.py)
ANCHOR_SEEDS: np.ndarray = np.array([
    [0.1, 0.1, 0.5, 0.9, 0.9],  # truth
    [0.3, 0.9, 0.5, 0.5, 0.2],  # ignorance
    [0.9, 0.3, 0.5, 0.5, 0.3],  # noise
    [0.5, 0.5, 0.9, 0.5, 0.5],  # complex
    [0.3, 0.3, 0.5, 0.1, 0.4],  # stale
    [0.2, 0.2, 0.3, 0.8, 0.8],  # ideal
], dtype=np.float32)


def load_metric_net(
    checkpoint_path: str, device: str = "cpu"
) -> Optional[object]:
    """Carrega MetricNet do checkpoint para computar G(x).

    Args:
        checkpoint_path: Caminho para checkpoint .pt
        device: Device alvo

    Returns:
        MetricNet ou None se falhar
    """
    try:
        import torch
        from aletheion_v2.config import AletheionV2Config
        from aletheion_v2.drm.metric_tensor import MetricNet

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = ckpt.get("config", {})
        if isinstance(config, dict):
            config = AletheionV2Config(**config)

        metric_net = MetricNet(
            dim=config.drm_dim,
            hidden_dim=config.metric_net_hidden,
            eps=config.metric_eps,
            n_quad=config.metric_net_n_quad,
            gravity_dim=config.metric_gravity_dim,
        )

        state_dict = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
        metric_state = {}
        prefix = "epistemic_head.metric_net."
        for k, v in state_dict.items():
            k_clean = k.replace("module.", "").replace("_orig_mod.", "")
            if k_clean.startswith(prefix):
                metric_state[k_clean[len(prefix):]] = v

        if metric_state:
            metric_net.load_state_dict(metric_state, strict=False)
            metric_net = metric_net.to(device).eval()
            logger.info("[METRIC] MetricNet carregada com %d params", len(metric_state))
            return metric_net

        logger.warning("[WARN] MetricNet nao encontrada no checkpoint")
        return None

    except Exception as e:
        logger.warning("[WARN] Falha ao carregar MetricNet: %s", e)
        return None


def riemannian_distance_batch(
    points_a: np.ndarray,
    points_b: np.ndarray,
    metric_net: Optional[object] = None,
    device: str = "cpu",
    batch_size: int = 4096,
) -> np.ndarray:
    """Computa distancia Riemanniana (ou euclidiana) entre pontos.

    Args:
        points_a: [N, D] pontos de partida
        points_b: [N, D] ou [D] ponto de chegada
        metric_net: MetricNet opcional
        device: Device para MetricNet
        batch_size: Tamanho do batch

    Returns:
        distances: [N] distancias
    """
    if metric_net is None:
        if points_b.ndim == 1:
            return np.linalg.norm(points_a - points_b, axis=-1)
        return cdist(points_a, points_b, metric="euclidean").squeeze()

    import torch
    a_t = torch.tensor(points_a, dtype=torch.float32, device=device)
    b_t = torch.tensor(points_b, dtype=torch.float32, device=device)
    if b_t.ndim == 1:
        b_t = b_t.unsqueeze(0).expand_as(a_t)

    results = []
    with torch.no_grad():
        for start in range(0, a_t.shape[0], batch_size):
            end = min(start + batch_size, a_t.shape[0])
            pa = a_t[start:end].unsqueeze(0)
            pb = b_t[start:end].unsqueeze(0)
            d = metric_net.line_integral_distance(pa, pb)
            results.append(d.squeeze(0).squeeze(-1).cpu().numpy())
    return np.concatenate(results, axis=0)


def riemannian_kmeans(
    vectors: np.ndarray,
    n_seeds: int = 30,
    metric_net: Optional[object] = None,
    device: str = "cpu",
    max_iter: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """K-means com distancia Riemanniana ou euclidiana.

    Inicializa com anchor seeds + K-means++ extras.

    Args:
        vectors: [N, D] pontos
        n_seeds: Numero de clusters
        metric_net: MetricNet opcional
        device: Device para MetricNet
        max_iter: Iteracoes maximas

    Returns:
        labels: [N] indices de cluster
        centers: [K, D] centroides
    """
    n, d = vectors.shape

    if n_seeds >= len(ANCHOR_SEEDS):
        n_extra = n_seeds - len(ANCHOR_SEEDS)
        km_init = KMeans(n_clusters=max(n_extra, 1), n_init=1, max_iter=10)
        km_init.fit(vectors)
        centers = np.vstack([ANCHOR_SEEDS, km_init.cluster_centers_])[:n_seeds]
    else:
        km = KMeans(n_clusters=n_seeds, n_init=3, max_iter=10)
        km.fit(vectors)
        centers = km.cluster_centers_

    centers = centers.astype(np.float32)

    for iteration in range(max_iter):
        if metric_net is not None:
            dists = np.zeros((n, n_seeds), dtype=np.float32)
            for k in range(n_seeds):
                dists[:, k] = riemannian_distance_batch(
                    vectors, centers[k], metric_net, device
                )
        else:
            dists = cdist(vectors, centers, metric="euclidean")

        labels = np.argmin(dists, axis=1)

        new_centers = np.zeros_like(centers)
        for k in range(n_seeds):
            mask = labels == k
            if mask.sum() > 0:
                new_centers[k] = vectors[mask].mean(axis=0)
            else:
                new_centers[k] = centers[k]

        shift = np.linalg.norm(new_centers - centers, axis=-1).max()
        centers = new_centers
        if shift < 1e-5:
            logger.info("[KMEANS] Convergiu iter %d (shift=%.6f)", iteration, shift)
            break

    unique, counts = np.unique(labels, return_counts=True)
    logger.info(
        "[KMEANS] %d clusters: min=%d, median=%d, max=%d",
        len(unique), counts.min(), int(np.median(counts)), counts.max(),
    )
    return labels, centers


def main() -> None:
    """Pipeline principal de Voronoi foliation."""
    parser = argparse.ArgumentParser(
        description="Riemannian Voronoi Tessellation e deteccao de foliacao"
    )
    parser.add_argument("--vectors", default="eval_results/foliation/finetuned_vectors.npy")
    parser.add_argument("--drm-coords", default=None,
                        help="Path to DRM coords .npy para correlacao folha-DRM")
    parser.add_argument("--checkpoint", default="checkpoints/350m_epistemic_finetune_v2/final.pt")
    parser.add_argument("--backbone-vectors", default=None)
    parser.add_argument("--output-dir", default="eval_results/foliation")
    parser.add_argument("--n-seeds", type=int, default=30)
    parser.add_argument("--use-metric-net", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-restarts", type=int, default=10)
    parser.add_argument("--homology-points", type=int, default=5000)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    t_start = time.time()

    logger.info("[LOAD] Carregando vetores de %s", args.vectors)
    vectors = np.load(args.vectors).astype(np.float32)
    logger.info("[LOAD] Shape: %s", vectors.shape)

    metric_net = None
    if args.use_metric_net and os.path.exists(args.checkpoint):
        metric_net = load_metric_net(args.checkpoint, args.device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: Dict = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    # 1. Voronoi Tessellation
    logger.info("=" * 60)
    logger.info("  FASE 1: Voronoi Tessellation (K=%d)", args.n_seeds)
    labels, centers = riemannian_kmeans(vectors, args.n_seeds, metric_net, args.device)
    np.save(str(out_dir / "voronoi_labels.npy"), labels)
    np.save(str(out_dir / "voronoi_centers.npy"), centers)

    # 2. LTSA
    logger.info("=" * 60)
    logger.info("  FASE 2: Local Tangent Space Analysis")
    ltsa = ltsa_per_cell(vectors, labels)
    np.save(str(out_dir / "eigenvalues.npy"), ltsa["eigenvalues"])
    np.save(str(out_dir / "eff_dims.npy"), ltsa["eff_dims"])
    results["ltsa"] = {
        "mean_eff_dim": float(ltsa["eff_dims"].mean()),
        "eff_dims_dist": np.bincount(ltsa["eff_dims"], minlength=6).tolist(),
        "cell_sizes": ltsa["cell_sizes"].tolist(),
    }

    # Filtrar centers para clusters nao-vazios (LTSA opera em unique labels)
    unique_labels = np.unique(labels)
    active_centers = centers[unique_labels] if unique_labels.max() < len(centers) else centers[:len(unique_labels)]

    # 3. Coerencia Tangencial
    logger.info("=" * 60)
    logger.info("  FASE 3: Coerencia Tangencial")
    coherence, _ = tangent_coherence(active_centers, ltsa["eigenvectors"], ltsa["eff_dims"])
    np.save(str(out_dir / "coherence_matrix.npy"), coherence)
    valid_coh = coherence[~np.isnan(coherence)]
    results["coherence"] = {
        "mean_angle": float(valid_coh.mean()) if len(valid_coh) > 0 else 90.0,
        "median_angle": float(np.median(valid_coh)) if len(valid_coh) > 0 else 90.0,
        "coherent_fraction": float((valid_coh < 30).mean()) if len(valid_coh) > 0 else 0.0,
    }

    # 4. Reeb Graphs
    logger.info("=" * 60)
    logger.info("  FASE 4: Reeb Graphs")
    reeb_conf = compute_reeb_graph(vectors, labels, func_idx=2, func_name="confidence")
    reeb_phi = compute_reeb_graph(vectors, labels, func_idx=4, func_name="phi_total")
    results["reeb"] = {
        "confidence": {k: v for k, v in reeb_conf.items() if k not in ("nodes", "edges")},
        "phi_total": {k: v for k, v in reeb_phi.items() if k not in ("nodes", "edges")},
    }
    for name, data in [("reeb_confidence", reeb_conf), ("reeb_phi", reeb_phi)]:
        with open(str(out_dir / f"{name}.json"), "w") as f:
            json.dump(data, f, indent=2, default=str)

    # 5. Persistent Homology
    logger.info("=" * 60)
    logger.info("  FASE 5: Persistent Homology")
    riemann_fn = None
    if metric_net is not None:
        def _dist_matrix(sub: np.ndarray) -> np.ndarray:
            n_s = len(sub)
            dm = np.zeros((n_s, n_s), dtype=np.float32)
            for i in range(n_s):
                dm[i] = riemannian_distance_batch(sub, sub[i], metric_net, args.device)
            dm = (dm + dm.T) / 2
            np.fill_diagonal(dm, 0.0)
            return dm
        riemann_fn = _dist_matrix

    homology = compute_persistent_homology(vectors, riemann_fn, args.homology_points)
    if homology is not None:
        results["homology"] = {k: v for k, v in homology.items() if k != "diagrams"}
        with open(str(out_dir / "homology.json"), "w") as f:
            json.dump(homology, f, indent=2, default=str)

    # 6. Null Models
    logger.info("=" * 60)
    logger.info("  FASE 6: Null Models")
    null_results = {}
    for nt in ["shuffled", "uniform"]:
        null_results[nt] = run_null_model(
            vectors, labels, centers, ltsa_per_cell, tangent_coherence, nt,
        )
    if args.backbone_vectors and os.path.exists(args.backbone_vectors):
        bb_vec = np.load(args.backbone_vectors).astype(np.float32)
        null_results["backbone"] = run_null_model(
            vectors, labels, centers, ltsa_per_cell, tangent_coherence,
            "backbone", bb_vec,
        )
    results["null_models"] = null_results

    # 7. Stability
    logger.info("=" * 60)
    logger.info("  FASE 7: Estabilidade (ARI)")
    mean_ari, ari_values = stability_test(vectors, args.n_seeds, args.n_restarts)
    results["stability"] = {
        "mean_ari": mean_ari,
        "std_ari": float(np.std(ari_values)),
        "n_restarts": args.n_restarts,
    }

    # 8. Foliation Score
    logger.info("=" * 60)
    logger.info("  FASE 8: Foliation Score")
    F = compute_foliation_score(ltsa["eff_dims"], coherence, mean_ari)
    results["foliation_score"] = F

    # 9. Correlacao folha-DRM (se drm_coords disponivel)
    if args.drm_coords and os.path.exists(args.drm_coords):
        logger.info("=" * 60)
        logger.info("  FASE 9: Correlacao Folha-DRM")
        drm = np.load(args.drm_coords).astype(np.float32)
        if drm.shape[0] == vectors.shape[0]:
            drm_axis_names = ["q1_aleat", "q2_epist", "q3_complex",
                              "q4_familiar", "q5_confid"]
            drm_per_cell = {}
            for k in range(len(centers)):
                mask = labels == k
                if mask.sum() < 10:
                    continue
                cell_drm = drm[mask]
                drm_per_cell[str(k)] = {
                    "n": int(mask.sum()),
                    "mean": [float(x) for x in cell_drm.mean(axis=0)],
                    "std": [float(x) for x in cell_drm.std(axis=0)],
                }
            # Separabilidade: ANOVA F-stat por eixo DRM entre celulas
            from scipy import stats as sp_stats
            f_stats = {}
            groups = [drm[labels == k] for k in range(len(centers))
                      if (labels == k).sum() >= 10]
            if len(groups) >= 2:
                for ax_i, ax_name in enumerate(drm_axis_names):
                    ax_groups = [g[:, ax_i] for g in groups]
                    f_val, p_val = sp_stats.f_oneway(*ax_groups)
                    f_stats[ax_name] = {
                        "F": float(f_val), "p": float(p_val),
                    }
            results["drm_correlation"] = {
                "per_cell": drm_per_cell,
                "anova_f_stats": f_stats,
            }
            np.save(str(out_dir / "drm_per_cell.npy"), drm)
            logger.info(
                "[DRM] Correlacao calculada para %d celulas, %d eixos",
                len(drm_per_cell), len(f_stats),
            )
            for ax, fs in f_stats.items():
                logger.info("  %s: F=%.2f p=%.2e", ax, fs["F"], fs["p"])
        else:
            logger.warning(
                "[DRM] Shape mismatch: vectors=%d, drm=%d",
                vectors.shape[0], drm.shape[0],
            )

    results["config"] = {
        "n_seeds": args.n_seeds,
        "use_metric_net": args.use_metric_net,
        "n_vectors": int(vectors.shape[0]),
        "checkpoint": args.checkpoint,
        "drm_coords": args.drm_coords,
    }

    results_path = out_dir / "foliation_results.json"
    with open(str(results_path), "w") as f:
        json.dump(results, f, indent=2, default=str)

    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info("  RESULTADO FINAL")
    logger.info("  F = %.4f | ARI = %.3f | Coherent = %.1f%%",
                F, mean_ari, results["coherence"]["coherent_fraction"] * 100)
    if homology:
        logger.info("  Topology: %s", homology["topology"])
    logger.info("  Tempo: %.1fs", elapsed)
    logger.info("[OK] %s", results_path)


if __name__ == "__main__":
    main()
