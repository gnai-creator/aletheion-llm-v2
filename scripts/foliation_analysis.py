"""
Funcoes de analise para deteccao de foliacao epistemica.

Modulo auxiliar de voronoi_foliation.py: LTSA, coerencia tangencial,
Reeb graph, persistent homology, null models, stability, foliation score.
"""

import logging
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


def ltsa_per_cell(
    vectors: np.ndarray, labels: np.ndarray, min_points: int = 100,
) -> Dict[str, np.ndarray]:
    """PCA local por celula Voronoi. Retorna eigenvalues, eigenvectors, eff_dims, cell_sizes."""
    unique_labels = np.unique(labels)
    k, d = len(unique_labels), vectors.shape[1]

    eigenvalues = np.zeros((k, d), dtype=np.float32)
    eigenvectors = np.zeros((k, d, d), dtype=np.float32)
    eff_dims = np.zeros(k, dtype=np.int32)
    cell_sizes = np.zeros(k, dtype=np.int32)

    for i, label in enumerate(unique_labels):
        mask = labels == label
        n_pts = mask.sum()
        cell_sizes[i] = n_pts
        if n_pts < min_points:
            eigenvalues[i] = np.ones(d) / d
            eigenvectors[i] = np.eye(d)
            eff_dims[i] = d
            continue

        cell_centered = vectors[mask] - vectors[mask].mean(axis=0)
        cov = (cell_centered.T @ cell_centered) / (n_pts - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1]
        eigenvalues[i] = eigvals[idx]
        eigenvectors[i] = eigvecs[:, idx].T

        total_var = eigvals.sum()
        if total_var < 1e-10:
            eff_dims[i] = d
        else:
            cumulative = np.cumsum(eigvals[idx] / total_var)
            eff_dims[i] = min(np.searchsorted(cumulative, 0.95) + 1, d)

    logger.info(
        "[LTSA] eff_dims: media=%.1f, mediana=%d, dist=%s",
        eff_dims.mean(), int(np.median(eff_dims)),
        np.bincount(eff_dims, minlength=d + 1)[1:],
    )
    return {"eigenvalues": eigenvalues, "eigenvectors": eigenvectors,
            "eff_dims": eff_dims, "cell_sizes": cell_sizes}


def tangent_coherence(
    centers: np.ndarray, eigenvectors: np.ndarray,
    eff_dims: np.ndarray, n_neighbors: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Angulo principal entre subespacos tangentes de celulas vizinhas."""
    # Garantir alinhamento: usar min(centers, eff_dims) como k
    k = min(centers.shape[0], len(eff_dims), eigenvectors.shape[0])
    centers = centers[:k]
    eff_dims = eff_dims[:k]
    eigenvectors = eigenvectors[:k]
    nn = NearestNeighbors(n_neighbors=min(n_neighbors + 1, k), metric="euclidean")
    nn.fit(centers)
    _, indices = nn.kneighbors(centers)

    coherence_matrix = np.full((k, k), np.nan, dtype=np.float32)
    neighbor_graph = np.zeros((k, k), dtype=bool)

    for i in range(k):
        d_i = max(eff_dims[i], 1)
        S_i = eigenvectors[i, :d_i, :]
        for j_idx in range(1, indices.shape[1]):
            j = indices[i, j_idx]
            neighbor_graph[i, j] = neighbor_graph[j, i] = True
            d_j = max(eff_dims[j], 1)
            S_j = eigenvectors[j, :d_j, :]
            svd_vals = np.linalg.svd(S_i @ S_j.T, compute_uv=False)
            angle = np.degrees(np.arccos(np.clip(svd_vals[0], -1.0, 1.0)))
            coherence_matrix[i, j] = coherence_matrix[j, i] = angle

    valid = coherence_matrix[~np.isnan(coherence_matrix)]
    logger.info("[COHERENCE] %d/%d pares < 30 graus (%.1f%%)",
                (valid < 30).sum(), len(valid), 100 * (valid < 30).mean() if len(valid) > 0 else 0)
    return coherence_matrix, neighbor_graph


def compute_reeb_graph(
    vectors: np.ndarray, labels: np.ndarray,
    func_idx: int = 2, func_name: str = "confidence",
    n_levels: int = 50, n_neighbors: int = 15,
) -> Dict:
    """Grafo Reeb via level sets com logit pre-conditioning automatico."""
    f_raw = vectors[:, func_idx].copy()
    f_std = f_raw.std()

    # Logit transform se saturada
    if f_std < 0.1:
        eps = 1e-4
        fc = np.clip(f_raw, eps, 1.0 - eps)
        f = np.log(fc / (1.0 - fc))
        transform = "logit"
        logger.info("[REEB] %s: std=%.4f, logit -> std=%.4f", func_name, f_std, f.std())
    else:
        f = f_raw
        transform = "none"
        logger.info("[REEB] %s: std=%.4f, no transform", func_name, f_std)

    f_std_final = f.std()
    if f_std_final < 0.5:
        logger.warning("[WARN] %s: std=%.4f, variacao insuficiente", func_name, f_std_final)

    level_edges = np.linspace(f.min(), f.max(), n_levels + 1)
    nn = NearestNeighbors(n_neighbors=min(n_neighbors, len(vectors) - 1))
    nn.fit(vectors)
    nn_graph = nn.kneighbors_graph(mode="connectivity")

    nodes: List[Dict] = []
    edges: List[Tuple[int, int]] = []
    prev_start, has_prev = 0, False

    for lvl in range(n_levels):
        lo, hi = level_edges[lvl], level_edges[lvl + 1]
        mask = (f >= lo) & (f <= hi) if lvl == n_levels - 1 else (f >= lo) & (f < hi)
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue

        sub_graph = nn_graph[indices][:, indices]
        n_comp, comp_labels = connected_components(sub_graph, directed=False)
        cur_start = len(nodes)

        for c in range(n_comp):
            nodes.append({"level": lvl, "level_value": float((lo + hi) / 2),
                          "n_points": int((comp_labels == c).sum())})

        if has_prev:
            for c_curr in range(n_comp):
                curr_pts = indices[comp_labels == c_curr]
                for pni in range(prev_start, cur_start):
                    pn = nodes[pni]
                    pl, plo, phi_ = pn["level"], level_edges[pn["level"]], level_edges[pn["level"] + 1]
                    pm = (f >= plo) & (f <= phi_) if pl == n_levels - 1 else (f >= plo) & (f < phi_)
                    p_idx = np.where(pm)[0]
                    connected = False
                    for cp in curr_pts[:50]:
                        if np.intersect1d(nn_graph[cp].indices, p_idx).size > 0:
                            connected = True
                            break
                    if connected:
                        edges.append((pni, cur_start + c_curr))

        has_prev = True
        prev_start = cur_start

    n_splits = n_merges = 0
    if edges:
        out_deg = Counter(e[0] for e in edges)
        in_deg = Counter(e[1] for e in edges)
        n_splits = sum(1 for v in out_deg.values() if v > 1)
        n_merges = sum(1 for v in in_deg.values() if v > 1)

    is_tree = (n_merges == 0 and n_splits == 0) if nodes else False
    logger.info("[REEB] %s: %d nodes, %d edges, splits=%d, merges=%d, tree=%s",
                func_name, len(nodes), len(edges), n_splits, n_merges, is_tree)

    return {"func_name": func_name, "func_idx": func_idx, "transform": transform,
            "n_nodes": len(nodes), "n_edges": len(edges),
            "n_splits": n_splits, "n_merges": n_merges, "is_tree": is_tree,
            "nodes": nodes, "edges": edges,
            "f_std_raw": float(f_std), "f_std_final": float(f_std_final)}


def compute_persistent_homology(
    vectors: np.ndarray,
    riemannian_dist_fn: Optional[object] = None,
    max_points: int = 5000,
    max_dim: int = 2,
) -> Optional[Dict]:
    """Persistent homology com validacao T^2 (H1=Z^2, H2=Z). Requer ripser."""
    try:
        from ripser import ripser
    except ImportError:
        logger.warning("[WARN] ripser nao instalado, pulando homology")
        return None

    n = vectors.shape[0]
    sub = vectors[np.random.choice(n, min(max_points, n), replace=False)] if n > max_points else vectors

    logger.info("[HOMOLOGY] %d pontos, max_dim=%d...", len(sub), max_dim)
    t0 = time.time()

    if riemannian_dist_fn is not None:
        result = ripser(riemannian_dist_fn(sub), maxdim=max_dim, distance_matrix=True)
    else:
        result = ripser(sub, maxdim=max_dim)

    logger.info("[HOMOLOGY] Concluido em %.1fs", time.time() - t0)
    diagrams = result["dgms"]
    report = {}

    for dim in range(min(len(diagrams), max_dim + 1)):
        dgm = diagrams[dim]
        if len(dgm) == 0:
            report[f"H{dim}"] = {"n_features": 0, "long_bars": 0, "max_persistence": 0.0}
            continue
        fin = dgm[np.isfinite(dgm[:, 1])]
        pers = (fin[:, 1] - fin[:, 0]) if len(fin) > 0 else np.array([0.0])
        thresh = (np.median(pers) + 2 * np.std(pers)) if len(pers) > 1 else 0.0
        report[f"H{dim}"] = {
            "n_features": len(dgm), "long_bars": int((pers > thresh).sum()),
            "max_persistence": float(pers.max()), "mean_persistence": float(pers.mean()),
            "persistence_values": pers.tolist(),
        }

    h1_long = report.get("H1", {}).get("long_bars", 0)
    h2_long = report.get("H2", {}).get("long_bars", 0)
    t2_valid = (h1_long == 2) and (h2_long == 1)

    if h1_long == 0:
        topo = "trivial (no loops)"
    elif h1_long == 1:
        topo = "cylinder (S^1 x R)"
    elif h1_long == 2:
        topo = "torus T^2 (validated)" if h2_long == 1 else "partial T^2 (H1=Z^2, H2 mismatch)"
    else:
        topo = f"complex (genus > 1, H1 rank={h1_long})"

    logger.info("[HOMOLOGY] H1=%d, H2=%d -> %s", h1_long, h2_long, topo)
    return {"homology": report, "t2_valid": t2_valid, "topology": topo,
            "n_points_used": len(sub), "diagrams": [d.tolist() for d in diagrams]}


def run_null_model(
    vectors: np.ndarray, labels: np.ndarray, centers: np.ndarray,
    ltsa_fn: object, coherence_fn: object,
    null_type: str = "shuffled", backbone_vectors: Optional[np.ndarray] = None,
) -> Dict:
    """Executa pipeline LTSA+coerencia em modelo nulo (shuffled/uniform/backbone)."""
    if null_type == "shuffled":
        null_vec = vectors.copy()
        for d in range(vectors.shape[1]):
            np.random.shuffle(null_vec[:, d])
    elif null_type == "uniform":
        null_vec = np.random.uniform(0, 1, size=vectors.shape).astype(np.float32)
    elif null_type == "backbone" and backbone_vectors is not None:
        null_vec = backbone_vectors
    else:
        return {"type": null_type, "error": "invalid or missing data"}

    null_labels = np.argmin(cdist(null_vec, centers), axis=1)
    ltsa = ltsa_fn(null_vec, null_labels, min_points=50)
    coh, _ = coherence_fn(centers, ltsa["eigenvectors"], ltsa["eff_dims"])
    valid = coh[~np.isnan(coh)]
    mean_coh = float(valid.mean()) if len(valid) > 0 else 90.0
    coh_frac = float((valid < 30).mean()) if len(valid) > 0 else 0.0

    dc = np.bincount(ltsa["eff_dims"], minlength=6)[1:]
    dp = dc / dc.sum()
    dp = dp[dp > 0]
    h_dk = float(-np.sum(dp * np.log(dp))) if len(dp) > 0 else 0.0

    logger.info("[NULL:%s] angle=%.1f, coh=%.2f, H=%.3f", null_type, mean_coh, coh_frac, h_dk)
    return {"type": null_type, "mean_coherence_angle": mean_coh,
            "coherent_fraction": coh_frac, "dim_entropy": h_dk,
            "eff_dims_dist": ltsa["eff_dims"].tolist(),
            "mean_eff_dim": float(ltsa["eff_dims"].mean())}


def stability_test(
    vectors: np.ndarray, n_seeds: int = 30, n_restarts: int = 10,
) -> Tuple[float, List[float]]:
    """Testa estabilidade via ARI entre restarts de K-means."""
    all_labels = []
    for r in range(n_restarts):
        km = KMeans(n_clusters=n_seeds, n_init=1, random_state=r * 42)
        km.fit(vectors)
        all_labels.append(km.labels_)

    ari_values = [adjusted_rand_score(all_labels[i], all_labels[j])
                  for i in range(len(all_labels)) for j in range(i + 1, len(all_labels))]

    mean_ari = float(np.mean(ari_values))
    logger.info("[STABILITY] ARI: mean=%.3f, std=%.3f (%d pares)",
                mean_ari, np.std(ari_values), len(ari_values))
    return mean_ari, ari_values


def compute_foliation_score(
    eff_dims: np.ndarray, coherence_matrix: np.ndarray, stability_ari: float,
) -> float:
    """F = (1 - H(dk)/log(5)) * coherent_fraction * ARI. Score em [0, 1]."""
    dc = np.bincount(eff_dims, minlength=6)[1:]
    dp = dc / dc.sum()
    dp = dp[dp > 0]
    h_dk = -np.sum(dp * np.log(dp))

    valid = coherence_matrix[~np.isnan(coherence_matrix)]
    coh = float((valid < 30).mean()) if len(valid) > 0 else 0.0

    F = float(np.clip((1.0 - h_dk / np.log(5)) * coh * stability_ari, 0.0, 1.0))
    logger.info("[SCORE] F=%.4f (1-H/Hmax=%.3f, coh=%.3f, ARI=%.3f)",
                F, 1.0 - h_dk / np.log(5), coh, stability_ari)
    return F
