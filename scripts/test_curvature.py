"""
Teste pratico de curvatura entre branches main/full_mahalanobis/real_geodesic.

Mede distancia geodesica para os mesmos probes em cada branch.
Se o espaco e plano, distancias sao similares entre branches.
Se tem curvatura real, real_geodesic vai mostrar distancias diferentes
(especialmente para pares context_sensitive).

Uso:
    # Na branch main:
    python scripts/test_curvature.py --checkpoint checkpoints/350m_epistemic_finetune_v2/final.pt --branch main

    # Na branch full_mahalanobis:
    python scripts/test_curvature.py --checkpoint checkpoints/350m_full_mahalanobis/final.pt --branch full_mahalanobis

    # Na branch real_geodesic:
    python scripts/test_curvature.py --checkpoint checkpoints/350m_real_geodesic/final.pt --branch real_geodesic

    # Comparar resultados salvos:
    python scripts/test_curvature.py --compare
"""

import argparse
import json
import sys
import logging
from pathlib import Path

import torch
import numpy as np

# Gauss-Legendre pre-computado para n=16 em [0,1]
def _gauss_legendre_01(n: int):
    """Pontos e pesos de Gauss-Legendre em [0, 1]."""
    points, weights = np.polynomial.legendre.leggauss(n)
    return (points + 1.0) / 2.0, weights / 2.0


# ============================================================================
# Probes
# ============================================================================

PROBES = {
    "high_confidence": [
        ("The capital of France is", "Paris"),
        ("2 + 2 =", "4"),
        ("Water is composed of hydrogen and", "oxygen"),
    ],
    "low_confidence": [
        ("The exact number of neurons in the human brain is", "86"),
        ("The GDP of Nauru in 2019 was", "132"),
    ],
    "context_sensitive": [
        # bank = margem vs bank = instituicao
        ("The bank was steep and", "muddy"),
        ("The bank was closed and", "dark"),
        # plant = planta vs plant = fabrica
        ("He left the plant near", "water"),
        ("He left the plant near", "the door"),
        # bat = morcego vs bat = taco
        ("The bat flew out of the", "cave"),
        ("The bat cracked on the", "pitch"),
    ],
}


# ============================================================================
# Estado epistemico
# ============================================================================

def get_epistemic_state(model, tokenizer, prompt, device):
    """Extrai coordenadas 5D e tomografia do ultimo token.

    Args:
        model: AletheionV2Model
        tokenizer: tiktoken encoding
        prompt: texto de entrada

    Returns:
        coords: [5] coordenadas no manifold
        tomo: EpistemicTomography completa
    """
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        output = model(input_ids, return_tomography=True)

    tomo = output.tomography
    if tomo is None:
        raise RuntimeError("Tomografia nao disponivel. Verifique enable_tomography=true.")

    # Ultimo token
    coords = tomo.drm_coords[0, -1, :].cpu().numpy()  # [5]
    return coords, tomo


def get_metric_at_point(model, coords_tensor):
    """Avalia MetricNet em um ponto arbitrario do manifold.

    So funciona na branch real_geodesic (metric_net habilitado).

    Args:
        model: AletheionV2Model
        coords_tensor: [5] tensor de coordenadas

    Returns:
        G: [5, 5] tensor metrico SPD naquele ponto
    """
    metric_net = model.epistemic_head.get_metric_net()
    if metric_net is None:
        return None

    with torch.no_grad():
        # MetricNet espera [..., 5], retorna [..., 5, 5]
        x = coords_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, 5]
        G = metric_net(x)  # [1, 1, 5, 5]
        return G[0, 0].cpu().numpy()


# ============================================================================
# Funcoes de distancia
# ============================================================================

def distance_diagonal(x1, x2, tau_sq):
    """Branch main: distancia diagonal via BayesianTau.

    d = sqrt(sum((x1_i - x2_i)^2 / tau_i^2))
    """
    diff = x1 - x2
    tau_sq_safe = np.maximum(tau_sq, 1e-8)
    return float(np.sqrt(np.sum(diff**2 / tau_sq_safe)))


def distance_mahalanobis(x1, x2, G):
    """Branch full_mahalanobis: Mahalanobis com G constante.

    d = sqrt(delta^T @ G @ delta)
    """
    diff = x1 - x2
    d_sq = diff @ G @ diff
    return float(np.sqrt(max(d_sq, 0.0)))


def distance_line_integral(x1, x2, metric_fn, n_quad=16):
    """Branch real_geodesic: integral de linha via Gauss-Legendre.

    d = integral_0^1 sqrt(dx^T G(gamma(t)) dx) dt
    onde gamma(t) = x1 + t*(x2 - x1)
    """
    t_nodes, weights = _gauss_legendre_01(n_quad)
    dx = x2 - x1
    total = 0.0

    for t, w in zip(t_nodes, weights):
        x_t = x1 + t * dx
        G_t = metric_fn(x_t)
        if G_t is None:
            # Fallback se metric_net nao disponivel
            return distance_mahalanobis(x1, x2, np.eye(5))
        ds2 = dx @ G_t @ dx
        total += w * np.sqrt(max(ds2, 1e-12))

    return float(total)


# ============================================================================
# Experimento principal
# ============================================================================

def run_experiment(model, tokenizer, branch, device):
    """Roda o experimento de curvatura para uma branch.

    Returns:
        dict com todos os resultados
    """
    results = {
        "branch": branch,
        "distances": {},
        "metric_variation": {},
    }

    print(f"\n{'='*60}")
    print(f"  CURVATURE TEST - branch: {branch}")
    print(f"{'='*60}")

    # --- Extrair G constante (main e full_mahalanobis) ---
    G_constant = model.epistemic_head.get_metric_tensor().cpu().numpy()
    tau_sq = model.epistemic_head.get_tau_sq().cpu().numpy()

    # --- Funcao de distancia por branch ---
    has_metric_net = model.epistemic_head.get_metric_net() is not None

    def compute_distance(x1, x2):
        """Computa distancia usando o metodo da branch."""
        if branch == "main":
            return distance_diagonal(x1, x2, tau_sq)
        elif branch == "full_mahalanobis":
            return distance_mahalanobis(x1, x2, G_constant)
        elif branch == "real_geodesic":
            if has_metric_net:
                def metric_fn(x):
                    xt = torch.tensor(x, dtype=torch.float32, device=device)
                    return get_metric_at_point(model, xt)
                return distance_line_integral(x1, x2, metric_fn)
            else:
                return distance_mahalanobis(x1, x2, G_constant)
        else:
            raise ValueError(f"Branch desconhecida: {branch}")

    # --- Teste 1: Context Sensitivity ---
    print("\n--- CONTEXT SENSITIVITY TEST ---")
    ctx_probes = PROBES["context_sensitive"]
    for i in range(0, len(ctx_probes), 2):
        prompt_a, token_a = ctx_probes[i]
        prompt_b, token_b = ctx_probes[i + 1]

        x_a, _ = get_epistemic_state(model, tokenizer, prompt_a, device)
        x_b, _ = get_epistemic_state(model, tokenizer, prompt_b, device)

        d = compute_distance(x_a, x_b)
        pair_key = f"{prompt_a[:20]}... vs {prompt_b[:20]}..."
        results["distances"][pair_key] = d
        print(f"  {pair_key}")
        print(f"    dist = {d:.6f}")

    # --- Teste 2: High vs Low Confidence ---
    print("\n--- HIGH vs LOW CONFIDENCE ---")
    for ph, _ in PROBES["high_confidence"]:
        for pl, _ in PROBES["low_confidence"][:1]:
            x_h, _ = get_epistemic_state(model, tokenizer, ph, device)
            x_l, _ = get_epistemic_state(model, tokenizer, pl, device)

            d = compute_distance(x_h, x_l)
            pair_key = f"HIGH:{ph[:20]}... vs LOW:{pl[:20]}..."
            results["distances"][pair_key] = d
            print(f"  {pair_key}")
            print(f"    dist = {d:.6f}")

    # --- Teste 3: Within-domain (controle) ---
    print("\n--- WITHIN-DOMAIN (controle) ---")
    hc = PROBES["high_confidence"]
    for i in range(len(hc) - 1):
        x_a, _ = get_epistemic_state(model, tokenizer, hc[i][0], device)
        x_b, _ = get_epistemic_state(model, tokenizer, hc[i + 1][0], device)

        d = compute_distance(x_a, x_b)
        pair_key = f"SAME:{hc[i][0][:15]}... vs {hc[i+1][0][:15]}..."
        results["distances"][pair_key] = d
        print(f"  {pair_key}")
        print(f"    dist = {d:.6f}")

    # --- Teste 4: Variacao de G ao longo do path (so real_geodesic) ---
    if has_metric_net:
        print("\n--- METRIC VARIATION TEST ---")
        for ph, _ in PROBES["high_confidence"][:1]:
            for pl, _ in PROBES["low_confidence"][:1]:
                x_h, _ = get_epistemic_state(model, tokenizer, ph, device)
                x_l, _ = get_epistemic_state(model, tokenizer, pl, device)

                n_samples = 20
                G_samples = []
                for t in np.linspace(0, 1, n_samples):
                    x_t = x_h + t * (x_l - x_h)
                    xt_tensor = torch.tensor(
                        x_t, dtype=torch.float32, device=device,
                    )
                    G_t = get_metric_at_point(model, xt_tensor)
                    if G_t is not None:
                        G_samples.append(G_t)

                if G_samples:
                    G_stack = np.stack(G_samples)
                    variation = np.std(G_stack, axis=0)
                    var_mean = float(variation.mean())
                    var_max = float(variation.max())

                    path_key = f"{ph[:20]}... -> {pl[:20]}..."
                    results["metric_variation"][path_key] = {
                        "var_mean": var_mean,
                        "var_max": var_max,
                        "is_curved": var_max > 0.01,
                    }

                    print(f"  Path: {path_key}")
                    print(f"    Variacao media de G(x): {var_mean:.6f}")
                    print(f"    Variacao maxima: {var_max:.6f}")
                    verdict = "CURVO" if var_max > 0.01 else "PLANO"
                    print(f"    -> Espaco {verdict}")

    # --- Teste 5: G diagnostico ---
    print("\n--- METRIC TENSOR DIAGNOSTICO ---")
    print(f"  G constante (diagonal): {np.diag(G_constant)}")
    print(f"  G constante (off-diag norm): "
          f"{np.linalg.norm(G_constant - np.diag(np.diag(G_constant))):.6f}")
    print(f"  G condition number: "
          f"{np.linalg.cond(G_constant):.2f}")
    print(f"  tau_sq: {tau_sq}")

    if has_metric_net:
        # G(x) em 3 pontos distintos
        test_points = [
            np.array([0.1, 0.1, 0.5, 0.9, 0.9]),  # truth-like
            np.array([0.9, 0.9, 0.5, 0.1, 0.1]),  # ignorance-like
            np.array([0.5, 0.5, 0.5, 0.5, 0.5]),  # centro
        ]
        print(f"\n  G(x) em 3 pontos do manifold:")
        for label, pt in zip(["truth", "ignorance", "centro"], test_points):
            pt_tensor = torch.tensor(pt, dtype=torch.float32, device=device)
            G_pt = get_metric_at_point(model, pt_tensor)
            if G_pt is not None:
                print(f"    [{label}] diag={np.diag(G_pt).round(4)}, "
                      f"cond={np.linalg.cond(G_pt):.2f}")

    results["G_constant_diag"] = np.diag(G_constant).tolist()
    results["G_constant_offdiag_norm"] = float(
        np.linalg.norm(G_constant - np.diag(np.diag(G_constant)))
    )
    results["G_condition_number"] = float(np.linalg.cond(G_constant))
    results["tau_sq"] = tau_sq.tolist()

    return results


def compare_results():
    """Compara resultados salvos das 3 branches."""
    result_dir = Path("eval_results/curvature")
    branches = ["main", "full_mahalanobis", "real_geodesic"]

    all_results = {}
    for branch in branches:
        path = result_dir / f"{branch}.json"
        if path.exists():
            with open(path) as f:
                all_results[branch] = json.load(f)
        else:
            print(f"[WARN] {path} nao encontrado, pulando {branch}")

    if len(all_results) < 2:
        print("[ERROR] Precisa de pelo menos 2 branches para comparar")
        return

    print(f"\n{'='*70}")
    print(f"  COMPARACAO DE CURVATURA ({len(all_results)} branches)")
    print(f"{'='*70}")

    # Coleta todas as chaves de distancia
    all_keys = set()
    for r in all_results.values():
        all_keys.update(r.get("distances", {}).keys())

    print(f"\n{'Probe pair':<50}", end="")
    for b in all_results:
        print(f"  {b:>15}", end="")
    print()
    print("-" * (50 + 17 * len(all_results)))

    for key in sorted(all_keys):
        print(f"{key:<50}", end="")
        for b in all_results:
            d = all_results[b].get("distances", {}).get(key, None)
            if d is not None:
                print(f"  {d:>15.6f}", end="")
            else:
                print(f"  {'N/A':>15}", end="")
        print()

    # Metric variation (so real_geodesic)
    if "real_geodesic" in all_results:
        mv = all_results["real_geodesic"].get("metric_variation", {})
        if mv:
            print(f"\n--- METRIC VARIATION (real_geodesic) ---")
            for path_key, v in mv.items():
                verdict = "CURVO" if v["is_curved"] else "PLANO"
                print(f"  {path_key}")
                print(f"    var_mean={v['var_mean']:.6f}, "
                      f"var_max={v['var_max']:.6f} -> {verdict}")

    # Diagnostico G
    print(f"\n--- G DIAGNOSTICO ---")
    for b, r in all_results.items():
        print(f"  [{b}]")
        print(f"    G diag: {r.get('G_constant_diag', 'N/A')}")
        print(f"    G off-diag norm: {r.get('G_constant_offdiag_norm', 'N/A')}")
        print(f"    G cond: {r.get('G_condition_number', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(description="Teste de curvatura epistemica")
    parser.add_argument("--checkpoint", type=str, help="Caminho do checkpoint")
    parser.add_argument("--branch", type=str,
                        choices=["main", "full_mahalanobis", "real_geodesic"],
                        help="Branch sendo testada")
    parser.add_argument("--compare", action="store_true",
                        help="Compara resultados salvos das 3 branches")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda ou cpu)")
    args = parser.parse_args()

    if args.compare:
        compare_results()
        return

    if not args.checkpoint or not args.branch:
        parser.error("--checkpoint e --branch sao obrigatorios (ou use --compare)")

    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[SETUP] Device: {device}")
    print(f"[SETUP] Checkpoint: {args.checkpoint}")
    print(f"[SETUP] Branch: {args.branch}")

    # Carrega modelo
    from aletheion_v2.config import AletheionV2Config
    from aletheion_v2.core.model import AletheionV2Model

    state = torch.load(args.checkpoint, map_location=device, weights_only=False)

    if "config" in state and isinstance(state["config"], AletheionV2Config):
        config = state["config"]
    else:
        config = AletheionV2Config.medium()

    model = AletheionV2Model(config).to(device)

    # Load com strict=False para compatibilidade entre branches
    missing, unexpected = model.load_state_dict(
        state["model"], strict=False,
    )
    if missing:
        metric_missing = [k for k in missing if "metric_net" in k]
        other_missing = [k for k in missing if "metric_net" not in k]
        if metric_missing:
            print(f"[LOAD] {len(metric_missing)} params de MetricNet "
                  f"inicializados do zero (esperado em branches sem MetricNet)")
        if other_missing:
            print(f"[WARN] {len(other_missing)} params faltando: "
                  f"{other_missing[:5]}...")
    if unexpected:
        print(f"[WARN] {len(unexpected)} params inesperados: "
              f"{unexpected[:5]}...")

    model.eval()
    model.float()  # fp32

    print(f"[LOAD] Modelo carregado: {model.count_parameters()['total']:,} params")

    # Tokenizer
    import tiktoken
    tokenizer = tiktoken.get_encoding(config.tokenizer_name or "gpt2")

    # Roda experimento
    results = run_experiment(model, tokenizer, args.branch, device)

    # Salva resultados
    result_dir = Path("eval_results/curvature")
    result_dir.mkdir(parents=True, exist_ok=True)
    out_path = result_dir / f"{args.branch}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVE] Resultados salvos em {out_path}")


if __name__ == "__main__":
    main()
