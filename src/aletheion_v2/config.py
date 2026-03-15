"""
Configuracao centralizada do AletheionV2.

Todos os hiperparametros do modelo, DRM, MAD, VI, MPC e treinamento
sao definidos aqui. Configs podem ser carregadas de YAML.
"""

from dataclasses import dataclass, field
from typing import Optional
import yaml


@dataclass
class AletheionV2Config:
    """Configuracao completa do AletheionV2."""

    # --- Modelo base ---
    vocab_size: int = 32000
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    d_ff: int = 3072  # 4 * d_model
    max_seq_len: int = 1024
    dropout: float = 0.1
    rope_base: float = 10000.0

    # --- Epistemic Gates ---
    gate_hidden_dim: int = 64
    gate_num_layers: int = 2
    gate_dropout: float = 0.1
    enable_fractal: bool = False

    # --- DRM Manifold ---
    drm_dim: int = 5  # Dimensoes do manifold epistemico
    drm_num_anchors: int = 6  # Pontos ancora fixos
    drm_coord_activation: str = "sigmoid"  # Coords em [0,1]

    # --- Metric Tensor ---
    metric_eps: float = 1e-6  # Regularizacao SPD
    metric_position_dependent: bool = True  # G(x) variavel vs G constante
    metric_net_hidden: int = 32  # Hidden dim da MetricNet
    metric_net_n_quad: int = 5  # Pontos de quadratura Gauss-Legendre
    metric_net_lr_multiplier: float = 10.0  # LR multiplicador para MetricNet
    lambda_metric_smoothness: float = 0.1  # Suavidade do campo G(x)
    metric_max_condition: float = 50.0  # Condition number maximo
    metric_gravity_dim: int = 0  # Dim do campo gravitacional (0 = desabilitado)
    gravity_decay: float = 0.99  # Decay temporal do campo gravitacional

    # --- Directional Field ---
    dir_num_bins: int = 16  # Bins de entropia de atencao

    # --- MAD ---
    mad_init_log_tau_sq: float = 0.0  # log(tau^2) inicial = log(1) = 0
    mad_per_axis: bool = True  # tau^2 aprendivel por eixo

    # --- VI ---
    vi_phi_weights: tuple = (0.35, 0.25, 0.25, 0.15)  # dim, disp, ent, conf
    vi_phi_critical: float = 0.5
    vi_injection_strength: float = 0.6
    vi_confidence_penalty: float = 0.4
    vi_ideal_conf_std: float = 0.15

    # --- MPC ---
    mpc_num_actions: int = 12
    mpc_beam_width: int = 4
    mpc_lookahead_depth: int = 3
    mpc_phi_floor: float = 0.45
    mpc_intervention_cost_weight: float = 0.20

    # --- Eidos Decay ---
    enable_eidos: bool = True
    eidos_base_decay: float = 0.95
    eidos_base_reinforce: float = 1.05
    eidos_dream_intensity: float = 3.0

    # --- Filosofia3 ---
    enable_filosofia3: bool = True
    filosofia3_quality_projection: tuple = (0.1, 0.1, 0.1, 0.7)
    filosofia3_analytical_weight: float = 0.7

    # --- Consciousness ---
    enable_consciousness: bool = True
    consciousness_hidden_dim: int = 32
    consciousness_energy_decay: float = 0.3

    # --- Grounding ---
    enable_grounding: bool = True
    grounding_task_hidden_dim: int = 64
    grounding_ambiguity_hidden_dim: int = 32

    # --- Plasticity ---
    enable_plasticity: bool = True
    plasticity_initial_budget: float = 1.0
    plasticity_depletion_rate: float = 0.02

    # --- MPL ---
    enable_mpl: bool = True
    mpl_resolution: int = 10
    mpl_bandwidth: float = 0.15
    mpl_hidden_dim: int = 16

    # --- MOPsi ---
    enable_mopsi: bool = True
    mopsi_hidden_dim: int = 32

    # --- CausalState ---
    enable_causal_state: bool = True
    causal_state_hidden_dim: int = 32

    # --- Metacognitive ---
    enable_metacognitive: bool = True
    metacognitive_hidden_dim: int = 128
    metacognitive_proj_dim: int = 0  # 0 = d_model // 2

    # --- Continual Learning ---
    enable_ewc: bool = False  # Elastic Weight Consolidation
    ewc_lambda: float = 100.0  # Peso da regularizacao EWC
    ewc_online: bool = True  # Online EWC (acumula Fisher)
    ewc_gamma: float = 0.9  # Decay para Fisher de fases anteriores
    ewc_fisher_samples: int = 256  # Amostras para estimar Fisher
    enable_replay: bool = False  # Experience Replay
    replay_buffer_size: int = 10000  # Tamanho do buffer em amostras
    replay_mix_ratio: float = 0.1  # Fracao do batch substituida por replay

    # --- Tomography ---
    enable_tomography: bool = True  # False = CE+STP only (faster, less VRAM)

    # --- Loss ---
    lambda_ce: float = 1.0
    lambda_varo: float = 0.1
    lambda_vi: float = 0.01
    lambda_mad: float = 0.05
    lambda_metric_reg: float = 0.001
    lambda_eidos: float = 0.005
    lambda_conflict: float = 0.005
    lambda_consciousness: float = 0.003
    lambda_grounding: float = 0.005
    lambda_plasticity: float = 0.002
    lambda_frontier: float = 0.002
    lambda_mopsi: float = 0.003
    lambda_contrastive: float = 0.003
    lambda_stp: float = 0.01  # Smooth Transition Penalty
    enable_stp: bool = True  # STP loss (smooth hidden state transitions)
    stp_num_triplets: int = 1  # Triplets amostrados por step
    loss_warmup_fraction: float = 0.1  # Fracao do treino so com CE
    loss_ramp_fraction: float = 0.5  # Ramp linear ate aqui
    lambda_decay_mode: str = "none"  # "none", "exponential"
    lambda_decay_k: float = 0.0003  # e^(-k*t) decay constant
    lambda_decay_factor: float = 1.0  # Legacy: discrete decay factor
    lambda_decay_interval: int = 0  # Legacy: discrete decay interval

    # --- Training ---
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    batch_size: int = 16
    max_epochs: int = 10
    grad_clip: float = 1.0
    warmup_steps: int = 500
    log_interval: int = 50
    eval_interval: int = 500
    early_stopping_patience: int = 0  # 0 = desabilitado; >0 = para apos N evals sem melhora

    # --- Tokenizer ---
    tokenizer_name: str = "gpt2"  # tiktoken encoding name
    tokenizer_path: str = ""  # Caminho para tokenizer custom (vazio = tiktoken)

    # --- Data Pipeline ---
    data_dir: str = "data"  # Diretorio raiz dos dados
    dataset_name: str = ""  # Nome HuggingFace (ex: "HuggingFaceFW/fineweb")
    dataset_subset: str = ""  # Subset (ex: "sample-10BT")
    data_mix_weights: str = ""  # "wiki:0.3,book:0.7" pesos de mixing
    num_workers: int = 4
    prefetch_factor: int = 2
    streaming: bool = True  # Streaming de datasets grandes

    # --- Distributed Training ---
    distributed: bool = False
    dist_backend: str = "nccl"  # nccl para GPU, gloo para CPU
    fsdp: bool = False  # Fully Sharded Data Parallel
    fsdp_sharding: str = "full"  # full, grad_op, no_shard
    gradient_checkpointing: bool = False
    gradient_accumulation_steps: int = 1

    # --- Mixed Precision ---
    mixed_precision: str = "none"  # none, fp16, bf16
    compile_model: bool = False  # torch.compile (PyTorch 2.0+)

    # --- Logging ---
    wandb_project: str = ""  # Vazio = desabilitado
    wandb_run_name: str = ""
    tensorboard_dir: str = ""
    save_dir: str = "checkpoints"
    save_interval: int = 1000  # Steps entre saves
    save_total_limit: int = 5  # Maximo de checkpoints mantidos

    # --- Scaling ---
    total_tokens: int = 0  # 0 = usa max_epochs; >0 = treina ate N tokens
    tokens_per_step: int = 0  # Calculado: batch_size * seq_len * grad_accum

    @classmethod
    def from_yaml(cls, path: str) -> "AletheionV2Config":
        """Carrega config de arquivo YAML."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        filtered = {}
        for k, v in data.items():
            if k in cls.__dataclass_fields__:
                # Converte lists de volta para tuples onde necessario
                if isinstance(v, list) and isinstance(cls.__dataclass_fields__[k].default, tuple):
                    v = tuple(v)
                filtered[k] = v
        return cls(**filtered)

    def to_yaml(self, path: str) -> None:
        """Salva config em arquivo YAML."""
        from dataclasses import asdict
        data = asdict(self)
        # Converte tuples para lists (yaml.safe_load nao suporta tuples)
        for k, v in data.items():
            if isinstance(v, tuple):
                data[k] = list(v)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def small(cls) -> "AletheionV2Config":
        """Config pequena para testes rapidos."""
        return cls(
            vocab_size=32000, d_model=256, n_heads=4, n_layers=4,
            d_ff=1024, max_seq_len=512, dropout=0.1,
        )

    @classmethod
    def medium(cls) -> "AletheionV2Config":
        """Config media para treinamento real."""
        return cls(
            vocab_size=32000, d_model=768, n_heads=12, n_layers=12,
            d_ff=3072, max_seq_len=1024, dropout=0.1,
        )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def total_params_estimate(self) -> int:
        """Estimativa grosseira de parametros totais."""
        # Embeddings
        emb = self.vocab_size * self.d_model
        # Transformer blocks
        attn = 4 * self.d_model * self.d_model  # Q, K, V, O
        ff = 2 * self.d_model * self.d_ff
        block = attn + ff
        transformer = self.n_layers * block
        # Epistemic head (estimativa ~2.2M)
        epistemic = 2_200_000
        return emb + transformer + epistemic
