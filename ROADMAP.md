# Roadmap

Plano de desenvolvimento do Aletheion LLM v2.

## v0.1.0 (Atual)

- [x] Arquitetura decoder-only com RoPE + SwiGLU
- [x] Sistema epistemico completo (11 sub-heads, 3 tiers)
- [x] Manifold DRM 5D com tensor metrico SPD
- [x] MAD Confidence com tau Bayesiano (InverseGamma prior)
- [x] Vetor de Intencionalidade (VI) com phi(M)
- [x] MPC Navigator com beam search (12 acoes)
- [x] Loss composta com 13 componentes + annealing
- [x] Continual Learning (EWC + Experience Replay)
- [x] Pipeline de treinamento distribuido (DDP/FSDP)
- [x] 15 configs de escala (1M - 640B)
- [x] 261 testes unitarios
- [x] Geracao com tomografia epistemica
- [x] Dashboard bridge para ATIC
- [x] Licenciamento dual (AGPL-3.0 + Comercial)

## v0.2.0 (Planejado)

- [ ] Tokenizer proprio treinado (BPE customizado)
- [ ] Pre-treinamento completo em FineWeb-Edu (350M, single RTX 4090)
- [ ] Benchmarks padrao (HellaSwag, ARC, MMLU) pos-treinamento
- [ ] Metricas epistemicas durante avaliacao (confidence vs accuracy)
- [ ] Export ONNX para inferencia otimizada
- [ ] Documentacao da API Python completa

## v0.3.0 (Futuro)

- [ ] Fine-tuning supervisionado (SFT) com tomografia
- [ ] RLHF/DPO com reward epistemico
- [ ] Treinamento em escala 1.3B+ (multi-GPU validado)
- [ ] Comparativo formal: Aletheion vs baselines (GPT-2, LLaMA)
  com foco em calibracao de incerteza
- [ ] Visualizacao interativa do manifold 5D (web UI)

## v1.0.0 (Longo Prazo)

- [ ] Treinamento em escala 7B+
- [ ] Integracao com frameworks de servico (vLLM, TGI)
- [ ] API publica de inferencia com tomografia
- [ ] Paper publicado em conferencia revisada por pares
- [ ] Ecossistema de plugins para novos sub-heads epistemicos

---

Este roadmap e indicativo e pode mudar conforme prioridades e recursos.
Sugestoes sao bem-vindas via issues.
