# Guia de Contribuicao

Obrigado pelo interesse em contribuir com o Aletheion LLM v2!

## Pre-requisitos

1. **Assinar o CLA**: todas as contribuicoes exigem assinatura do
   [Contributor License Agreement](CLA.md). Isso e necessario para manter
   o licenciamento dual (AGPL-3.0 + Comercial).

2. **Ambiente de desenvolvimento**:
   ```bash
   git clone https://github.com/gnai-creator/aletheion-llm-v2.git
   cd aletheion-llm-v2
   pip install -e ".[dev,data]"
   ```

3. **Verificar que os testes passam**:
   ```bash
   pytest tests/ -v
   ```

## Fluxo de Trabalho

1. **Fork** o repositorio
2. **Crie uma branch** a partir de `main`:
   ```bash
   git checkout -b feature/minha-feature
   ```
3. **Faca suas alteracoes** seguindo as convencoes do projeto
4. **Escreva testes** para novas funcionalidades
5. **Execute os testes**:
   ```bash
   pytest tests/ -v
   ```
6. **Commit** com mensagem descritiva
7. **Abra um Pull Request** incluindo a declaracao do CLA

## Convencoes de Codigo

### Estilo

- Python 3.10+
- Modulos entre 300-500 linhas (maximo)
- Docstrings e comentarios em portugues
- Nao usar emojis em codigo-fonte (incompatibilidade com Windows cp1252)
- Usar formato `[TAG]` para logs (ex: `[ERROR]`, `[OK]`, `[INFO]`)

### Estrutura

- Um modulo por responsabilidade (Single Responsibility)
- Novos `nn.Module` devem ter loss correspondente em `src/aletheion_v2/loss/`
- Novos modulos devem ter suite de testes em `tests/`
- Configs devem ser adicionadas ao `AletheionV2Config` em `config.py`

### Testes

- Minimo: testes de forward pass (shape correto) + backward (gradientes fluem)
- Para losses: testar valor positivo, gradientes, e edge cases
- Para modulos com estado: testar reset/init
- Nomear como `test_<modulo>.py`

### Commits

- Mensagens concisas em ingles
- Formato: `Add/Fix/Update/Remove <o que>` + descricao se necessario
- Exemplo: `Add plasticity gate module with depletion mechanism`

## Tipos de Contribuicao

### Aceitas

- Correcao de bugs
- Novos modulos epistemicos (com justificativa teorica)
- Melhorias de performance
- Novas configs de escala
- Testes adicionais
- Correcoes de documentacao

### Precisam de Discussao Previa

- Mudancas na arquitetura core (model.py, epistemic_head.py)
- Novos componentes de loss
- Alteracoes no pipeline de treinamento
- Mudancas que afetem compatibilidade de checkpoints

Abra uma issue antes de comecar a trabalhar em mudancas estruturais.

## Reportando Bugs

Abra uma issue com:

1. Versao do Python e PyTorch
2. Sistema operacional
3. Passos para reproduzir
4. Comportamento esperado vs observado
5. Logs relevantes (sem dados sensiveis)

## Duvidas

Abra uma issue com a tag `question` ou entre em contato via
felipemuniz.grsba@gmail.com.
