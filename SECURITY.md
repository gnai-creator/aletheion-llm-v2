# Politica de Seguranca

## Versoes Suportadas

| Versao | Suporte |
|--------|---------|
| 0.1.x  | Sim     |

## Reportando Vulnerabilidades

**NAO abra issues publicas para vulnerabilidades de seguranca.**

Se voce descobrir uma vulnerabilidade de seguranca, por favor reporte
de forma responsavel:

1. **E-mail**: felipemuniz.grsba@gmail.com
2. **Assunto**: `[SECURITY] Aletheion LLM v2 - <breve descricao>`
3. **Inclua**:
   - Descricao da vulnerabilidade
   - Passos para reproduzir
   - Impacto potencial
   - Sugestao de correcao (se tiver)

## Prazo de Resposta

- **Confirmacao de recebimento**: ate 48 horas
- **Avaliacao inicial**: ate 7 dias
- **Correcao (se aplicavel)**: ate 30 dias para severidade alta

## Escopo

### Incluido

- Codigo-fonte em `src/aletheion_v2/`
- Scripts de treinamento e inferencia
- Pipeline de dados
- Dependencias diretas

### Nao Incluido

- Modelos treinados (checkpoints) - sao dados, nao codigo
- Infraestrutura de terceiros (PyTorch, CUDA, etc.)
- Uso indevido intencional do modelo (adversarial prompts, etc.)

## Consideracoes de Seguranca do Modelo

O Aletheion LLM v2 e um modelo de linguagem generativo. Como todo LLM:

- Pode gerar conteudo incorreto, enviesado ou prejudicial
- Nao deve ser usado como fonte unica de verdade
- O sistema epistemico (Q1/Q2, confidence) fornece indicadores de
  incerteza, mas nao garante corretude
- Outputs do modelo nao devem ser usados diretamente para decisoes
  criticas sem revisao humana

## Divulgacao

Apos correcao, a vulnerabilidade sera divulgada publicamente com credito
ao reporter (salvo se preferir anonimato). Seguimos o modelo de
divulgacao responsavel com prazo coordenado.
