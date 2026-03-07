# 007 - Licenca AGPL-3.0 + Dual Licensing

**Data:** 2026-03-07

## Resumo

Migrado de licenca proprietaria simples para AGPL-3.0 com licenciamento dual
(open-source + comercial).

## Motivacao

- Permitir que o codigo seja verificado/auditado publicamente
- Proteger contra uso comercial/SaaS nao autorizado (AGPL obriga abrir codigo)
- Habilitar monetizacao via licenca comercial para empresas

## Arquivos Criados/Modificados

### `LICENSE` (substituido)
- Antes: 5 linhas proprietarias ("All rights reserved")
- Agora: Header com copyright + texto completo AGPL-3.0
- Inclui referencia a licenca comercial

### `LICENSE-COMMERCIAL.md` (novo)
- Termos da licenca comercial
- 3 modelos: Startup/Individual, Empresa, OEM/Redistribuicao
- FAQ sobre uso academico, uso interno, e modificacoes
- Contato para licenciamento

### `CLA.md` (novo)
- Contributor License Agreement
- Necessario para manter copyright unificado (pre-requisito de dual licensing)
- Processo de assinatura via Pull Request
- Tabela de contribuidores

### `pyproject.toml` (modificado)
- `license` alterado de `"Proprietary"` para `"AGPL-3.0-or-later"`

## Modelo de Licenciamento

```
Uso open-source (AGPL-3.0)     -> Gratuito, copyleft forte
Uso comercial/SaaS/proprietario -> Licenca comercial (paga)
Contribuicoes externas          -> Requerem CLA assinado
```

## Proximos Passos Sugeridos

- Adicionar header de licenca nos arquivos-fonte (.py)
- Registrar copyright formalmente (opcional, mas recomendado)
- Configurar bot de CLA automatico no GitHub (ex: CLA Assistant)
