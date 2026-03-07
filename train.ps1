# ============================================================
# AletheionV2 - Treinamento 350M RTX 4090
# PowerShell script para Windows 11
# ============================================================
#
# Uso:
#   .\train.ps1                    # Setup completo interativo
#   .\train.ps1 -Step setup        # So instala dependencias
#   .\train.ps1 -Step data         # So prepara dados
#   .\train.ps1 -Step train        # So treina
#   .\train.ps1 -TestData          # Usa TinyStories (teste rapido)
#   .\train.ps1 -Resume path.pt    # Resume de checkpoint
# ============================================================

param(
    [ValidateSet("all", "setup", "data", "train")]
    [string]$Step = "all",

    [switch]$TestData,
    [string]$Resume = "",
    [string]$DataDir = ""
)

$ErrorActionPreference = "Continue"
$ROOT = Split-Path -Parent $PSScriptRoot
if (-not $ROOT) { $ROOT = (Get-Location).Path }
# Se o script esta na raiz, ROOT = diretorio atual
if (Test-Path "$ROOT\train.ps1") {
    # Script esta na raiz
} elseif (Test-Path "$ROOT\scripts\setup_and_train.py") {
    # OK
} else {
    $ROOT = (Get-Location).Path
}

$CONFIG = "$ROOT\configs\scaling\350m_rtx4090.yaml"
$SCRIPTS = "$ROOT\scripts"
$PYTHON = "python"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  AletheionV2 - Treinamento 350M RTX 4090" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# -------------------------------------------------------
# STEP 1: Verificar CUDA
# -------------------------------------------------------
function Test-CudaAvailable {
    try {
        $output = & $PYTHON -c "import torch; print(torch.cuda.is_available())" 2>&1
        return $output.Trim() -eq "True"
    } catch {
        return $false
    }
}

function Show-GpuInfo {
    & $PYTHON -c @"
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f'  GPU: {name} ({mem:.0f} GB VRAM)')
    print(f'  torch: {torch.__version__}')
    print(f'  CUDA: {torch.version.cuda}')
else:
    print(f'  torch: {torch.__version__} (CPU only)')
"@
}

function Install-CudaTorch {
    Write-Host ""
    Write-Host "[SETUP] Instalando torch com CUDA 12.4..." -ForegroundColor Yellow
    Write-Host "[SETUP] Isso pode demorar alguns minutos." -ForegroundColor Yellow
    Write-Host ""

    & $PYTHON -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERRO] Falha ao instalar torch CUDA." -ForegroundColor Red
        Write-Host "[ERRO] Tente manualmente:" -ForegroundColor Red
        Write-Host "  pip install torch --index-url https://download.pytorch.org/whl/cu124"
        return $false
    }

    # Reinstala projeto
    & $PYTHON -m pip install -e "$ROOT" 2>$null
    return $true
}

if ($Step -eq "all" -or $Step -eq "setup") {
    Write-Host "--- Step 1: Verificando CUDA ---" -ForegroundColor Green

    if (Test-CudaAvailable) {
        Write-Host "[OK] CUDA disponivel" -ForegroundColor Green
        Show-GpuInfo
    } else {
        Write-Host "[WARN] torch CUDA nao disponivel" -ForegroundColor Yellow
        Show-GpuInfo

        $resp = Read-Host "Instalar torch com CUDA 12.4? [S/n]"
        if ($resp -eq "" -or $resp -match "^[SsYy]") {
            $ok = Install-CudaTorch
            if (-not $ok) { exit 1 }

            if (Test-CudaAvailable) {
                Write-Host "[OK] CUDA instalado com sucesso!" -ForegroundColor Green
                Show-GpuInfo
            } else {
                Write-Host "[ERRO] CUDA nao detectado apos instalacao." -ForegroundColor Red
                Write-Host "[ERRO] Verifique se os drivers NVIDIA estao atualizados." -ForegroundColor Red
                Write-Host "[ERRO] Download: https://www.nvidia.com/drivers" -ForegroundColor Red
                exit 1
            }
        } else {
            Write-Host "[INFO] Continuando sem CUDA (treinamento sera CPU - MUITO LENTO)" -ForegroundColor Yellow
        }
    }

    # Verifica dependencias de dados
    Write-Host ""
    Write-Host "--- Step 2: Verificando dependencias ---" -ForegroundColor Green
    & $PYTHON -m pip install tiktoken datasets -q 2>$null
    Write-Host "[OK] tiktoken + datasets instalados" -ForegroundColor Green

    # Instala projeto
    & $PYTHON -m pip install -e "$ROOT" -q 2>$null
    Write-Host "[OK] aletheion-v2 instalado" -ForegroundColor Green
}

if ($Step -eq "setup") {
    Write-Host ""
    Write-Host "[DONE] Setup completo. Execute '.\train.ps1 -Step data' para preparar dados." -ForegroundColor Cyan
    exit 0
}

# -------------------------------------------------------
# STEP 2: Preparar dados
# -------------------------------------------------------
if ($Step -eq "all" -or $Step -eq "data") {
    Write-Host ""
    Write-Host "--- Step 3: Preparando dados ---" -ForegroundColor Green

    if ($TestData) {
        $targetDir = "$ROOT\data\350m_test"
        Write-Host "[DATA] Preparando TinyStories (teste rapido, ~500M tokens)..." -ForegroundColor Yellow
        & $PYTHON "$SCRIPTS\prepare_data.py" `
            --dataset tinystories `
            --output $targetDir `
            --max-tokens 500000000

        if ($LASTEXITCODE -ne 0) {
            Write-Host "[ERRO] Falha na preparacao de dados" -ForegroundColor Red
            exit 1
        }
        $DataDir = $targetDir
    } else {
        $targetDir = "$ROOT\data\350m"

        # Verifica se dados ja existem
        $metaFile = "$targetDir\metadata.json"
        $hasData = $false

        if (Test-Path $metaFile) {
            $binFiles = Get-ChildItem "$targetDir\*.bin" -ErrorAction SilentlyContinue |
                Where-Object { $_.Length -gt 1MB }
            if ($binFiles.Count -gt 0) {
                $meta = Get-Content $metaFile | ConvertFrom-Json
                Write-Host "[OK] Dados encontrados: $($meta.total_tokens.ToString('N0')) tokens, $($binFiles.Count) shards" -ForegroundColor Green
                $hasData = $true
                $DataDir = $targetDir
            }
        }

        if (-not $hasData) {
            Write-Host "[DATA] Dados nao encontrados em $targetDir" -ForegroundColor Yellow
            Write-Host ""
            Write-Host "  1. Preparar fineweb-edu (7B tokens, ~horas de download)"
            Write-Host "  2. Preparar TinyStories (500M tokens, ~minutos)"
            Write-Host "  3. Pular"
            Write-Host ""
            $resp = Read-Host "Escolha [1/2/3]"

            switch ($resp) {
                "1" {
                    Write-Host ""
                    Write-Host "[DATA] Preparando fineweb-edu/sample-10BT..." -ForegroundColor Yellow
                    Write-Host "[DATA] Output: $targetDir" -ForegroundColor Yellow
                    Write-Host "[DATA] Max tokens: 7,000,000,000" -ForegroundColor Yellow
                    Write-Host ""
                    & $PYTHON "$SCRIPTS\prepare_data.py" `
                        --dataset fineweb-edu `
                        --subset sample-10BT `
                        --output $targetDir `
                        --max-tokens 7000000000
                    $DataDir = $targetDir
                }
                "2" {
                    $testDir = "$ROOT\data\350m_test"
                    & $PYTHON "$SCRIPTS\prepare_data.py" `
                        --dataset tinystories `
                        --output $testDir `
                        --max-tokens 500000000
                    $DataDir = $testDir
                }
                default {
                    Write-Host "[INFO] Pulando preparacao de dados" -ForegroundColor Yellow
                }
            }
        }
    }
}

if ($Step -eq "data") {
    Write-Host ""
    Write-Host "[DONE] Dados preparados. Execute '.\train.ps1 -Step train' para treinar." -ForegroundColor Cyan
    exit 0
}

# -------------------------------------------------------
# STEP 3: Treinar
# -------------------------------------------------------
if ($Step -eq "all" -or $Step -eq "train") {
    Write-Host ""
    Write-Host "--- Step 4: Treinamento ---" -ForegroundColor Green

    # Determina diretorio de dados
    if (-not $DataDir) {
        # Tenta em ordem: 350m, 350m_test, 125m, 50m
        $candidates = @(
            "$ROOT\data\350m",
            "$ROOT\data\350m_test",
            "$ROOT\data\125m",
            "$ROOT\data\50m"
        )
        foreach ($dir in $candidates) {
            if (Test-Path "$dir\metadata.json") {
                $bins = Get-ChildItem "$dir\*.bin" -ErrorAction SilentlyContinue |
                    Where-Object { $_.Length -gt 1MB }
                if ($bins.Count -gt 0) {
                    $DataDir = $dir
                    break
                }
            }
        }
    }

    if (-not $DataDir) {
        Write-Host "[ERRO] Nenhum dado encontrado. Execute '.\train.ps1 -Step data' primeiro." -ForegroundColor Red
        exit 1
    }

    Write-Host "[TRAIN] Config: $CONFIG" -ForegroundColor Cyan
    Write-Host "[TRAIN] Data:   $DataDir" -ForegroundColor Cyan
    Write-Host "[TRAIN] Output: checkpoints\350m_rtx4090\" -ForegroundColor Cyan
    Write-Host ""

    $trainArgs = @(
        "$SCRIPTS\train_distributed.py",
        "--config", $CONFIG,
        "--data-dir", $DataDir
    )

    if ($Resume) {
        $trainArgs += @("--resume", $Resume)
        Write-Host "[TRAIN] Resuming de: $Resume" -ForegroundColor Yellow
    }

    & $PYTHON @trainArgs

    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "[DONE] Treinamento completo!" -ForegroundColor Green
        Write-Host "[DONE] Checkpoints em: checkpoints\350m_rtx4090\" -ForegroundColor Green

        # Gera plots se possivel
        $logFile = "$ROOT\checkpoints\350m_rtx4090\training_log.json"
        if (Test-Path $logFile) {
            Write-Host ""
            Write-Host "[PLOT] Gerando visualizacoes..." -ForegroundColor Yellow
            & $PYTHON "$SCRIPTS\plot_training.py" --log $logFile --output "$ROOT\plots\350m_rtx4090"
        }
    } else {
        Write-Host ""
        Write-Host "[ERRO] Treinamento falhou (exit code: $LASTEXITCODE)" -ForegroundColor Red
    }
}
