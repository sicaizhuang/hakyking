param(
    [switch]$SkipQa,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand) { $python = $pythonCommand.Source }
}
$spec = Join-Path $root "Hakyking.spec"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python not found. Install Python 3.11+ and add it to PATH."
}

Push-Location $root
try {
    if ($Clean) {
        Remove-Item -LiteralPath (Join-Path $root "build") -Recurse -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath (Join-Path $root "dist\Hakyking") -Recurse -Force -ErrorAction SilentlyContinue
    }

    & $python -m compileall -q hakyking dev_tools
    if (-not $SkipQa) {
        & $python (Join-Path $root "dev_tools\qa_whitepaper_v1.py")
    }

    & $python -m PyInstaller --noconfirm --clean $spec
    Write-Host ""
    Write-Host "Built: $root\dist\Hakyking\Hakyking.exe"
    Write-Host "Note: edit source files first, then rerun this script to rebuild the exe."
}
finally {
    Pop-Location
}
