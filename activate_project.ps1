$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvActivate = Join-Path $projectRoot ".venv\Scripts\Activate.ps1"
$matplotlibConfig = Join-Path $projectRoot ".matplotlib"
$projectModulePaths = "$projectRoot;$projectRoot\truck_uav_modules"

if (-not (Test-Path -LiteralPath $venvActivate)) {
    throw "Project virtual environment not found: $venvActivate"
}

. $venvActivate
New-Item -ItemType Directory -Path $matplotlibConfig -Force | Out-Null
$env:MPLCONFIGDIR = $matplotlibConfig
$env:PYTHONPATH = if ($env:PYTHONPATH) {
    "$projectModulePaths;$env:PYTHONPATH"
} else {
    $projectModulePaths
}

Write-Host "Dynamic Patrol environment activated: $env:VIRTUAL_ENV"
Write-Host "Python: $((Get-Command python).Source)"
