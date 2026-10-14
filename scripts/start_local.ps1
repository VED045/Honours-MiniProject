param([switch]$Install)
$ErrorActionPreference="Stop"
$project=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $project
$python=Join-Path $project ".venv\Scripts\python.exe"
if(!(Test-Path -LiteralPath $python)){throw "Create .venv and install requirements first (see README.md)."}
if($Install){& $python -m pip install -r requirements.txt; & $python -m pip install -r requirements-torch.txt; npm ci --prefix frontend}
# Background helpers are hidden; logs stay in ignored project work/.
New-Item -ItemType Directory -Path (Join-Path $project "work") -Force | Out-Null
Start-Process -FilePath $python -ArgumentList @("-m","uvicorn","backend.app:app","--host","127.0.0.1","--port","8000") -WorkingDirectory $project -WindowStyle Hidden -RedirectStandardOutput (Join-Path $project "work/api.log") -RedirectStandardError (Join-Path $project "work/api-error.log")
Start-Process -FilePath $python -ArgumentList @("-m","backend.worker") -WorkingDirectory $project -WindowStyle Hidden -RedirectStandardOutput (Join-Path $project "work/worker.log") -RedirectStandardError (Join-Path $project "work/worker-error.log")
$node=(Get-Command node).Source
Start-Process -FilePath $node -ArgumentList @("node_modules/next/dist/bin/next","dev","--webpack","--hostname","127.0.0.1") -WorkingDirectory (Join-Path $project "frontend") -WindowStyle Hidden -RedirectStandardOutput (Join-Path $project "work/frontend.log") -RedirectStandardError (Join-Path $project "work/frontend-error.log")
Write-Output "Started API, worker and dashboard. Open http://localhost:3000. Logs: project/work."
