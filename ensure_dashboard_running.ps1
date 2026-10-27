# Keepalive for the MLB DFS dashboard.
# Probes the Streamlit health endpoint and starts start_dashboard_detached.bat
# only if it is not answering. Safe to run on a schedule -- it never starts a
# second copy while one is healthy.

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$log  = Join-Path $root 'dashboard_keepalive.log'

function Write-Log($msg) {
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $msg" | Add-Content -Path $log -Encoding utf8
}

function Test-Healthy {
    try {
        $r = Invoke-WebRequest -Uri 'http://localhost:8600/_stcore/health' -UseBasicParsing -TimeoutSec 5
        return ($r.Content -match 'ok')
    } catch {
        return $false
    }
}

if (Test-Healthy) { exit 0 }

Write-Log 'health check failed - starting dashboard'
Start-Process -FilePath (Join-Path $root 'start_dashboard_detached.bat') -WorkingDirectory $root -WindowStyle Hidden

for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 2
    if (Test-Healthy) { Write-Log "started ok after $(($i+1)*2)s"; exit 0 }
}

Write-Log 'FAILED to come up within 60s - see dashboard_server.log'
exit 1
