# Enslaved - encerra somente processos cuja posse pelo Enslaved e comprovada. Chamado por stop.bat.
#  - Backend: processo cujo executavel e o .venv\Scripts\python.exe DESTE projeto rodando
#    "-m backend.main" (com seus filhos).
#  - Janela do start.bat: o cmd.exe pai desse backend, apenas se ele executa o start.bat DESTE projeto.
#  - Ollama: apenas a instancia registrada em .run\ollama.json pelo start.bat, e so se PID, instante de
#    inicio e executavel conferirem (protege contra PID reutilizado).
# Todo o resto (Ollama preexistente ou compartilhado, app da bandeja, backends de outras pastas) e
# preservado e listado. -DryRun lista o que seria feito sem encerrar nada e sem alterar arquivos.
param([switch]$DryRun, [string]$Root)

if (-not $Root) { $Root = $PSScriptRoot }  # no PS 5.1, $PSScriptRoot nem sempre vale como padrao de parametro
$Root = [IO.Path]::GetFullPath($Root)
$venvPython = [IO.Path]::GetFullPath((Join-Path $Root '.venv\Scripts\python.exe'))
$startBat = [IO.Path]::GetFullPath((Join-Path $Root 'start.bat'))
$statePath = Join-Path $Root '.run\ollama.json'
$preserved = 0

function Same([string]$a, [string]$b) { return [string]::Equals($a, $b, [StringComparison]::OrdinalIgnoreCase) }

function Encerrar($id, $label) {
    if ($DryRun) { Write-Host ("  [dry run] would stop {0,-10} PID {1}" -f $label, $id); return }
    # shortcut: taskkill /T apos a verificacao deixa uma janela de milissegundos para reuso de PID; aceitavel aqui.
    taskkill /PID $id /T /F 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) { Write-Host ("  stopped    {0,-10} PID {1}" -f $label, $id) }
    else { Write-Host ("  [warning] failed to stop {0} PID {1} (exit code {2})" -f $label, $id, $LASTEXITCODE) }
}

function Preservar($id, $name, $reason) {
    $script:preserved++
    Write-Host ("  preserved  {0,-16} PID {1}: {2}" -f $name, $id, $reason)
}

function PortaAberta($ip, $port) {
    $c = New-Object System.Net.Sockets.TcpClient
    try { return $c.ConnectAsync($ip, $port).Wait(700) -and $c.Connected } catch { return $false } finally { $c.Close() }
}

if ($DryRun) { Write-Host '  (dry run: nothing will be stopped or changed)' }
$procs = @(Get-CimInstance Win32_Process)

# --- Backend e janela do start.bat -------------------------------------------------------------
$backends = @($procs | Where-Object {
    $_.ExecutablePath -and (Same $_.ExecutablePath $venvPython) -and $_.CommandLine -match '(^|\s)-m\s+backend\.main(\s|$)'
})
$ownedIds = @($backends | ForEach-Object { $_.ProcessId })
if (-not $backends) { Write-Host '  no backend from this project is running' }
foreach ($b in $backends) {
    $parent = $procs | Where-Object { $_.ProcessId -eq $b.ParentProcessId } | Select-Object -First 1
    $isOurWindow = $parent -and $parent.Name -eq 'cmd.exe' -and $parent.CreationDate -le $b.CreationDate -and
        $parent.CommandLine -and $parent.CommandLine.IndexOf($startBat, [StringComparison]::OrdinalIgnoreCase) -ge 0
    if ($isOurWindow) { Encerrar $parent.ProcessId 'start.bat' } else { Encerrar $b.ProcessId 'backend' }
}
# Outros "backend.main" (outra pasta/ambiente) nunca sao tocados; filhos do nosso backend saem com /T.
$others = @($procs | Where-Object {
    $_.Name -match '^pythonw?\.exe$' -and $_.CommandLine -match 'backend\.main' -and
    $ownedIds -notcontains $_.ProcessId -and $ownedIds -notcontains $_.ParentProcessId
})
$otherIds = @($others | ForEach-Object { $_.ProcessId })
foreach ($p in $others) {
    # o filho do lancador do venv e o mesmo backend: lista so o pai
    if ($otherIds -notcontains $p.ParentProcessId) { Preservar $p.ProcessId $p.Name 'backend.main from another environment/project' }
}

# --- Ollama ------------------------------------------------------------------------------------
$ownedOllama = $null
if (-not (Test-Path $statePath)) {
    Write-Host '  no Ollama started by Enslaved is recorded'
} else {
    $state = $null
    try { $state = Get-Content -Raw -Path $statePath | ConvertFrom-Json } catch { }
    $proc = $null
    if ($state -and $state.pid) { $proc = Get-Process -Id ([int]$state.pid) -ErrorAction SilentlyContinue }
    $ticks = $null; $path = $null
    if ($proc) { try { $ticks = $proc.StartTime.ToUniversalTime().Ticks; $path = $proc.Path } catch { } }
    $stale = $true
    if (-not $state -or -not $state.pid) {
        Write-Host '  Ollama record is unreadable; nothing will be stopped'
    } elseif (-not $proc) {
        Write-Host "  recorded Ollama (PID $($state.pid)) is no longer running"
    } elseif ($null -eq $ticks -or -not $path) {
        $stale = $false
        Preservar $proc.Id $proc.ProcessName 'cannot prove ownership (start time/executable unreadable)'
    } elseif ([math]::Abs($ticks - [long]$state.startTimeTicks) -gt 10000 -or -not (Same $path $state.path)) {
        Preservar $proc.Id $proc.ProcessName 'PID reused by another process; not the Ollama started by Enslaved'
    } else {
        $stale = $false
        $ownedOllama = $proc.Id
        Encerrar $proc.Id 'ollama'
    }
    if (-not $DryRun -and ($stale -or $ownedOllama)) { Remove-Item -Force -Path $statePath -ErrorAction SilentlyContinue }
}
foreach ($p in $procs) {
    if ($p.Name -match '^ollama' -and $p.ProcessId -ne $ownedOllama -and $p.ParentProcessId -ne $ownedOllama) {
        Preservar $p.ProcessId $p.Name 'Ollama not started by Enslaved (pre-existing or shared)'
    }
}
if ($preserved) { Write-Host '  Preserved processes were not touched. Close them manually if you want.' }

if ($DryRun) { return }

# --- Conferencia final -------------------------------------------------------------------------
Start-Sleep -Seconds 1
$ips = @('127.0.0.1') + @(
    Get-NetIPConfiguration -ErrorAction SilentlyContinue |
        Where-Object { $_.IPv4DefaultGateway -and $_.NetAdapter.Status -eq 'Up' } |
        ForEach-Object { $_.IPv4Address.IPAddress })
$abertos = @($ips | Where-Object { PortaAberta $_ 8000 })
if ($abertos) { Write-Host ('  [warning] port 8000 still answers on: ' + ($abertos -join ', ') + ' (process not proven to belong to this project)') }
else { Write-Host '  port 8000 closed (this PC and local network)' }
if ($ownedOllama -and (PortaAberta '127.0.0.1' 11434)) { Write-Host '  [warning] port 11434 still answers (another Ollama running?)' }
