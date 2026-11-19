# Enslaved - inicia "ollama serve" e registra a posse do processo. Chamado por start.bat.
# O registro (.run\ollama.json) guarda PID, instante exato de inicio e executavel. O stop.ps1 so
# encerra um Ollama se os tres conferirem; assim um PID reutilizado ou um Ollama aberto por outro
# programa nunca e confundido com o do Enslaved.
param(
    [Parameter(Mandatory = $true)][string]$Exe,
    [string[]]$ArgumentList = @('serve'),
    [string]$Root
)
$ErrorActionPreference = 'Stop'
# No PowerShell 5.1, $PSScriptRoot chega vazio como padrao de parametro neste tipo de script: resolver aqui.
if (-not $Root) { $Root = $PSScriptRoot }

# Tudo que pode falhar antes de iniciar e resolvido primeiro: nunca deixar um Ollama sem registro.
try {
    $cmd = Get-Command -Name $Exe -CommandType Application | Select-Object -First 1
    $exePath = [IO.Path]::GetFullPath($cmd.Source)
    $stateDir = Join-Path $Root '.run'
    $statePath = Join-Path $stateDir 'ollama.json'
    New-Item -ItemType Directory -Force -Path $stateDir | Out-Null
} catch {
    Write-Host "[Enslaved] Ollama NOT started: $($_.Exception.Message)"
    exit 1
}

try {
    $p = Start-Process -FilePath $exePath -ArgumentList $ArgumentList -WindowStyle Minimized -PassThru
} catch {
    Write-Host "[Enslaved] Could not start Ollama: $($_.Exception.Message)"
    exit 1
}

try {
    # O caminho vem do Get-Command: logo apos o Start-Process, Process.Path pode ainda estar vazio.
    $info = [ordered]@{
        pid            = $p.Id
        startTimeTicks = $p.StartTime.ToUniversalTime().Ticks
        path           = $exePath
        startedBy      = 'Enslaved start.bat'
    }
    $tmp = "$statePath.tmp"
    ($info | ConvertTo-Json) | Set-Content -Path $tmp -Encoding ASCII
    Move-Item -Force -Path $tmp -Destination $statePath
} catch {
    Write-Host "[Enslaved] Ollama started (PID $($p.Id)), but recording it failed: $($_.Exception.Message)"
    Write-Host "            Without a record, stop.bat will leave it running. Close it manually."
    exit 1
}
Write-Host "[Enslaved] Ollama started by Enslaved (PID $($p.Id)). stop.bat will close only this instance."
