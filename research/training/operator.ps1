param(
    [Parameter(Mandatory=$true)][string]$DataRoot,
    [ValidateSet('preflight','smoke','train','evaluate')][string]$Action='preflight',
    [ValidateSet('cpu','cuda')][string]$Device='cpu',
    [string]$Python='python',
    [string]$RunName='candidate_v3_pair_replay',
    [switch]$Resume
)
$ErrorActionPreference='Stop'
$taskArgs=@((Join-Path $PSScriptRoot 'runner.py'),$Action,'--data-root',$DataRoot,'--run-name',$RunName,'--device',$Device)
if($Resume){$taskArgs+='--resume'}
& $Python @taskArgs
if($LASTEXITCODE -ne 0){throw "Experiment action failed ($LASTEXITCODE); no acceptance or promotion."}
