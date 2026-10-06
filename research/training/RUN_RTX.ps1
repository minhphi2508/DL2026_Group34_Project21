param(
    [string]$Python='python',
    [string]$DataRoot=(Join-Path (Split-Path $PSScriptRoot -Parent) 'data'),
    [string]$RunName='candidate_v3_pair_replay',
    [switch]$DryRun,
    [switch]$Resume
)
# The coordinator's fresh-machine bootstrap supplies the validated Python3.12 /
# torch2.8-cu128 environment. This launcher never modifies another environment.
$ErrorActionPreference='Stop'
$taskArgs=@((Join-Path $PSScriptRoot 'run_rtx.py'),'--data-root',$DataRoot,'--run-name',$RunName)
if($DryRun){$taskArgs+='--dry-run'}
if($Resume){$taskArgs+='--resume'}
& $Python @taskArgs
if($LASTEXITCODE -ne 0){throw 'Research run failed; send RETURN_LIGHT and preserve BACKUP_FULL_RESUME. Do not accept candidate.'}
