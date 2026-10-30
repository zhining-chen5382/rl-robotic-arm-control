param([string]$Python = 'python', [int]$Port = 5000)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONUTF8 = '1'
$env:IK_PORT = "$Port"
& $Python deployment.py
if ($LASTEXITCODE -ne 0) { throw '模型或依赖检查失败。请使用安装了 requirements.txt 的 Python。' }
& $Python app.py
