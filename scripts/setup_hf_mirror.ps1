# Configure Hugging Face mirror settings for the current PowerShell session only.
# Keep using the same PowerShell terminal after running this script, then run:
#   python scripts\test_bge_download.py
# or start the Gradio demo manually.

$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HOME = "$PWD\.hf_cache"
Remove-Item Env:\HF_HUB_OFFLINE -ErrorAction SilentlyContinue

# Clear invalid proxy variables for the current PowerShell session.
# Some local environments set these to 127.0.0.1:9, which breaks Hugging Face downloads.
Remove-Item Env:\HTTP_PROXY -ErrorAction SilentlyContinue
Remove-Item Env:\HTTPS_PROXY -ErrorAction SilentlyContinue
Remove-Item Env:\ALL_PROXY -ErrorAction SilentlyContinue
Remove-Item Env:\http_proxy -ErrorAction SilentlyContinue
Remove-Item Env:\https_proxy -ErrorAction SilentlyContinue
Remove-Item Env:\all_proxy -ErrorAction SilentlyContinue
Remove-Item Env:\GIT_HTTP_PROXY -ErrorAction SilentlyContinue
Remove-Item Env:\GIT_HTTPS_PROXY -ErrorAction SilentlyContinue
$env:HF_HUB_DISABLE_XET = "1"

Write-Host "HF mirror configured for this PowerShell session."
Write-Host "HF_ENDPOINT=$env:HF_ENDPOINT"
Write-Host "HF_HOME=$env:HF_HOME"
Write-Host "HF_HUB_OFFLINE=$env:HF_HUB_OFFLINE"
Write-Host "HF_HUB_DISABLE_XET=$env:HF_HUB_DISABLE_XET"
Write-Host "HTTP_PROXY=$env:HTTP_PROXY"
Write-Host "HTTPS_PROXY=$env:HTTPS_PROXY"
Write-Host "ALL_PROXY=$env:ALL_PROXY"
