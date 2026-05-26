# Configure npm for this PowerShell session only.
# Run this before npm install if npm reports:
# "cache mode is 'only-if-cached' but no cached response is available".

$env:NPM_CONFIG_OFFLINE = "false"
$env:npm_config_offline = "false"

$proxyVars = @(
  "HTTP_PROXY",
  "HTTPS_PROXY",
  "ALL_PROXY",
  "http_proxy",
  "https_proxy",
  "all_proxy"
)

foreach ($name in $proxyVars) {
  Remove-Item "Env:\$name" -ErrorAction SilentlyContinue
}

Write-Host "NPM_CONFIG_OFFLINE=$env:NPM_CONFIG_OFFLINE"
Write-Host "HTTP/HTTPS proxy environment variables cleared for this session."
Write-Host "npm cache directory:"
npm config get cache
Write-Host "npm offline setting:"
npm config get offline
