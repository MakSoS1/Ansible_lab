param(
    [string]$ProjectRoot = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

$mobileDir = Join-Path $ProjectRoot "mobile"
$vendorDir = Join-Path $mobileDir "vendor"
$poseDir = Join-Path $vendorDir "pose"
$tmpDir = Join-Path $ProjectRoot "_tmp_pose_vendor"
$poseVersion = "0.5.1675469404"

if (!(Test-Path $mobileDir)) {
    throw "mobile directory not found: $mobileDir"
}

if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    throw "Node.js is required. Install Node.js and run again."
}

if (Test-Path $tmpDir) {
    Remove-Item -Recurse -Force $tmpDir
}
New-Item -ItemType Directory -Path $tmpDir | Out-Null

Push-Location $tmpDir
try {
    npm init -y | Out-Null
    npm i "@mediapipe/pose@$poseVersion" --silent
} finally {
    Pop-Location
}

if (!(Test-Path $vendorDir)) {
    New-Item -ItemType Directory -Path $vendorDir | Out-Null
}
if (Test-Path $poseDir) {
    Remove-Item -Recurse -Force $poseDir
}
Copy-Item -Recurse -Force (Join-Path $tmpDir "node_modules\@mediapipe\pose") $poseDir
Remove-Item -Recurse -Force $tmpDir

Write-Host "MediaPipe vendor installed:"
Write-Host $poseDir
