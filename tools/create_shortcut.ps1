# Creates a "Report Program" shortcut on the Desktop that starts the app.
#   powershell -ExecutionPolicy Bypass -File tools\create_shortcut.ps1
$project = Split-Path -Parent $PSScriptRoot
$desktop = [Environment]::GetFolderPath("Desktop")
$link = Join-Path $desktop "Report Program.lnk"

$shell = New-Object -ComObject WScript.Shell
$s = $shell.CreateShortcut($link)
$s.TargetPath = Join-Path $project "run.bat"
$s.WorkingDirectory = $project
$s.WindowStyle = 7   # start minimised; the server window stays out of the way
$s.Description = "Check, edit and research your report"
$icon = Join-Path $project "static\icon.ico"
if (Test-Path $icon) { $s.IconLocation = "$icon,0" }
$s.Save()
Write-Output "Created $link"
