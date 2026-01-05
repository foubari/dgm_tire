# GPU Monitoring Script for Training
# Run this in a separate terminal while training runs

Write-Host "=== GPU Monitoring (Press Ctrl+C to stop) ===" -ForegroundColor Cyan
Write-Host ""

while ($true) {
    Clear-Host
    Write-Host "=== GPU STATUS $(Get-Date -Format 'HH:mm:ss') ===" -ForegroundColor Green
    Write-Host ""

    # NVIDIA GPU
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv
        Write-Host ""
        Write-Host "Temperature Guide:" -ForegroundColor Yellow
        Write-Host "  60-75C: Perfect" -ForegroundColor Green
        Write-Host "  75-85C: Normal but warm" -ForegroundColor Yellow
        Write-Host "  85-90C: Hot, check cooling" -ForegroundColor DarkYellow
        Write-Host "  >90C: Too hot, stop training!" -ForegroundColor Red
    } else {
        Write-Host "nvidia-smi not found. Install NVIDIA drivers to monitor GPU." -ForegroundColor Red
    }

    Write-Host ""
    Write-Host "Disk Space:" -ForegroundColor Cyan
    Get-PSDrive C | Select-Object @{Name="Drive";Expression={$_.Name}}, @{Name="Used (GB)";Expression={[math]::Round($_.Used/1GB,2)}}, @{Name="Free (GB)";Expression={[math]::Round($_.Free/1GB,2)}}, @{Name="Total (GB)";Expression={[math]::Round(($_.Used+$_.Free)/1GB,2)}}

    Start-Sleep -Seconds 5
}
