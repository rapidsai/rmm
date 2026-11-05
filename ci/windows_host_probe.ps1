# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# TEMP: probe a native Windows GPU runner's memory settings and CUDA allocation ceiling (rapidsai/rmm#2339).
$ErrorActionPreference = "Continue"

Write-Host "== Windows host memory"
Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory, AutomaticManagedPagefile, Model, Manufacturer, HypervisorPresent | Format-List
Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, TotalVisibleMemorySize, FreePhysicalMemory, TotalVirtualMemorySize, FreeVirtualMemory, SizeStoredInPagingFiles | Format-List
Get-CimInstance Win32_PageFileUsage | Select-Object Name, AllocatedBaseSize, CurrentUsage, PeakUsage | Format-List
Get-CimInstance Win32_PageFileSetting | Format-List
(Get-Counter '\Memory\Committed Bytes', '\Memory\Commit Limit', '\Memory\Available MBytes').CounterSamples | Select-Object Path, CookedValue | Format-List
Get-CimInstance Win32_VideoController | Select-Object Name, AdapterRAM, DriverVersion | Format-List

Write-Host "== WSL"
wsl.exe --version 2>&1
wsl.exe --list --verbose 2>&1
Get-Content "$env:USERPROFILE\.wslconfig" -ErrorAction SilentlyContinue
Get-ChildItem C:\Users -Directory -ErrorAction SilentlyContinue | ForEach-Object { $f = Join-Path $_.FullName ".wslconfig"; if (Test-Path $f) { Write-Host "-- $f"; Get-Content $f } }

Write-Host "== nvidia-smi"
nvidia-smi
nvidia-smi -q | Select-String -Pattern "Driver Version|CUDA Version|Product Name|Driver Model|EGM|Addressing Mode|Virtualization Mode|Total|Reserved|BAR1"

Write-Host "== GPU adapter memory (WDDM)"
Get-CimInstance -Namespace root\cimv2 -ClassName Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory -ErrorAction SilentlyContinue | Format-List
(Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage', '\GPU Adapter Memory(*)\Shared Usage' -ErrorAction SilentlyContinue).CounterSamples | Select-Object Path, CookedValue | Format-List
