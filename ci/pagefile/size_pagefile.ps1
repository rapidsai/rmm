# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# TEMP (rapidsai/rmm#2339): size C:\pagefile.sys's maximum to the least value that lets all GPU memory be
# allocated while host memory is full, with every input measured on this machine.
#
# Every WDDM/MCDM GPU allocation is charged 1:1 against Windows commit, and the commit limit is
# RAM + pagefile, so the pagefile must hold whatever part of VRAM doesn't fit in RAM left over after
# the host's own commit:
#   windows: MaximumSize = VRAM + headroom
#            (all of Windows' system memory may be in use, so RAM backs none of VRAM)
#   wsl:     MaximumSize = VRAM - (RAM - WSL_memory - Windows_baseline - headroom)
#            (the WSL VM may be full; the rest of Windows stays at its baseline)
# where
#   VRAM             = sum over GPUs of nvidia-smi memory.total - memory.reserved
#   RAM              = Win32_ComputerSystem.TotalPhysicalMemory
#   WSL_memory       = the WSL VM's size: .wslconfig memory=, else 50% of RAM (or the VM's MemTotal if larger)
#   Windows_baseline = Committed Bytes now - vmmemWSL private bytes (Windows' commit outside the VM; the VM
#                      returns freed guest pages to Windows lazily, so its commit is read from the Windows side)
# The result is never below the current pagefile size.
param(
    [Parameter(Mandatory = $true)][ValidateSet("windows", "wsl")][string]$Mode,
    [double]$HeadroomGiB = 1.0,
    [long]$WslMemTotalBytes = 0,
    [switch]$Apply
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$GiB = [double](1L -shl 30)
$MiB = [double](1L -shl 20)

function Get-Commit {
    $s = (Get-Counter '\Memory\Committed Bytes', '\Memory\Commit Limit').CounterSamples
    [pscustomobject]@{ Committed = [double]$s[0].CookedValue; Limit = [double]$s[1].CookedValue }
}

function Get-WslVmBytes([double]$ram) {
    # WSL2's VM size is .wslconfig's [wsl2] memory=, else 50% of Windows RAM.
    $cfg = @(Get-ChildItem C:\Users -Directory -ErrorAction SilentlyContinue |
            ForEach-Object { Join-Path $_.FullName ".wslconfig" } | Where-Object { Test-Path $_ })
    foreach ($f in $cfg) {
        $m = Select-String -Path $f -Pattern '^\s*memory\s*=\s*([0-9.]+)\s*([KMGT]?B?)\s*$' | Select-Object -First 1
        if ($m) {
            $n = [double]$m.Matches[0].Groups[1].Value
            $u = $m.Matches[0].Groups[2].Value.ToUpper().TrimEnd("B")
            $mult = @{ "" = 1.0; "K" = 1KB; "M" = 1MB; "G" = 1GB; "T" = 1TB }[$u]
            return [pscustomobject]@{ Bytes = $n * $mult; Source = "$f" }
        }
    }
    return [pscustomobject]@{ Bytes = $ram / 2; Source = "default (50% of RAM)" }
}

$cs = Get-CimInstance Win32_ComputerSystem
$os = Get-CimInstance Win32_OperatingSystem
$ram = [double]$cs.TotalPhysicalMemory
$ramAvail = [double]$os.FreePhysicalMemory * 1KB
$pfUsage = @(Get-CimInstance Win32_PageFileUsage)
$pfNow = [double](($pfUsage | Measure-Object AllocatedBaseSize -Sum).Sum) * $MiB
$commit = Get-Commit

$gpuRows = @(& nvidia-smi --query-gpu=name,memory.total,memory.reserved --format=csv,noheader,nounits)
$gpu = 0.0
$gpuNames = @()
foreach ($row in $gpuRows) {
    $p = $row -split ',\s*'
    $reserved = 0.0
    [double]::TryParse($p[2], [ref]$reserved) | Out-Null
    $gpu += ([double]$p[1] - $reserved) * $MiB
    $gpuNames += $p[0]
}

$disk = Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='C:'"
$vram = $gpu
$headroom = $HeadroomGiB * $GiB
$vm = $null
$wslMemory = 0.0
$windowsBaseline = 0.0

if ($Mode -eq "windows") {
    $maximumSize = $vram + $headroom
} else {
    if ($WslMemTotalBytes -le 0) { throw "wsl mode needs -WslMemTotalBytes from the WSL VM's /proc/meminfo" }
    $vm = Get-WslVmBytes $ram
    $wslMemory = [math]::Max($vm.Bytes, [double]$WslMemTotalBytes)
    $vmProc = @(Get-Process -Name vmmemWSL, vmmem -ErrorAction SilentlyContinue)
    if ($vmProc.Count -eq 0) { throw "no vmmemWSL/vmmem process found" }
    $vmCommit = [double](($vmProc | Measure-Object PrivateMemorySize64 -Sum).Sum)
    $windowsBaseline = $commit.Committed - $vmCommit
    $maximumSize = $vram - ($ram - $wslMemory - $windowsBaseline - $headroom)
}

$pfMax = [math]::Max($pfNow, [math]::Ceiling($maximumSize / $MiB) * $MiB)
$growth = $pfMax - $pfNow

$result = [ordered]@{
    mode                  = $Mode
    gpus                  = ($gpuNames -join "; ")
    VRAM_gib              = [math]::Round($vram / $GiB, 3)
    RAM_gib               = [math]::Round($ram / $GiB, 3)
    WSL_memory_gib        = $(if ($vm) { [math]::Round($wslMemory / $GiB, 3) } else { $null })
    Windows_baseline_gib  = $(if ($vm) { [math]::Round($windowsBaseline / $GiB, 3) } else { $null })
    headroom_gib          = $HeadroomGiB
    formula_gib           = [math]::Round($maximumSize / $GiB, 3)
    MaximumSize_gib       = [math]::Round($pfMax / $GiB, 3)
    ram_available_gib     = [math]::Round($ramAvail / $GiB, 3)
    committed_now_gib     = [math]::Round($commit.Committed / $GiB, 3)
    commit_limit_now_gib  = [math]::Round($commit.Limit / $GiB, 3)
    pagefile_now_gib      = [math]::Round($pfNow / $GiB, 3)
    pagefile_growth_gib   = [math]::Round($growth / $GiB, 3)
    wsl_memory_source     = $(if ($vm) { $vm.Source } else { $null })
    wsl_memtotal_gib      = $(if ($vm) { [math]::Round($WslMemTotalBytes / $GiB, 3) } else { $null })
    wsl_vm_commit_gib     = $(if ($vm) { [math]::Round($vmCommit / $GiB, 3) } else { $null })
    c_free_gib            = [math]::Round([double]$disk.FreeSpace / $GiB, 3)
    automatic_managed     = $cs.AutomaticManagedPagefile
}
Write-Output ("PAGEFILE_PLAN " + ($result | ConvertTo-Json -Compress))
if ($growth -gt [double]$disk.FreeSpace) { throw "C: has $([math]::Round($disk.FreeSpace / $GiB, 1)) GiB free; the pagefile needs up to $([math]::Round($growth / $GiB, 1)) GiB more" }

if ($Apply -and $growth -le 0) {
    Write-Output ("PAGEFILE_APPLY " + ([ordered]@{ status = "skipped: current pagefile already covers MaximumSize" } | ConvertTo-Json -Compress))
} elseif ($Apply) {
    Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class PageFile {
    [StructLayout(LayoutKind.Sequential)]
    public struct UNICODE_STRING { public ushort Length; public ushort MaximumLength; public IntPtr Buffer; }
    [DllImport("ntdll.dll")] public static extern int RtlAdjustPrivilege(int Privilege, bool Enable, bool CurrentThread, out bool Enabled);
    [DllImport("ntdll.dll")] public static extern int NtCreatePagingFile(ref UNICODE_STRING Name, ref long MinimumSize, ref long MaximumSize, uint Priority);
    public static int Create(string ntPath, long min, long max) {
        bool was;
        int s = RtlAdjustPrivilege(15, true, false, out was); // SE_CREATE_PAGEFILE_PRIVILEGE
        if (s != 0) return s;
        UNICODE_STRING u = new UNICODE_STRING();
        u.Buffer = Marshal.StringToHGlobalUni(ntPath);
        u.Length = (ushort)(ntPath.Length * 2);
        u.MaximumLength = (ushort)(u.Length + 2);
        try { return NtCreatePagingFile(ref u, ref min, ref max, 0); } finally { Marshal.FreeHGlobal(u.Buffer); }
    }
}
"@
    # Runtime change only: no registry write, no reboot; the initial size stays as is so no disk is reserved up front.
    $status = [PageFile]::Create('\??\C:\pagefile.sys', [long]$pfNow, [long]$pfMax)
    $after = Get-Commit
    $pf = @(Get-CimInstance Win32_PageFileUsage)
    Write-Output ("PAGEFILE_APPLY " + ([ordered]@{
                status               = ("0x{0:X8}" -f $status)
                min_gib              = [math]::Round($pfNow / $GiB, 3)
                max_gib              = [math]::Round($pfMax / $GiB, 3)
                committed_gib        = [math]::Round($after.Committed / $GiB, 3)
                commit_limit_gib     = [math]::Round($after.Limit / $GiB, 3)
                max_commit_limit_gib = [math]::Round(($commit.Limit - $pfNow + $pfMax) / $GiB, 3)
                pagefile_alloc_mib   = ($pf | Measure-Object AllocatedBaseSize -Sum).Sum
            } | ConvertTo-Json -Compress))
    if ($status -ne 0) { throw ("NtCreatePagingFile failed with status 0x{0:X8}" -f $status) }
}
