#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# TEMP: probe the Windows host of a WSL runner (rapidsai/rmm#2339).
set -uo pipefail
grep -qi microsoft /proc/version || { echo "not WSL, skipping"; exit 0; }

rapids-logger "WSL host probe: from job container"
echo "cmdline: $(cat /proc/cmdline)"
grep -E "MemTotal|SwapTotal" /proc/meminfo
cat /proc/sys/vm/overcommit_memory 2>&1
stat -c "%n" /var/run/docker.sock /run/WSL /mnt/c /init /proc/sys/fs/binfmt_misc/WSLInterop 2>&1

HOSTPROBE=$(cat <<'EOS'
set -x
cat /proc/cmdline
grep -E "MemTotal|SwapTotal|CommitLimit" /proc/meminfo
ls / /mnt /mnt/wsl /run/WSL 2>&1 | head -40
ls /proc/sys/fs/binfmt_misc/ 2>&1; cat /proc/sys/fs/binfmt_misc/WSLInterop 2>&1 | head -3
cat /etc/wsl.conf 2>&1 | head -30
mount | grep -E "drvfs|9p|virtiofs|/mnt/" | head -20
PS=""
for p in /mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe /mnt/host/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe; do [ -e "$p" ] && PS=$p && break; done
echo "PS=$PS"
if [ -n "$PS" ]; then
  timeout 120 "$PS" -NoProfile -Command '
    Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory,AutomaticManagedPagefile,Model,Manufacturer | Format-List
    Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,TotalVirtualMemorySize,FreeVirtualMemory,TotalVisibleMemorySize,FreePhysicalMemory | Format-List
    Get-CimInstance Win32_PageFileUsage | Select-Object Name,AllocatedBaseSize,CurrentUsage,PeakUsage | Format-List
    Get-CimInstance Win32_PageFileSetting | Format-List
    (Get-Counter "\Memory\Committed Bytes","\Memory\Commit Limit").CounterSamples | Select-Object Path,CookedValue | Format-List
    Get-CimInstance Win32_LogicalDisk | Select-Object DeviceID,DriveType,FileSystem,Size,FreeSpace | Format-List
    Get-CimInstance Win32_DiskDrive | Select-Object Model,Size,InterfaceType | Format-List
    Get-ChildItem C:\\ -Force -Filter *.sys -ErrorAction SilentlyContinue | Select-Object Name,Length | Format-List
    Get-ChildItem "$env:LOCALAPPDATA\\wsl","C:\\Users\\*\\AppData\\Local\\wsl" -Recurse -Filter *.vhdx -ErrorAction SilentlyContinue | Select-Object FullName,Length | Format-List
    Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM,DriverVersion | Format-List
    Get-Content "$env:USERPROFILE\.wslconfig" -ErrorAction SilentlyContinue
    wsl.exe --version
  ' 2>&1 | tr -d '\r'
fi
EOS
)

if [ -S /var/run/docker.sock ]; then
  # The job container has the runner's docker socket but no docker CLI, so drive the Engine API with curl.
  rapids-logger "WSL host probe: privileged sibling container in the WSL VM's root namespaces"
  IMG=$(curl -s --unix-socket /var/run/docker.sock http://localhost/containers/json | python -c 'import json,sys; c=[x["Image"] for x in json.load(sys.stdin) if "ci-conda" in x["Image"]]; print(c[0] if c else "")')
  echo "image=${IMG}"
  BODY=$(HOSTPROBE="$HOSTPROBE" IMG="$IMG" python -c '
import json, os
print(json.dumps({"Image": os.environ["IMG"], "User": "root", "Entrypoint": ["nsenter", "-t", "1", "-m", "-u", "-i", "-n", "-p", "--", "/bin/sh", "-c", os.environ["HOSTPROBE"]],
  "HostConfig": {"Privileged": True, "PidMode": "host", "NetworkMode": "host"}}))')
  CID=$(curl -s --unix-socket /var/run/docker.sock -H "Content-Type: application/json" -d "$BODY" http://localhost/containers/create | python -c 'import json,sys; d=json.load(sys.stdin); print(d.get("Id", d))')
  echo "cid=${CID}"
  curl -s --unix-socket /var/run/docker.sock -X POST "http://localhost/containers/${CID}/start"
  curl -s --unix-socket /var/run/docker.sock -X POST "http://localhost/containers/${CID}/wait?condition=not-running" --max-time 300
  echo
  curl -s --unix-socket /var/run/docker.sock "http://localhost/containers/${CID}/logs?stdout=1&stderr=1" | tr -d '\000-\010\016-\037' | head -300
  curl -s --unix-socket /var/run/docker.sock -X DELETE "http://localhost/containers/${CID}?force=1"
fi
