# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# TEMP (rapidsai/rmm#2339): run PowerShell on a WSL runner's Windows host from inside the job container.
# The job container has the runner's docker socket but no WSL interop, so start a privileged sibling
# container that enters the WSL VM's root namespaces, where powershell.exe is reachable.
import base64
import http.client
import json
import socket
import struct
import sys

SOCK = "/var/run/docker.sock"
PS = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"


class _Conn(http.client.HTTPConnection):
    def __init__(self):
        super().__init__("localhost", timeout=900)

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(900)
        self.sock.connect(SOCK)


def _api(method, path, body=None):
    c = _Conn()
    headers = {"Content-Type": "application/json"} if body is not None else {}
    c.request(
        method, path, json.dumps(body) if body is not None else None, headers
    )
    r = c.getresponse()
    data = r.read()
    if r.status >= 400:
        raise RuntimeError(f"{method} {path}: {r.status} {data[:500]!r}")
    return data


def _demux(raw):
    out, i = [], 0
    while i + 8 <= len(raw):
        _, n = struct.unpack(">BxxxL", raw[i : i + 8])
        out.append(raw[i + 8 : i + 8 + n])
        i += 8 + n
    return b"".join(out).decode("utf-8", "replace").replace("\r", "")


def _image():
    for c in json.loads(_api("GET", "/containers/json")):
        if "ci-conda" in c["Image"]:
            return c["Image"]
    raise RuntimeError("no ci-conda container found")


def run(script):
    script = "$ProgressPreference = 'SilentlyContinue'\n" + script
    enc = base64.b64encode(script.encode("utf-16-le")).decode()
    cmd = f'"{PS}" -NoProfile -NonInteractive -ExecutionPolicy Bypass -OutputFormat Text -EncodedCommand {enc}'
    body = {
        "Image": _image(),
        "User": "root",
        "Entrypoint": [
            "nsenter",
            "-t",
            "1",
            "-m",
            "-u",
            "-i",
            "-n",
            "-p",
            "--",
            "/bin/sh",
            "-c",
            cmd,
        ],
        "HostConfig": {
            "Privileged": True,
            "PidMode": "host",
            "NetworkMode": "host",
        },
    }
    cid = json.loads(_api("POST", "/containers/create", body))["Id"]
    try:
        _api("POST", f"/containers/{cid}/start")
        code = json.loads(_api("POST", f"/containers/{cid}/wait"))[
            "StatusCode"
        ]
        return code, _demux(
            _api("GET", f"/containers/{cid}/logs?stdout=1&stderr=1")
        )
    finally:
        _api("DELETE", f"/containers/{cid}?force=1")


if __name__ == "__main__":
    code, out = run(sys.stdin.read())
    sys.stdout.write(out)
    sys.exit(code)
