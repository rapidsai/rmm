#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import datetime
import json
import re
import subprocess
from pathlib import Path

POOL_POLICIES = ("separate", "unified")
STREAM_POLICIES = ("same_stream", "cross_stream")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_text(command: list[str], cwd: Path) -> str:
    try:
        return subprocess.check_output(command, cwd=cwd, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "unknown"


def benchmark_min_time(value: str) -> str:
    return value if value[-1].isalpha() else f"{value}s"


def cuda_device_name(root: Path) -> str:
    return run_text(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", "0"],
        root,
    ).splitlines()[0]


def benchmark_json_summary(path: Path) -> list[dict[str, object]]:
    data = json.loads(path.read_text())
    return [
        {
            "name": item["name"],
            "real_time": item.get("real_time"),
            "cpu_time": item.get("cpu_time"),
            "time_unit": item.get("time_unit"),
        }
        for item in data.get("benchmarks", [])
    ]


def write_markdown_report(
    report_path: Path,
    *,
    workload: str,
    command: list[str],
    json_path: Path,
    pool_policy: str,
    stream_policy: str,
    git_commit: str,
    device_name: str,
    timestamp: str,
) -> None:
    rows = benchmark_json_summary(json_path)
    lines = [
        f"# Caching Allocator Benchmark: {workload}",
        "",
        f"- Git commit: `{git_commit}`",
        f"- CUDA device: `{device_name}`",
        f"- Timestamp UTC: `{timestamp}`",
        f"- Pool policy: `{pool_policy}`",
        f"- Stream policy: `{stream_policy}`",
        f"- Raw JSON: `{json_path.name}`",
        "",
        "## Command",
        "",
        "```bash",
        " ".join(command),
        "```",
        "",
        "## Results",
        "",
        "| Benchmark | Real Time | CPU Time | Unit |",
        "| --- | ---: | ---: | --- |",
    ]
    lines.extend(
        f"| `{row['name']}` | {row['real_time']} | {row['cpu_time']} | {row['time_unit']} |"
        for row in rows
    )
    report_path.write_text("\n".join(lines) + "\n")


def run_benchmark_report(
    *,
    binary: Path,
    workload: str,
    extra_args: list[str],
    output_dir: Path,
    root: Path,
    pool_policy: str,
    stream_policy: str,
    git_commit: str,
    device_name: str,
    timestamp: str,
) -> None:
    name = sanitize(
        f"{workload}_{pool_policy}_{stream_policy}_{git_commit}_{device_name}_{timestamp}"
    )
    json_path = output_dir / f"{name}.json"
    report_path = output_dir / f"{name}.md"
    command = [
        str(binary),
        "--resource=caching",
        f"--pool-policy={pool_policy}",
        f"--stream-policy={stream_policy}",
        "--benchmark_out_format=json",
        f"--benchmark_out={json_path}",
        *extra_args,
    ]
    subprocess.run(command, cwd=root, check=True)
    write_markdown_report(
        report_path,
        workload=workload,
        command=command,
        json_path=json_path,
        pool_policy=pool_policy,
        stream_policy=stream_policy,
        git_commit=git_commit,
        device_name=device_name,
        timestamp=timestamp,
    )


def main() -> None:
    root = repo_root()
    default_build = root / "cpp" / "build-bench-opencode" / "gbenchmarks"
    parser = argparse.ArgumentParser(
        description="Generate caching allocator benchmark reports."
    )
    parser.add_argument(
        "--random-bench",
        type=Path,
        default=default_build / "RANDOM_ALLOCATIONS_BENCH",
    )
    parser.add_argument(
        "--multi-stream-bench",
        type=Path,
        default=default_build / "MULTI_STREAM_ALLOCATIONS_BENCH",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=root / "caching-allocator-reports"
    )
    parser.add_argument("--benchmark-min-time", default="0.2s")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    git_commit = run_text(["git", "rev-parse", "--short", "HEAD"], root)
    device_name = cuda_device_name(root)
    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    min_time = benchmark_min_time(args.benchmark_min_time)

    workloads = [
        (
            args.random_bench,
            "mixed_allocations",
            [
                "--numallocs=1000",
                "--maxsize=64",
                f"--benchmark_min_time={min_time}",
            ],
        ),
        (
            args.multi_stream_bench,
            "multi_stream_churn",
            [
                "--kernels=4",
                "--streams=4",
                "--warm=true",
                f"--benchmark_min_time={min_time}",
            ],
        ),
    ]

    for binary, workload, extra_args in workloads:
        for pool_policy in POOL_POLICIES:
            for stream_policy in STREAM_POLICIES:
                run_benchmark_report(
                    binary=binary,
                    workload=workload,
                    extra_args=extra_args,
                    output_dir=args.output_dir,
                    root=root,
                    pool_policy=pool_policy,
                    stream_policy=stream_policy,
                    git_commit=git_commit,
                    device_name=device_name,
                    timestamp=timestamp,
                )


if __name__ == "__main__":
    main()
