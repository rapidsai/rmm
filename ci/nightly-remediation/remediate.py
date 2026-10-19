# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import agent
import jsonschema

ROOT = Path(__file__).resolve().parent
FAILED = {
    "failure",
    "timed_out",
    "cancelled",
    "startup_failure",
    "action_required",
}


def run(*args, **kwargs) -> str:
    return subprocess.run(
        args, text=True, capture_output=True, check=True, timeout=120, **kwargs
    ).stdout.strip()


def api(endpoint: str, data=None):
    args = ["gh", "api", endpoint]
    if data is not None:
        args.extend(["--input", "-"])
    return json.loads(
        run(*args, input=json.dumps(data) if data is not None else None)
    )


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2)
    for name in ("INFERENCE_API_KEY", "GH_TOKEN", "SLACK_WEBHOOK_URL"):
        secret = os.environ.get(name)
        if secret:
            text = text.replace(secret, "[REDACTED]")
    path.write_text(text + "\n")


def validate(value, schema: str) -> None:
    jsonschema.validate(
        value, json.loads((ROOT / f"{schema}.schema.json").read_text())
    )


def excerpt(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    lines = text.splitlines()
    selected = set(range(min(20, len(lines))))
    for i, line in enumerate(lines):
        if re.search(
            r"error:|fatal error|FAILED|Traceback|Exception|timed out", line
        ):
            selected.update(range(max(0, i - 5), min(len(lines), i + 15)))
    evidence = "\n".join(lines[i] for i in sorted(selected))
    return (
        evidence[: limit * 2 // 3]
        + "\n[log truncated]\n"
        + text[-limit // 3 :]
    )


def collect(repo: str, run_ids: list[int]) -> tuple[list[dict], list[dict]]:
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        raise ValueError("Invalid source repository")
    if (
        not isinstance(run_ids, list)
        or not run_ids
        or any(type(i) is not int or i <= 0 for i in run_ids)
    ):
        raise ValueError(
            "run-ids must be a nonempty JSON array of positive integers"
        )
    runs, jobs = [], []
    for run_id in sorted(set(run_ids)):
        metadata = api(f"repos/{repo}/actions/runs/{run_id}")
        if metadata["status"] != "completed":
            raise ValueError(f"Run {run_id} has not completed")
        runs.append({"id": run_id, "url": metadata["html_url"]})
        pages = json.loads(
            run(
                "gh",
                "api",
                "--paginate",
                "--slurp",
                f"repos/{repo}/actions/runs/{run_id}/attempts/{metadata['run_attempt']}/jobs?per_page=100",
            )
        )
        for page in pages:
            for job in page["jobs"]:
                if job["conclusion"] in FAILED:
                    jobs.append(
                        {
                            "id": job["id"],
                            "name": job["name"],
                            "url": job["html_url"],
                            "run_id": run_id,
                            "conclusion": job["conclusion"],
                            "failed_steps": [
                                s["name"]
                                for s in job["steps"]
                                if s["conclusion"] in FAILED
                            ],
                        }
                    )
        if metadata["conclusion"] in FAILED and not any(
            j["run_id"] == run_id for j in jobs
        ):
            raise ValueError(
                f"Failed run {run_id} has no failed jobs to analyze"
            )
    budget = max(1000, 180000 // max(1, len(jobs)))
    for job in jobs:
        try:
            log = run(
                "gh", "api", f"repos/{repo}/actions/jobs/{job['id']}/logs"
            )
            job["log"] = excerpt(log, min(24000, budget))
        except subprocess.SubprocessError:
            job["log"] = (
                "Logs unavailable; do not infer a code defect from this."
            )
    return runs, jobs


def plan(
    analysis: dict, jobs: list[dict], repo: str, sha: str, branch: str
) -> list[dict]:
    validate(analysis, "analysis")
    known = {j["id"]: j for j in jobs}
    covered, signatures, failures = set(), set(), []
    for failure in analysis["failures"]:
        ids = set(failure["job_ids"])
        signature = " ".join(failure["signature"].lower().split())
        if not ids <= known.keys() or not signature or signature in signatures:
            raise ValueError(
                "Unknown jobs or duplicate/empty failure signature"
            )
        if (
            failure["kind"] in {"build", "dependency"}
            and failure["runner"] != "cpu"
        ):
            raise ValueError("Build/dependency failures must use CPU runners")
        covered.update(ids)
        signatures.add(signature)
        key = json.dumps([repo, branch, sha, signature])
        failures.append(
            {
                **failure,
                "id": hashlib.sha256(key.encode()).hexdigest()[:16],
                "repo": repo,
                "sha": sha,
                "base": branch,
                "jobs": [known[i] for i in sorted(ids)],
            }
        )
    if covered != known.keys():
        raise ValueError("Analysis did not cover every failed job")
    return failures


def analyze(output: Path) -> None:
    repo, sha, branch = (
        os.environ[k] for k in ("SOURCE_REPO", "SOURCE_SHA", "TARGET_BRANCH")
    )
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("source-sha must be the full tested source commit")
    run("git", "check-ref-format", "--branch", branch)
    runs, jobs = collect(repo, json.loads(os.environ["RUN_IDS"]))
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp:
        context = {"repo": repo, "sha": sha, "branch": branch, "jobs": jobs}
        raw = (
            agent.invoke(
                (ROOT / "prompts/analyze.md").read_text()
                + "\n"
                + json.dumps(context),
                Path(temp),
                Path(temp) / "transcript.jsonl",
            )
            if jobs
            else {"failures": []}
        )
    failures = plan(raw, jobs, repo, sha, branch)
    save(output / "analysis.json", {"runs": runs, "failures": failures})
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as stream:
        for runner in ("cpu", "gpu"):
            matrix = {
                "include": [
                    {"id": f["id"]} for f in failures if f["runner"] == runner
                ]
            }
            stream.write(
                f"{runner}={json.dumps(matrix, separators=(',', ':'))}\n"
            )
            stream.write(
                f"has-{runner}={str(bool(matrix['include'])).lower()}\n"
            )


def load_failure(artifacts: Path, failure_id: str) -> dict:
    data = json.loads((artifacts / "analysis.json").read_text())
    return next(f for f in data["failures"] if f["id"] == failure_id)


def fix(artifacts: Path, output: Path, failure_id: str, source: Path) -> None:
    failure = load_failure(artifacts, failure_id)
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "id": failure_id,
        "outcome": "unresolved",
        "summary": "Fixer did not complete",
        "validation": [],
    }
    if failure["runner"] == "gpu":
        try:
            devices = run(
                "nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"
            )
            if not any(
                line.startswith("GPU-") for line in devices.splitlines()
            ):
                raise ValueError("No GPU devices are visible")
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            result.update(
                summary=f"GPU preflight blocked: {exc}",
                validation=[
                    {
                        "command": "GPU tests",
                        "status": "not_run",
                        "details": f"nvidia-smi device check failed: {exc}",
                    }
                ],
            )
            save(output / f"{failure_id}.json", result)
            return
    try:
        run(
            "git",
            "clone",
            "--no-checkout",
            f"https://github.com/{failure['repo']}.git",
            str(source),
        )
        run("git", "checkout", "--detach", failure["sha"], cwd=source)
        run(
            "git",
            "merge-base",
            "--is-ancestor",
            failure["sha"],
            f"origin/{failure['base']}",
            cwd=source,
        )
        with tempfile.TemporaryDirectory() as temp:
            prompt = (
                (ROOT / "prompts/fix.md").read_text()
                + "\n"
                + json.dumps(failure)
            )
            proposal = agent.invoke(
                prompt,
                Path(temp),
                Path(temp) / "transcript.jsonl",
                cwd=source,
                tools=True,
                timeout=5400,
            )
            validate(proposal, "result")
            result.update(proposal)
        if result["outcome"] == "proposed":
            run("git", "add", "--all", cwd=source)
            changes = run(
                "git",
                "diff",
                "--cached",
                "--name-only",
                failure["sha"],
                cwd=source,
            )
            if not changes:
                raise ValueError("Agent proposed a fix without source changes")
            run(
                "git",
                "diff",
                "--cached",
                "--check",
                failure["sha"],
                cwd=source,
            )
            with (output / f"{failure_id}.patch").open("wb") as patch:
                subprocess.run(
                    [
                        "git",
                        "diff",
                        "--cached",
                        "--binary",
                        "--no-ext-diff",
                        "--no-textconv",
                        failure["sha"],
                    ],
                    stdout=patch,
                    cwd=source,
                    check=True,
                    timeout=120,
                )
    except (
        ValueError,
        jsonschema.ValidationError,
        subprocess.SubprocessError,
    ) as exc:
        result.update(
            outcome="unresolved",
            summary=f"Fixer failed: {type(exc).__name__}: {exc}",
        )
    save(output / f"{failure_id}.json", result)


def prepare(
    artifacts: Path,
    proposal: Path,
    output: Path,
    failure_id: str,
    source: Path,
) -> None:
    result = {
        "id": failure_id,
        "outcome": "unresolved",
        "summary": "No fixer result available",
        "validation": [],
    }
    try:
        failure = load_failure(artifacts, failure_id)
        if (failure["repo"], failure["sha"], failure["base"]) != (
            "rapidsai/rmm",
            os.environ["SOURCE_SHA"],
            os.environ["TARGET_BRANCH"],
        ):
            raise ValueError(
                "Analysis source does not match the workflow inputs"
            )
        data = json.loads((proposal / f"{failure_id}.json").read_text())
        if not isinstance(data, dict) or data.get("id") != failure_id:
            raise ValueError("Fixer result has an unexpected failure ID")
        del data["id"]
        validate(data, "result")
        result.update(data)
        if result["outcome"] == "proposed":
            run(
                "git",
                "clone",
                "--no-checkout",
                "https://github.com/rapidsai/rmm.git",
                str(source),
            )
            run("git", "checkout", "--detach", failure["sha"], cwd=source)
            run(
                "git",
                "merge-base",
                "--is-ancestor",
                failure["sha"],
                f"origin/{failure['base']}",
                cwd=source,
            )
            run(
                "git",
                "apply",
                "--index",
                "--whitespace=error",
                "--",
                str((proposal / f"{failure_id}.patch").resolve()),
                cwd=source,
            )
            if not run("git", "diff", "--cached", "--name-only", cwd=source):
                raise ValueError("Fixer patch contains no source changes")
    except (
        OSError,
        ValueError,
        KeyError,
        StopIteration,
        jsonschema.ValidationError,
        subprocess.SubprocessError,
    ) as exc:
        result.update(
            outcome="unresolved",
            summary=f"Publication preparation failed: {type(exc).__name__}: {exc}",
        )
    save(output / f"{failure_id}.json", result)
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as stream:
        stream.write(
            f"proposed={str(result['outcome'] == 'proposed').lower()}\n"
        )


def target_fork(source_repo: str, owner: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?", owner):
        raise ValueError("Invalid fork owner")
    fork = f"{owner}/{source_repo.split('/')[1]}"
    if fork.lower() == source_repo.lower():
        raise ValueError(
            "Refusing to open remediation PRs in the source repository"
        )
    metadata = api(f"repos/{fork}")
    if (
        not metadata["fork"]
        or metadata["parent"]["full_name"].lower() != source_repo.lower()
    ):
        raise ValueError("Destination must be a fork of the source repository")
    return fork


def pr_body(failure: dict, result: dict) -> str:
    sources = "\n".join(j["url"] for j in failure["jobs"])
    checks = (
        "\n".join(
            f"- {v['status']}: {v['command']}\n  {v['details']}"
            for v in result["validation"]
        )
        or "No validation reported."
    )
    return (
        f"## Nightly failure\n{failure['root_cause']}\n\n{sources}\n\n"
        f"## Proposed fix\n{result['summary']}\n\n"
        f"## Validation (agent-reported)\n{checks}\n\n"
        f"Source revision: `{failure['sha']}`\n\n"
        "AI-generated draft; requires human review. Not independently validated by this workflow."
    )


def publish(
    artifacts: Path, output: Path, failure_id: str, source: Path
) -> None:
    failure = load_failure(artifacts, failure_id)
    path = output / f"{failure_id}.json"
    result = json.loads(path.read_text())
    if result["outcome"] != "proposed":
        return
    try:
        fork = target_fork(failure["repo"], os.environ["FORK_OWNER"])
        branch = f"nightly-fix/{failure_id}"
        existing = api(
            f"repos/{fork}/pulls?state=all&head={fork.split('/')[0]}:{branch}"
        )
        if existing:
            result.update(
                pr_url=existing[0]["html_url"],
                publication="existing",
                pr_state=existing[0]["state"],
                summary="Existing PR left unchanged; this attempt: "
                + result["summary"],
            )
        else:
            remote = f"https://github.com/{fork}.git"
            run("git", "fetch", remote, failure["base"], cwd=source)
            run(
                "git",
                "merge-base",
                "--is-ancestor",
                failure["sha"],
                "FETCH_HEAD",
                cwd=source,
            )
            if run(
                "git",
                "ls-remote",
                "--heads",
                remote,
                f"refs/heads/{branch}",
                cwd=source,
            ):
                run("git", "fetch", remote, branch, cwd=source)
                run(
                    "git",
                    "diff",
                    "--cached",
                    "--no-ext-diff",
                    "--no-textconv",
                    "--exit-code",
                    "FETCH_HEAD",
                    cwd=source,
                )
            else:
                owner = os.environ["FORK_OWNER"]
                run(
                    "git",
                    "-c",
                    f"user.name={owner}",
                    "-c",
                    f"user.email={owner}@users.noreply.github.com",
                    "-c",
                    "core.hooksPath=/dev/null",
                    "-c",
                    "commit.gpgsign=false",
                    "commit",
                    "-m",
                    failure["title"],
                    cwd=source,
                )
                run(
                    "git",
                    "-c",
                    "credential.helper=",
                    "-c",
                    "credential.helper=!gh auth git-credential",
                    "push",
                    remote,
                    f"HEAD:refs/heads/{branch}",
                    cwd=source,
                )
            pr = api(
                f"repos/{fork}/pulls",
                {
                    "title": failure["title"],
                    "body": pr_body(failure, result),
                    "head": branch,
                    "base": failure["base"],
                    "draft": True,
                },
            )
            result.update(
                pr_url=pr["html_url"], publication="created", pr_state="open"
            )
        result.pop("publication_error", None)
    except (ValueError, subprocess.SubprocessError) as exc:
        result.update(
            publication="failed",
            publication_error=f"{type(exc).__name__}: {exc}",
        )
    save(path, result)


def report_payloads(
    artifacts: Path, workflow_url: str, statuses: dict
) -> list[dict]:
    analysis_path = artifacts / "analysis.json"
    blocks = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Nightly remediation* — <{workflow_url}|workflow run>",
            },
        }
    ]
    pr_links = []
    if not analysis_path.exists():
        summaries = [
            "Analyzer did not produce a report. No conclusion about nightly health is available."
        ]
    else:
        analysis = json.loads(analysis_path.read_text())
        summaries = []
        if not analysis["failures"]:
            summaries.append("No failed jobs in the selected runs.")
        for failure in analysis["failures"]:
            path = artifacts / f"{failure['id']}.json"
            result = json.loads(path.read_text()) if path.exists() else None
            summary = f"{failure['title']} [{failure['runner']}]\n{failure['root_cause'][:1000]}\n"
            if result is None:
                summary += "No fixer result available (failed, cancelled, or skipped)."
            else:
                summary += f"{result['outcome']}: {result['summary'][:1000]}\n"
                summary += "Validation (agent-reported): " + (
                    ", ".join(v["status"] for v in result["validation"])
                    or "not reported"
                )
                if result.get("pr_url"):
                    url = result["pr_url"]
                    if not re.fullmatch(
                        r"https://github\.com/[\w.-]+/[\w.-]+/pull/[0-9]+", url
                    ):
                        raise ValueError("Unexpected PR URL in fixer result")
                    pr_links.append(f"<{url}|Draft proposal {failure['id']}>")
                    summary += f"\nDraft proposal ({result.get('pr_state', 'unknown')}, {result.get('publication', 'unknown')}): {url}"
                if result.get("publication_error"):
                    summary += f"\nPR publication failed: {result['publication_error'][:300]}"
                elif result["outcome"] == "proposed" and not result.get(
                    "pr_url"
                ):
                    summary += "\nPR publication did not complete."
            summaries.append(summary)
    summaries.append(
        "Pipeline jobs: "
        + ", ".join(f"{k}={v['result']}" for k, v in statuses.items())
    )
    blocks.extend(
        {
            "type": "section",
            "text": {"type": "plain_text", "text": s[:3000], "emoji": False},
        }
        for s in summaries
    )
    blocks.extend(
        {"type": "section", "text": {"type": "mrkdwn", "text": link}}
        for link in pr_links
    )
    return [
        {
            "text": "Nightly remediation report",
            "blocks": blocks[i : i + 10],
            "unfurl_links": False,
        }
        for i in range(0, len(blocks), 10)
    ]


def report(artifacts: Path, output: Path) -> None:
    url = f"{os.environ['GITHUB_SERVER_URL']}/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
    payloads = report_payloads(
        artifacts, url, json.loads(os.environ["JOB_STATUSES"])
    )
    for i, payload in enumerate(payloads):
        save(output / f"slack-{i}.json", payload)
        request = urllib.request.Request(
            os.environ["SLACK_WEBHOOK_URL"],
            data=(output / f"slack-{i}.json").read_bytes(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.read().strip() != b"ok":
                raise ValueError("Slack did not acknowledge the report")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=["analyze", "fix", "prepare", "publish", "report"]
    )
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--proposal", type=Path, default=Path("proposal"))
    parser.add_argument("--source", type=Path, default=Path("source"))
    parser.add_argument("--failure-id", default="")
    args = parser.parse_args()
    if args.stage == "analyze":
        analyze(args.output)
    elif args.stage == "fix":
        fix(
            args.artifacts, args.output, args.failure_id, args.source.resolve()
        )
    elif args.stage == "prepare":
        prepare(
            args.artifacts,
            args.proposal,
            args.output,
            args.failure_id,
            args.source.resolve(),
        )
    elif args.stage == "publish":
        publish(
            args.artifacts, args.output, args.failure_id, args.source.resolve()
        )
    else:
        report(args.artifacts, args.output)


if __name__ == "__main__":
    main()
