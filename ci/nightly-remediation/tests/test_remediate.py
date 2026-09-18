# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agent
import remediate as r

SHA = "a" * 40
JOBS = [
    {
        "id": 11,
        "name": "build",
        "url": "https://github.com/upstream/project/actions/runs/1/job/11",
    }
]
FAILURE = {
    "signature": "missing-header",
    "title": "Missing header",
    "root_cause": "Missing include",
    "evidence": "fatal error: foo.h missing",
    "job_ids": [11],
    "kind": "build",
    "runner": "cpu",
    "validation_plan": "Rebuild the affected target",
}
PROPOSAL = {
    "outcome": "proposed",
    "summary": "Added include",
    "validation": [
        {
            "command": "cmake --build build",
            "status": "passed",
            "details": "Build completed",
        }
    ],
}


class PlanTests(unittest.TestCase):
    def plan(self, failures=None, jobs=None):
        return r.plan(
            {
                "failures": (
                    failures
                    if failures is not None
                    else [copy.deepcopy(FAILURE)]
                )
            },
            JOBS if jobs is None else jobs,
            "upstream/project",
            SHA,
            "main",
        )

    def test_empty_plan(self):
        self.assertEqual(self.plan([], []), [])

    def test_grouped_jobs_and_stable_id(self):
        failure = copy.deepcopy(FAILURE)
        failure["job_ids"] = [12, 11]
        jobs = [*JOBS, {**JOBS[0], "id": 12}]
        first = self.plan([failure], jobs)[0]
        failure["job_ids"].reverse()
        failure["signature"] = "  MISSING-HEADER "
        second = self.plan([failure], jobs)[0]
        self.assertEqual(first["id"], second["id"])
        self.assertEqual(first["sha"], SHA)
        self.assertEqual([j["id"] for j in first["jobs"]], [11, 12])

    def test_different_base_branches_have_distinct_ids(self):
        other = r.plan(
            {"failures": [FAILURE]},
            JOBS,
            "upstream/project",
            SHA,
            "release/test",
        )
        self.assertNotEqual(self.plan()[0]["id"], other[0]["id"])

    def test_gpu_tests_and_cpu_builds(self):
        test = {
            **FAILURE,
            "signature": "test-assertion",
            "kind": "test",
            "runner": "gpu",
        }
        self.assertEqual(
            [f["runner"] for f in self.plan([FAILURE, test])], ["cpu", "gpu"]
        )

    def test_rejects_invalid_or_incomplete_plans(self):
        cases = [
            [],
            [dict(FAILURE, job_ids=[999])],
            [FAILURE, FAILURE],
            [dict(FAILURE, runner="gpu")],
            [dict(FAILURE, kind="dependency", runner="gpu")],
            [dict(FAILURE, runner="arbitrary-runner")],
            [dict(FAILURE, surprise=True)],
            [dict(FAILURE, job_ids=[])],
            [dict(FAILURE, signature=" ")],
        ]
        for failures in cases:
            with (
                self.subTest(failures=failures),
                self.assertRaises((ValueError, r.jsonschema.ValidationError)),
            ):
                self.plan(failures)

    def test_log_excerpts_retain_early_error_and_end(self):
        text = (
            "preamble\n" * 100
            + "fatal error: missing\n"
            + "noise\n" * 1000
            + "final symptom"
        )
        result = r.excerpt(text, 1200)
        self.assertIn("fatal error: missing", result)
        self.assertIn("final symptom", result)
        self.assertIn("truncated", result)
        self.assertEqual(r.excerpt("small", 1200), "small")


class AgentTests(unittest.TestCase):
    def test_json_events_use_final_assistant_not_tool_or_partial_messages(
        self,
    ):
        events = [
            {
                "type": "message_end",
                "message": {"role": "assistant", "stopReason": "toolUse"},
            },
            {
                "type": "message_end",
                "message": {"role": "toolResult", "content": []},
            },
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "stopReason": "stop",
                    "content": [
                        {"type": "thinking", "thinking": "private"},
                        {"type": "text", "text": json.dumps(PROPOSAL)},
                    ],
                },
            },
            {"type": "agent_end"},
        ]
        with tempfile.TemporaryDirectory() as temp:
            transcript = Path(temp) / "events.jsonl"
            transcript.write_text("\n".join(map(json.dumps, events)))
            self.assertEqual(agent.response(transcript), PROPOSAL)
            for reason in ("error", "aborted", "length", "toolUse"):
                events[-2]["message"]["stopReason"] = reason
                transcript.write_text("\n".join(map(json.dumps, events)))
                with (
                    self.subTest(reason=reason),
                    self.assertRaises(ValueError),
                ):
                    agent.response(transcript)
            transcript.write_text(json.dumps({"type": "session"}))
            with self.assertRaises(ValueError):
                agent.response(transcript)

    @patch.dict(
        os.environ,
        {
            "MODEL_CONFIG": '{"id":"model"}',
            "INFERENCE_URL": "https://example.com/v1",
            "INFERENCE_API": "openai-completions",
            "INFERENCE_API_KEY": "private-key",
        },
    )
    def test_isolated_configuration_and_tool_allowlist(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            agent.configure(directory)
            config = (directory / "models.json").read_text()
            self.assertNotIn("private-key", config)
            self.assertIn("$INFERENCE_API_KEY", config)
        for flag in (
            "--no-extensions",
            "--no-context-files",
            "--no-approve",
            "--no-session",
            "--offline",
        ):
            self.assertIn(flag, agent.command(False))
        self.assertIn("--no-tools", agent.command(False))
        self.assertIn("--tools", agent.command(True))

    @patch.dict(
        os.environ,
        {
            "MODEL_CONFIG": '{"id":"model"}',
            "INFERENCE_URL": "https://example.com/v1",
            "INFERENCE_API": "openai-completions",
            "INFERENCE_API_KEY": "private-key",
            "RAPIDS_CUDA_VERSION": "13.3.0",
            "NVIDIA_VISIBLE_DEVICES": "GPU-test",
            "LD_LIBRARY_PATH": "/usr/local/nvidia/lib64",
            "GH_TOKEN": "github-secret",
            "SLACK_WEBHOOK_URL": "slack-secret",
            "ACTIONS_RUNTIME_TOKEN": "runtime-secret",
        },
    )
    @patch("agent.response", return_value={})
    @patch("agent.subprocess.run")
    def test_invocation_preserves_build_environment_without_service_tokens(
        self, run, response
    ):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            for tools in (False, True):
                with self.subTest(tools=tools):
                    agent.invoke(
                        "prompt",
                        directory,
                        directory / "events.jsonl",
                        cwd=Path("/source"),
                        tools=tools,
                        timeout=5400,
                    )
                    args, kwargs = run.call_args
                    self.assertEqual(args[0][0], "pi")
                    self.assertIn(
                        "--tools" if tools else "--no-tools", args[0]
                    )
                    self.assertEqual(kwargs["cwd"], Path("/source"))
                    self.assertEqual(kwargs["timeout"], 5400)
                    env = kwargs["env"]
                    for name in (
                        "RAPIDS_CUDA_VERSION",
                        "NVIDIA_VISIBLE_DEVICES",
                        "LD_LIBRARY_PATH",
                        "INFERENCE_API_KEY",
                    ):
                        self.assertEqual(env[name], os.environ[name])
                    for name in (
                        "GH_TOKEN",
                        "SLACK_WEBHOOK_URL",
                        "ACTIONS_RUNTIME_TOKEN",
                    ):
                        self.assertNotIn(name, env)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.failure = r.plan(
            {"failures": [FAILURE]}, JOBS, "upstream/project", SHA, "main"
        )[0]
        r.save(
            self.root / "analysis.json",
            {"runs": [], "failures": [self.failure]},
        )

    @patch("remediate.run")
    @patch("remediate.api")
    def test_collect_paginates_jobs_and_handles_missing_logs(self, api, run):
        api.return_value = {
            "status": "completed",
            "html_url": "run-url",
            "run_attempt": 2,
            "conclusion": "failure",
        }
        job = {
            **JOBS[0],
            "html_url": JOBS[0]["url"],
            "conclusion": "failure",
            "steps": [{"name": "build", "conclusion": "failure"}],
        }
        run.side_effect = [
            json.dumps([{"jobs": [job]}, {"jobs": [dict(job, id=12)]}]),
            "compiler error",
            subprocess.CalledProcessError(1, "gh"),
        ]
        runs, jobs = r.collect("upstream/project", [1, 1])
        self.assertEqual(len(runs), 1)
        self.assertEqual(len(jobs), 2)
        self.assertIn("--paginate", run.call_args_list[0].args)
        self.assertIn("attempts/2/jobs", run.call_args_list[0].args[-1])
        self.assertIn("unavailable", jobs[1]["log"])

    @patch("remediate.run", return_value='[{"jobs": []}]')
    @patch("remediate.api")
    def test_failed_run_without_jobs_is_not_reported_as_healthy(
        self, api, run
    ):
        api.return_value = {
            "status": "completed",
            "conclusion": "failure",
            "html_url": "run-url",
            "run_attempt": 1,
        }
        with self.assertRaisesRegex(ValueError, "no failed jobs"):
            r.collect("upstream/project", [1])

    @patch("remediate.api")
    def test_incomplete_runs_and_invalid_ids_are_rejected(self, api):
        api.return_value = {"status": "in_progress"}
        for ids in ([], [True], [-1], ["1"], [1], 1, {"1": 1}):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                r.collect("upstream/project", ids)

    @patch("remediate.agent.invoke")
    @patch("remediate.collect", return_value=([], []))
    @patch("remediate.run")
    def test_healthy_runs_do_not_call_model_and_emit_empty_matrices(
        self, run, collect, invoke
    ):
        with patch.dict(
            os.environ,
            {
                "SOURCE_REPO": "upstream/project",
                "SOURCE_SHA": SHA,
                "TARGET_BRANCH": "main",
                "RUN_IDS": "[1]",
                "GITHUB_OUTPUT": str(self.root / "outputs"),
            },
        ):
            r.analyze(self.root)
        invoke.assert_not_called()
        outputs = (self.root / "outputs").read_text()
        self.assertIn('cpu={"include":[]}', outputs)
        self.assertIn("has-gpu=false", outputs)

    @patch(
        "remediate.run", side_effect=subprocess.CalledProcessError(1, "git")
    )
    def test_fixer_errors_produce_structured_unresolved_result(self, run):
        with patch.dict(
            os.environ, {"GITHUB_OUTPUT": str(self.root / "outputs")}
        ):
            r.fix(
                self.root, self.root, self.failure["id"], self.root / "source"
            )
        result = json.loads(
            (self.root / f"{self.failure['id']}.json").read_text()
        )
        self.assertEqual(result["outcome"], "unresolved")
        self.assertIn("proposed=false", (self.root / "outputs").read_text())

    @patch("remediate.agent.invoke", return_value=PROPOSAL)
    @patch("remediate.run", return_value="changed-file")
    def test_fixer_runs_pi_directly_in_the_source_checkout(self, run, invoke):
        source = self.root / "source"
        with patch.dict(
            os.environ, {"GITHUB_OUTPUT": str(self.root / "outputs")}
        ):
            r.fix(self.root, self.root, self.failure["id"], source)
        self.assertEqual(
            invoke.call_args.kwargs,
            {"cwd": source, "tools": True, "timeout": 5400},
        )
        result = json.loads(
            (self.root / f"{self.failure['id']}.json").read_text()
        )
        self.assertEqual(result["outcome"], "proposed")
        self.assertIn("proposed=true", (self.root / "outputs").read_text())

    @patch("remediate.agent.invoke")
    @patch("remediate.collect", return_value=([], JOBS))
    @patch("remediate.run")
    def test_analysis_routes_independent_failures_into_matrices(
        self, run, collect, invoke
    ):
        invoke.return_value = {
            "failures": [
                FAILURE,
                dict(FAILURE, signature="gpu-test", kind="test", runner="gpu"),
            ]
        }
        with patch.dict(
            os.environ,
            {
                "SOURCE_REPO": "upstream/project",
                "SOURCE_SHA": SHA,
                "TARGET_BRANCH": "main",
                "RUN_IDS": "[1]",
                "GITHUB_OUTPUT": str(self.root / "outputs"),
            },
        ):
            r.analyze(self.root)
        outputs = dict(
            line.split("=", 1)
            for line in (self.root / "outputs").read_text().splitlines()
        )
        self.assertEqual(outputs["has-cpu"], "true")
        self.assertEqual(outputs["has-gpu"], "true")
        self.assertNotEqual(
            json.loads(outputs["cpu"]), json.loads(outputs["gpu"])
        )
        self.assertEqual(
            len(
                json.loads((self.root / "analysis.json").read_text())[
                    "failures"
                ]
            ),
            2,
        )

    @patch("remediate.api")
    def test_refuses_upstream_and_unrelated_destinations(self, api):
        with self.assertRaises(ValueError):
            r.target_fork("upstream/project", "upstream")
        api.assert_not_called()
        api.return_value = {
            "fork": True,
            "parent": {"full_name": "other/project"},
        }
        with self.assertRaises(ValueError):
            r.target_fork("upstream/project", "bot")
        api.return_value["parent"]["full_name"] = "upstream/project"
        self.assertEqual(
            r.target_fork("upstream/project", "bot"), "bot/project"
        )

    @patch.dict(os.environ, {"FORK_OWNER": "another-bot"})
    @patch("remediate.run", return_value="")
    @patch("remediate.api")
    def test_publication_is_draft_in_configured_fork(self, api, run):
        result_path = self.root / f"{self.failure['id']}.json"
        r.save(result_path, PROPOSAL)
        api.side_effect = [
            {"fork": True, "parent": {"full_name": "upstream/project"}},
            [],
            {"html_url": "https://github.com/another-bot/project/pull/1"},
        ]
        r.publish(self.root, self.root, self.failure["id"], self.root)
        endpoint, body = api.call_args.args
        self.assertEqual(endpoint, "repos/another-bot/project/pulls")
        self.assertTrue(body["draft"])
        self.assertEqual(body["base"], "main")
        self.assertIn("agent-reported", body["body"])
        self.assertIn(
            "another-bot",
            json.dumps([call.args for call in run.call_args_list]),
        )
        self.assertNotIn("--force", str(run.call_args_list))
        self.assertTrue(
            any("push" in call.args for call in run.call_args_list)
        )
        self.assertEqual(
            json.loads(result_path.read_text())["publication"], "created"
        )

    @patch.dict(os.environ, {"FORK_OWNER": "bot"})
    @patch("remediate.run")
    @patch("remediate.api")
    def test_existing_pr_is_not_overwritten_or_reopened(self, api, run):
        path = self.root / f"{self.failure['id']}.json"
        r.save(path, PROPOSAL)
        api.side_effect = [
            {"fork": True, "parent": {"full_name": "upstream/project"}},
            [
                {
                    "html_url": "https://github.com/bot/project/pull/1",
                    "state": "closed",
                }
            ],
        ]
        r.publish(self.root, self.root, self.failure["id"], self.root)
        run.assert_not_called()
        self.assertEqual(json.loads(path.read_text())["pr_state"], "closed")

    def test_reports_missing_analysis_and_fixer_results(self):
        statuses = {
            "analyze": {"result": "success"},
            "fix-gpu": {"result": "failure"},
        }
        payload = r.report_payloads(
            self.root,
            "https://github.com/upstream/project/actions/runs/2",
            statuses,
        )
        self.assertIn("No fixer result", json.dumps(payload))
        (self.root / "analysis.json").unlink()
        payload = r.report_payloads(self.root, "workflow-url", statuses)
        self.assertIn("No conclusion", json.dumps(payload))

    def test_report_contains_pr_and_does_not_interpret_agent_slack_markup(
        self,
    ):
        result = {
            **PROPOSAL,
            "summary": "<!channel> look here",
            "pr_url": "https://github.com/bot/project/pull/1",
        }
        r.save(self.root / f"{self.failure['id']}.json", result)
        payload = r.report_payloads(self.root, "workflow-url", {})[0]
        text = payload["blocks"][1]["text"]
        self.assertEqual(text["type"], "plain_text")
        self.assertIn(result["pr_url"], text["text"])
        self.assertIn("agent-reported", text["text"])

    def test_long_slack_reports_keep_every_pr_link_within_message_limits(self):
        failures = [
            dict(self.failure, id=str(i), root_cause="cause" * 800)
            for i in range(32)
        ]
        r.save(self.root / "analysis.json", {"runs": [], "failures": failures})
        for failure in failures:
            r.save(
                self.root / f"{failure['id']}.json",
                {
                    **PROPOSAL,
                    "summary": "summary" * 500,
                    "pr_url": f"https://github.com/bot/project/pull/{int(failure['id']) + 1}",
                },
            )
        payloads = r.report_payloads(self.root, "workflow-url", {})
        self.assertGreater(len(payloads), 1)
        links = []
        for payload in payloads:
            self.assertLessEqual(len(payload["blocks"]), 50)
            texts = [block["text"]["text"] for block in payload["blocks"]]
            self.assertTrue(all(len(text) <= 3000 for text in texts))
            self.assertLess(sum(map(len, texts)), 40000)
            links.extend(text for text in texts if "|Draft proposal" in text)
        self.assertEqual(len(links), 32)

    @patch.dict(os.environ, {"INFERENCE_API_KEY": "secret-token"})
    def test_artifacts_redact_known_credentials(self):
        path = self.root / "redacted.json"
        r.save(path, {"summary": "secret-token"})
        self.assertNotIn("secret-token", path.read_text())


class PublicationGitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        env = patch.dict(
            os.environ,
            {
                "FORK_OWNER": "bot",
                "GIT_CONFIG_GLOBAL": "/dev/null",
                "GIT_CONFIG_NOSYSTEM": "1",
            },
        )
        env.start()
        self.addCleanup(env.stop)
        self.source = self.root / "source"
        self.fork = self.root / "fork.git"
        r.run("git", "init", "--initial-branch=main", str(self.source))
        r.run("git", "config", "user.name", "Test", cwd=self.source)
        r.run(
            "git", "config", "user.email", "test@example.com", cwd=self.source
        )
        (self.source / "file.txt").write_text("before\n")
        r.run("git", "add", "file.txt", cwd=self.source)
        r.run("git", "commit", "-m", "Initial", cwd=self.source)
        sha = r.run("git", "rev-parse", "HEAD", cwd=self.source)
        r.run("git", "init", "--bare", str(self.fork))
        r.run(
            "git",
            "push",
            str(self.fork),
            "HEAD:refs/heads/main",
            cwd=self.source,
        )
        r.run(
            "git",
            "config",
            f"url.{self.fork}.insteadOf",
            "https://github.com/bot/project.git",
            cwd=self.source,
        )
        self.failure = r.plan(
            {"failures": [FAILURE]}, JOBS, "upstream/project", sha, "main"
        )[0]
        r.save(
            self.root / "analysis.json",
            {"runs": [], "failures": [self.failure]},
        )
        self.result = self.root / f"{self.failure['id']}.json"
        r.save(self.result, PROPOSAL)
        (self.source / "file.txt").write_text("after\n")
        r.run("git", "add", "file.txt", cwd=self.source)
        self.reject_pr = False
        self.created = []

    def api(self, endpoint, data=None):
        if endpoint == "repos/bot/project":
            return {"fork": True, "parent": {"full_name": "upstream/project"}}
        if data is None:
            return []
        if self.reject_pr:
            raise subprocess.CalledProcessError(1, "gh")
        self.created.append(data)
        return {"html_url": "https://github.com/bot/project/pull/1"}

    def publish(self):
        with patch("remediate.api", side_effect=self.api):
            r.publish(self.root, self.root, self.failure["id"], self.source)
        return json.loads(self.result.read_text())

    def test_real_git_push_contains_only_fix_and_leaves_base_unchanged(self):
        result = self.publish()
        self.assertEqual(result["publication"], "created")
        base = r.run("git", "rev-parse", "main", cwd=self.fork)
        self.assertEqual(base, self.failure["sha"])
        branch = f"nightly-fix/{self.failure['id']}"
        self.assertEqual(
            r.run("git", "diff", "--name-only", "main", branch, cwd=self.fork),
            "file.txt",
        )
        self.assertEqual(
            r.run("git", "show", f"{branch}:file.txt", cwd=self.fork), "after"
        )
        self.assertTrue(self.created[0]["draft"])

    def test_retry_after_push_recovers_without_overwriting_branch(self):
        self.reject_pr = True
        self.assertEqual(self.publish()["publication"], "failed")
        branch = f"nightly-fix/{self.failure['id']}"
        before = r.run("git", "rev-parse", branch, cwd=self.fork)
        self.reject_pr = False
        result = self.publish()
        self.assertEqual(result["publication"], "created")
        self.assertNotIn("publication_error", result)
        self.assertEqual(
            r.run("git", "rev-parse", branch, cwd=self.fork), before
        )

    def test_existing_branch_with_different_patch_is_not_overwritten(self):
        self.reject_pr = True
        self.publish()
        branch = f"nightly-fix/{self.failure['id']}"
        before = r.run("git", "rev-parse", branch, cwd=self.fork)
        (self.source / "file.txt").write_text("different\n")
        r.run("git", "add", "file.txt", cwd=self.source)
        self.reject_pr = False
        self.assertEqual(self.publish()["publication"], "failed")
        self.assertEqual(
            r.run("git", "rev-parse", branch, cwd=self.fork), before
        )
        self.assertEqual(self.created, [])


if __name__ == "__main__":
    unittest.main()
