# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import agent
import remediate


class PiIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.requests = []
        requests = self.requests

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                request = json.loads(
                    self.rfile.read(int(self.headers["Content-Length"]))
                )
                requests.append(request)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                tool_results = [
                    m for m in request["messages"] if m["role"] == "tool"
                ]
                if request.get("tools") and not tool_results:
                    command = (
                        'test -z "$GH_TOKEN" && test -z "$SLACK_WEBHOOK_URL" && '
                        'test "$RAPIDS_CUDA_VERSION" = "test-cuda" && '
                        "printf fixed > repaired.txt"
                    )
                    delta = {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "bash",
                                    "arguments": json.dumps(
                                        {"command": command}
                                    ),
                                },
                            }
                        ],
                    }
                    reason = "tool_calls"
                else:
                    delta = {
                        "role": "assistant",
                        "content": json.dumps({"failures": []}),
                    }
                    reason = "stop"
                for content, finish in ((delta, None), ({}, reason)):
                    event = {
                        "id": "test",
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "model": "test-model",
                        "choices": [
                            {
                                "index": 0,
                                "delta": content,
                                "finish_reason": finish,
                            }
                        ],
                    }
                    self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(
            target=self.server.serve_forever, daemon=True
        )
        self.thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.port = self.server.server_address[1]
        self.environment = {
            "MODEL_CONFIG": '{"id":"test-model"}',
            "INFERENCE_API": "openai-completions",
            "INFERENCE_URL": f"http://127.0.0.1:{self.port}/v1",
            "INFERENCE_API_KEY": "fake-test-key",
            "RAPIDS_CUDA_VERSION": "test-cuda",
            "GH_TOKEN": "must-not-reach-agent",
            "SLACK_WEBHOOK_URL": "must-not-reach-agent",
        }

    def test_real_pi_cli_tool_free_analysis(self):
        with (
            tempfile.TemporaryDirectory() as temp,
            patch.dict(os.environ, self.environment),
        ):
            result = agent.invoke(
                "Return JSON with an empty failures array.",
                Path(temp),
                Path(temp) / "events.jsonl",
            )
        self.assertEqual(result, {"failures": []})
        self.assertEqual(len(self.requests), 1)
        self.assertFalse(self.requests[0].get("tools"))

    def test_real_pi_fixer_edits_source_without_inheriting_service_tokens(
        self,
    ):
        with (
            tempfile.TemporaryDirectory() as temp,
            patch.dict(os.environ, self.environment),
        ):
            root = Path(temp)
            source = root / "source"
            source.mkdir()
            remediate.run("git", "init", str(source))
            result = agent.invoke(
                "Fix repaired.txt.",
                root / "config",
                root / "events.jsonl",
                cwd=source,
                tools=True,
                timeout=120,
            )
            self.assertEqual(result, {"failures": []})
            self.assertEqual((source / "repaired.txt").read_text(), "fixed")
        self.assertEqual(len(self.requests), 2)
        self.assertIn(
            "bash",
            [tool["function"]["name"] for tool in self.requests[0]["tools"]],
        )


if __name__ == "__main__":
    unittest.main()
