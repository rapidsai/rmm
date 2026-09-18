# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
from pathlib import Path

FLAGS = [
    "--print",
    "--mode",
    "json",
    "--no-session",
    "--no-approve",
    "--no-extensions",
    "--no-skills",
    "--no-prompt-templates",
    "--no-themes",
    "--no-context-files",
    "--offline",
]


def configure(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    model = json.loads(os.environ["MODEL_CONFIG"])
    if not isinstance(model, dict) or not isinstance(model.get("id"), str):
        raise ValueError("MODEL_CONFIG must be a pi model object with an id")
    provider = {
        "baseUrl": os.environ["INFERENCE_URL"],
        "api": os.environ["INFERENCE_API"],
        "apiKey": "$INFERENCE_API_KEY",
        "authHeader": True,
        "models": [model],
    }
    (directory / "models.json").write_text(
        json.dumps({"providers": {"inference-hub": provider}})
    )
    (directory / "settings.json").write_text(
        json.dumps(
            {"defaultProjectTrust": "never", "enableInstallTelemetry": False}
        )
    )


def command(tools: bool) -> list[str]:
    model = json.loads(os.environ["MODEL_CONFIG"])["id"]
    return [
        "pi",
        *FLAGS,
        "--provider",
        "inference-hub",
        "--model",
        model,
        *(
            ["--tools", "read,write,edit,bash,grep,find,ls"]
            if tools
            else ["--no-tools"]
        ),
    ]


def response(transcript: Path) -> dict:
    last = None
    ended = False
    with transcript.open() as stream:
        for line in stream:
            event = json.loads(line)
            if event.get("type") == "message_end":
                message = event["message"]
                if message.get("role") == "assistant":
                    last = message
            if event.get("type") == "agent_end":
                ended = True
    if not ended or last is None or last.get("stopReason") != "stop":
        raise ValueError(
            "pi did not complete successfully or omitted its final response"
        )
    text = "".join(
        block["text"] for block in last["content"] if block["type"] == "text"
    )
    result = json.loads(text)
    if not isinstance(result, dict):
        raise ValueError("pi must return a JSON object")
    return result


def invoke(prompt: str, directory: Path, transcript: Path) -> dict:
    configure(directory)
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(directory),
        "PI_CODING_AGENT_DIR": str(directory),
        "INFERENCE_API_KEY": os.environ["INFERENCE_API_KEY"],
    }
    with transcript.open("w") as output:
        subprocess.run(
            command(False),
            input=prompt,
            text=True,
            stdout=output,
            env=env,
            cwd=directory,
            check=True,
            timeout=900,
        )
    return response(transcript)
