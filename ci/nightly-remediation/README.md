# Nightly remediation with pi

Investigate RMM nightly failures, attempt independent fixes on CPU or GPU, and
send draft proposals to a fork for human review. The implementation lives in RMM
for this pilot; it does not depend on an unmerged shared workflow.

## Workflow

1. **Analyze (CPU):** fetch the selected runs' latest-attempt job logs and ask a
   tool-free pi session for independent root causes. Validate the JSON schema,
   coverage of all failed jobs, and CPU routing for build/dependency failures.
2. **Fix (CPU/GPU matrices):** one pi session per root cause. GPU fixers
   have `max-parallel: 4`; workflow-level concurrency prevents overlapping runs
   in this repository. Each agent investigates, edits, builds, and tests. A
   separate publication step creates a draft PR whose head and base are both in
   `bdice-bot/rmm`, or the fork selected by `fork-owner`.
3. **Report (CPU):** collect structured results, including unresolved failures
   and missing fixer results, and post problems and PR links to Slack. Reports
   explicitly label validation as **agent-reported**, not independently verified.

The initial workflow is manual-only. It requires run IDs and the actual tested
source SHA because RMM's nightly `workflow_dispatch` input `sha` can differ from
GitHub's workflow run `head_sha`. Select runs that tested the same source SHA.
Automatic nightly discovery and scheduling are intentionally not enabled yet.

## Configure before dispatching

Repository variables:

| Variable | Value |
| --- | --- |
| `NIGHTLY_REMEDIATION_MODEL_CONFIG` | A pi model JSON object, including `id`, with the context window, reasoning, and compatibility settings required by your InferenceHub model |
| `NIGHTLY_REMEDIATION_INFERENCE_URL` | InferenceHub API base URL; use the URL appropriate for the chosen API, not a full request endpoint |
| `NIGHTLY_REMEDIATION_INFERENCE_API` | Pi API type, such as `openai-completions` or `openai-responses` |
| `NIGHTLY_REMEDIATION_APP_ID` | GitHub App ID |

Repository secrets:

| Secret | Purpose |
| --- | --- |
| `NIGHTLY_REMEDIATION_INFERENCE_API_KEY` | InferenceHub service credential |
| `NIGHTLY_REMEDIATION_APP_PRIVATE_KEY` | GitHub App private key |
| `SLACK_WEBHOOK_NIGHTLY_STATUS_URL` | Incoming webhook for the nightly Slack channel |

Pi 0.85.1 is installed from `@earendil-works/pi-coding-agent` with npm lifecycle
scripts disabled. Its generated `models.json` references the inference credential
through an environment variable; the credential is not written to that file.
Provider configuration is independent of the agent harness.

Create the destination fork in advance. Install the App on that fork with
**Contents**, **Pull requests**, and **Workflows** write permissions. The last
permission allows proposed fixes to CI workflow files. Do not grant the App
write access to the upstream repository. The workflow also checks that the
configured destination is a fork of `rapidsai/rmm`, not upstream itself.
Publishing with an App can trigger workflows in the fork: do not give that fork
privileged runners or secrets that unreviewed proposals could access.

The fork must have the selected base branch and contain the tested source SHA
on that branch. Synchronize it before dispatching. The workflow never rewrites
the fork's base branch and never force-pushes a fix branch. A repeated signature
at the same source SHA and base branch reuses an existing PR, including a closed PR, without
changing it. Semantic deduplication across different signatures or source SHAs
is not implemented.

All jobs use `rapidsai/ci-conda:26.12-latest` as their GitHub Actions job image.
Node.js, pi, and Python dependencies are installed directly in that environment;
there is no custom Dockerfile or nested Docker invocation. CPU and GPU runner
labels are configured in `.github/workflows/nightly-remediation.yaml`. The fixer
job forwards the runner's `NVIDIA_VISIBLE_DEVICES` into its job container.
The pilot uses amd64 and the CUDA/toolchain versions supplied by the CI image;
failures requiring another architecture or version must be reported as blocked
rather than claimed fixed. No build-cluster credentials are supplied.

### Runner safety

Use only runners approved for agent workloads, with ephemeral machines and
network policy preventing access to instance metadata, internal services, and
other workloads. The job container is **not** an agent sandbox. Confirm runner
isolation policy before a live run. The agent has internet access, its inference
credential, and access to the job filesystem, including writable git metadata
and workflow helpers.

Pi receives a disposable configuration directory and a limited environment that
preserves CUDA/build settings without deliberately passing GitHub or Slack
credentials. The fork App token is minted after pi returns, but subsequent steps
share the job environment: this is not a security boundary against agent code.
Pi's extensions, project settings, automatic context files, skills, and startup
network traffic are disabled; the fixer may explicitly read relevant repository
guidance.

Only structured analysis, results, and Slack payloads are uploaded, for seven
days. Raw pi transcripts are temporary and are not uploaded. Treat all proposed
patches, diagnoses, and validation claims as untrusted until human review.

## Run manually

The workflow must exist on the repository's default branch before GitHub will
accept `workflow_dispatch`. A PR that only adds the workflow cannot itself be
manually dispatched. Bootstrap the reviewed manual-only workflow on the default
branch, or test in a fork where it exists on the default branch and separately
configure that fork's runners and secrets.

```bash
gh workflow run nightly-remediation.yaml -R rapidsai/rmm \
  -f run-ids='[BUILD_RUN_ID,TEST_RUN_ID]' \
  -f source-sha=FULL_TESTED_SOURCE_SHA \
  -f target-branch=main \
  -f fork-owner=bdice-bot
```

Replace the run ID placeholders with integers and the SHA with the nightly
source commit. This is a live operation: it can consume model/GPU resources,
create draft PRs, and post to the configured Slack channel.

## Local tests

```bash
python -m pip install -r ci/nightly-remediation/requirements.txt
python -m unittest discover -s ci/nightly-remediation/tests -v
```

To exercise the actual pi CLI against a local mock inference API, without
calling InferenceHub, GitHub, or Slack:

```bash
npm install --ignore-scripts --prefix /tmp/nightly-pi @earendil-works/pi-coding-agent@0.85.1
export PATH="/tmp/nightly-pi/node_modules/.bin:$PATH"
python ci/nightly-remediation/test_pi_integration.py -v
```

These tests cover protocol handling, source edits, and environment propagation,
not RMM compilation, GPU execution, InferenceHub compatibility, GitHub App
permissions, or Slack delivery. Those require an approved, configured end-to-end
pilot run.
