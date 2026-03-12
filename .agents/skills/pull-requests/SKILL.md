---
name: pull-requests
description: Open pull requests against rapidsai/rmm. Use when creating or editing PRs.
---

## Rules for pull requests
- Target `main` unless told otherwise.
- Always push branches to a user's fork of rapidsai/rmm, not to the upstream repository.
- PR descriptions should follow `.github/pull_request_template.md`.
- The PR title appears in the CHANGELOG; keep it concise.
- Every PR should reference an issue.
  - Check for an issue first.
  - If no issue exists, ask the user whether to create one. Suggest relevant context for the issue.
  - If an issue exists or is created, mention it in the PR description like "Closes #1234."
