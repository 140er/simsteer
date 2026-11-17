## Verification before a PR

- Run the verify skill (`.claude/skills/verify/SKILL.md`) before opening a PR and put its evidence in the PR body.
- Run the cheap checks (`scripts/check.sh`: ruff, pytest, CLI smoke) before every push.
- A failing check gets at most 3 fix rounds. Then stop and ask the maintainer for a decision, with what failed, where, and the evidence.
- Never retry a flaky test or sleep around it. Find the root cause, or delete the test and explain why in the commit.
- A PR never modifies the checks or rules that judge it. The merge gate reads REVIEW.md and its config from the base branch, never from the PR head.
- Review focus for this repo: see `REVIEW.md`.
