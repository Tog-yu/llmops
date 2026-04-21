# AGENTS.md

This file defines the working agreement for this repository.

## Default Execution Policy

- Prefer direct execution over repeated confirmation.
- For coding, debugging, dependency checks, local service startup, configuration changes, test runs, and documentation updates:
  act directly and continue until the task is fully handled.
- Do not stop at analysis if a reasonable implementation path exists.
- When a task involves multiple steps, keep going through inspection, code change, verification, and documentation unless blocked by a real external dependency.

## Approval Boundary

- The user grants broad permission for normal repository work.
- Do not ask for confirmation again for ordinary actions such as:
  - reading and editing repo files
  - updating `.env` and local config for development
  - installing or checking dependencies
  - starting or restarting local services
  - running tests, linters, migrations, and local verification commands
  - inspecting Docker containers and local processes
  - updating docs
- Still pause for clearly destructive or irreversible actions, including:
  - dropping databases
  - deleting large amounts of user data
  - force-resetting git history
  - removing files outside the task scope
  - any action that would destroy important state

## Working Style

- Make reasonable assumptions and proceed.
- Surface blockers only when they are real blockers.
- If a command fails because of environment or permission issues, try the next practical path instead of stopping early.
- Prefer fixing root causes over adding temporary workarounds.
- Keep the user informed briefly, but do not require step-by-step permission for normal progress.

## Scope

- This policy applies to work inside this repository.
- Repository safety still matters:
  never intentionally revert unrelated user changes,
  and never perform destructive cleanup unless required by the task.
