# GitHub Release Preflight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Audit and organize the repository for a future GitHub release without pushing, creating a PR, retraining, or evaluating new test sets.

**Architecture:** First establish local/remote Git and provenance facts, then audit tracked/ignored content, secrets, paths, Markdown links, configuration, and optional deployment dependencies. Apply only explicit release-blocking fixes, add provenance/index/model-card/release-preflight documentation, run the existing quality gates, and create one local documentation commit.

**Tech Stack:** Git, GitHub CLI when available, PowerShell, Python 3.10, pytest, compileall, Markdown, existing YAML configuration and experiment artifacts.

---

### Task 1: Capture GitHub and branch state

**Files:**
- Read only: Git metadata, remote refs, GitHub CLI state

- [ ] Run `gh --version`, `gh auth status`, `git remote -v`, current branch/status/HEAD, then `git fetch --all --prune` with the required safe-directory prefix.
- [ ] Record origin, default branch, tracking branch, ahead/behind counts, and any existing PR without creating or modifying one.
- [ ] Audit `branch -vv`, `branch -a`, `log --graph --all -80`, and `show-branch --all`; do not merge, cherry-pick, delete, rebase, or rewrite history.

### Task 2: Build commit and experiment provenance

**Files:**
- Create: `docs/experiments/commit_provenance.md`
- Read: experiment docs, configs, and Git history

- [ ] Map each requested Stage A/B/C/D, parity, pretrained, resume-fix, and current-documentation milestone to actual commits.
- [ ] Verify every recorded SHA exists in the current history and distinguish training commits from evaluation/documentation commits.
- [ ] State whether outputs or weights were committed; keep all generated artifacts outside Git.

### Task 3: Audit repository contents, paths, and secrets

**Files:**
- Modify only if required: `.gitignore`, documentation/config path references
- Read: all tracked files and ignored status

- [ ] Inspect `ls-files`, `status --ignored`, `count-objects`, tracked files over 1 MB, and tracked model/data/cache/IDE/temp files.
- [ ] Verify `.gitignore` covers datasets, outputs, checkpoints, pretrained weights, exports, caches, `.env`, and local IDE/environment files.
- [ ] Run bounded `git grep` scans for credentials, Windows user paths, private network paths, and personal information without printing secret values.
- [ ] Apply only explicit release-blocking path or ignore fixes; never delete user files or silently filter findings.

### Task 4: Audit and repair Markdown release surfaces

**Files:**
- Create or modify: `README.md`, `docs/experiments/README.md`, `docs/model_card_d2_seed42.md`
- Modify as needed: tracked Markdown documents with broken absolute links or stale claims

- [ ] Check repository links are relative and do not point to ignored `outputs/` as if they were GitHub files.
- [ ] Reconcile current candidate metrics, split labels, strict versus EI legacy metrics, locked-test policy, D2 multi-seed statistics, pretrained H5 provenance, and ONNX optional dependency instructions.
- [ ] Add the experiment index and model card with architecture, tensor shapes, classes, selection rules, hashes, evaluator definitions, limitations, and Raspberry Pi 5 unknowns.

### Task 5: Review code/config release blockers

**Files:**
- Read and test: `src/`, `scripts/`, `configs/`, `tests/`
- Modify only for clear release blockers, with a paired pytest test

- [ ] Verify config schema and relocation through environment/config paths, pretrained hash checking, resume RNG restoration, evaluator isolation, cleaning-manifest fail-closed behavior, locked-test restrictions, CLI diagnostics, and optional ONNX imports.
- [ ] Do not change model design, training recipe, loss, weights, augmentation, optimizer, scheduler, checkpoint selection, or test-set protocol.

### Task 6: Run quality gates and produce release preflight report

**Files:**
- Create: `docs/release_preflight_report.md`
- Read: final Git state and test outputs

- [ ] Run full pytest, compileall, diff-check, and the existing targeted config/locked/pretrained/cleaning/resume tests.
- [ ] Record ONNX skips only when their reason is the missing optional dependency; do not install dependencies just to remove skips.
- [ ] Classify changes as required release fixes, documentation, deferred improvements, or excluded artifacts.
- [ ] Create one local documentation commit with explicit file paths only; rerun tests and confirm clean status.
- [ ] Stop before `git push`, `gh pr create`, merge, rebase, branch deletion, or any other remote mutation.
