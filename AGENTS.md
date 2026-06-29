# AGENTS.md

Operating manual for contributors and AI agents touching this repo. Read it before opening a PR.

The repo is a curriculum, not a SaaS app. The lessons are the product. Every rule below keeps 435 lessons coherent over time.

---

## Philosophy

435 lessons. 20 phases. Every algorithm built from raw math before a single framework gets imported. You write backprop, the tokenizer, the attention mechanism, and the agent loop by hand in Python, TypeScript, Rust, or Julia. Then you run the same operation through the production library so the framework stops being a black box. The "Build It / Use It" split is the spine. Each lesson ships a reusable artifact you can plug into your daily workflow.

---

## Repo layout

```
phases/
  NN-phase-slug/
    NN-lesson-slug/
      docs/en.md              # lesson explainer
      code/                   # implementation + tests
      quiz.json               # 6 questions
      outputs/                # reusable artifact (skill / prompt / agent / MCP server)
README.md                     # public face; lesson counts auto-synced
ROADMAP.md                    # phase/lesson status
glossary/terms.md             # canonical term definitions
site/
  build.js                    # parses README + ROADMAP + glossary -> data.js
  data.js                     # generated; rebuilt by CI on main push
scripts/                      # automation
.github/workflows/
  curriculum.yml              # invariant + auto-sync workflow
```

---

## Hard rules

1. **One commit per lesson directory.** Never batch multiple lessons into one commit. A 10-lesson PR has 10 commits.
2. **Conventional commit subjects** ≤72 chars: `feat(phase-NN/MM): <slug>`. Body explains why, not what.
3. **Mermaid or SVG only** for diagrams. No ASCII / Unicode box-drawing.
4. **Every fenced code block needs a language tag.** Use `text`, `json`, `python`, `typescript`, `rust`, `julia`, `bash`, `console`, `mermaid`, `yaml` as appropriate.
5. **Original implementations only.** Don't cite external curriculum repos in docs, code comments, or commit text. Cite RFCs, official specs, and academic papers when they are the canonical source.
6. **Dependency allowlist** (see `Dependencies` below). Stdlib-first.
7. **Never commit generated files**: `catalog.json` is gitignored, `site/data.js` is rebuilt by CI, `package-lock.json` is never tracked.

---

## Dependencies

| Language   | Allowed                                                                  |
|------------|--------------------------------------------------------------------------|
| Python     | `numpy`, `torch`, `h5py`, `zstandard`, `safetensors`, stdlib              |
| TypeScript | `hono`, `zod`, `ws` (only when WebSockets needed), `@hono/node-server`, Node 20+ stdlib |
| Rust       | stdlib only (single-file `rustc --edition 2021`)                          |
| Julia      | `Random`, `Statistics`, `LinearAlgebra`, `Printf` (Julia stdlib)          |

If a finding suggests a banned dep, skip it with the reason "stays stdlib-first for educational clarity."

---

## Lesson contract

### docs/en.md frontmatter

```markdown
# <Title>

> <One-line hook>

**Type:** <Learn | Build | Reference>
**Languages:** <comma-list matching the main.* files in code/>
**Prerequisites:** <comma-list of upstream lessons, or "None">
**Time:** ~<estimate in minutes>

## Learning Objectives
- <4-6 bullet points starting with a verb>
```

The `**Languages:**` field must match the languages with a `main.*` file in `code/`.

### quiz.json schema

```json
{
  "lesson": "<dir-slug>",
  "title": "<Lesson Title>",
  "questions": [
    {"stage": "pre",   "question": "...", "options": ["a","b","c","d"], "correct": 0, "explanation": ""},
    {"stage": "check", "question": "...", "options": ["a","b","c","d"], "correct": 1, "explanation": ""},
    {"stage": "check", "question": "...", "options": ["a","b","c","d"], "correct": 2, "explanation": ""},
    {"stage": "check", "question": "...", "options": ["a","b","c","d"], "correct": 1, "explanation": ""},
    {"stage": "post",  "question": "...", "options": ["a","b","c","d"], "correct": 3, "explanation": ""},
    {"stage": "post",  "question": "...", "options": ["a","b","c","d"], "correct": 0, "explanation": ""}
  ]
}
```

Exactly 6 questions: 1 pre + 3 check + 2 post. `correct` is zero-indexed. The site renderer only understands this shape — legacy `q/choices/answer` schemas crash silently.

### code/

- Runs end-to-end and exits 0 on the canonical command for the language.
- Self-terminating demo. No infinite stdin loops, no hangs on missing API keys.
- 4-6 line header comment citing the lesson's `docs/en.md` path and any spec or RFC sources.

### code/tests/

- 5+ unit tests minimum.
- Runs via the language's stdlib runner (`python3 -m unittest discover`, `npx tsx --test`, Rust/Julia inline).

### outputs/ — Prompts and skills

Each lesson ships zero or more reusable artifacts in `outputs/`. These are surfaced to learners via `scripts/install_skills.py` and the site catalog. Two artifact types exist today:

**Prompt artifact** (`outputs/prompt-<slug>.md`):

```yaml
---
name: prompt-<slug>
description: <One-line summary of what the prompt teaches or helps with>
phase: <N>
lesson: <M>
---

<System prompt body — instructions for an AI to adopt a specific teaching
persona, workflow, or decision framework.>
```

Prompts are AI tutor/system prompts that encode a teaching methodology. They are persona-driven: the body is a "You are a..." directive followed by principles, response structures, and anti-patterns. See `phases/01-math-foundations/02-vectors-matrices-operations/outputs/prompt-matrix-operations.md` for the canonical example.

**Skill artifact** (`outputs/skill-<slug>.md`):

```yaml
---
name: skill-<slug>
description: <One-line summary>
version: 1.0.0
phase: <N>
lesson: <M>
tags: [<comma-separated>]
---

# <Title>

<Reference content — decision checklists, lookup tables, recipes, or
step-by-step workflows the learner can consult during real work.>
```

Skills are reference material, not personas. They answer "how do I decide X?" or "what are the options for Y?" with structured, scannable content.

**Prompt writing conventions** (distilled from the matrix-operations example):

1. **Persona-first.** Start with "You are a..." — define the role, not the task.
2. **Principles before procedures.** List 4-6 design principles the AI must follow, each with a concrete behavioral rule.
3. **Structured response format.** Define exactly how the AI should organize its answers (e.g., geometry → formula → example → NN connection → pitfall).
4. **Scope explicitly.** List the operations/concepts the prompt covers. List what to avoid.
5. **Concrete over abstract.** Every principle needs a "show, don't tell" behavioral rule. "Start with geometry" is abstract; "Show what happens to a unit square before writing any equations" is concrete.
6. **Ban weasel words.** Forbid "obviously", "trivially", "it can be shown that" — these undermine learner confidence.

**Skill writing conventions:**

1. **Decision-first.** Structure around choices the learner faces, not around concepts.
2. **Tables over prose.** Decision matrices, comparison tables, and checklists scan faster than paragraphs.
3. **Version explicitly.** Every skill has a `version` in its frontmatter. Bump it on content changes.
4. **Tag for discovery.** Tags enable `install_skills.py --tag <tag>` filtering. Use lowercase, hyphenated tags.

Artifacts are installed via `python3 scripts/install_skills.py` (see `--help` for options). The script discovers all `outputs/prompt-*.md` and `outputs/skill-*.md` files, parses frontmatter, and copies them to a target directory.

---

## Per-PR validation

Run locally before pushing:

```bash
python3 scripts/audit_lessons.py
python3 scripts/check_readme_counts.py        # advisory — CI fixes on merge

# For each lesson touched:
cd phases/NN-phase/MM-lesson/code
python3 main.py && python3 -m unittest discover tests -v   # or the lang equivalent
```

CI gates (`.github/workflows/curriculum.yml`):

| Job                              | Trigger      | Behavior                                              |
|----------------------------------|--------------|-------------------------------------------------------|
| `audit`                          | push + PR    | Runs `audit_lessons.py`. Blocking.                    |
| `readme-counts-sync` (main only) | push to main | Rebuilds catalog + auto-fixes README counts.         |
| `site-rebuild` (main only)       | push to main | Re-runs `node site/build.js`, commits `site/data.js`. |
| `readme-counts-drift`            | PR           | Advisory only — main self-heals on merge.             |

---

## Automation contract

**CI handles automatically — do not touch in your PR:**

| Surface              | Bot                            | When                |
|----------------------|--------------------------------|---------------------|
| `catalog.json`       | rebuilt on demand (gitignored) | every CI job        |
| `README.md` counts   | `readme-counts-sync`           | on push to main     |
| `site/data.js`       | `site-rebuild`                 | on push to main     |

**You handle:**

| Surface                       | When                                                             |
|-------------------------------|------------------------------------------------------------------|
| `README.md` lesson-link rows  | when adding a new lesson — link `[Title](phases/NN-phase/MM-lesson/)` |
| `ROADMAP.md` status           | when marking a lesson complete or WIP                            |
| `glossary/terms.md`           | when introducing a term used by more than one lesson             |

**Common bug**: if `grep -c 'tree/main/phases/NN-' site/data.js` is 0 after merge, the Phase NN README rows are plain text and missing the `[Title](phases/NN-...)` markdown link. `site/build.js` derives the URL from that link.

---

## Conflict resolution

```bash
git fetch origin main
git merge --no-edit origin/main

# Catalog conflict (legacy branches only — catalog.json is gitignored now):
git rm catalog.json
git commit --no-edit

# README count conflict:
git checkout --theirs README.md
python3 scripts/build_catalog.py
python3 scripts/check_readme_counts.py --fix
git add README.md && git commit --no-edit

# site/data.js conflict:
git checkout --theirs site/data.js
node site/build.js
git add site/data.js && git commit --no-edit

git push origin <your-branch>
```

Avoid `git push --force` to a branch with open review comments. Force-push detaches them.

---

## New-lesson onboarding

```bash
mkdir -p phases/NN-phase-slug/MM-new-lesson/{docs,code/tests,outputs}

# 1. Write docs/en.md with the frontmatter above.
# 2. Write code/main.<lang> with the 4-6 line header.
# 3. Write code/tests/test_main.* with 5+ tests.
# 4. Write quiz.json with the schema above.
# 5. (Optional) Add outputs/skill-<slug>.md if the lesson ships a skill.

# 6. Add to README.md:
#    | MM | [Lesson Title](phases/NN-phase-slug/MM-new-lesson/) | Type | Lang |

# 7. Update ROADMAP.md status row.

# 8. Validate locally.

# 9. Atomic commit:
git add phases/NN-phase-slug/MM-new-lesson README.md ROADMAP.md
git commit -m "feat(phase-NN/MM): add <slug>"
git push -u origin <your-branch>
gh pr create --title "feat(phase-NN/MM): add <slug>" --body "<5-line summary>"
```

`site/data.js` regenerates on merge — leave it for CI.

---

Last reviewed: 2026-05-27.
