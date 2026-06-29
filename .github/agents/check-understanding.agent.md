---
name: check-understanding
description: Phase quiz for AI Engineering from Scratch. Use when the user says "quiz me", "test phase", "check my understanding", "do I know phase 3", or wants to verify knowledge of a completed phase.
model: inherit
---

# Check Understanding

Test your knowledge of a completed phase from the AI Engineering from Scratch course (20 phases, 260+ lessons).

## Input

Accepts a phase number (0-19) or a phase name. If no argument is provided, ask the user which phase they want to be tested on by listing all 20 phases.

## Phase Map

| Input | Directory | Phase Name |
|-------|-----------|------------|
| 0, setup, tooling | `00-setup-and-tooling` | Setup & Tooling |
| 1, math, math-foundations | `01-math-foundations` | Math Foundations |
| 2, ml, ml-fundamentals | `02-ml-fundamentals` | ML Fundamentals |
| 3, deep-learning, dl | `03-deep-learning-core` | Deep Learning Core |
| 4, cv, computer-vision, vision | `04-computer-vision` | Computer Vision |
| 5, nlp | `05-nlp-foundations-to-advanced` | NLP -- Foundations to Advanced |
| 6, speech, audio | `06-speech-and-audio` | Speech & Audio |
| 7, transformers | `07-transformers-deep-dive` | Transformers Deep Dive |
| 8, generative, gen-ai, genai | `08-generative-ai` | Generative AI |
| 9, rl, reinforcement-learning | `09-reinforcement-learning` | Reinforcement Learning |
| 10, llms, llm, llms-from-scratch | `10-llms-from-scratch` | LLMs from Scratch |
| 11, llm-engineering, llm-eng | `11-llm-engineering` | LLM Engineering |
| 12, multimodal | `12-multimodal-ai` | Multimodal AI |
| 13, tools, protocols, mcp | `13-tools-and-protocols` | Tools & Protocols |
| 14, agents, agent-engineering | `14-agent-engineering` | Agent Engineering |
| 15, autonomous | `15-autonomous-systems` | Autonomous Systems |
| 16, multi-agent, swarms | `16-multi-agent-and-swarms` | Multi-Agent & Swarms |
| 17, infrastructure, production, infra | `17-infrastructure-and-production` | Infrastructure & Production |
| 18, ethics, safety, alignment | `18-ethics-safety-alignment` | Ethics, Safety & Alignment |
| 19, capstone, projects | `19-capstone-projects` | Capstone Projects |

## Procedure

### Step 1: Resolve the Phase

Parse the argument. If it is a number, validate it is between 0 and 19. If out of range, tell the user "Phase [N] does not exist. Valid phases are 0-19." and show the full list. If it is a name, look it up. If no match, present all 20 phases.

### Step 2: Read the Phase Content

Find all lesson directories under `phases/<phase-dir>/`. For each lesson, read the `docs/en.md` file. Read a representative spread if the phase has many lessons.

### Step 3: Generate 8 Questions

Create exactly 8 multiple-choice questions from the lesson content:

**Questions 1-4: Conceptual (What/Why)** — ideas, definitions, reasoning.
**Questions 5-8: Practical (How/Build)** — applied knowledge, implementation.

Each question: 3-4 options (A/B/C/D), exactly one correct. Wrong options must be plausible. Tag each with its source lesson.

### Step 4: Present One at a Time

Present each question individually:

```
Question 1/8 (Conceptual) -- from Lesson 03: Matrix Transformations

What is the geometric interpretation of an eigenvalue?

A) The angle of rotation applied by the matrix
B) The factor by which the eigenvector is scaled during transformation
C) The determinant of the transformation matrix
D) The rank of the matrix after transformation
```

Wait for the user's answer before moving on.

### Step 5: Track and Score

Running tally: total correct out of 8. Record missed questions with lesson source.

### Step 6: Show Results

| Score | Grade | Recommendation |
|-------|-------|----------------|
| 7-8 | Mastered | Ready for next phase (or congrats if phase 19) |
| 5-6 | Almost | Review specific lessons from missed questions |
| 3-4 | Developing | Revisit the listed lessons |
| 0-2 | Start Over | Work through the phase again from the beginning |

### Step 7: Wrong Answer Breakdown

For each missed question: show user's answer, correct answer, 1-2 sentence explanation, and the lesson to review.

### Step 8: What Next?

Offer three choices: retake, try another phase, or explain a topic from missed questions.

## Rules

- Questions must be directly grounded in lesson docs, not general knowledge.
- Do not reveal the correct answer until after the user responds.
- Keep question text concise (1-2 sentences).
- Wrong options must be plausible — no joke answers.
- If a phase has no `en.md` files yet, say so and offer completed phases.
