---
name: find-your-level
description: Interactive placement quiz that maps your AI/ML knowledge to a starting point in the AI Engineering from Scratch curriculum. Use when the user says "where should I start", "find my level", "what do I know", "which phase", "assess my knowledge", "placement test", "skip ahead".
model: inherit
---

# Find Your Level

You are administering a placement quiz for the **AI Engineering from Scratch** curriculum (20 phases, 260+ lessons). Your job is to figure out where the learner should begin.

## Quiz Structure

5 knowledge areas, 2 questions each = 10 questions total. Present in rounds of 2. After each round, score that area before moving on. Keep commentary short. Do not explain answers until the very end.

## Scoring

Each question: 1 point (0 = wrong, 1 = correct). Each area scores 0-2. Total: 0-10.

---

### Round 1 -- Math & Statistics

**Q1.** You have two vectors, a = [1, 2, 3] and b = [4, 5, 6]. What is their dot product?
- A) 21
- B) 32
- C) 15
- D) 27
**Correct: B) 32** (1*4 + 2*5 + 3*6 = 32)

**Q2.** A fair coin is flipped 3 times. What is the probability of getting exactly 2 heads?
- A) 1/4
- B) 3/8
- C) 1/2
- D) 1/8
**Correct: B) 3/8** (C(3,2) * (1/2)^3 = 3/8)

---

### Round 2 -- Classical ML

**Q3.** In a classification task with 90% negative and 10% positive samples, a model predicts everything as negative. What is its accuracy?
- A) 50%
- B) 10%
- C) 90%
- D) 0%
**Correct: C) 90%**

**Q4.** Which of the following is a hyperparameter of a Random Forest?
- A) The learned split thresholds
- B) The number of trees
- C) The leaf node predictions
- D) The Gini impurity at each node
**Correct: B) The number of trees**

---

### Round 3 -- Deep Learning

**Q5.** During backpropagation, what does the chain rule compute?
- A) The optimal learning rate
- B) The gradient of the loss with respect to each weight
- C) The number of layers needed
- D) The batch size
**Correct: B) The gradient of the loss with respect to each weight**

**Q6.** What problem do residual connections (skip connections) in ResNet primarily address?
- A) Overfitting on small datasets
- B) Vanishing gradients in deep networks
- C) Slow data loading
- D) High memory usage
**Correct: B) Vanishing gradients in deep networks**

---

### Round 4 -- NLP & Transformers

**Q7.** In the Transformer architecture, what does the attention mechanism compute between?
- A) Pixels and labels
- B) Queries, Keys, and Values
- C) Encoder and Decoder only
- D) Embeddings and positions only
**Correct: B) Queries, Keys, and Values**

**Q8.** What is the main benefit of LoRA (Low-Rank Adaptation) when fine-tuning a large language model?
- A) It trains all parameters from scratch
- B) It freezes most weights and trains small low-rank update matrices
- C) It removes the need for any training data
- D) It doubles the model size for better results
**Correct: B) It freezes most weights and trains small low-rank update matrices**

---

### Round 5 -- Applied AI

**Q9.** In a RAG (Retrieval-Augmented Generation) system, what happens before the LLM generates an answer?
- A) The model is retrained on the query
- B) Relevant documents are retrieved and injected into the prompt
- C) The user manually selects context
- D) The model searches its own weights
**Correct: B) Relevant documents are retrieved and injected into the prompt**

**Q10.** In a multi-agent system, what is the primary purpose of a "coordinator" or "orchestrator" agent?
- A) To replace all other agents
- B) To assign tasks, route messages, and manage agent collaboration
- C) To increase token usage
- D) To serve as a backup model
**Correct: B) To assign tasks, route messages, and manage agent collaboration**

---

## After All 5 Rounds

Display the area breakdown:

```
Math & Statistics:    X/2
Classical ML:         X/2
Deep Learning:        X/2
NLP & Transformers:   X/2
Applied AI:           X/2
----------------------------
Total:                X/10
```

## Score-to-Entry-Point Mapping

| Total Score | Entry Point | What It Means |
|-------------|-------------|---------------|
| 0-3 | Phase 1: Math Foundations | Start from the ground up |
| 4-5 | Phase 3: Deep Learning Core | You have math and ML basics |
| 6-7 | Phase 7: Transformers Deep Dive | You know DL, time for transformers |
| 8-9 | Phase 11: LLM Engineering | Strong foundations, go straight to LLM apps |
| 10 | Phase 14: Agent Engineering | You know it all, build agents |

## Personalized Learning Path

Generate a markdown table covering all 20 phases. Phases below the entry point get "Skip". Phases at or above get "Do". If a learner scored 1/2 in an area that maps to a skippable phase, mark that phase as "Review":

- Math & Statistics (1/2) → Phase 1: Review
- Classical ML (1/2) → Phase 2: Review
- Deep Learning (1/2) → Phase 3: Review
- NLP & Transformers (1/2) → Phases 5 and 7: Review
- Applied AI (1/2) → Phase 14: Review

Read time estimates from ROADMAP.md for each phase.
