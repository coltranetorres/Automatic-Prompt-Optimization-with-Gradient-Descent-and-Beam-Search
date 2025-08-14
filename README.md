# Automatic-Prompt-Optimization-with-Gradient-Descent-and-Beam-Search

### ProTeGi-Style Automatic Prompt Optimization (APO) for Text-to-Video

This repo implements Automatic Prompt Optimization for text-to-video generation inspired by:

- Automatic Prompt Optimization with Gradient Descent and Beam Search (ProTeGi) — https://arxiv.org/abs/2305.03495

It performs *textual “gradient descent”* over prompts using LLM feedback, explores with beam search, and exploits with a bandit selector.

✨ What this does

Given an initial prompt (e.g., “A cat and a dog running on a treadmill.”), the system:

1. Evaluates the prompt on a small minibatch of criteria and detects concrete errors (e.g., missing lighting, weak action).

2. Uses an LLM to produce targeted criticism (∇) and suggestions.

3. Applies semantic edits (δ) via the LLM to generate multiple improved prompts.

4. Expands locally via paraphrases (LLM_mc).

5. Pre-evaluates children once (fast path, no LLM) and picks the next beam via UCB (empirical means).

6. Repeats for a few iterations, returning the best prompt + history.

It’s fully automatic out of the box, with an optional human-in-the-loop mode for interactive scoring.

✅ Alignment with ProTeGi

- Textual gradients (∇): targeted criticism produced from concrete errors collected during minibatch evaluation.

- Edits (δ): LLM generates several improved prompts from the criticism/suggestions.

- Local exploration (LLM_mc): explicit paraphrasing step with multiple paraphrases per edit.

- Beam search: expand each prompt to many candidates, then select only from children for the next beam.

- Bandit selection: UCB using empirical mean reward over multiple pulls (not just the last score).
