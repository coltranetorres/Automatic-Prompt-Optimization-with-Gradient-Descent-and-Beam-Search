# Automatic Prompt Optimization with Gradient Descent and Beam Search

### ProTeGi-Style Automatic Prompt Optimization (APO) for Text-to-Video

This repo implements Automatic Prompt Optimization for text-to-video generation inspired by:

- Automatic Prompt Optimization with Gradient Descent and Beam Search (ProTeGi): https://arxiv.org/abs/2305.03495

It performs textual "gradient descent" over prompts using LLM feedback, explores with beam search, and exploits with a bandit selector.

## ✨ What this does

Given an initial prompt (e.g., "A cat and a dog running on a treadmill."), the system:

1. Evaluates the prompt on a small minibatch of criteria and detects concrete errors (e.g., missing lighting, weak action).
2. Uses an LLM to produce targeted criticism (nabla) and suggestions.
3. Applies semantic edits (delta) via the LLM to generate multiple improved prompts.
4. Expands locally via paraphrases (`LLM_mc`).
5. Pre-evaluates children once (fast path, no LLM) and picks the next beam via UCB (empirical means).
6. Repeats for a few iterations, returning the best prompt plus history.

It is fully automatic out of the box, with an optional human-in-the-loop mode for interactive scoring.

## ✅ Alignment with ProTeGi

- Textual gradients (nabla): targeted criticism produced from concrete errors collected during minibatch evaluation.
- Edits (delta): the LLM generates several improved prompts from the criticism/suggestions.
- Local exploration (`LLM_mc`): explicit paraphrasing step with multiple paraphrases per edit.
- Beam search: expand each prompt to many candidates, then select only from children for the next beam.
- Bandit selection: UCB using empirical mean reward over multiple pulls (not just the last score).

## 🧩 Key Components (code map)

- `evaluate_on_minibatch`: Minibatch evaluation with concrete error detection. Produces numeric `overall_score` and an error list. Caches LLM text only (criticism/suggestions), not numeric scores (so bandit pulls remain meaningful).
- `_generate_criticism_from_errors` (nabla): Turns the error list into targeted criticism via the LLM.
- `_apply_gradient` (delta): Uses criticism/suggestions to create multiple edited prompts.
- `_paraphrase_prompts` (`LLM_mc`): Creates k paraphrases per edited prompt with robust parsing for "1.", "1)", "1 -", etc.
- `_update_candidate_stats`: Tracks `generation_count`, `all_scores`, and updates `mean_score` (empirical mean).
- `_select_candidates_bandit`: UCB over empirical means with a finite prior for unseen children to avoid crowding.
- `optimize_prompt`: The outer loop (Algorithm 1): expand -> pre-eval children (fast) -> select only from expansions -> repeat.

## 🧪 Evaluation (current vs. production)

Current repo uses a simulated evaluator built from simple checks (presence/absence of elements, random draws for quality criteria). It is fast and shows the optimization mechanics.

To go production:
- Swap `evaluate_on_minibatch` with your real video generation + scoring (e.g., CLIP similarity, VQA, shot quality metrics, or human ratings).
- Keep the same interfaces (return a `VideoFeedback`) and you will get ProTeGi's full loop with real data.

## 🗂 Output

- Console logs for each iteration (scores, errors found, selections).
- `prompt_optimization_results.json` with:
  - `best_prompt`, `best_score`, `initial_prompt`, `improvement`
  - `iterations_completed`
  - `optimization_history` (per-iteration results)
  - `final_beam`

## ⚠️ Notes

- This repo uses a simulated evaluator as a proxy. For meaningful results, plug in your real video generation + scoring pipeline in `evaluate_on_minibatch()` and use real training/evaluation data. Keep the `VideoFeedback` return shape unchanged.
- Set `llm_feedback=False` during child pre-eval to cut costs; enable it when you need targeted criticism.
- Example metrics: CLIP similarity to targets, aesthetic/motion quality, VMAF, or human/KPI-based scores (watch time, CTR).
