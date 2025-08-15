"""
Automatic Prompt Optimization (APO) for Text-to-Video Generation
Inspired by "Automatic Prompt Optimization with Gradient Descent and Beam Search"
https://arxiv.org/abs/2305.03495

This implementation follows the ProTeGi methodology with:
1. Natural language "gradients" (criticism)
2. Semantic prompt editing (propagation)
3. Beam search for exploration
4. Bandit selection for exploitation

ProTeGi Improvements Implemented:
- Minibatch evaluation with concrete error detection
- Explicit paraphrasing step (LLM_mc) for local exploration
- UCB bandit selection using empirical means over pulls
- Targeted criticism based on specific detected errors

USAGE:
------
export OPENROUTER_API_KEY='your-openrouter-key-here'
python test_prompt_opt.py
"""

import os
from dotenv import load_dotenv
import openai
import random
import time
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np
import re

load_dotenv()

@dataclass
class PromptCandidate:
    """A candidate prompt with its performance metrics"""
    text: str
    score: float = 0.0
    generation_count: int = 0
    feedback_history: List[Dict] = field(default_factory=list)
    parent_prompt: Optional[str] = None
    edit_description: Optional[str] = None
    # ProTeGi improvements
    all_scores: List[float] = field(default_factory=list)  # Track all evaluations for mean
    mean_score: float = 0.0  # Running mean for UCB
    concrete_errors: List[str] = field(default_factory=list)  # Specific errors for better criticism

@dataclass
class VideoFeedback:
    """Feedback for a generated video"""
    visual_quality: float  # 0-10
    prompt_adherence: float  # 0-10
    creativity: float  # 0-10
    technical_quality: float  # 0-10
    overall_score: float  # 0-10
    criticism: str  # Natural language feedback
    suggestions: str  # Improvement suggestions
    concrete_errors: List[str] = field(default_factory=list)  # Specific issues found

@dataclass
class MinibatchSample:
    """A sample for minibatch evaluation (ProTeGi-style)"""
    description: str  # What the video should show
    expected_elements: List[str]  # Key elements that should be present
    quality_criteria: Dict[str, str]  # Quality aspects to evaluate

class PromptOptimizer:
    """
    Automatic Prompt Optimization using gradient descent and beam search.
    
    Implements the ProTeGi methodology with natural language gradients,
    semantic prompt editing, beam search exploration, and UCB bandit selection.
    """
    
    def __init__(self, 
                 api_key: str,
                 base_url: str = "https://api.openai.com/v1",
                 model: str = "gpt-4o-mini",
                 beam_size: int = 5,
                 max_iterations: int = 10,
                 learning_rate: float = 0.7,
                 minibatch_size: int = 3,
                 ucb_c: float = 2.0,
                 ucb_prior: float = 10.0,
                 k_paraphrases: int = 2):
        """
        Initialize the prompt optimizer.
        
        Args:
            api_key: API key (OpenRouter).
            base_url: API base URL (OpenRouter).
            model: Model name to use.
            beam_size: Number of prompt candidates to maintain.
            max_iterations: Maximum optimization iterations.
            learning_rate: How aggressively to apply gradients (0-1).
            minibatch_size: Size of minibatch for evaluation (ProTeGi-style).
            ucb_c: UCB exploration weight for bandit selection.
            ucb_prior: Prior score for unseen arms in UCB selection.
            k_paraphrases: Number of paraphrases to generate per prompt.
            
        Performance Notes:
            - Higher beam_size = more exploration but slower convergence.
            - Higher learning_rate = more aggressive prompt changes.
            - Larger minibatch_size = more stable evaluation but slower.
            - Higher ucb_c = more exploration, lower ucb_c = more exploitation.
            - Higher ucb_prior = more exploration of new candidates.
        """
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model
        self.beam_size = beam_size
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.minibatch_size = minibatch_size
        self.ucb_c = ucb_c
        self.ucb_prior = ucb_prior
        self.k_paraphrases = k_paraphrases
        
        # OpenRouter specific headers
        self.extra_headers = {}
        
        # Track optimization history
        self.optimization_history = []
        self.current_beam = []
        self.iteration = 0
        
        # ProTeGi improvements
        self.evaluation_dataset = self._create_evaluation_dataset()
        self.llm_text_cache = {}  # Cache only expensive LLM text (criticism/suggestions), not numeric scores
        
    def _create_evaluation_dataset(self) -> List[MinibatchSample]:
        """
        Create a dataset for minibatch evaluation (ProTeGi-style).
        
        Returns:
            List of MinibatchSample objects for concrete error detection.
        """
        return [
            MinibatchSample(
                description="Two pets exercising together indoors",
                expected_elements=["cat", "dog", "treadmill", "indoor setting", "movement/action"],
                quality_criteria={
                    "specificity": "Are the animals clearly described?",
                    "environment": "Is the setting detailed?", 
                    "action": "Is the movement/activity clear?",
                    "visual_richness": "Are there sufficient visual details?"
                }
            ),
            MinibatchSample(
                description="Dynamic indoor fitness scene with animals",
                expected_elements=["pets", "exercise equipment", "indoor gym", "energetic movement"],
                quality_criteria={
                    "energy": "Does the prompt convey energy/motion?",
                    "composition": "Is the scene well-composed?",
                    "details": "Are there sufficient descriptive elements?",
                    "cinematic": "Does it suggest good video quality?"
                }
            ),
            MinibatchSample(
                description="Playful animal workout scenario", 
                expected_elements=["animals", "fitness", "playful mood", "workout equipment"],
                quality_criteria={
                    "mood": "Is the emotional tone clear?",
                    "context": "Is the scenario well-established?",
                    "visual_appeal": "Would this create an appealing video?",
                    "coherence": "Do all elements work together?"
                }
            )
        ]
    
    def evaluate_on_minibatch(self, prompt: str, llm_feedback: bool = True) -> VideoFeedback:
        """
        ProTeGi-style minibatch evaluation with concrete errors.
        
        Args:
            prompt: Text prompt to evaluate.
            llm_feedback: Whether to generate LLM criticism/suggestions (expensive).
            
        Returns:
            VideoFeedback with scores and concrete error analysis.
            
        Notes:
            - Set llm_feedback=False for fast pre-evaluation during UCB selection.
            - LLM calls occur only when llm_feedback=True or during δ/LLM_mc.
            - Pre-evaluation is LLM-free and uses cached error signatures.
        """
        
        # Always compute fresh numeric scores (for proper bandit "multiple pulls")
        minibatch = random.sample(self.evaluation_dataset, min(self.minibatch_size, len(self.evaluation_dataset)))
        
        total_score = 0.0
        all_errors = []
        
        for sample in minibatch:
            sample_score, sample_errors = self._evaluate_against_sample(prompt, sample)
            total_score += sample_score
            all_errors.extend(sample_errors)
        
        overall_score = total_score / len(minibatch)
        
        # Cache only expensive LLM text generation, not numeric scores
        if llm_feedback and all_errors:
            # Create error signature to avoid stale criticism for different error patterns
            error_signature = "|".join(sorted(set([e.split(':',1)[0] for e in all_errors[:5]])))
            cache_key = f"{prompt}||{error_signature}"
            
            if cache_key in self.llm_text_cache:
                criticism, suggestions = self.llm_text_cache[cache_key]
            else:
                criticism = self._generate_criticism_from_errors(prompt, overall_score, all_errors)
                suggestions = self._generate_suggestions(prompt, criticism)
                self.llm_text_cache[cache_key] = (criticism, suggestions)
        else:
            # Fast path: just use error summary without LLM calls
            criticism = f"Issues found: {'; '.join(all_errors[:3])}" if all_errors else "No major issues detected"
            suggestions = "—"
        
        return VideoFeedback(
            visual_quality=overall_score,
            prompt_adherence=overall_score,
            creativity=overall_score,
            technical_quality=overall_score,
            overall_score=overall_score,
            criticism=criticism,
            suggestions=suggestions,
            concrete_errors=all_errors
        )
    
    def _evaluate_against_sample(self, prompt: str, sample: MinibatchSample) -> Tuple[float, List[str]]:
        """
        Evaluate prompt against a single sample with concrete error detection.
        
        Args:
            prompt: Text prompt to evaluate.
            sample: MinibatchSample containing expected elements and quality criteria.
            
        Returns:
            Tuple of (score, errors) for the prompt against this sample.
        """
        errors = []
        scores = []
        
        # Check for expected elements
        for element in sample.expected_elements:
            if element.lower() not in prompt.lower():
                errors.append(f"Missing expected element: {element}")
                scores.append(3.0)  # Low score for missing elements
            else:
                scores.append(7.0)  # Good score for present elements
        
        # Evaluate quality criteria
        for criterion, question in sample.quality_criteria.items():
            score = random.uniform(4.0, 8.5)  # Simulated evaluation
            if score < 6.0:
                errors.append(f"Low {criterion}: {question}")
            scores.append(score)
        
        return sum(scores) / len(scores), errors
    
    def _generate_criticism_from_errors(self, prompt: str, score: float, errors: List[str]) -> str:
        """
        Generate criticism based on concrete errors (more targeted than before).
        
        Uses error signature caching (top-5 keys) to avoid stale criticism text
        for different error patterns.
        
        Args:
            prompt: Text prompt to criticize.
            score: Current evaluation score.
            errors: List of concrete errors found during evaluation.
            
        Returns:
            Targeted criticism addressing specific detected issues.
        """
        if not errors:
            return self._generate_criticism(prompt, score)  # Fallback to original method
        
        system_prompt = """You are an expert video generation critic. Based on concrete errors found during evaluation, provide focused criticism."""
        
        error_list = "\n".join([f"- {error}" for error in errors[:5]])  # Limit to top 5 errors
        
        user_prompt = f"""
        Prompt: "{prompt}"
        Score: {score:.1f}/10
        
        Concrete errors found:
        {error_list}
        
        Provide targeted criticism addressing these specific issues:
        """
        
        try:
            response = self.client.chat.completions.create(
                extra_headers=self.extra_headers,
                extra_body={},
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=200,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating targeted criticism: {e}")
            return f"Issues found: {'; '.join(errors[:3])}"
    
    def _paraphrase_prompts(self, prompts: List[str], k_per_prompt: int = 2) -> List[str]:
        """
        ProTeGi's explicit paraphrasing step (LLM_mc) - multiple paraphrases per edit.
        
        This implements the paper's local exploration through semantic variation
        while maintaining core meaning and visual elements.
        
        Args:
            prompts: List of prompts to paraphrase.
            k_per_prompt: Number of paraphrases to generate per prompt.
            
        Returns:
            List of paraphrased prompts (total = len(prompts) × k_per_prompt).
            
        Notes:
            - Parsing: Uses robust regex to handle "1.", "1)", "1 -" formats.
            - LLM calls occur during δ/LLM_mc for local exploration.
            - Pre-evaluation is LLM-free and uses cached error signatures.
        """
        
        paraphrased = []
        
        for prompt in prompts:
            system_prompt = f"""Return EXACTLY {k_per_prompt} paraphrases, numbered 1..{k_per_prompt}. Keep the same core meaning and visual elements while using different vocabulary and sentence structure."""
            
            user_prompt = f'Original: "{prompt}"'
            
            try:
                response = self.client.chat.completions.create(
                    extra_headers=self.extra_headers,
                    extra_body={},
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=250,
                    temperature=0.8
                )
                
                # Robust regex parsing for numbered items: "1.", "1)", "1 -", etc.
                response_text = response.choices[0].message.content.strip()
                items = re.findall(r'^\s*\d+[\.\)\-]\s*(.+)$', response_text, flags=re.MULTILINE)
                
                if not items:
                    # Fallback: split by lines and take non-empty ones
                    lines = [line.strip() for line in response_text.split('\n') if line.strip()]
                    items = lines[:k_per_prompt] if lines else [response_text.strip()]
                
                # Ensure exactly k_per_prompt items
                paraphrased.extend(items[:k_per_prompt])
                    
            except Exception as e:
                print(f"Error paraphrasing: {e}")
                paraphrased.append(prompt)  # Fallback to original
        
        return paraphrased

    # def simulate_video_feedback(self, prompt: str) -> VideoFeedback:
    #     """
    #     DEPRECATED: Legacy function - use evaluate_on_minibatch() instead.
        
    #     This was the original simulation before ProTeGi improvements.
    #     All evaluation now flows through evaluate_on_minibatch() which provides:
    #     - Concrete error detection via dataset samples.
    #     - Targeted criticism based on specific issues.
    #     - More realistic evaluation methodology.
        
    #     Args:
    #         prompt: Text prompt to evaluate.
            
    #     Returns:
    #         VideoFeedback object (redirects to evaluate_on_minibatch).
    #     """
    #     print("⚠️  WARNING: Using deprecated simulate_video_feedback - use evaluate_on_minibatch() instead")
        
    #     # Redirect to improved method
    #     return self.evaluate_on_minibatch(prompt)
    
    def _generate_criticism(self, prompt: str, score: float) -> str:
        """
        Generate natural language criticism of the prompt.
        
        Args:
            prompt: Text prompt to criticize.
            score: Current evaluation score.
            
        Returns:
            Constructive criticism focusing on specificity, composition, and technical feasibility.
        """
        
        system_prompt = """You are an expert video generation critic. Analyze text-to-video prompts and provide constructive criticism focusing on:
        1. Specificity and detail level
        2. Visual composition elements
        3. Action/motion descriptions
        4. Cinematic qualities
        5. Technical feasibility
        
        Be specific and actionable in your criticism."""
        
        user_prompt = f"""
        Analyze this text-to-video prompt and provide criticism:
        
        Prompt: "{prompt}"
        Video Quality Score: {score:.1f}/10
        
        What are the main weaknesses and areas for improvement?
        """
        
        try:
            response = self.client.chat.completions.create(
                extra_headers=self.extra_headers,
                extra_body={},
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=200,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating criticism: {e}")
            return f"Score of {score:.1f}/10 suggests room for improvement in prompt specificity and visual details."
    
    def _generate_suggestions(self, prompt: str, criticism: str) -> str:
        """
        Generate improvement suggestions based on criticism.
        
        Args:
            prompt: Original text prompt.
            criticism: Criticism text to base suggestions on.
            
        Returns:
            Specific improvement suggestions for video generation quality.
        """
        
        system_prompt = """You are an expert at improving text-to-video prompts. Based on criticism, suggest specific improvements that would make the prompt generate better videos."""
        
        user_prompt = f"""
        Original prompt: "{prompt}"
        Criticism: "{criticism}"
        
        Suggest 2-3 specific improvements to make this prompt better for video generation:
        """
        
        try:
            response = self.client.chat.completions.create(
                extra_headers=self.extra_headers,
                extra_body={},
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=150,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating suggestions: {e}")
            return "Add more visual details, specify camera movements, and include environmental context."
    
    def _apply_gradient(self, prompt: str, feedback: VideoFeedback) -> List[str]:
        """
        Apply 'gradient descent' by editing prompt based on feedback.
        
        This is the core of APO - using criticism to generate improved prompts
        through semantic editing and LLM-based refinement.
        
        Args:
            prompt: Original prompt to improve.
            feedback: VideoFeedback containing criticism and suggestions.
            
        Returns:
            List of improved prompt variations (beam_size count).
            
        Algorithm:
            1. Generate LLM-based improvements addressing specific criticism.
            2. Parse numbered response using robust regex.
            3. Fallback to manual variations if parsing fails.
        """
        
        system_prompt = f"""You are an expert prompt engineer. Your task is to improve a text-to-video prompt based on criticism.

        IMPORTANT: Generate {self.beam_size} DIFFERENT improved versions of the prompt. Each should address the criticism while maintaining the core concept.

        Guidelines:
        1. Address specific issues mentioned in the criticism
        2. Keep the core visual concept intact
        3. Add cinematic and technical details
        4. Make each version distinctly different
        5. Ensure prompts are 15-30 words each

        Learning Rate: {self.learning_rate} (higher = more aggressive changes)"""
        
        user_prompt = f"""
        Original prompt: "{prompt}"
        
        Feedback Score: {feedback.overall_score:.1f}/10
        Criticism: "{feedback.criticism}"
        Suggestions: "{feedback.suggestions}"
        
        Generate {self.beam_size} improved versions of this prompt. Format as:
        1. [improved prompt 1]
        2. [improved prompt 2]
        3. [improved prompt 3]
        etc.
        """
        
        try:
            response = self.client.chat.completions.create(
                extra_headers=self.extra_headers,
                extra_body={},
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=400,
                temperature=0.8
            )
            
            # Robust regex parsing for numbered items (same as paraphrases): "1.", "1)", "1 -", etc.
            content = response.choices[0].message.content.strip()
            items = re.findall(r'^\s*\d+[\.\)\-]\s*(.+)$', content, flags=re.MULTILINE)
            
            # If parsing failed or not enough items, create variations manually
            if len(items) < self.beam_size:
                print(f"Parsing yielded {len(items)} items, need {self.beam_size}. Creating manual variations...")
                improved_prompts = self._create_manual_variations(prompt, feedback)
            else:
                improved_prompts = items[:self.beam_size]
            
            return improved_prompts
            
        except Exception as e:
            print(f"Error applying gradient: {e}")
            return self._create_manual_variations(prompt, feedback)
    
    def _create_manual_variations(self, prompt: str, feedback: VideoFeedback) -> List[str]:
        """
        Create manual prompt variations as fallback.
        
        This method is used when parsing δ output fails or yields insufficient results.
        Provides basic cinematic variations to maintain beam diversity.
        
        Args:
            prompt: Original prompt to create variations for.
            feedback: VideoFeedback containing criticism and suggestions.
            
        Returns:
            List of prompt variations including the original.
        """
        variations = [prompt]  # Include original
        
        # Add cinematic terms
        cinematic_additions = [
            f"Cinematic {prompt} with dramatic lighting",
            f"High-quality {prompt} in 4K resolution",
            f"Professional {prompt} with depth of field",
            f"Stunning {prompt} with dynamic camera movement"
        ]
        
        variations.extend(cinematic_additions[:self.beam_size-1])
        return variations
    
    def _select_candidates_bandit(self, candidates: List[PromptCandidate]) -> List[PromptCandidate]:
        """
        ProTeGi-improved UCB bandit selection using empirical means.
        
        Algorithm: Upper Confidence Bound (UCB) with empirical mean rewards.
        Balances exploitation of good prompts with exploration of new ones.
        
        Args:
            candidates: List of prompt candidates to select from.
            
        Returns:
            Top beam_size candidates selected by UCB scores.
            
        Notes:
            - UCB Formula: mean_reward + ucb_c * sqrt(ln(total_trials) / trials_i).
            - Uses ucb_prior for unseen arms to encourage exploration.
            - Higher ucb_c increases exploration, lower ucb_c increases exploitation.
        """
        if len(candidates) <= self.beam_size:
            return candidates
        
        # Calculate UCB scores using empirical means (ProTeGi improvement)
        total_trials = sum(c.generation_count for c in candidates)
        ucb_scores = []
        
        for candidate in candidates:
            if candidate.generation_count == 0:
                # Use high finite prior instead of infinity to avoid crowding out good incumbents
                ucb_score = self.ucb_prior  # Use the new parameter
            else:
                # ProTeGi improvement: Use empirical mean, not just latest score
                mean_reward = candidate.mean_score
                exploration_bonus = self.ucb_c * np.sqrt(np.log(total_trials + 1) / candidate.generation_count)
                ucb_score = mean_reward + exploration_bonus
            
            ucb_scores.append(ucb_score)
        
        # Select top candidates by UCB score
        candidate_scores = list(zip(candidates, ucb_scores))
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        
        selected = [candidate for candidate, _ in candidate_scores[:self.beam_size]]
        return selected
    
    def _update_candidate_stats(self, candidate: PromptCandidate, feedback: VideoFeedback):
        """
        Update candidate statistics for proper UCB (ProTeGi improvement).
        
        Args:
            candidate: PromptCandidate to update statistics for.
            feedback: VideoFeedback containing evaluation results.
            
        Notes:
            - Updates generation count for UCB exploration bonus calculation.
            - Maintains running mean score for empirical UCB selection.
            - Tracks concrete errors for targeted criticism generation.
        """
        candidate.generation_count += 1
        candidate.all_scores.append(feedback.overall_score)
        candidate.concrete_errors.extend(feedback.concrete_errors)
        
        # Calculate running mean (empirical mean for UCB)
        candidate.mean_score = sum(candidate.all_scores) / len(candidate.all_scores)
        candidate.score = feedback.overall_score  # Keep latest for compatibility
    
    def optimize_prompt(self, initial_prompt: str) -> Dict:
        """
        Main optimization loop - the APO algorithm.
        
        Algorithm Complexity: O(iterations × beam_size × minibatch_size).
        
        Args:
            initial_prompt: Starting text-to-video prompt.
            
        Returns:
            Dictionary with optimization results and history.
            
        Returns (dict):
            best_prompt: str
            best_score: float
            initial_prompt: str
            initial_score: float
            improvement: float
            iterations_completed: int
            optimization_history: List[...]
            final_beam: List[{"prompt": str, "score": float}]
            
        Example:
            >>> optimizer = PromptOptimizer(api_key="...")
            >>> result = optimizer.optimize_prompt("A cat running")
            >>> print(f"Best: {result['best_prompt']}")
        """
        
        print(f"🚀 Starting Automatic Prompt Optimization")
        print(f"Initial prompt: \"{initial_prompt}\"")
        print(f"Beam size: {self.beam_size}, Max iterations: {self.max_iterations}")
        print("=" * 60)
        
        # Initialize beam with the original prompt
        initial_candidate = PromptCandidate(text=initial_prompt)
        self.current_beam = [initial_candidate]
        
        best_prompt = initial_candidate
        best_score = 0.0
        
        for iteration in range(self.max_iterations):
            print(f"\n🔄 Iteration {iteration + 1}/{self.max_iterations}")
            
            iteration_results = []
            new_candidates = []
            
            # ProTeGi-style evaluation on minibatch  
            for candidate in self.current_beam:
                print(f"  📊 Evaluating: \"{candidate.text[:50]}...\"")
                
                # ProTeGi improvement: Minibatch evaluation with concrete errors
                feedback = self.evaluate_on_minibatch(candidate.text)
                
                # ProTeGi improvement: Update stats properly for UCB
                self._update_candidate_stats(candidate, feedback)
                
                candidate.feedback_history.append({
                    'iteration': iteration + 1,
                    'feedback': feedback,
                    'timestamp': time.time()
                })
                
                iteration_results.append({
                    'prompt': candidate.text,
                    'score': feedback.overall_score,
                    'feedback': feedback,
                    'concrete_errors': feedback.concrete_errors  # ProTeGi addition
                })
                
                print(f"    Score: {feedback.overall_score:.2f}/10 (Mean: {candidate.mean_score:.2f})")
                if feedback.concrete_errors:
                    print(f"    Errors: {len(feedback.concrete_errors)} found")
                
                # Track best prompt using mean score (ProTeGi improvement)
                if candidate.mean_score > best_score:
                    best_score = candidate.mean_score
                    best_prompt = candidate
                    print(f"    🏆 New best score!")
                
                # Generate improved versions using gradient descent
                if iteration < self.max_iterations - 1:  # Don't generate on last iteration
                    improved_prompts = self._apply_gradient(candidate.text, feedback)
                    
                    # ProTeGi improvement: Add explicit paraphrasing step (LLM_mc) with k=2 per edit
                    paraphrased_prompts = self._paraphrase_prompts(improved_prompts, k_per_prompt=self.k_paraphrases)
                    
                    # Combine both improved and paraphrased versions
                    all_new_prompts = improved_prompts + paraphrased_prompts
                    
                    for i, improved_text in enumerate(all_new_prompts):
                        edit_type = "Gradient" if i < len(improved_prompts) else "Paraphrase"
                        new_candidate = PromptCandidate(
                            text=improved_text,
                            parent_prompt=candidate.text,
                            edit_description=f"{edit_type} from iter {iteration+1}"
                        )
                        new_candidates.append(new_candidate)
            
            # Store iteration results
            self.optimization_history.append({
                'iteration': iteration + 1,
                'results': iteration_results,
                'best_score': best_score,
                'beam_size': len(self.current_beam)
            })
            
            # ProTeGi Algorithm 1: Select only from expansions (Bi+1 = Select_b(C, m))
            if new_candidates and iteration < self.max_iterations - 1:  # Don't select on last iteration
                # Evaluate each new candidate once before UCB (fast path - no LLM calls)
                for new_candidate in new_candidates:
                    if new_candidate.generation_count == 0:
                        feedback = self.evaluate_on_minibatch(new_candidate.text, llm_feedback=False)
                        self._update_candidate_stats(new_candidate, feedback)
                        print(f"    📋 Pre-evaluated new candidate: {feedback.overall_score:.2f}/10 (fast)")
                
                # Select from expansions only (not current_beam + new_candidates)
                self.current_beam = self._select_candidates_bandit(new_candidates)
                print(f"  🎯 Selected {len(self.current_beam)} candidates from {len(new_candidates)} expansions")
            
            # Early stopping if we're doing really well
            if best_score >= 9.0:
                print(f"  🎉 Excellent score achieved! Early stopping.")
                break
        
        # Final results
        print("\n" + "=" * 60)
        print("🏁 Optimization Complete!")
        print(f"Best prompt: \"{best_prompt.text}\"")
        print(f"Best score: {best_score:.2f}/10")
        
        # Fix: Use initial score as baseline (not arbitrary final beam member)
        initial_score = self.optimization_history[0]['results'][0]['score'] if self.optimization_history else 0
        print(f"Improvement: {best_score - initial_score:.2f} points")
        
        return {
            'best_prompt': best_prompt.text,
            'best_score': best_score,
            'initial_prompt': initial_prompt,
            'initial_score': self.optimization_history[0]['results'][0]['score'] if self.optimization_history else 0,
            'improvement': best_score - (self.optimization_history[0]['results'][0]['score'] if self.optimization_history else 0),
            'iterations_completed': len(self.optimization_history),
            'optimization_history': self.optimization_history,
            'final_beam': [{'prompt': c.text, 'score': c.score} for c in self.current_beam]
        }

def demo_optimization():
    """
    Demo the prompt optimization system.
    
    This function demonstrates the complete APO workflow:
    1. Initialize optimizer with OpenRouter configuration.
    2. Run optimization on test prompts.
    3. Display results and save to JSON file.
    
    Configuration:
        - Uses OpenRouter with openai/gpt-4o-mini.
        - Beam size: 3, Max iterations: 5.
        - Learning rate: 0.8, Minibatch size: 3.
        - UCB parameters: ucb_c=2.0, ucb_prior=10.0, k_paraphrases=2.
        
    Returns:
        List of optimization results for each test prompt.
    """
    
    # Example prompts to optimize
    test_prompts = [
        "A cat and a dog running on a treadmill.",
        #"Beautiful landscape with mountains",
        #"Person walking in the city",
        #"Sunrise over the ocean",
        #"Flying through clouds"
    ]
    
    # Configuration - OpenRouter only with openai/gpt-4o-mini
    API_KEY = os.getenv("OPENROUTER_API_KEY")
    BASE_URL = "https://openrouter.ai/api/v1"
    MODEL = "openai/gpt-4o-mini"  # Only using this model
    
    if not API_KEY:
        print("⚠️  Please set your OpenRouter API key as environment variable OPENROUTER_API_KEY")
        print("💡 Run: export OPENROUTER_API_KEY='your-openrouter-key-here'")
        print("\n🔑 Get OpenRouter API key at: https://openrouter.ai/keys")
        return
    
    print(f"🔗 Using OpenRouter with model: {MODEL}")
    
    optimizer = PromptOptimizer(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL,
        beam_size=3,
        max_iterations=5,
        learning_rate=0.8,
        minibatch_size=3,  # ProTeGi improvement
        ucb_c=2.0,
        ucb_prior=10.0,
        k_paraphrases=2
    )
    
    # Optimize each test prompt
    results = []
    
    for prompt in test_prompts:
        print(f"\n{'='*80}")
        print(f"OPTIMIZING: {prompt}")
        print(f"{'='*80}")
        
        result = optimizer.optimize_prompt(prompt)
        results.append(result)
        
        print(f"\nRESULT SUMMARY:")
        print(f"  Original: {result['initial_prompt']}")
        print(f"  Optimized: {result['best_prompt']}")
        print(f"  Improvement: +{result['improvement']:.2f} points")
    
    # Overall summary
    print(f"\n{'='*80}")
    print("OVERALL RESULTS")
    print(f"{'='*80}")
    
    total_improvement = sum(r['improvement'] for r in results)
    avg_improvement = total_improvement / len(results)
    
    print(f"Average improvement: +{avg_improvement:.2f} points")
    print(f"Best performing prompt: {max(results, key=lambda x: x['best_score'])['best_prompt']}")
    
    return results

if __name__ == "__main__":
    # Run the demo
    results = demo_optimization()
    
    # Save results to file
    if results:
        with open('prompt_optimization_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n📁 Results saved to prompt_optimization_results.json")
