pip install torch

# Core imports for baseline evaluation
import os
import json
import argparse
import random
import time
from pathlib import Path
from typing import Optional

import torch
import numpy as np
import wandb
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# math_utils
# Math answer extraction and equivalence checking
# Handles numeric equality, fractions, LaTeX and symbolic expressions

import re
import math
from typing import Optional, Union
from fractions import Fraction

try:
  from sympy import sympify, simplify, N, Abs
  from sympy.parsing.latex import parse_latex
  SYMPY_AVAILABLE = True
except ImportError:
  SYMPY_AVAILABLE = False
  print("Warning: sympy not available. Falling back to numeric comparison only.")


# Answer extraction
def extract_answer_gsm8k(text: str) -> Optional[str]:
  """
  Extract the final numeric answer from a GSMK8-style response.
  GSM8K ground truth answers follow the pattern: '#### <number>'
  Model answers may use 'The answer is X', '\\boxed{X}', or plain numbers.
  """
  # Try #### pattern (ground truth format)
  match = re.search(r'####\s*([\d,\.\-]+)', text)
  if match:
    return _normalize_number(match.group(1))

  # Try \boxed{} (LaTeX format)
  match = re.search(r'\\boxed\{([^}]+)\}', text)
  if match:
      return _normalize_latex(match.group(1))

  # Try "The answer is X" / "= X" patterns
  patterns = [
      r'[Tt]he\s+answer\s+is\s*([\d,\.\-\/]+)',
      r'[Tt]he\s+final\s+answer\s+is\s*([\d,\.\-\/]+)',
      r'=\s*([\d,\.\-]+)\s*$',
      r'≈\s*([\d,\.\-]+)',
  ]
  for pattern in patterns:
      match = re.search(pattern, text)
      if match:
          return _normalize_number(match.group(1))

  # Last number in the text (fallback)
  numbers = re.findall(r'-?[\d,]+\.?\d*', text)
  if numbers:
      return _normalize_number(numbers[-1])

  return None
def extract_answer_math(text: str) -> Optional[str]:
    """
    Extract the final answer from a MATH-500-style response.
    MATH answers are more complex (expressions, fractions, sets).
    """
    # Last \boxed{} in the text (accounts for intermediate boxed steps)
    matches = re.findall(r'\\boxed\{([^}]+)\}', text)
    if matches:
        return _normalize_latex(matches[-1])

    # After </think> tag if present
    think_match = re.search(r'</think>\s*(.*)', text, re.DOTALL)
    if think_match:
        remainder = think_match.group(1).strip()
        boxed = re.findall(r'\\boxed\{([^}]+)\}', remainder)
        if boxed:
            return _normalize_latex(boxed[-1])
        # Try plain answer after </think>
        return remainder.split('\n')[0].strip() if remainder else None

    return extract_answer_gsm8k(text)


def _normalize_number(s: str) -> str:
    """Strip commas, trailing zeros, normalize to float string."""
    s = s.replace(',', '').strip()
    try:
        val = float(s)
        # Return as int string if it's a whole number
        if val == int(val):
            return str(int(val))
        return str(val)
    except ValueError:
        return s


def _normalize_latex(s: str) -> str:
    """Normalize a LaTeX math string for comparison."""
    s = s.strip()
    # Remove text wrappers
    s = re.sub(r'\\text\{([^}]+)\}', r'\1', s)
    # Normalize fractions: \frac{a}{b} → a/b
    s = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'\1/\2', s)
    # Remove LaTeX formatting
    s = s.replace('\\,', '').replace('\\ ', '').replace('$', '')
    return s.strip()

# ---------------------------------------------------------------------------
# Answer Equivalence
# ---------------------------------------------------------------------------

def math_equal(pred: Optional[str], gold: Optional[str], tolerance: float = 1e-6) -> bool:
    """
    Check if pred and gold are mathematically equivalent.
    Handles: integers, floats, fractions, percentages, LaTeX expressions.
    """
    if pred is None or gold is None:
        return False

    pred = str(pred).strip()
    gold = str(gold).strip()

    # Exact string match (fastest path)
    if pred == gold:
        return True

    # Normalize and retry
    pred_norm = _normalize_latex(_normalize_number(pred) if _is_numeric_string(pred) else pred)
    gold_norm = _normalize_latex(_normalize_number(gold) if _is_numeric_string(gold) else gold)
    if pred_norm == gold_norm:
        return True

    #  Numeric comparison
    pred_num = _try_parse_number(pred)
    gold_num = _try_parse_number(gold)
    if pred_num is not None and gold_num is not None:
        if gold_num == 0:
            return abs(pred_num - gold_num) < tolerance
        return abs(pred_num - gold_num) / (abs(gold_num) + 1e-8) < tolerance

    # Fraction comparison
    pred_frac = _try_parse_fraction(pred)
    gold_frac = _try_parse_fraction(gold)
    if pred_frac is not None and gold_frac is not None:
        return pred_frac == gold_frac

    # Sympy symbolic comparison (last resort + expensive!)
    if SYMPY_AVAILABLE:
        return _sympy_equal(pred, gold)

    return False


def _is_numeric_string(s: str) -> bool:
    try:
        float(s.replace(',', ''))
        return True
    except ValueError:
        return False


def _try_parse_number(s: str) -> Optional[float]:
    """Try to parse s as a float, handling percentages and commas."""
    s = s.replace(',', '').strip()
    if s.endswith('%'):
        try:
            return float(s[:-1]) / 100
        except ValueError:
            return None
    try:
        return float(s)
    except ValueError:
        return None


def _try_parse_fraction(s: str) -> Optional[Fraction]:
    """Try to parse s as a Fraction."""
    s = s.strip()
    # Handle LaTeX fraction
    match = re.match(r'(\-?\d+)/(\-?\d+)$', s)
    if match:
        try:
            return Fraction(int(match.group(1)), int(match.group(2)))
        except (ValueError, ZeroDivisionError):
            return None
    return None


def _sympy_equal(pred: str, gold: str) -> bool:
    """Use sympy to check symbolic equality."""
    try:
        # Try direct sympify
        pred_expr = sympify(pred)
        gold_expr = sympify(gold)
        diff = simplify(pred_expr - gold_expr)
        return diff == 0
    except Exception:
        try:
            # Try LaTeX parsing
            pred_expr = parse_latex(pred)
            gold_expr = parse_latex(gold)
            diff = simplify(pred_expr - gold_expr)
            return diff == 0
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Reward Hacking Detectors
# ---------------------------------------------------------------------------

def detect_reward_hacking(completion: str, reward: float) -> dict:
    """
    Detect common reward hacking patterns.
    Returns a dict with flags and details.
    """
    flags = {}

    # Empty or trivial <think> section
    think_match = re.search(r'<think>(.*?)</think>', completion, re.DOTALL)
    if think_match:
        think_content = think_match.group(1).strip()
        flags['empty_think'] = len(think_content) < 20
        flags['think_length'] = len(think_content.split())
    else:
        flags['no_think_tag'] = True
        flags['think_length'] = 0

    # Answer repetition (e.g., "42 42 42")
    words = completion.split()
    if len(words) > 5:
        last_words = words[-10:]
        flags['answer_repetition'] = len(set(last_words)) < len(last_words) * 0.4

    # Excessive length (padding)
    flags['excessive_length'] = len(completion.split()) > 1500

    # High reward but suspicious
    flags['reward'] = reward
    flags['suspicious'] = (
        reward > 0.8
        and flags.get('empty_think', False)
    )

    return flags


# Few-shot CoT Examples (GSM8K style)
# These examples are used only for the few-shot baseline condition; they are not used for training or fine-tuning

GSM8K_FEW_SHOT = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the workers plant today?",
        "answer": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6 trees planted.\n#### 6"
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "answer": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.\n#### 5"
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "answer": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.\n#### 39"
    },
    {
        "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "answer": "Jason started with 20 lollipops. Then he gave some to Denny and now has 12. So he gave Denny 20 - 12 = 8 lollipops.\n#### 8"
    },
    {
        "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "answer": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9.\n#### 9"
    },
    {
        "question": "There were nine computers in the server room. Five more computers were installed each day, from Monday to Thursday. How many computers are now in the server room?",
        "answer": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 = 29.\n#### 29"
    },
    {
        "question": "Michael had 58 golf balls. On Tuesday, he lost 23 golf balls. On Wednesday, he lost 2 more. How many golf balls did he have at the end of Wednesday?",
        "answer": "Michael started with 58 golf balls. After losing 23 on Tuesday, he had 58 - 23 = 35. After losing 2 more on Wednesday, he had 35 - 2 = 33 golf balls.\n#### 33"
    },
    {
        "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "answer": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 * 3 = 15 dollars. 23 - 15 = 8.\n#### 8"
    },
]

# Prompt builders
# Zero-shot uses only the target question
# Few-shot prepends 8 solved GSM8K-style examples

def build_zero_shot_prompt(question: str) -> str:
    return (
        "Solve the following math problem step by step. "
        "Show your reasoning clearly and state the final answer. \n\n"
        f"Problem: {question}\n\n"
        "Solution:"
    )

def build_few_shot_prompt(question: str, examples: list) -> str:
  prompt = "Solve each math problem step by step. \n\n"
  for ex in examples:
    prompt += f"Problem: {ex['question']}\nSolution: {ex['answer']}\n\n"
  prompt += f"Problem: {question}\nSolution:"
  return prompt


def build_chat_prompt(tokenizer, question: str, system: str = None) -> str:
  """
  Build a prompt using the model's chat template if available.
  Falls back to a raw prompt for base (non-instruct) models.
  """
  if not hasattr(tokenizer, 'apply_chat_template') or tokenizer.chat_template is None:
    return build_zero_shot_prompt(question)

  messages = []
  if system:
    messages.append({"role": "system", "content": system})
  messages.append({"role": "user", "content": question})

  return tokenizer.apply_chat_template(
      messages,
      tokenize=False,
      add_generation_prompt=True,
  )


# Dataset Loaders
# GSM8K is the primary benchmark used in the reported baseline results
# MATH-500 support is included for extension, but may require additional cleanup
def load_gsm8k(split: str = "test", num_samples: Optional[int] = None, seed: int = 42):
  """Load GSM8K test set."""
  ds = load_dataset("openai/gsm8k", "main", split=split)
  if num_samples and num_samples < len(ds):
    ds = ds.shuffle(seed=seed).select(range(num_samples))
  return [{"question": ex["question"], "answer": ex["answer"]} for ex in ds]

def load_math500(num_samples: Optional[int] = None, seed: int = 42):
  """
  Load MATH-500 benchmark.
  Falls back to a MATH subset if the specific split isn't available.
  """
  try:
    ds = load_dataset("hendrycks/competition_math", split="test")
    # MATH-500 is a curated 500-problem subset
    if num_samples and num_samples < len(ds):
      ds = ds.shuffle(seed=seed).select(range(num_samples))
    return [
        {"question": ex["problem"], "answer": ex["solution"], "level": ex.get("level", "?")}
        for ex in ds
    ]
  except Exception as e:
    print(f"Warning: Could not load MATH-500 ({e}). Falling back to GSM8K only.")
    return []

# Model Inference
# BaselineEvaluator wraps tokenizer/model loading and batched generation
# The model is used as-is with no SFT, LoRA, or RL updates in Phase 1


class BaselineEvaluator:
  def __init__(
      self,
      model_name: str,
      device: str = "auto",
      max_new_tokens: int = 512,
      dtype: torch.dtype = torch.bfloat16
  ):
    self.model_name = model_name
    self.max_new_tokens = max_new_tokens

    print(f"Loading tokenizer {model_name}")
    self.tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",
    )
    if self.tokenizer.pad_token is None:
      self.tokenizer.pad_token = self.tokenizer.eos_token

    print(f"Loading model: {model_name} (dtype={dtype})")
    self.model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    self.model.eval()
    print(f"Model loaded. Parameters: {sum(p.numel() for p in self.model.parameters()) / 1e9:.2f}B")

  @torch.no_grad()
  def generate(
      self,
      prompts: list[str],
      temperature: float = 0.0,
      num_return_sequences: int = 1,
      batch_size: int = 4,
  ) -> list[list[str]]:
    """
    Generate completions for a list of prompts.
    Returns list of lists: outer=prompts, inner = sequences per prompt
    """
    all_outputs = []

    for i in range(0, len(prompts), batch_size):
      batch_prompts = prompts[i: i + batch_size]

      inputs = self.tokenizer(
          batch_prompts,
          return_tensors="pt",
          padding=True,
          truncation=True,
          max_length=1024,
      ).to(self.model.device)

      gen_kwargs = dict(
          **inputs,
          max_new_tokens=self.max_new_tokens,
          pad_token_id=self.tokenizer.pad_token_id,
          eos_token_id=self.tokenizer.eos_token_id,
          num_return_sequences=num_return_sequences
      )

      if temperature == 0.0:
        # Greedy decoding
        gen_kwargs.update(do_sample=False)
      else:
        # Sampling
        gen_kwargs.update(
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
        )

      output_ids = self.model.generate(**gen_kwargs)

      # Decode only the new tokens (strip the input prompt)
      input_len = inputs["input_ids"].shape[1]
      new_tokens = output_ids[:, input_len:]
      decoded = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

      # Reshape: (batch_size * num_seqs, ) -> (batch_size, num_seqs)
      for j in range(len(batch_prompts)):
        seqs = decoded[j * num_return_sequences: (j+1) * num_return_sequences]
        all_outputs.append(seqs)
    return all_outputs

# ---------------------------------------------------------------------------
# Evaluation Logic
# This section computes baseline metrics used for comparison with Phase 2 SFT and Phase 3 GRPO
# pass@1 uses greedy decoding; pass@k uses sampling.
# ---------------------------------------------------------------------------

def evaluate_on_benchmark(
    evaluator: BaselineEvaluator,
    dataset: list[dict],
    mode: str,               # "zero_shot" or "few_shot"
    benchmark: str,          # "gsm8k" or "math500"
    pass_k: int = 1,
    temperature: float = 0.7,
    batch_size: int = 4,
) -> dict:
    """
    Run evaluation and return metrics dict.
    pass_k > 1 triggers sampling pass@k evaluation.
    """
    extract_fn = extract_answer_gsm8k if benchmark == "gsm8k" else extract_answer_math

    prompts = []
    gold_answers = []

    for ex in dataset:
        question = ex["question"]
        if mode == "zero_shot":
            prompt = build_zero_shot_prompt(question)
        elif mode == "few_shot":
            prompt = build_few_shot_prompt(question, GSM8K_FEW_SHOT[:8])
        else:
            raise ValueError(f"Unknown mode: {mode}")

        prompts.append(prompt)
        raw_gold = ex["answer"]
        gold_answers.append(extract_fn(raw_gold) or raw_gold)

    # determine sampling strategy
    if pass_k == 1:
        num_seqs = 1
        temp = 0.0   # greedy for pass@1
    else:
        num_seqs = pass_k
        temp = temperature

    print(f"\n→ Generating {len(prompts)} prompts × {num_seqs} sequences each...")
    t0 = time.time()

    all_completions = evaluator.generate(
        prompts,
        temperature=temp,
        num_return_sequences=num_seqs,
        batch_size=batch_size,
    )

    elapsed = time.time() - t0
    print(f"  Generation done in {elapsed:.1f}s ({len(prompts) * num_seqs / elapsed:.1f} samples/s)")

    # Score
    results = []
    pass1_correct = 0
    passk_correct = 0
    response_lengths = []
    has_reasoning = 0
    # --- Phase 2 compatible tracking ---
    think_tag_flags = []
    think_lengths = []        # word count of <think>...</think> content
    empty_think_count = 0     # <think> present but nearly empty (< 10 words)

    for i, (completions, gold) in enumerate(zip(all_completions, gold_answers)):
        pred_answers = [extract_fn(c) for c in completions]
        corrects = [math_equal(p, gold) for p in pred_answers]

        # Pass@1: first completion (greedy / first sample)
        if corrects[0]:
            pass1_correct += 1

        # Pass@k: any completion correct
        if any(corrects):
            passk_correct += 1

        # Diagnostics on first completion only
        c = completions[0]
        response_lengths.append(len(c.split()))

        # Heuristic: "has reasoning" = math ops or connective language
        reasoning_patterns = [
            r'\d+\s*[\+\-\×\÷\*\/]\s*\d+',
            r'therefore', r'so\s', r'thus', r'step',
        ]
        import re
        has_reasoning += any(re.search(p, c, re.IGNORECASE) for p in reasoning_patterns)

        # --- <think> tag analysis (Phase 2 compatible keys) ---
        has_open = "<think>" in c
        has_close = "</think>" in c
        has_both = has_open and has_close
        think_tag_flags.append(has_both)

        # Extract think-block word count
        think_match = re.search(r'<think>(.*?)</think>', c, re.DOTALL)
        think_word_count = len(think_match.group(1).split()) if think_match else 0
        think_lengths.append(think_word_count)

        # Empty-think: tag present but trivially short
        if has_both and think_word_count < 10:
            empty_think_count += 1

        results.append({
            "question": dataset[i]["question"][:100] + "...",
            "gold": gold,
            "predictions": pred_answers[:3],
            "corrects": corrects[:3],
            "completion_preview": c[:200],
        })

    n = len(dataset)
    metrics = {
        # --- identity ---
        "mode": mode,
        "benchmark": benchmark,
        "n_samples": n,

        # --- accuracy (Phase 2 reads these exact keys) ---
        "pass_at_1": pass1_correct / n,
        f"pass_at_{pass_k}": passk_correct / n,

        # --- think-tag metrics (Phase 2 reads these exact keys) ---
        "think_tag_rate": float(np.mean(think_tag_flags)),          # was pct_with_think_tags
        "avg_think_length_words": float(np.mean(think_lengths)),     # new
        "empty_think_rate": empty_think_count / n,                   # new

        # --- response quality ---
        "avg_response_length_words": float(np.mean(response_lengths)),
        "pct_with_reasoning": has_reasoning / n,

        # --- kept for backwards compatibility / display ---
        "pct_with_think_tags": float(np.mean(think_tag_flags)),      # alias of think_tag_rate

        # --- timing ---
        "generation_time_s": elapsed,
        "throughput_samples_per_s": len(prompts) * num_seqs / elapsed,
    }

    return metrics, results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 1: LLM Baseline Evaluation")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B",
                        help="HuggingFace model name or local path")
    parser.add_argument("--benchmarks", nargs="+", default=["gsm8k"],
                        choices=["gsm8k", "math500"],
                        help="Benchmarks to evaluate on")
    parser.add_argument("--mode", nargs="+", default=["zero_shot", "few_shot"],
                        choices=["zero_shot", "few_shot"],
                        help="Evaluation modes")
    parser.add_argument("--num_samples", type=int, default=500,    # The reported run used 500 GSM8K examples and pass@8; for quick debugging, lower num_samples and pass_k to reduce GPU time.
                        help="Number of test examples (None = full set)")
    parser.add_argument("--pass_k", type=int, default=8,
                        help="k for pass@k evaluation (uses temperature sampling)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature for pass@k > 1")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for model inference")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--output_dir", type=str, default="./results/baseline",
                        help="Directory to save results")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="W&B project name (optional)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    # Use parse_known_args to ignore unrecognized arguments from the Colab kernel
    args, unknown = parser.parse_known_args()

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # W&B
    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            name=f"baseline-{args.model.split('/')[-1]}",
            config=vars(args),
        )

    # Load model
    evaluator = BaselineEvaluator(
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
    )

    # Load datasets
    datasets_map = {}
    if "gsm8k" in args.benchmarks:
        print("\nLoading GSM8K...")
        datasets_map["gsm8k"] = load_gsm8k(num_samples=args.num_samples, seed=args.seed)
        print(f"  Loaded {len(datasets_map['gsm8k'])} examples")

    if "math500" in args.benchmarks:
        print("\nLoading MATH-500...")
        datasets_map["math500"] = load_math500(num_samples=args.num_samples, seed=args.seed)
        print(f"  Loaded {len(datasets_map['math500'])} examples")

    # Run evaluations
    all_metrics = {}

    for benchmark, dataset in datasets_map.items():
        for mode in args.mode:
            run_name = f"{benchmark}_{mode}"
            print(f"\n{'='*60}")
            print(f"Evaluating: {run_name}")
            print(f"{'='*60}")

            metrics, results = evaluate_on_benchmark(
                evaluator=evaluator,
                dataset=dataset,
                mode=mode,
                benchmark=benchmark,
                pass_k=args.pass_k,
                temperature=args.temperature,
                batch_size=args.batch_size,
            )

            all_metrics[run_name] = metrics

            # Print summary
            print(f"\nResults for {run_name}:")
            print(f"  Pass@1:              {metrics['pass_at_1']:.3f} ({metrics['pass_at_1']*100:.1f}%)")
            print(f"  Pass@{args.pass_k}:              {metrics[f'pass_at_{args.pass_k}']:.3f} ({metrics[f'pass_at_{args.pass_k}']*100:.1f}%)")
            print(f"  Avg response length: {metrics['avg_response_length_words']:.0f} words")
            print(f"  Has reasoning (%):   {metrics['pct_with_reasoning']*100:.1f}%")
            print(f"  Has <think> tags (%): {metrics['pct_with_think_tags']*100:.1f}%")

            # Save detailed results
            results_path = output_dir / f"{run_name}_results.json"
            with open(results_path, "w") as f:
                json.dump({"metrics": metrics, "samples": results[:50]}, f, indent=2)
            print(f"\n  Saved results → {results_path}")

            # Log to W&B
            if args.wandb_project:
                wandb.log({f"{run_name}/{k}": v for k, v in metrics.items()})

    # ---------------------------------------------------------------------------
    # Build the "baseline" block with the exact schema Phase 2 expects.
    # Priority: gsm8k_zero_shot -> gsm8k_few_shot -> first available run.
    # ---------------------------------------------------------------------------
    ref_run = (
        all_metrics.get("gsm8k_zero_shot")
        or all_metrics.get("gsm8k_few_shot")
        or next(iter(all_metrics.values()))
    )
    pass_k_key = f"pass_at_{args.pass_k}"

    # baseline block — Phase 2 reads this directly via summary["baseline"]
    baseline_block = {
        # accuracy
        "pass_at_1":                  ref_run["pass_at_1"],
        pass_k_key:                   ref_run[pass_k_key],
        # think-tag metrics (all 0.0 at baseline — no <think> training yet)
        "think_tag_rate":             ref_run["think_tag_rate"],
        "avg_think_length_words":     ref_run["avg_think_length_words"],
        "empty_think_rate":           ref_run["empty_think_rate"],
        # response quality
        "avg_response_length_words":  ref_run["avg_response_length_words"],
        "pct_with_reasoning":         ref_run["pct_with_reasoning"],
        # provenance — lets Phase 2 know which run was used as reference
        "source_run": (
            "gsm8k_zero_shot" if "gsm8k_zero_shot" in all_metrics
            else list(all_metrics.keys())[0]
        ),
        "pass_k": args.pass_k,
    }

    # Save summary
    summary_path = output_dir / "summary.json"
    summary = {
        "model":       args.model,
        "num_samples": args.num_samples,
        "pass_k":      args.pass_k,
        # Top-level "baseline" key — consumed directly by phase2_eval.py
        "baseline":    baseline_block,
        # Full per-run breakdown kept for analysis / ablations
        "metrics":     all_metrics,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("BASELINE SUMMARY")
    print(f"{'='*60}")
    for run_name, metrics in all_metrics.items():
        print(f"  {run_name:30s}  Pass@1: {metrics['pass_at_1']*100:5.1f}%  Pass@{args.pass_k}: {metrics[f'pass_at_{args.pass_k}']*100:5.1f}%")
    print(f"\nFull summary saved → {summary_path}")

    if args.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    main()
