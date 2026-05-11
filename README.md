# Reproducing Reasoning Emergence in Small LLMs via GRPO Post-Training

This project investigates whether structured reasoning behavior can be improved in a small open-source language model through post-training. Inspired by DeepSeek-R1, the project explores a lightweight reasoning pipeline using a sub-3B parameter model, supervised fine-tuning (SFT), and Group Relative Policy Optimization (GRPO) with verifiable math rewards.

The broader goal is to test whether a small base model can be nudged toward more reliable mathematical reasoning without relying on expensive proprietary models or human-labeled preference data. The project uses public math datasets such as GSM8K and NuminaMath-CoT, and evaluates both final-answer accuracy and reasoning-format behavior.

## Repository Organization

```text
CS272FinalProject/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── phase1_baseline_model.ipynb
│   ├── phase2_sft_warmup.ipynb
│   └── phase3_grpo.ipynb
├── scripts/
│   ├── phase1_baseline_model.py
│   ├── phase2_sft_warmup.py
│   └── phase3_grpo.py
├── results/
│   ├── phase1_baseline/
│   ├── phase2_sft/
│   └── phase3_grpo/
└── configs/
    ├── phase1_baseline_config.json
    ├── phase2_sft_training_config.json
    └── phase3_grpo_config.json
```

The repository is organized by project phase:

- `notebooks/` contains the main experimental notebooks for each phase.
- `scripts/` contains Python script exports or cleaned versions of the notebook code for easier review.
- `results/` contains JSON outputs, evaluation summaries, and sample generations.
- `configs/` contains experiment and training configuration files.

## Project Phases

- **Phase 1: Baseline Evaluation**  
  Evaluate the raw base model on GSM8K before any post-training.

- **Phase 2: SFT Warm-Up**  
  Fine-tune the base model on filtered chain-of-thought math examples to encourage structured, parseable reasoning outputs.

- **Phase 3: GRPO Post-Training**  
  Build on the SFT checkpoint using GRPO with exact-answer and format rewards.

---

# Phase 1: Baseline Evaluation

Phase 1 establishes the baseline performance of the raw `Qwen/Qwen2.5-1.5B` model before supervised fine-tuning or GRPO. The purpose of this phase is to measure how well the base model performs on mathematical reasoning tasks in zero-shot and few-shot settings.

The baseline results are used as the reference point for later phases:

- Phase 2: SFT warm-up
- Phase 3: GRPO post-training

## What This Phase Does

- Loads the raw `Qwen/Qwen2.5-1.5B` model from Hugging Face.
- Evaluates the model on GSM8K-style math reasoning problems.
- Runs two prompting settings:
  - zero-shot prompting
  - 8-shot chain-of-thought prompting
- Computes baseline metrics:
  - Pass@1
  - Pass@8
  - average response length
  - percentage of responses with reasoning-like steps
  - percentage of responses containing `<think>` tags
- Saves detailed results and summary metrics.

## Phase 1 Files

```text
notebooks/phase1_baseline_model.ipynb
scripts/phase1_baseline_model.py
results/phase1_baseline/summary.json
results/phase1_baseline/gsm8k_zero_shot_results.json
results/phase1_baseline/gsm8k_few_shot_results.json
```

## How To Run Phase 1

Install dependencies from the repo root:

```bash
pip install -r requirements.txt
```

Open the notebook:

```text
notebooks/phase1_baseline_model.ipynb
```

Run the cells from top to bottom.

The reported baseline run used:

```text
Model: Qwen/Qwen2.5-1.5B
Benchmark: GSM8K
Samples: 500
Pass@k: 8
Modes: zero-shot and few-shot chain-of-thought
```

If using the Python script version, run:

```bash
python scripts/phase1_baseline_model.py \
  --model Qwen/Qwen2.5-1.5B \
  --benchmarks gsm8k \
  --mode zero_shot few_shot \
  --num_samples 500 \
  --pass_k 8 \
  --output_dir ./results/phase1_baseline
```

For a faster smoke test:

```bash
python scripts/phase1_baseline_model.py \
  --model Qwen/Qwen2.5-1.5B \
  --benchmarks gsm8k \
  --mode zero_shot \
  --num_samples 20 \
  --pass_k 1 \
  --output_dir ./results/phase1_baseline_smoke_test
```

## Reported Phase 1 Results

The reported baseline run used 500 GSM8K examples.

```text
GSM8K zero-shot:
Pass@1: 58.4%
Pass@8: 85.2%

GSM8K few-shot CoT:
Pass@1: 59.2%
Pass@8: 88.0%
```

The base model already showed meaningful GSM8K performance, especially under Pass@8 sampling. However, it did not naturally produce `<think>...</think>` reasoning blocks or standardized final-answer lines, which are needed for the later GRPO reward pipeline.

## Phase 1 Takeaway

The raw base model already has some mathematical reasoning ability, but its outputs are not formatted for verifiable reward training. Phase 2 therefore focuses on teaching the model a structured reasoning format that can be parsed and rewarded during GRPO.

---

# Phase 2: SFT Warm-Up

Phase 2 performs supervised fine-tuning as a warm-up step before GRPO. The goal is not to fully solve mathematical reasoning with SFT alone. Instead, the goal is to make the raw base model produce structured, parseable outputs that are easier to score with rewards in Phase 3.

We kept the raw `Qwen/Qwen2.5-1.5B` base model rather than switching to an instruct model. This keeps the project aligned with the goal of post-training a base model into better reasoning behavior.

## What This Phase Does

- Loads and filters NuminaMath-CoT examples.
- Removes examples that are too long, too symbolic, non-English, multiple-choice, or too difficult for the small model setting.
- Converts examples into a plain completion format.
- Fine-tunes `Qwen/Qwen2.5-1.5B` using LoRA.
- Saves a Phase 2 SFT checkpoint for use in Phase 3 GRPO.
- Evaluates the SFT-only model on a small GSM8K subset.

## SFT Formatting

An initial chat-style format using Qwen chat tokens was unstable for the raw base model. The model sometimes generated role labels, unrelated prompts, malformed text, or repeated examples.

The final Phase 2 notebook therefore uses a plain completion format:

```text
Problem:
{problem}

Solution:
<think>
{reasoning}
</think>
The final answer is {answer}
```

This format better matches the behavior of a base language model and makes the output easier to parse for GRPO.

## Phase 2 Files

```text
notebooks/phase2_sft_warmup.ipynb
scripts/phase2_sft_warmup.py
configs/phase2_sft_training_config.json
results/phase2_sft/sft_eval_summary_format_full.json
results/phase2_sft/sft_eval_results_format_full.json
results/phase2_sft/sample_generations.json
```

The notebook also produces local files such as:

```text
data/sft/train
data/sft/val
data/sft/prep_summary.json
data/sft/sample_formatted_examples.json
checkpoints/sft_plain_base_format/final
checkpoints/sft_plain_base_format/training_config.json
checkpoints/sft_plain_base_format/train_result.json
checkpoints/sft_plain_base_format/trainer_log_history.json
checkpoints/sft_plain_base_format/sample_generations.json
```

The checkpoint directory is not committed to GitHub because model weights are large. It can be regenerated by running the notebook.

## How To Run Phase 2

Install dependencies from the repo root:

```bash
pip install -r requirements.txt
```

Open the notebook:

```text
notebooks/phase2_sft_warmup.ipynb
```

Run the notebook from top to bottom.

The notebook has three main sections:

1. Dataset preparation
2. LoRA SFT training
3. SFT-only evaluation

## Phase 2 Training Configuration

The final Phase 2 training run used:

```text
Base model: Qwen/Qwen2.5-1.5B
Training method: LoRA SFT
Train examples: 1500
Validation examples: 150
Epochs: 1
LoRA rank: 16
LoRA alpha: 32
LoRA dropout: 0.05
Learning rate: 2e-5
Batch size: 1
Gradient accumulation: 16
Max sequence length: 768
Output checkpoint: checkpoints/sft_plain_base_format/final
```

To run a quick smoke test, set:

```python
SMALL_TEST = True
```

To reproduce the final Phase 2 run, set:

```python
SMALL_TEST = False
```

## Phase 2 Evaluation

The final Phase 2 evaluation used a small GSM8K subset to check whether the model produced GRPO-compatible outputs.

The final reported Phase 2 result was:

```text
Examples evaluated: 10
Exact-answer accuracy: 0.10
Complete think-tag rate: 1.00
Final answer line rate: 1.00
GRPO format rate: 1.00
```

This means the model learned the required output format, but it still made many mathematical mistakes.

## Phase 2 Takeaway

Phase 2 improved structure and parseability more than correctness. The final SFT model reliably produced complete `<think>...</think>` blocks and final-answer lines, which makes it a useful warm start for GRPO. However, exact-answer accuracy remained low, showing that SFT alone did not reliably improve mathematical reasoning. This motivates Phase 3, where exact-answer rewards can directly optimize correctness.

---

# Phase 3: GRPO Post-Training

Phase 3 implements Group Relative Policy Optimization (GRPO) on top of the Phase 2 SFT checkpoint. The goal is to improve mathematical correctness using verifiable rewards rather than human preference labels.

GRPO is used because it does not require a learned critic or value model. Instead, it samples multiple completions for the same prompt, scores each completion with deterministic reward functions, normalizes rewards within the group, and updates the policy toward higher-reward completions.

## What This Phase Does

- Loads the Phase 2 SFT checkpoint as the starting policy.
- Loads a frozen reference model for KL regularization.
- Builds GRPO training examples from GSM8K.
- Samples multiple completions per prompt.
- Scores each completion using verifiable rewards.
- Computes group-relative advantages.
- Applies a clipped policy-gradient objective with KL penalty.
- Saves the GRPO-trained checkpoint and merged model.

## GRPO Reward Design

The reward function combines:

- **Correctness reward**: exact match against the normalized ground-truth answer.
- **Format reward**: rewards outputs with complete `<think>...</think>` blocks and a parseable final answer.
- **Missing-answer penalty**: penalizes outputs that do not contain an extractable answer.
- **Repetition penalty**: penalizes degenerate repeated text.
- **Length penalty**: discourages very long reasoning traces that drift away from the problem.

The main reward signal is correctness. Format reward is included because GRPO needs outputs to remain parseable during training.

## Phase 3 Files

```text
notebooks/phase3_grpo.ipynb
scripts/phase3_grpo.py
configs/phase3_grpo_config.json
results/phase3_grpo/grpo_config.json
results/phase3_grpo/evaluation_results.json
```

The notebook may also produce local checkpoint files such as:

```text
checkpoints/grpo_final_clean_v2/
checkpoints/grpo_final_clean_v2/merged/
checkpoints/grpo_final_clean_v2/grpo_config.json
```

Large GRPO checkpoint files should not be committed to GitHub.

## How To Run Phase 3

Before running Phase 3, run Phase 2 first and make sure the SFT checkpoint exists locally:

```text
checkpoints/sft_plain_base_format/final
```

Then open:

```text
notebooks/phase3_grpo.ipynb
```

In the GRPO config cell, make sure the SFT checkpoint path points to the Phase 2 checkpoint:

```python
config = GRPOConfig(
    sft_checkpoint="./checkpoints/sft_plain_base_format/final",
    model_name="Qwen/Qwen2.5-1.5B",
    output_dir="./checkpoints/grpo_final_clean_v2",
)
```

Then run the cells from top to bottom.

## Phase 3 Training Configuration

The current Phase 3 experiment uses a small-compute GRPO configuration:

```text
Starting checkpoint: Phase 2 SFT checkpoint
Base model: Qwen/Qwen2.5-1.5B
Dataset: GSM8K
Number of generations per prompt: 3
Training steps: 120
Learning rate: 5e-7
KL coefficient: 0.02
Clip epsilon: 0.2
Batch size: 1
Gradient accumulation: 3
LoRA rank: 12
LoRA alpha: 24
Max new tokens: 256
Output directory: checkpoints/grpo_final_clean_v2
```

This setup is intentionally small because GPU time was limited.

## Phase 3 Evaluation

Phase 3 evaluates the GRPO-trained model on GSM8K-style examples using exact-answer matching.

The evaluation checks:

- whether the model produces a parseable final answer
- whether the predicted answer matches the GSM8K gold answer
- whether the output remains in a structured reasoning format
- whether GRPO improves correctness compared with the SFT-only checkpoint

## Phase 3 Takeaway

Phase 3 directly targets the limitation observed in Phase 2. SFT made the model’s outputs more structured, but it did not reliably improve correctness. GRPO adds a verifiable reward signal for exact answers, so the model is optimized toward solving the problem correctly rather than only imitating a reasoning format.

Because this project uses a small model and limited GPU time, the Phase 3 results should be interpreted as a small-scale reproduction attempt rather than a full DeepSeek-R1-style training run.

---

# Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Main dependencies:

```text
torch
datasets
transformers
accelerate
peft
trl
numpy
sympy
wandb
tqdm
pyarrow
huggingface_hub
```

If using Google Colab, use a GPU runtime for Phase 2 and Phase 3.

---

# Reproducibility Notes

- Large checkpoints are excluded from GitHub.
- Small JSON result files and configuration files are included for documentation.
- Phase 1 can be rerun from the raw base model.
- Phase 2 can regenerate the SFT checkpoint from NuminaMath-CoT.
- Phase 3 requires the Phase 2 checkpoint before GRPO training.
- Random seeds are set where possible, but exact results may vary slightly because of GPU kernels, sampling, and library versions.

---

# Summary

This project shows that a small base model can already perform some GSM8K reasoning before post-training. SFT then improves output structure and makes the model more compatible with a GRPO reward pipeline. However, SFT alone does not reliably improve correctness. GRPO is therefore used as the final post-training phase to optimize the model with verifiable math rewards.
