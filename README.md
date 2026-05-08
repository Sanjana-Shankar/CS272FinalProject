# Phase 2: SFT Warm-Up

This phase performs supervised fine-tuning (SFT) as a warm-up step before GRPO. The goal is to adapt the raw `Qwen/Qwen2.5-1.5B` base model toward structured mathematical reasoning outputs using chain-of-thought style examples from NuminaMath-CoT.

Unlike an instruction-tuned model, the raw Qwen base model is primarily a text-completion model. An initial chat-style formatting approach was unstable, so this phase uses a simpler plain-completion format:

```text
Problem:
{problem}

Solution:
<think>
{reasoning}
</think>
The final answer is {answer}

