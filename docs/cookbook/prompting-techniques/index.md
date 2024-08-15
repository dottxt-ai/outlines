# Prompting Techniques

This part of the documentation provides links to various prompting techniques featured in [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608). We follow closely the presentation of the Paper. We propose an implementation of some of these prompting techniques using Outlines. Contributions for the remaining techniques are welcome!


# Text-based Techniques

## Few-shots prompting


- [Few-shot Prompting](few-shot-prompting.md) - Provide the model a small number of examples.

### Example selection

- K-Nearest Neighbour - [Paper](https://arxiv.org/abs/2101.06804)
- Vote-K ([Paper](https://arxiv.org/abs/2209.01975))
- [Self-Generated In-Context Learning (SG-ICL)](self-generated-in-context-learning-sg-icl.md) - Uses the model to generate its own in-context learning examples.
- [Prompt Mining](prompt-mining.md) - Extracts effective prompts from existing data or model outputs.
- LENS - [Paper](https://arxiv.org/abs/2302.13539)
- UDR - [Paper](https://arxiv.org/abs/2305.04320)
- Active Example Selection - [Paper](https://arxiv.org/abs/2211.04486)

## Zero-shot prompting

Zero-shot prompting uses zero exemplars.

- [Zero-Shot Prompting](zero-shot-prompting.md) - Generates answers without any task-specific examples or fine-tuning.
- Role Prompting - [Paper 1](https://arxiv.org/abs/2307.05300), [paper 2](https://arxiv.org/abs/2305.16291), [paper 3](https://arxiv.org/abs/2311.10054), [paper 4](https://www.dre.vanderbilt.edu/~schmidt/PDF/ADA_Europe_Position_Paper.pdf)
- Style prompting - [Paper](https://arxiv.org/abs/2302.09185)
- [Emotion Prompting](emotion-prompting.md) - Incorporates emotional context into prompts.
- System 2 Attention (S2A) - [Paper](https://arxiv.org/abs/2311.11829)
- [Simulation Theory of Mind (SimToM)](simtom-simulation-theory-of-mind.md) - Simulates different perspectives or thought processes.
- Rephrase and Respond (RaR) - [Paper](https://arxiv.org/abs/2311.04205)
- [Re-Reading (Re2)](re-reading-re2.md) - Encourages the model to review and refine its own outputs.
- [Self-Ask](self-ask.md) - Prompts the model to ask and answer its own follow-up questions.

## Though generation

- [Chain of Thought (CoT) Prompting](chain-of-thought.md) - Encourages the model to show its reasoning step-by-step.

### Zero-short CoT

- [Zero-Shot Chain of Thought (CoT)](zero-shot-chain-of-thought.md) - Applies chain of thought reasoning without specific examples.
- Step-Back prompting - [Paper](https://arxiv.org/abs/2310.06117)
- [Analogical Prompting](analogical-prompting.md) - Uses analogies to guide the model's reasoning.
- Thread-of-Thought - [Paper](https://arxiv.org/abs/2311.08734)
- Tabular Chain-of-Thought - [Paper](https://arxiv.org/abs/2305.17812)

### Few-shot CoT


- [Contrastive CoT Prompting](contrastive-cot-prompting.md) - Uses contrasting examples to improve chain of thought reasoning.
- [Uncertainty-Routed CoT prompting](uncertainty-routed-cot-prompting.md) - Selects reasoning paths based on a confidence threshold.
- [Complexity-based prompting](complexity-based-prompting.md) - Enhances CoT by focusing on complex examples.
- [Active Prompting](active-prompting.md) - Refine prompts dynamically.
- Memory-of-Thought prompting - [Paper](https://arxiv.org/abs/2305.05181)
- [Automatic CoT](automatic-chain-of-thought.md) - Automate the choice of examples for CoT prompting.

## Decomposition

- Least-to-Most prompting - [Paper](https://arxiv.org/abs/2205.10625)
- [Decomposed Prompting (DeComp)](decomposed-prompting-decomp.md) - Breaks down complex tasks into smaller, manageable steps.
- Plan-and-solve prompting - [Paper](http://arxiv.org/abs/2305.04091)
- Tree-of-Thought (ToT) - [Paper 1](http://arxiv.org/abs/2305.10601), [paper 2](http://arxiv.org/abs/2305.08291)
- Recursion-of-Thought - [Paper](http://arxiv.org/abs/2306.06891)
- Program-of-Thought - [Paper](https://arxiv.org/abs/2211.12588)
- Faithful Chain-of-Thought - [Paper](http://arxiv.org/abs/2301.13379)
- [Skeleton-of-Thought](skeleton-of-thought.md) - Provides a structural framework for the model's reasoning.

## Ensembling

- [Demonstration Ensembling (DENSE)](demonstration-ensembling-dense.md) - Combines multiple demonstrations to improve performance.
- Mixture of Reasoning Experts (MoRE) - [Paper](http://umiacs.umd.edu/~jbg//docs/2023_findings_more.pdf)
- [Max Mutual Information Method](max-mutual-information-method.md) - Maximizes mutual information between prompts and desired outputs.
- [Self-Consistency](self-consistency.md) - Generates multiple outputs and selects the most consistent one.
- Universal self-consistency - [Paper](http://arxiv.org/abs/2311.17311)
- Meta-reasoning over multiple CoTs - [Paper](http://arxiv.org/abs/2304.13007)
- [DiVeRSe](diverse-diversity-focused-self-consistency.md) - Combine multiple prompts with self-consistency.
- Consistency-based Self-adaptive Prompting (COSP) - [Paper](http://arxiv.org/abs/2305.14106)
- [Universal Self-Adaptive Prompting (USP)](universal-self-adaptive-prompting-usp.md) - Adapts prompts across different tasks and domains.
- Prompt paraphrasing - [Paper](https://doi.org/10.1162/tacl_a_00324)

## Self-criticism

- [Self-Calibration](self-calibration.md) - Helps the model adjust its own confidence and accuracy.
- Self-Refine - [Paper](https://arxiv.org/abs/2303.17651)
- [Reversing Chain of Thought (RCoT)](reversing-chain-of-thought-rcot.md) - Applies chain of thought reasoning in reverse order.
- Self-Verification - [Paper](https://arxiv.org/abs/2212.09561)
- Chain-of-Verification (COVE) - [Paper](https://arxiv.org/pdf/2406.06608)
- [Cumulative Reasoning](cumulative-reasoning.md) - Builds upon previous reasoning steps to reach a conclusion.


# Prompt Engineering


- [Meta Prompting](meta-prompting.md) - Uses prompts to generate or improve other prompts.
- AutoPrompt - [Paper](https://doi.org/10.18653/v1/2020.emnlp-main.346)
- Automatic Prompt Engineering (APE) - [Paper](http://arxiv.org/abs/2211.01910)
- Gradientfree Instructional Prompt Search (GrIPS) - [Paper](https://aclanthology.org/2023.eacl-main.277)
