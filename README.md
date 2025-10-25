# NAG Time: Temporal Dynamics of Negative Prompting in Few-Step Diffusion Models

## Overview

This repository contains the experimental framework for investigating the temporal dynamics of negative prompt adherence in few-step diffusion models, specifically focusing on the optimization of Normalized Attention Guidance (NAG) timing for improved object removal while preserving image consistency.

## Primary Research Question

**How well do few-step diffusion models follow negative-prompt controls, and how can we improve this adherence?**

Our investigation centers on the hypothesis that timing insights from negative prompt research apply critically in few-step regimes, particularly when combined with NAG techniques.

## Background & Methodology

### Normalized Attention Guidance (NAG)

We build upon the NAG framework, which steers generation by:
1. Pushing positive cross-attention features away from negative ones
2. L1-normalizing the result
3. α-blending back toward the positive baseline to prevent drift

This approach enables stable, training-free negative prompting in few-step settings.

**Paper:** [Normalized Attention Guidance](https://arxiv.org/abs/2505.21179)

### Key Hypothesis: Timing Matters

Our central hypothesis draws from "Understanding the Impact of Negative Prompts" research, proposing that timing insights apply equally in few-step regimes when using NAG:

> **Core Finding:** Negative prompts act with a delayed effect, primarily neutralizing the positive prompt after relevant content has formed. Activating negatives in a mid-process window enables effective object removal while preserving background.

**Paper:** [Understanding the Impact of Negative Prompts](https://arxiv.org/abs/2406.02965)

## Experimental Design

### Scope
- **Models:** SDXL-derived models
- **Task:** Single-object removal via negative prompts
- **Focus:** Temporal scheduling optimization

### Experiment 1: Baseline Comparison

For each tested model, across multiple seeds and prompt/negative-prompt pairs, we generate:

1. **Baseline:** Original model with no negative prompt
   - Example: "older woman librarian"
   
2. **Standard Negative:** Original model with negative prompt
   - Example: "older woman librarian" + negative "glasses"
   
3. **NAG with Negative:** Standard NAG implementation
   
4. **Timed NAG:** Our technique with timing set to the "noun" point (~0.2 timestep)

#### Metrics
- **Object Deletion Rate:** Presence/absence of target object
- **Image Consistency:** Similarity between deletion and no-deletion images
- **Presentation:** Averaged results in tabular format

### Experiment 2: Temporal Sweep Analysis

Comprehensive sweep of start timing across multiple prompts and seeds to validate:
- Range of effective object deletion timing
- Optimal preservation of non-target image elements

**Output:** Plots of start time vs. metrics (aggregated across models or per-model if significant differences exist)

## Repository Structure

```
nag-schedule/
├── README.md                 # This file
├── experiments/             # Experimental scripts and notebooks
├── data/                   # Dataset and prompt collections
├── results/               # Experimental outputs and analysis
├── src/                  # Core implementation
│   ├── models/          # Model implementations and wrappers
│   ├── scheduling/      # Timing and scheduling algorithms
│   └── evaluation/      # Metrics and evaluation tools
└── configs/             # Configuration files for experiments
```

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch
- Diffusers library
- CUDA-compatible GPU

### Installation
```bash
git clone https://github.com/llevera/nag-schedule.git
cd nag-schedule
pip install -r requirements.txt
```

### Quick Start
```bash
# Run baseline comparison experiment
python experiments/baseline_comparison.py --config configs/experiment1.yaml

# Run temporal sweep analysis
python experiments/temporal_sweep.py --config configs/experiment2.yaml
```

## Key Research Questions

1. **Temporal Sensitivity:** How sensitive is object removal effectiveness to the timing of negative prompt activation?

2. **Model Consistency:** Do timing effects generalize across different SDXL-derived models?

3. **Background Preservation:** What is the trade-off between object removal effectiveness and background preservation across different timing strategies?

4. **Few-Step Optimization:** How do timing insights translate from multi-step to few-step generation regimes?

## Expected Contributions

- Empirical validation of timing effects in few-step diffusion models
- Optimized scheduling strategies for NAG in few-step settings
- Comprehensive benchmark for negative prompt effectiveness evaluation
- Guidelines for temporal parameter selection in practical applications

## Citations

```bibtex
@article{nag2024,
  title={Normalized Attention Guidance},
  url={https://arxiv.org/abs/2505.21179},
  year={2024}
}

@article{negative_prompts2024,
  title={Understanding the Impact of Negative Prompts},
  url={https://arxiv.org/abs/2406.02965},
  year={2024}
}
```

## License

[Add appropriate license information]

## Contact

[Add contact information for the research team]

---

**Status:** Active Development | **Last Updated:** October 2025