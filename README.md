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

## Research Hypotheses

We test three complementary hypotheses across few-step SDXL families:

### H1: Diversity Distillation → Adherence
**Hypothesis:** Running a base diffusion model (SDXL) for the first denoising step before handing off to a few-step distilled model improves negative guidance compared with using the few-step model alone (hybrid-k vs. student-only).

**Implementation:** `diversity-handoff.ipynb`

### H2: NAG Improves Adherence/Preservation
**Hypothesis:** Normalized Attention Guidance (NAG), applied across all few-step models under test, improves negative guidance over baseline performance.

**Implementation:** `nag-time.ipynb`

### H3: NAG Time ≥ NAG on Adherence; Better Preservation
**Hypothesis:** NAG Time (time-gated NAG), applied across all few-step models, improves negative guidance over baseline performance, is non-inferior to NAG on adherence, and yields superior preservation.

**Implementation:** `nag-time.ipynb` (with temporal gating parameters)

## Experimental Design

### Scope
- **Models:** SDXL-derived few-step families (base, DMD, Turbo, Lightning, LCM, Hyper, PCM)
- **Task:** Single-object removal via negative prompts
- **Focus:** Temporal scheduling optimization and adherence measurement

### Core Metrics
- **Object Deletion Rate (Adherence):** Presence/absence of target object
- **Image Consistency (Preservation):** Similarity between deletion and no-deletion images
- **Generation Speed:** Latency and GPU memory usage across models and techniques

## Notebooks

This repository includes three main Jupyter notebooks implementing different approaches:

### 1. `diversity-handoff.ipynb` — Diversity Distillation Approach
Tests H1 by implementing a hybrid handoff strategy:
- Runs base SDXL for initial denoising steps to build diversity
- Transfers to few-step distilled models (student models) for remaining steps
- Compares adherence (object removal) vs. student-only baseline

**Key Variables:**
- `handoff_step`: Number of base model steps before transfer
- `student_model`: Target distilled model (DMD, Lightning, LCM, etc.)
- Metrics: deletion rate, consistency vs. baseline

### 2. `nag-time.ipynb` — NAG and NAG Time Implementation
Tests H2 and H3 by implementing Normalized Attention Guidance with optional temporal gating:

**NAG (H2 baseline):**
- Applies attention steering across all denoising steps
- `nag_scale`: Magnitude of negative guidance (e.g., 3.0)
- `nag_tau`: Normalization threshold (e.g., 2.5)
- `nag_alpha`: Blend factor toward positive (e.g., 0.5)

**NAG Time (H3 - our contribution):**
- Time-gated NAG activation with delayed start
- `nag_start`: Fractional timestep to begin NAG (e.g., 0.17 for ~20% through trajectory)
- `nag_end`: Fractional timestep to disable NAG (e.g., 0.5 for halfway)
- `nag_ramp_steps`: Optional linear ramp-up of `nag_scale` for soft activation
- `nag_cooldown`: Post-NAG CFG reduction window for improved preservation

**Key Benchmarks:**
- Generation latency (seconds/image) across models
- Peak GPU memory usage
- Adherence and preservation scores

### 3. `experiment_results_analysis.ipynb` — Analysis & Visualization
Aggregates results from both approaches and generates:
- Comparative plots: H1 vs. H2 vs. H3 across models
- Adherence vs. preservation trade-off curves
- Temporal sensitivity analysis (start time sweeps)
- Latency and memory comparison tables
- Statistical summaries and error bars

## Repository Structure

```
nag-time/
├── README.md                           # This file
├── nag-time.ipynb                     # NAG & NAG Time implementation (H2, H3)
├── diversity-handoff.ipynb            # Diversity distillation approach (H1)
├── experiment_results_analysis.ipynb  # Analysis, visualization, and comparisons
├── data/
│   └── prompts_general.json          # Prompt collection for experiments
├── metrics_results.csv                # Aggregated results
├── clipscore_results.csv              # CLIP-based adherence metrics
├── fid_scores.csv                     # Fréchet Inception Distance (diversity)
├── kid_scores.csv                     # Kernel Inception Distance
├── ssim_raw_scores.csv                # Structural Similarity (preservation)
├── clip_raw_scores.csv                # Raw CLIP scores
└── requirements.txt                   # Python dependencies
```

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch
- Diffusers library with SDXL support
- Hugging Face Hub access (for model downloads)
- CUDA-compatible GPU

### Installation
```bash
git clone https://github.com/llevera/nag-time.git
cd nag-time
pip install -r requirements.txt
```

### Quick Start

1. **Test H1 (Diversity Distillation):**
   ```python
   jupyter notebook diversity-handoff.ipynb
   ```
   - Adjust `handoff_step` and `student_model` parameters
   - Compare adherence metrics vs. baseline

2. **Test H2/H3 (NAG & NAG Time):**
   ```python
   jupyter notebook nag-time.ipynb
   ```
   - Set `use_nag=True` for H2 (standard NAG across all steps)
   - Set `nag_start` and `nag_end` for H3 (time-gated NAG)
   - Benchmark latency and memory with the timing suite

3. **Analyze Results:**
   ```python
   jupyter notebook experiment_results_analysis.ipynb
   ```
   - Load CSV metrics from both experiments
   - Generate comparative visualizations

## Key Implementation Details

### NAG Attention Processor
Located in `nag-time.ipynb`, the `NAGAttnProcessor2_0` class implements:
- Efficient cross-attention steering using PyTorch 2.0's `scaled_dot_product_attention`
- L1 normalization with configurable thresholds (`nag_tau`)
- α-blending to prevent feature drift

### NAG Time Scheduler
The `NAGTimeStableDiffusionXLPipeline` extends SDXL with:
- Fractional timestep mapping to DDPM indices
- Delayed NAG activation (`nag_start`)
- Linear ramp-up for soft transitions (`nag_ramp_steps`)
- Optional cool-down window post-NAG (`nag_cooldown`)

### Model Support
Tested across:
- **Base SDXL:** Reference baseline (30 steps)
- **DMD2:** Distilled, 4 steps
- **SDXL-Turbo:** 4 steps
- **SDXL-Lightning:** 4 steps
- **LCM-SDXL:** 4 steps
- **Hyper-SD:** 8 steps
- **PCM:** 4 steps

## Results & Metrics

### CSV Output Files
- `metrics_results.csv`: Aggregated adherence, preservation, and speed metrics
- `clipscore_results.csv`: CLIP-based object adherence scores
- `fid_scores.csv`: Diversity across generated images
- `kid_scores.csv`: Kernel-based image diversity
- `ssim_raw_scores.csv`: Structural similarity (background preservation)
