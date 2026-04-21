# PHATGOOSE-Modified: Expert Ablation and Routing Analysis

This repository is a modified version of [PHATGOOSE](https://github.com/r-three/phatgoose), extended for controlled expert intervention, ablation experiments, and routing-distribution analysis.

The goal of this fork is to make expert behavior easier to study by allowing explicit control over which experts are active during inference, and by providing utilities for analyzing how expert usage relates to downstream task performance.

## Overview

Compared with the original PHATGOOSE implementation, this fork adds support for:

- restricting active experts during inference
- single-expert and leave-one-out ablation experiments
- cumulative top-expert dropping experiments
- random expert removal experiments across multiple seeds
- routing-distribution aggregation and analysis
- custom result-processing and visualization scripts

This codebase is intended for studying expert specialization and expert importance, especially on BBH-style reasoning evaluations.

## Relation to the Original Repository

This project is based on the original [PHATGOOSE](https://github.com/r-three/phatgoose) repository.

The base model architecture, pretrained checkpoints, and core evaluation framework come from the original authors.  
This fork focuses on extending the codebase for expert-level intervention and analysis experiments.

If you are looking for the original training setup, model design, or released checkpoints, please refer to the original PHATGOOSE repository.

## Main Modifications

### 1. Expert Restriction During Inference

This fork adds support for explicitly controlling which experts are allowed to be active during inference.

Implemented in:
- `src/models/addons/moe.py`

Added options:
- `allowed_experts_str`: a comma-separated list of allowed expert IDs
- `allowed_experts_path`: a JSON file specifying allowed experts

This makes it possible to run experiments such as:
- keep only one expert active
- remove one expert at a time
- remove a selected subset of experts
- restrict inference to a manually chosen expert group

### 2. Expert Ablation Experiments

This repository supports several kinds of expert ablation studies:

- **Single-expert evaluation**: keep only one expert and disable all others
- **Leave-one-out ablation**: remove one expert at a time and measure performance drop
- **Cumulative top-expert dropping**: remove the most frequently routed experts incrementally
- **Random removal experiments**: remove random subsets of experts for multiple values of `M` and multiple seeds

These experiments are useful for identifying:
- which experts are most important
- which experts are robustly helpful across tasks
- whether routing concentrates performance on a small subset of experts

### 3. Routing Analysis Utilities

This fork includes analysis utilities for aggregating and comparing routing behavior across runs.

Examples of supported analyses:
- routing-distribution collection
- removed-vs-kept expert comparison
- per-expert average score when removed
- per-expert average score when kept
- expert ranking by performance delta
- visualization of top/bottom experts by impact

### 4. Custom Result Processing

Additional scripts were added to parse experiment outputs and summarize results across:
- tasks
- random seeds
- ablation settings
- expert subsets

Typical outputs include:
- sorted expert importance tables
- aggregate score summaries
- scatter plots / bar plots for top and bottom experts
- comparisons between removed and kept conditions

## Repository Structure

A high-level overview of important directories and files:

- `src/models/addons/moe.py`  
  Core MoE implementation with added expert restriction logic.

- `src/procedures/utils/analysis_processors.py`  
  Utilities for routing-distribution aggregation and analysis.

- `colm/experiments/bash_scripts/`  
  Evaluation scripts used for generation and multitask experiments.

- `plot/`  
  Custom analysis and visualization scripts for postprocessing experiment outputs.

- `exp_out/`  
  Directory containing experiment results, including metrics and routing data.

- `logs/`  
  Slurm output and error logs.