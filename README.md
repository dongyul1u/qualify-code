# Qualify Active Learning Code

This folder contains the main experiment code for the qualify-related active learning study.

## Main components

- `HyperparamConfig`: stores the key experiment hyperparameters and weighting settings.
- `EnhancedQuantumSamplingCircuit`: defines the quantum-inspired sampling circuit and entropy-based scoring logic.
- `TraditionalModels`: trains the classical baseline models used for comparison.
- `run_active_learning_experiment(...)`: the main entry point that runs the active learning loop and returns experiment results.

## What the code does

The program loads benchmark datasets, splits them into labeled and unlabeled pools, trains classical models, and evaluates different active learning strategies. It compares classical sampling with quantum-inspired methods, tracks performance across rounds, and saves the resulting statistics and figures for analysis.

This file is structured as a research workflow rather than a reusable library: the key functions and classes are organized around configuration, quantum sampling, scoring, baselines, and the overall active learning loop.