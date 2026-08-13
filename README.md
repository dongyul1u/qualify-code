# Qualify Active Learning Code

This folder contains the main experiment code for the qualify-related active learning study.

## Main components

- `HyperparamConfig`: stores the key experiment hyperparameters and weighting settings.
- `EnhancedQuantumSamplingCircuit`: defines the quantum-inspired sampling circuit and entropy-based scoring logic.
- `TraditionalModels`: trains the classical baseline models used for comparison.
- `run_active_learning_experiment(...)`: the main entry point that runs the active learning loop and returns experiment results.

