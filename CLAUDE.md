# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

TOMATOS (auTOMATed Optimization of Sensitivity) is a fully differentiable High Energy Physics (HEP) analysis optimization framework built with JAX. It optimizes statistical significance for HEP analyses by simultaneously optimizing variable cuts, histogram bin edges, and neural network parameters while accounting for systematic uncertainties through a pyhf-based statistical model.

## Installation and Setup

```bash
# Create and activate virtual environment
python3.9 -m venv ./tomatos_env
source ./tomatos_env/bin/activate
pip install --upgrade pip

# Avoid sklearn conflict
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

# Install dependencies
pip install --editable .

# Install JAX backend (choose CPU or GPU)
# CPU:
pip install jaxlib==0.3.14 -f https://storage.googleapis.com/jax-releases/jax_releases.html
# GPU (CUDA 11):
# pip install jaxlib==0.3.14+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Development Commands

### Running the Framework
```bash
# Basic workflow: preprocess data, train model, generate plots
tomatos --config ./tomatos/configs/demo_cls_nn.yaml --prep --train --plot

# Individual stages:
tomatos --config <config.yaml> --prep     # Preprocessing only
tomatos --config <config.yaml> --train    # Training only
tomatos --config <config.yaml> --plot     # Plotting only

# Debug mode (uses tomatos_debug/ output folder)
tomatos --config <config.yaml> --debug --prep --train --plot
```

### Code Quality
```bash
# Run pre-commit hooks (black, flake8, isort, yamllint)
pre-commit run --all-files

# Format code
black --preview .

# Sort imports
isort --profile black .
```

### Testing
```bash
# Generate test files (required before running tests)
python tests/generate_test_files.py
```

## Architecture

### Core Pipeline Flow

The optimization pipeline follows this sequence:

1. **Preprocessing** (`preprocess.py`): Loads ROOT files, applies min-max scaling, splits into train/valid/test sets, saves to HDF5
2. **Training Loop** (`training.py`): Iterates through batches, updates parameters via JAX gradient descent
3. **Loss Calculation** (`pipeline.py`): Computes statistical significance (CLs) from histograms and pyhf model
4. **Plotting** (`plotting.py`): Visualizes results, histograms, and optimization trajectories

### Key Components

**Configuration System** (`config.py`):
- `Setup` class parses YAML configs and sets up paths, sample handling, variable indices
- Main data array shape: `(n_samples, n_events, n_vars)` where vars = `[input_vars, event_weight, aux_vars]`
- Key indices: `nn_inputs_idx_end`, `weight_idx`, `cls_var_idx`
- Sample/systematic structure: expects `ntuple_path/SAMPLE/SYSTEMATIC.root`

**Differentiable Operations**:
- **Cuts** (`select.py`): Uses `relaxed.cut()` for smooth sigmoid approximation of hard cuts during training, switches to sharp cuts (slope=1e20) for validation
- **Histograms** (`histograms.py`): Implements soft histogramming with kernel density estimation for differentiability
- **Neural Networks** (`nn.py`): Equinox-based models (`NeuralNetwork`, `NeuralNetworkFeatureAttention`)

**Loss Function** (`pipeline.py`):
- `loss_fn()` orchestrates: cuts → event selection → histogram filling → pyhf model → CLs loss
- Supports objectives: `cls_nn` (NN-based), `cls_var` (variable cuts), `bce` (binary cross-entropy)
- Applies constraint penalties during training via `constraints.penalize_loss()`

**Statistical Model** (`workspace.py`):
- `pyhf_model()`: Constructs HistFactory model with systematics
- `hist_transforms()`: Implements ABCD background estimation method
- Zero-protection and symmetrized uncertainties

**Optimization** (`solver.py`, `training.py`):
- Uses JAX's `OptaxSolver` with Adam optimizer
- Parameters: `opt_pars = {nn: ..., bw: bandwidth, bins: [...], cut_<var>: ...}`
- Constraints applied post-update via `constraints.opt_pars()`
- Batching via generators in `batcher.py`

### Data Flow

```
ROOT files → preprocess.py → HDF5 (train/valid/test splits)
                ↓
HDF5 → batcher.py → batches → pipeline.loss_fn()
                                    ↓
                              cuts + selection + histograms
                                    ↓
                              pyhf model → CLs loss
                                    ↓
                              gradient update → constraints
```

### Important Configuration Details

**YAML Config Structure**:
- `ntuple_path`: Path to ROOT files (expects `/SAMPLE/SYSTEMATIC.root` structure)
- `results_path`: Output directory for models, plots, logs
- `vars`: List of input variables (order matters - defines array indices)
- `objective`: `cls_nn`, `cls_var`, or `bce`
- `opt_cuts`: Dictionary of cuts to optimize with `keep: "above"/"below"` and `init` value
- `n_bins`: Number of histogram bins
- `include_bins`: Whether to optimize bin edges

**Variable Ordering**:
The main data array concatenates: `vars + [event_weight_var] + aux_vars`. The framework relies on fixed indices (`nn_inputs_idx_end`, `weight_idx`) for slicing, making variable order critical.

**Batching**:
- Total batch size split across all samples and systematics
- Uses HDF5 chunking for memory efficiency (`chunk_size = batch_size / n_samples / n_chunk_combine`)

### JAX Specifics

- Backend configured in `config.py`: 32-bit precision, CPU platform by default
- JIT compilation caching monitored via `utils.clear_caches()` when memory exceeds threshold
- Debugging flags available: `jax_disable_jit`, `jax_debug_nans`, `jax_check_tracer_leaks`

### Dependencies

**Core**: JAX (0.3.14), Equinox (0.5.5), Optax (0.1.2), pyhf (custom fork with JAX support), neos (0.3.0)

**Data**: uproot (ROOT file reading), h5py (HDF5 storage), numpy (1.25.0), scipy (1.9.2)

**Custom Forks**:
- `pyhf@fix_jax`: JAX-compatible version of pyhf statistical framework
- `relaxed@mle_fix`: Provides differentiable relaxations for cuts

## Documentation

Full documentation: https://tomatos.readthedocs.io

Documentation built with Sphinx (config in `docs/conf.py`, uses `sphinx-book-theme`)
