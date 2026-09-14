# BayesCL — Bayesian Continual Learning

A small research-oriented PyTorch project comparing sequential fine-tuning with approximate Bayesian sequential posterior updating on Split-MNIST.

## Research question

Can Bayesian knowledge updating help a neural network learn new tasks while forgetting less of its previous knowledge?

## Tasks

Five sequential binary tasks:

- 0 vs 1
- 2 vs 3
- 4 vs 5
- 6 vs 7
- 8 vs 9

## Methods

**Baseline:** deterministic MLP trained sequentially without resetting weights.

**Bayesian:** Bayesian MLP with diagonal Gaussian variational parameters, ELBO-style loss, and approximate posterior-to-prior updating after each task.

This is an independent educational/research implementation inspired by Bayesian continual-learning research. It is not an official project of any research group.

## Install

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

## Run the main experiment

```bash
python experiments/run_all.py
```

For a quick test, edit `configs/default.yaml` to use `epochs: 1` and `seeds: 1` via the command line:

```bash
python experiments/run_all.py --seeds 1
```

The script downloads MNIST automatically and writes results to `results/`.

## Data efficiency

```bash
python experiments/run_data_efficiency.py
```

## KL-weight ablation

```bash
python experiments/run_ablation.py
```

## Important interpretation note

The Bayesian method implemented here uses a diagonal Gaussian variational posterior and carries its learned posterior parameters forward as the next task's prior. This is an approximation to sequential Bayesian learning, not exact inference.

## Results integrity

The repository intentionally does not contain fabricated performance numbers. Run the experiments to generate actual tables and figures.

## Suggested next steps

- add Permuted-MNIST
- compare against experience replay
- test more posterior samples
- investigate calibration and OOD uncertainty
- add a small uncertainty-guided replay extension
