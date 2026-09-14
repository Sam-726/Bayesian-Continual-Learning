# Limitations

1. Split-MNIST is a toy benchmark and does not represent all real-world continual-learning settings.
2. The Bayesian posterior is a diagonal Gaussian approximation.
3. The posterior-to-prior transfer is approximate.
4. The same architecture is not necessarily optimal for either method.
5. The initial implementation does not include experience replay.
6. Predictive uncertainty from a small number of posterior samples can be noisy.
7. Results depend on training hyperparameters and random seeds.
8. No claim of scientific novelty should be made without a complete literature review.
