# Mathematical Notes

## Bayesian inference

Bayes' rule is:

`p(w | D) ∝ p(D | w) p(w)`

where p(w) is the prior, p(D|w) is the likelihood, and p(w|D) is the posterior.

## Variational inference

The true posterior can be difficult to calculate. We approximate it with a tractable distribution q(w), here a diagonal Gaussian.

## Reparameterization

A sample can be generated as:

`w = mu + sigma * epsilon`

with epsilon sampled from a standard normal distribution.

## Variational objective

The implementation minimizes:

`NLL + beta * KL(q(w) || p(w))`

which corresponds to maximizing an ELBO-like objective up to constants and scaling.

## Sequential update

After task 1, the learned q1(w) is used as the prior for task 2. Then q2(w) becomes the prior for task 3, and so forth.

This is an approximate procedure because the implementation uses a restricted diagonal-Gaussian family and does not perform exact Bayesian inference.
