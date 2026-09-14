from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, hidden1=256, hidden2=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, hidden1), nn.ReLU(),
            nn.Linear(hidden1, hidden2), nn.ReLU(),
            nn.Linear(hidden2, 2)
        )
    def forward(self, x):
        return self.net(x)

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_mu=0.0, prior_sigma=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mu = nn.Parameter(torch.empty(out_features, in_features))
        self.rho = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_rho = nn.Parameter(torch.empty(out_features))
        self.register_buffer("prior_mu", torch.tensor(float(prior_mu)))
        self.register_buffer("prior_sigma", torch.tensor(float(prior_sigma)))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mu, 0, 0.05)
        nn.init.constant_(self.rho, -3.0)
        nn.init.normal_(self.bias_mu, 0, 0.05)
        nn.init.constant_(self.bias_rho, -3.0)

    @property
    def sigma(self):
        return F.softplus(self.rho) + 1e-6

    @property
    def bias_sigma(self):
        return F.softplus(self.bias_rho) + 1e-6

    def sample(self):
        w = self.mu + self.sigma * torch.randn_like(self.mu)
        b = self.bias_mu + self.bias_sigma * torch.randn_like(self.bias_mu)
        return w, b

    def forward(self, x):
        w, b = self.sample()
        return F.linear(x, w, b)

    def kl_divergence(self):
        # KL between diagonal Gaussians q and N(prior_mu, prior_sigma^2).
        ps = self.prior_sigma
        pm = self.prior_mu
        s = self.sigma
        kl_w = torch.log(ps/s) + (s.pow(2) + (self.mu-pm).pow(2))/(2*ps.pow(2)) - 0.5
        bs = self.bias_sigma
        kl_b = torch.log(ps/bs) + (bs.pow(2) + (self.bias_mu-pm).pow(2))/(2*ps.pow(2)) - 0.5
        return kl_w.sum() + kl_b.sum()

    def set_prior_from_posterior(self):
        # Store current posterior parameters as the next task's prior.
        self.prior_mu = self.mu.detach().clone()
        self.prior_sigma = self.sigma.detach().clone()

class BayesianMLP(nn.Module):
    def __init__(self, hidden1=256, hidden2=128):
        super().__init__()
        self.flatten = nn.Flatten()
        self.l1 = BayesianLinear(784, hidden1)
        self.l2 = BayesianLinear(hidden1, hidden2)
        self.l3 = BayesianLinear(hidden2, 2)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)

    def kl_divergence(self):
        return self.l1.kl_divergence() + self.l2.kl_divergence() + self.l3.kl_divergence()

    def update_priors(self):
        for layer in (self.l1, self.l2, self.l3):
            layer.set_prior_from_posterior()

    @torch.no_grad()
    def predictive_statistics(self, x, samples=20):
        probs = []
        for _ in range(samples):
            probs.append(torch.softmax(self(x), dim=-1))
        p = torch.stack(probs)
        mean_p = p.mean(0)
        variance = p.var(0, unbiased=False)
        entropy = -(mean_p.clamp_min(1e-8) * mean_p.clamp_min(1e-8).log()).sum(-1)
        return mean_p, variance, entropy
