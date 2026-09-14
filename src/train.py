from __future__ import annotations
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def device_from_config(name="auto"):
    if name != "auto": return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_deterministic(model, loader, optimizer, device, epochs):
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            # Map the two original labels to 0/1.
            y = y % 2
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()

def train_bayesian(model, loader, optimizer, device, epochs, beta=0.001):
    model.train()
    start = time.time()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), (y % 2).to(device)
            optimizer.zero_grad()
            logits = model(x)
            nll = F.cross_entropy(logits, y)
            # Normalize KL by number of training examples so beta is interpretable.
            kl = model.kl_divergence() / len(loader.dataset)
            loss = nll + beta * kl
            loss.backward()
            optimizer.step()
    return time.time() - start

@torch.no_grad()
def evaluate_deterministic(model, loader, device):
    model.eval()
    correct = total = 0
    confidences, correctness = [], []
    for x,y in loader:
        x,y=x.to(device),(y%2).to(device)
        p = torch.softmax(model(x),-1)
        conf,pred = p.max(-1)
        correct += (pred==y).sum().item(); total += y.numel()
        confidences.extend(conf.cpu().numpy()); correctness.extend((pred==y).cpu().numpy())
    return correct/total, np.asarray(confidences), np.asarray(correctness)

@torch.no_grad()
def evaluate_bayesian(model, loader, device, samples=20):
    model.eval()
    correct = total = 0
    confidences, correctness, entropies = [], [], []
    for x,y in loader:
        x,y=x.to(device),(y%2).to(device)
        mean_p, _, entropy = model.predictive_statistics(x, samples)
        conf,pred = mean_p.max(-1)
        correct += (pred==y).sum().item(); total += y.numel()
        confidences.extend(conf.cpu().numpy()); correctness.extend((pred==y).cpu().numpy())
        entropies.extend(entropy.cpu().numpy())
    return correct/total, np.asarray(confidences), np.asarray(correctness), np.asarray(entropies)
