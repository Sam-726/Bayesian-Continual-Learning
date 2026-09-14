from __future__ import annotations
import numpy as np

def average_accuracy(matrix):
    vals = []
    for row in matrix:
        vals.append(np.mean([x for x in row if x is not None]))
    return float(np.mean(vals))

def forgetting(matrix):
    # matrix[stage][task], with None for not-yet-seen tasks.
    final = matrix[-1]
    out = []
    for task in range(len(final)):
        history = [row[task] for row in matrix if row[task] is not None]
        if len(history) >= 2:
            out.append(max(history[:-1]) - final[task])
    return float(np.mean(out)) if out else 0.0

def ece(confidence, correctness, bins=10):
    confidence = np.asarray(confidence)
    correctness = np.asarray(correctness).astype(float)
    edges = np.linspace(0,1,bins+1)
    score = 0.0
    for i in range(bins):
        mask = (confidence > edges[i]) & (confidence <= edges[i+1])
        if mask.any():
            score += mask.mean() * abs(confidence[mask].mean() - correctness[mask].mean())
    return float(score)
