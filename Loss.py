import numpy as np

def crossEntropyLoss(logits, labels):
    # logits: (B, C)
    # labels: (B,)

    logits = logits - logits.max(axis=1, keepdims=True) 
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    loss = -np.log(probs[np.arange(len(labels)), labels] + 1e-10)
    loss = loss.mean()

    y = np.zeros_like(probs)
    y[np.arange(len(labels)), labels] = 1

    grad = (probs - y) / len(labels)
    return loss, grad
