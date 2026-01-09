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

def crossEntropyWithLogits(logits, labels):
    labels = labels.astype(np.int64)  # (B,)

    # stable log-softmax
    m = np.max(logits, axis=1, keepdims=True)        # (B,1)
    shifted = logits - m                              # (B,C)
    exp_shifted = np.exp(shifted)                     # (B,C)
    probs = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)  # (B,C)

    # loss per sample
    loss = -np.log(probs[np.arange(labels.shape[0]), labels] + 1e-12) # (B,)

    # gradient w.r.t logits
    grad = probs.copy()
    grad[np.arange(labels.shape[0]), labels] -= 1
    grad /= labels.shape[0]

    return loss, grad
