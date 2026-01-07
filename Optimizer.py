import numpy as np

class Adam:
    def __init__(self, network, lr=1e-3, warmupSteps = 100, betas=(0.9, 0.999), eps=1e-8):
        self.network = network
        self.targetlr = lr
        self.warmupSteps = warmupSteps
        self.beta1, self.beta2 = betas
        self.eps = eps

        self.t = 0
        self.params = self.network.parameters()
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]

    def step(self):
        self.t += 1
                
        if self.t <= self.warmup_steps:
            lr = self.targetlr * self.t / self.warmup_steps
        else:
            lr = self.targetlr
            
        parameters = self.params
        gradients = self.network.gradients()

        for i, (p, g) in enumerate(zip(parameters, gradients)):
            if g is None:
                continue

            # Update biased moments
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)

            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Parameter update (in-place)
            p -= lr * m_hat / (np.sqrt(v_hat) + self.eps)
