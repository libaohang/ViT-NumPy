import numpy as np
from Layer import Layer

class Dropout(Layer):
    def __init__(self, p=0.1):
        assert 0 <= p < 1
        self.p = p
        self.training = True

    def forward(self, x):
        if not self.training or self.p == 0:
            return x

        self.mask = (np.random.rand(*x.shape) > self.p) / (1 - self.p)
        return x * self.mask

    def backward(self, dE_dY):
        if not self.training or self.p == 0:
            return dE_dY
        return dE_dY * self.mask

    def parameters(self):
        return []

    def gradients(self):
        return []
