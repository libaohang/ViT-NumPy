from Layer import Layer
import numpy as np

class GetCLS(Layer):
    def forward(self, x):
        # (batch, numPatches + 1, modelDim)
        return x[:, 0, :]
    
    def backward(self, dE_dY):
        dE_dX = np.zeros_like(dE_dY)
        dE_dX[:, 0, :] = dE_dY
        return dE_dX
    
    def parameters(self):
        return []
    
    def gradients(self):
        return []