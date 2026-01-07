from Layer import Layer
import numpy as np

class GetCLS(Layer):
    def forward(self, x):
        # (batch, numPatches + 1, modelDim)
        self.seqLength = x.shape[1]
        return x[:, 0, :]
    
    def backward(self, dE_dY):
        dE_dX = np.zeros((dE_dY.shape[0], self.seqLength, dE_dY.shape[1]))
        dE_dX[:, 0, :] = dE_dY
        return dE_dX
    
    def parameters(self):
        return []
    
    def gradients(self):
        return []