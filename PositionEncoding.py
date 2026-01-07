from Layer import Layer
import numpy as np

class PositionEncoding(Layer):
    def __init__(self, numPatches, modelDim):
        self.cls = np.random.randn(1, 1, modelDim) * np.sqrt(1 / modelDim)
        self.posEmbedding = np.random.randn(1, numPatches + 1, modelDim) * np.sqrt(2 / ((numPatches + 1) * modelDim))

    def forward(self, patchEmbedding):
        # patchEmbedding : (batch, numPatches, modelDim)
        batch = patchEmbedding.shape[0]

        # Reshape cls to match batch dimension
        clsTokens = np.repeat(self.cls, batch, axis=0)

        # Combine cls and patch embeddings, add with postion embedding
        tokens = np.concatenate([clsTokens, patchEmbedding], axis=1) # tokens -> (batch, numPatches + 1, modelDim)
        tokens += self.posEmbedding
        return tokens
    
    def backward(self, dE_dY):
        self.dE_dPos = dE_dY.sum(axis=0, keepdims = True)

        # Get the cls gradient from every image
        self.dE_dCls = dE_dY[:, 0:1, :].sum(axis=0, keepdims = True)
        dE_dPatch = dE_dY[:, 1:, :]
        return dE_dPatch
    
    def parameters(self):
        return [(self.cls, "cls weight"), (self.posEmbedding, "pos weight")]
    
    def gradients(self):
        return [(self.dE_dCls, "cls weight"), (self.dE_dPos, "pos weight")]