from Layer import Layer
import numpy as np

class PatchEmbedding(Layer):
    def __init__(self, patchSize, channels, modelDim):
        self.patchSize = patchSize
        self.channels = channels
        self.modelDim = modelDim
        self.patchWeights = (np.random.randn(patchSize * patchSize * channels, modelDim) * np.sqrt(2 / (patchSize * patchSize * channels))).astype(np.float32)

    def forward(self, image):
        self.batch, height, width, channels = image.shape
        assert channels == self.channels
        assert height % self.patchSize == 0 and width % self.patchSize == 0

        self.outHeight = height // self.patchSize
        self.outWidth = width // self.patchSize

        patches = image.reshape(self.batch, self.outHeight, self.patchSize, self.outWidth, self.patchSize, channels)
        patches = patches.transpose(0, 1, 3, 2, 4, 5) # patches -> (batch, outHeight, outWidth, patchSize, patchSize, channels)
        self.patches = patches.reshape(self.batch, self.outHeight * self.outWidth, self.patchSize * self.patchSize * channels)

        embeddings = self.patches @ self.patchWeights # embeddings -> (batch, outHeight * outWidth, modelDim)

        return embeddings

    def backward(self, dE_dY):
        patchesTransposed = self.patches.transpose(2, 0, 1) # patches -> (patchSize * patchSize * channels, batch, outHeight * outWidth)
        patchesReshaped = patchesTransposed.reshape(self.patchSize * self.patchSize * self.channels, 
                                                    self.batch * self.outHeight * self.outWidth)
        dE_dY = dE_dY.reshape(self.batch * self.outHeight * self.outWidth, 
                              self.modelDim) # dE_dY -> (batch * outHeight * outWidth, modelDim)

        self.dE_dW = patchesReshaped @ dE_dY # dE_dW -> (patchSize * patchSize * channels, modelDim)

        # Don't actually need to return anything since this is the first layer (last layer of backward prop)

        dE_dX = dE_dY @ self.patchWeights.T # dE_dX -> (batch, outHeight * outWidth, patchSize * patchSize * channels)
        dE_dX = dE_dX.reshape(self.batch, self.outHeight, self.outWidth, self.patchSize, self.patchSize, self.channels)
        dE_dX = dE_dX.transpose(0, 1, 3, 2, 4, 5)
        dE_dX = dE_dX.reshape(self.batch, self.outHeight * self.patchSize, self.outWidth * self.patchSize, self.channels)
        return dE_dX
    
    def parameters(self):
        return [(self.patchWeights, "patch weight")]
    
    def gradients(self):
        return [(self.dE_dW, "patch weight")]