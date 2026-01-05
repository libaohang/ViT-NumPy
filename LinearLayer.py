from Layer import Layer
import numpy as np

class Linear(Layer):
    def __init__(self, inputDim, outputDim):
        self.weights = np.random.randn(outputDim, inputDim)  * np.sqrt(1 / (inputDim))
        # Add axis so bias broadcasts in the batch dimension
        self.bias = np.zeros((1, outputDim))
    
    def forward(self, input):
        self.input = input

        # Weight each input dimension by the corresponding weight dimension and add bias
        return input @ self.weights.T + self.bias
    
    def backward(self, dE_dY):
        # Gradient of weight
        self.dE_dW = dE_dY.T @ self.input

        # Gradient of bias
        self.dE_dB = np.sum(dE_dY, axis=0, keepdims=True)
        
        # Gradient of input
        dE_dX = dE_dY @ self.weights

        return dE_dX
    
    def parameters(self):
        return [self.weights, self.bias]
    
    def gradients(self):
        return [self.dE_dW, self.dE_dB]