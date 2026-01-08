from Layer import Layer
from LinearLayer import Linear
from Dropout import Dropout

class MLP(Layer):
    def __init__(self, inputDim, hiddenChannels, activation, dropout=False, p=None):
        # len(hiddenChannels) is the total number of linear layers
        assert len(hiddenChannels) > 0

        # Initialize with the input layer, then add remaining (len(hiddenChannels) - 1) activation+linear layers
        self.layers = [Linear(inputDim, hiddenChannels[0])]
        for i in range(len(hiddenChannels) - 1):
            self.layers.append(activation())
            if(dropout and (p is not None)):
                self.layers.append(Dropout(p))
            self.layers.append(Linear(hiddenChannels[i], hiddenChannels[i + 1]))

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input
    
    def backward(self, gradient):
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        return gradient
    
    def parameters(self):
        parameters = []
        for layer in self.layers:
            for parameter in layer.parameters():
                parameters.append(parameter)
        return parameters
    
    def gradients(self):
        gradients = []
        for layer in self.layers:
            for gradient in layer.gradients():
                gradients.append(gradient)
        return gradients
    
# MLP structure specific for ViT: Linear(modelDim, hiddenDim) -> activation -> Linear(hiddenDim, modelDim)
class ViTMLP(MLP):
    def __init__(self, modelDim, hiddenDim, activation, p=0.1):
        super().__init__(modelDim, [hiddenDim, modelDim], activation, dropout=True, p=p)