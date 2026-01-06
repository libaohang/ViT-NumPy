from Layer import Layer
from CLS import GetCLS
from LinearLayer import Linear
from PatchEmbedding import PatchEmbedding
from PositionEncoding import PositionEncoding
from Transformer import TransformerEncoder

class VisionTransformer(Layer):
    def __init__(self, patchSize, numPatches, channels, modelDim, numHeads, numTrans, mlpWidth, numClass, activation):
        self.patch = PatchEmbedding(patchSize, channels, modelDim)
        self.position = PositionEncoding(numPatches, modelDim)
        self.layers = [self.patch, self.position]

        for _ in range(numTrans):
            self.layers.append(TransformerEncoder(modelDim, numHeads, mlpWidth, activation))

        # Classification head
        self.mlp = Linear(modelDim, numClass)
        # Softmax done in training loop
        self.layers += [GetCLS(), self.mlp]

    
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