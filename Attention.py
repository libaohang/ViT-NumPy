from Layer import Layer
from LinearLayer import Linear
from Softmax import Softmax
import numpy as np

# Scaled Dot-Product Attention
class AttentionHead(Layer):
    def __init__(self, modelDim, headSize):
        self.headSize = headSize

        self.query = Linear(modelDim, headSize)
        self.key = Linear(modelDim, headSize)
        self.value = Linear(modelDim, headSize)
        self.softmax = Softmax()
        # Softmax is not needed here but just to be consistent
        self.layers = [self.query, self.key, self.value, self.softmax]

    def forward(self, tokens):
        # tokens : (batch, numPatches + 1, modelDim)
        self.Q = self.query(tokens)
        self.K = self.key(tokens)
        self.V = self.value(tokens)
        # Q, K, V : (batch, numPatches + 1, headSize)

        score = self.Q @ self.K.transpose(0, 2, 1) 
        score /= self.headSize ** 0.5

        # self.attention : (batch, numPatches + 1, numPatches + 1)
        self.attention = self.softmax.forward(score, axis=-1) 
        attention = self.attention @ self.V 
        # attention : (batch, numPatches + 1, headSize)

        return attention
    
    def backward(self, dE_dY):
        # dE_dY : (batch, numPatches + 1, headSize)
        # 2nd dimension of dE_dY correspond 2nd dimension of self.attention
        dE_dV = self.attention.transpose(0, 2, 1) @ dE_dY # dE_dV : (batch, numPatches + 1, headSize)
        dE_dAtt = dE_dY @ self.V.transpose(0, 2, 1) # dE_dAtt : (batch, numPatches + 1, numPatches + 1)
        dE_dScore = self.softmax.backward(dE_dAtt)
        dE_dScore /= self.headSize ** 0.5

        # 2nd dimension of score correspond to 2nd dimension of Q, 3rd correspond to 2nd dimension of K
        dE_dQ = dE_dScore @ self.K
        dE_dK = dE_dScore.transpose(0, 2, 1) @ self.Q

        dE_dTokensQ = self.query.backward(dE_dQ)
        dE_dTokensK = self.key.backward(dE_dK)
        dE_dTokensV = self.value.backward(dE_dV)
        # (batch, numPatches + 1, modelDim)
        dE_dTokens = dE_dTokensQ + dE_dTokensK + dE_dTokensV

        return dE_dTokens
    
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


class MultiHeadAttention(Layer):
    def __init__(self, modelDim, numHeads):
        assert modelDim % numHeads == 0
        self.headSize = modelDim // numHeads
        self.linear = Linear(modelDim, modelDim)

        self.heads = [AttentionHead(modelDim, self.headSize) for _ in range(numHeads)]
        self.layers = self.heads + [self.linear]

    # Feed tokens to numHeads and concatenate outputs together; output has same shape as input
    def forward(self, tokens):
        outTokens = []
        for head in self.heads:
            outTokens.append(head.forward(tokens))
        outTokens = np.concatenate(outTokens, axis=-1)

        outTokens = self.linear.forward(outTokens)
        return outTokens
    
    def backward(self, dE_dY):
        dE_dOut = self.linear.backward(dE_dY)

        dE_dTokens = np.zeros_like(dE_dY)
        splitGradient = np.split(dE_dOut, len(self.heads), axis=-1)
        for head, tokens in zip(self.heads, splitGradient):
            dE_dTokens += head.backward(tokens)
        
        return dE_dTokens
    
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

