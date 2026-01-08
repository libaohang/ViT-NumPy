from TrainTest import trainNetwork, testNetwork
from Loss import crossEntropyLoss
from Optimizer import Adam, AdamW
from Activations import GELU, ReLU
from VisionTransformer import VisionTransformer
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import cifar10
import numpy as np

# Toggle dropout
def train(network):
    for layer in network.layers:
        if hasattr(layer, "training"):
            layer.training = True

def eval(network):
    for layer in network.layers:
        if hasattr(layer, "training"):
            layer.training = False

def classifyMNIST():
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()

    xTrain = xTrain.astype(np.float32) / 255.0
    xTest  = xTest.astype(np.float32) / 255.0

    xTrain = xTrain[:, :, :, None]
    xTest = xTest[:, :, :, None]

    # Reach 98.22% after 30 epochs
    network = VisionTransformer(patchSize=4,
                                numPatches=49,
                                channels=1,
                                modelDim=32,
                                numHeads=4,
                                numTrans=3,
                                mlpWidth=64,
                                numClass=10,
                                activation=ReLU,
                                dropout=0.1)

    optimizer = AdamW(network, warmupSteps=200, lr=0.005, weight_decay=0.003)

    train(network)
    classifier = trainNetwork(network, crossEntropyLoss, optimizer, xTrain, yTrain, 30, 100, lrDecayStart=15)

    eval(network)
    testNetwork(classifier, crossEntropyLoss, xTest, yTest)

if __name__ == '__main__':
    classifyMNIST()