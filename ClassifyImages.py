from TrainTest import trainNetwork, testNetwork
from Loss import crossEntropyLoss
from Optimizer import Adam
from Activations import GELU
from VisionTransformer import VisionTransformer
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import cifar10
import numpy as np

def classifyMNIST():
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()

    xTrain = xTrain.astype(np.float32) / 255.0
    xTest  = xTest.astype(np.float32) / 255.0

    xTrain = xTrain[:, :, :, None]
    xTest = xTest[:, :, :, None]

    network = VisionTransformer(patchSize=7,
                                numPatches=16,
                                channels=1,
                                modelDim=9,
                                numHeads=3,
                                numTrans=3,
                                mlpWidth=32,
                                numClass=10,
                                activation=GELU)

    optimizer = Adam(network)
    classifier = trainNetwork(network, crossEntropyLoss, optimizer, xTrain, yTrain, 10, 100)

    testNetwork(classifier, crossEntropyLoss, xTest, yTest)

if __name__ == '__main__':
    classifyMNIST()