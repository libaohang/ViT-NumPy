from TrainTest import testNetwork
from Loss import crossEntropyLoss, crossEntropyWithLogits
from Activations import GELU, ReLU
from VisionTransformer import VisionTransformer
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import cifar10
import numpy as np
import os

def loadModel(network, path="vit_checkpoint.npz"):
    data = np.load(path)
    params = network.parameters()

    assert len(params) == len(data.files), "Mismatch in parameter count"

    for p, k in zip(params, data.files):
        p[0][...] = data[k]

def eval(network):
    for layer in network.layers:
        if hasattr(layer, "training"):
            layer.training = False

def testMNIST():
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()

    xTest  = xTest.astype(np.float32) / 255.0

    xTest = xTest[:, :, :, None]

    network1 = VisionTransformer(patchSize=7,
                                numPatches=16,
                                channels=1,
                                modelDim=9,
                                numHeads=3,
                                numTrans=3,
                                mlpWidth=32,
                                numClass=10,
                                activation=ReLU,
                                dropout=0.1)

    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "Saved models", "vit_mnist.npz")

    loadModel(network1, path)

    eval(network1)
    print("Testing Network 1 on MNIST...")
    testNetwork(network1, crossEntropyLoss, xTest, yTest)


def testCIFAR10():
    (xTrain, yTrain), (xTest, yTest) = cifar10.load_data()

    xTest  = xTest.astype(np.float32) / 255.0

    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
    std  = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)

    xTest  = (xTest  - mean) / std

    yTest  = yTest.reshape(-1).astype(np.int64)

    network3 = VisionTransformer(patchSize=4,
                                numPatches=64,
                                channels=3,
                                modelDim=64,
                                numHeads=8,
                                numTrans=4,
                                mlpWidth=128,
                                numClass=10,
                                activation=GELU,
                                dropout=0.1,
                                classifierLN=True)
    
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "Saved models", "vit_cifar10.npz")
    
    loadModel(network3, path)
    
    eval(network3)
    print("Testing Network 3 on CIFAR-10...")
    testNetwork(network3, crossEntropyWithLogits, xTest, yTest)

if __name__ == '__main__':
    testMNIST()
    testCIFAR10()