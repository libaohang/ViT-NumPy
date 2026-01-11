from TrainTest import trainNetwork, testNetwork
from Loss import crossEntropyLoss, crossEntropyWithLogits
from Optimizer import Adam, AdamW
from Activations import GELU, ReLU
from VisionTransformer import VisionTransformer
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import cifar10
import numpy as np
import time

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

    # Reach 96.15% test accuracy after 20 epochs; runtime 5 minutes; lr decay at epoch 10
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
    
    optimizer1 = AdamW(network1, warmupSteps=100, lr=0.01, weightDecay=0.001)

    # Reach 98.22% test accuracy after 30 epochs; runtime 90 minutes; lr decay at epoch 15
    network2 = VisionTransformer(patchSize=4,
                                numPatches=49,
                                channels=1,
                                modelDim=32,
                                numHeads=4,
                                numTrans=3,
                                mlpWidth=64,
                                numClass=10,
                                activation=ReLU,
                                dropout=0.1)

    optimizer2 = AdamW(network2, warmupSteps=200, lr=0.005, weightDecay=0.003)

    open("results.txt", "w").close()

    start_time = time.perf_counter()
    train(network2)
    classifier = trainNetwork(network2, crossEntropyLoss, optimizer2, xTrain, yTrain, 20, 100, lrDecayStart=10)
    end_time = time.perf_counter()
    
    saveModel(classifier, "vit_mnist.npz")

    eval(network2)
    testNetwork(classifier, crossEntropyLoss, xTest, yTest)
    print(f"Total training time: {end_time - start_time:.2f} seconds")

    result = open("results.txt", "a")
    result.write(f"Total training time: {end_time - start_time:.2f} seconds\n")
    result.close()

def classifyCIFAR10():
    (xTrain, yTrain), (xTest, yTest) = cifar10.load_data()

    xTrain = xTrain.astype(np.float32) / 255.0
    xTest  = xTest.astype(np.float32) / 255.0

    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
    std  = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)

    xTrain = (xTrain - mean) / std
    xTest  = (xTest  - mean) / std

    yTrain = yTrain.reshape(-1).astype(np.int64)
    yTest  = yTest.reshape(-1).astype(np.int64)

    xTrain = xTrain[:, :, :]
    xTest = xTest[:, :, :]
    yTrain = yTrain[:]
    yTest = yTest[:]

    #Reach 77.5% test accuracy after 40 epochs; runtime 6.5 hours; lr decay at epoch 20
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
    
    optimizer3 = AdamW(network3, warmupSteps=500, lr=0.001, weightDecay=0.01)

    open("results.txt", "w").close()

    start_time = time.perf_counter()
    train(network3)
    classifier = trainNetwork(network3, crossEntropyWithLogits, optimizer3, xTrain, yTrain, 40, 50, lrDecayStart=20, augment=True, printBatch=True)
    end_time = time.perf_counter()

    saveModel(classifier, "vit_cifar10.npz")

    eval(network3)
    testNetwork(classifier, crossEntropyWithLogits, xTest, yTest)
    print(f"Total training time: {end_time - start_time:.2f} seconds")

    result = open("results.txt", "a")
    result.write(f"Total training time: {end_time - start_time:.2f} seconds\n")
    result.close()

    
def saveModel(network, path="vit_checkpoint.npz"):
    params = [arr for (arr, name) in network.parameters()]
    np.savez(path, *params)

if __name__ == '__main__':
    classifyMNIST()
    classifyCIFAR10()