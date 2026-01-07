import numpy as np
from Softmax import Softmax

def trainNetwork(network, loss, optimizer, xTrain, yTrain, epochs = 5, batchSize = 100):
    sampleSize = xTrain.shape[0]

    for epoch in range(epochs):
        epochLoss = 0.0
        numBatches = 0

        indices = np.random.permutation(sampleSize)
        xTrain = xTrain[indices]
        yTrain = yTrain[indices]

        for batch in range(0, sampleSize, batchSize):   
            batchX = xTrain[batch : batch + batchSize]
            batchY = yTrain[batch : batch + batchSize]
            
            logits = network.forward(batchX)
            
            batchLoss, gradient = loss(logits, batchY)
            epochLoss += np.mean(batchLoss)
            numBatches += 1

            network.backward(gradient)
            optimizer.step()

            network.zeroGradient()

        epochLoss /= numBatches
        print(f"Training error on epoch {1 + epoch} is {epochLoss}")
    
    return network

def testNetwork(network, loss, xTest, yTest):
    logits = network.forward(xTest)

    error, _ = loss(logits, yTest)

    predictions = logits.argmax(axis=1)

    accuracy = np.mean(predictions == yTest)

    print(f"Testing error is {np.mean(error)}, with accuracy of {accuracy}")