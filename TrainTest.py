import numpy as np
from Augmentation import augmentCIFAR10

def trainNetwork(network, loss, optimizer, xTrain, yTrain, epochs = 5, batchSize = 100, lrDecayStart=10, augment=False, printBatch=False):
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

            if(augment):
                batchX = augmentCIFAR10(batchX)
            
            # Skip zeroing for the first batch since gradients have not been initialized
            if(batch > 0):
                network.zeroGradient()
            logits = network.forward(batchX)

            batchLoss, gradient = loss(logits, batchY)
            epochLoss += np.mean(batchLoss)
            numBatches += 1

            network.backward(gradient)
            optimizer.step()

            if numBatches % 10 == 0 and printBatch:
                print(f"epoch {epoch + 1} batch {numBatches}", flush=True)
                print("batchLoss shape/min/max/mean:",
                    np.shape(batchLoss), np.min(batchLoss), np.max(batchLoss), np.mean(batchLoss))

            if epoch == 0 and numBatches == 1:
                print("logits shape:", logits.shape)
                print("labels shape/dtype:", batchY.shape, batchY.dtype)
                print("batchLoss shape/min/max/mean:",
                    np.shape(batchLoss), np.min(batchLoss), np.max(batchLoss), np.mean(batchLoss))
                print("logits min/max/meanabs:",
          np.min(logits), np.max(logits), np.mean(np.abs(logits)))
                print("x dtype:", batchX.dtype)
                print("first weight dtype:", network.parameters()[0][0].dtype if isinstance(network.parameters()[0], tuple) else network.parameters()[0].dtype)
                print("logits dtype:", logits.dtype)

        if (epoch == lrDecayStart - 1):
            print("lr decay")
            optimizer.lrDecay()
        epochLoss /= numBatches
        print(f"Training error on epoch {1 + epoch} is {epochLoss}")

        result = open("results.txt", "a")
        result.write(f"Training error on epoch {1 + epoch} is {epochLoss}\n")
        result.close()
    
    return network

def testNetwork(network, loss, xTest, yTest, batchSize=100):
    n = xTest.shape[0]
    total_loss = 0.0
    total_correct = 0

    for i in range(0, n, batchSize):
        bx = xTest[i:i+batchSize]
        by = yTest[i:i+batchSize]

        logits = network.forward(bx)              # (B, 10)
        loss_vals, _ = loss(logits, by)        # (B,)
        total_loss += float(np.sum(loss_vals))

        preds = np.argmax(logits, axis=1)
        total_correct += int(np.sum(preds == by))

    avg_loss = total_loss / n
    acc = total_correct / n

    print(f"Testing error is {avg_loss}, with accuracy of {acc}")
    result = open("results.txt", "a")
    result.write(f"Testing error is {avg_loss}, with accuracy of {acc}\n")
    result.close()