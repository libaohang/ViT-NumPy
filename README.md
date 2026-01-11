# Vision Transformer from Scratch in NumPy
<br>

**Please read this by opening README.md in full screen for proper formatting** <br>

## Description
**A complete Vision Transformer (ViT) implemented using entirely NumPy**
<br>
I built the backpropagation, AdamW, data augmentation, and dropout with only NumPy to deeply understand attention mechanisms and transformer internals without relying on deep learning frameworks.
<br>

## Features
* Patch embedding with learnable linear projection
* Position embedding with CLS token, both learnable
* Multi-head attention from scratch
* Layer normalization with learnable scale and shift
* Residual connections
* Multi-layer perceptron with ReLU / GELU activations
* Dropout for regularization
* Adam and AdamW optimizer
* Learning rate warmup and decay
* Data augmentation
* Save and load model

## Model Architecture
### Vision Transformer Structure: 
ᅟᅟ    **Patch embedding**ᅟᅟᅟSplit image into patches and project into tokens <br>
ᅟᅟᅟᅟᅟᅟ↓ <br>
ᅟ  ᅟ**Position encoding**ᅟᅟᅟEncode position information for each token <br>
ᅟᅟᅟᅟᅟᅟ↓ <br>
 **Transformer encoder block** ᅟMulti-head attention followed by MLP, both paired with layer norm and residual; repeated *numTrans* times <br>
ᅟᅟᅟᅟᅟᅟ↓ <br>
ᅟᅟ ᅟ    **Layer norm** ᅟᅟᅟᅟ   Stablize logits (optional) <br>
ᅟᅟᅟᅟᅟᅟ↓ <br>
ᅟᅟᅟ     **Extract CLS**           ᅟᅟ         Get classification encodings from first row of sequence (2nd) dimension<br>
ᅟᅟᅟᅟᅟᅟ↓ <br>
ᅟᅟᅟᅟ ᅟ **MLP**               ᅟᅟᅟᅟMap CLS to class values <br>

### Networks and Variables:

I assembled 3 different configurations of vision transformer using the layers I implemented. Network 1 and Network 2 are designed for classifying MNIST while Network 3 is configured for classifying CIFAR-10. For context, MNIST is a dataset of grayscale images of handwritten numbers 0 to 9, and CIFAR is a dataset of colored images of 10 types of objects, such as birds, planes, trucks, etc. <br>
Below is a table detailing some influential variables and each network's respective value for them. <br>

**Variable**ᅟᅟ          **Description**ᅟᅟᅟᅟᅟᅟᅟᅟᅟᅟᅟᅟᅟᅟᅟᅟᅟᅟᅟᅟᅟᅟᅟᅟᅟ   ᅟ             **Network 1**ᅟᅟ**Network 2**ᅟᅟ**Network 3** <br>
-Parameters of *VisionTransformer*: <br>
*patchSize*ᅟᅟ      ​Side length of a patch, must be divisible by image dimensionsᅟᅟᅟᅟᅟ      7ᅟᅟᅟᅟᅟ        4ᅟᅟᅟᅟᅟ          4<br>
*numPatches*ᅟ     Number of square patches of size *patchSize* in one imageᅟᅟᅟᅟᅟᅟᅟ    16ᅟᅟᅟᅟᅟ    49ᅟᅟᅟᅟᅟ     64<br>
*channels*ᅟᅟᅟ Number of channels of input, 1 for MNIST and 3 for CIFAR-10ᅟᅟᅟᅟᅟᅟ1ᅟᅟᅟᅟᅟ        1ᅟᅟᅟᅟᅟ          3<br>
*modelDim*ᅟᅟ   Depth of embedding on each patch; must be dividible by *numHeads*ᅟᅟ     9ᅟᅟᅟᅟᅟ        32ᅟᅟᅟᅟᅟ     64<br>
*numHeads*ᅟᅟ  Number of attention heads in each transformer blockᅟᅟᅟᅟᅟᅟᅟᅟᅟ   3ᅟᅟᅟᅟᅟ        4ᅟᅟᅟᅟᅟ          8<br>
*numTrans*ᅟᅟ     Number of layers of transformer encoders to stackᅟᅟᅟᅟᅟᅟᅟᅟᅟᅟ    3ᅟᅟᅟᅟᅟ        3ᅟᅟᅟᅟᅟ          4<br>
*mlpWidth*ᅟᅟ     Hidden dimension of MLP in each transformer blockᅟᅟᅟᅟᅟᅟᅟᅟᅟ     32ᅟᅟᅟᅟᅟ    64ᅟᅟᅟᅟᅟ     128<br>
*numClasses*ᅟ       Number of classes to classify samples intoᅟᅟᅟᅟᅟᅟᅟᅟᅟᅟᅟᅟᅟ       10ᅟᅟᅟᅟᅟ   10ᅟᅟᅟᅟᅟ      10<br>
*activation*ᅟᅟ     Type of activation to use in MLP: ReLu or GELUᅟᅟᅟᅟᅟᅟᅟᅟᅟᅟ            ReLuᅟᅟᅟᅟ   ReLUᅟᅟᅟ            GELU<br>
*dropout*ᅟᅟᅟ    Percentage of MLP dropped; higher means more regularizationᅟᅟᅟ           0.1ᅟᅟᅟᅟᅟ  0.1ᅟᅟᅟᅟᅟ   0.1<br>
*classifierLN*ᅟᅟ Whether a layer norm is placed before classificationᅟᅟᅟᅟᅟᅟᅟᅟᅟ      Falseᅟᅟᅟᅟ   Falseᅟᅟᅟ            True<br>
<br>
-Parameters of *AdamW*: <br>
*warmupSteps*     Number of starting steps with reduced learning rate   ⁢⁬⁬⁬                   ⁢⁢⁢100ᅟᅟᅟᅟᅟ200ᅟᅟᅟᅟᅟ 500<br>
*lr*                                                  The learning rate of the optimizer                                              0.01ᅟᅟᅟ              0.005ᅟᅟᅟ           0.001<br>
*weightDecay*           Amount of weight decay on weights of linear and patch embedding         0.001ᅟᅟᅟᅟ  0.003ᅟᅟᅟᅟ   0.01<br>
<br>
-Parameters of *TrainNetwork*: <br>
*epochs*                                Number of epochs to train for                                              20ᅟᅟᅟᅟᅟ    30ᅟᅟ  ᅟᅟ           40<br>
*batchSize*                       Number of images to use in each update step                             100ᅟᅟᅟᅟᅟ100ᅟᅟᅟᅟᅟ 50<br>
*lrDecayStart*             The epoch to apply learning rate decay (x0.1) on                          10ᅟᅟᅟᅟᅟ     15ᅟᅟᅟᅟᅟ     20<br>
*augment*                         Whether to use data augmentation or not                               Falseᅟᅟᅟᅟ    Falseᅟᅟᅟᅟ   True<br>

## Training and Testing

I trained and tested Network 1 and Network 2 on MNIST, and Network 3 on CIFAR-10. Their performance and results are described below, with error measured using cross-entropy loss. <br>
### Network 1

**Training Error over Epochs Trained for Network 1 on MNIST:**

<img width="1400" height="882" alt="vit-mnist-1" src="https://github.com/user-attachments/assets/e8d7accb-aafb-4942-994c-7c7b6a313db2" />

__Key:__ <br>
Green line: training error vs epoch<br>
Red line: final test error after 20 epochs<br>

Network 1 is the most lightweight of the 3 and has the shortest training time of 5 minutes on my computer. It achieves final test accuracy of **96.7%** on MNIST after 20 epochs. The sharp decrease in loss between the 10th and 11th epoch is due to lr decay. <br>

### Network 2
**Training Error over Epochs Trained for Network 2 on MNIST:**

<img width="1400" height="865" alt="vit-mnist-2" src="https://github.com/user-attachments/assets/5b6e361c-8508-4e07-a28a-698d48c6a026" />

__Key:__ <br>
Orange line: training error vs epoch<br>
Blue line: final test error after 30 epochs<br>

Network 2 achieves **98.2%** test accuracy on MNIST after 30 epochs. It has a smaller patch size and more attention heads than Network 2, so it took 1.5 hours to train. Bumps and oscillations in the training curve are due to noise introduced by dropout.

### Network 3
**Training Error over Epochs Trained for Network 3 on CIFAR-10:**

<img width="1400" height="879" alt="vit-cifar10" src="https://github.com/user-attachments/assets/8fd3ed8f-7995-4fcc-a461-ed3dddb9749a" />

__Key:__ <br>
Purple line: training error vs epoch<br>
Black line: final test error after 40 epochs<br>

Network 3 is much bigger than the previous 2 networks, and because implementing everything in NumPy means it can only rely on CPU to run, it took 6.5 hours to train. After 40 epochs, it has a final test accuracy of **77.5%** on CIFAR-10. Considering the complexity of CIFAR-10, this accuracy is satisfactory for a ViT implemented from NumPy. Initially, training the network would cause the memory to explode to 6-8 GB, so I had to make some optimizations, such as forcing weight matrices to have type float32 instead of float64 and reducing batch size. After memory optimization, training took about 2 GB of memory and trained for 6.5 hours using 2-3 CPU cores. <br>

### Comparison to CNN:
For MNIST, ViT appears to converge faster than the CNN that I also implemented in NumPy. The lightweight ViT of Network 1 achieves 96.7% after 20 epochs, which is slightly better than the CNN with 1 convolution layer at 96% test accuracy on MNIST. The same pattern is observed for CIFAR-10 where Network 3 achieves 77.5% compared to the 60% achieved by CNN with 3 convolution layers. However, CNN theoretically would do better on small datasets such as MNIST and CIFAR-10, so these observed differences are certainly because I implemented data augmentation and layer normalization for ViT while I didn't do so for CNN. In addition, the CNN with 3 layers was only trained for 25 epochs, and its training curve was still steep towards the end, so it would have likely achieved a similar accuracy of 70-80% on CIFAR-10 if trained for more epochs. <br>

## Running and Loading

To run the training and testing process, open the file *ClassifyImages.py* and run *classifyMNIST()* to train on MNIST and run *classifyCIFAR10()* to train on CIFAR-10. To choose which network to use, configure the parameters of the *trainNetwork* function to be the defined network, optimizer, and corresponding epoch, batch size, and lr decay. <br>

To load and test the saved models, run the file *LoadNetworks.py*. This will recreate a trained Network 1 by loading in saved parameters to test on MNIST, and also load a trained Network 3 to test on CIFAR-10. <br>

## References
- Matt Nguyen, article: https://medium.com/correll-lab/building-a-vision-transformer-model-from-scratch-a3054f707cc6. Referenced the structure and mechanism of a general vision transformer, attention head, and transformer encoder.
- ViT: Dosovitskiy et al., *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale*, 2020. 
- Training: Touvron et al., *Training Data-Efficient Image Transformers & Distillation through Attention (DeiT)*, 2021.
- Attention: Vaswani et al., *Attention Is All You Need*, 2017.
- Optimizer: Loshchilov & Hutter, *Decoupled Weight Decay Regularization (AdamW)*, 2019.
- Data: Krizhevsky, *Learning Multiple Layers of Features from Tiny Images (CIFAR-10)*.
