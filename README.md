# Vision Transformer from Scratch in NumPy
<br>

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

### Networks and Variables

I assembled 3 different configurations of vision transformer using the layers I implemented. Network 1 and Network 2 are designed for classifying MNIST while Network 3 is configured for classifying CIFAR-10. For context, MNIST is a dataset of grayscale images of handwritten numbers 0 to 9, and CIFAR is a dataset of colored images of 10 types of objects, such as birds, planes, trucks, etc. <br>
Below is a table detailing some influential variables and each network's respective value for them. <br>

**Variable name**ᅟᅟ**Description**ᅟᅟᅟᅟᅟᅟᅟᅟᅟᅟᅟᅟᅟᅟᅟᅟᅟᅟᅟᅟᅟᅟᅟᅟᅟ   ᅟ**Network 1**ᅟᅟ**Network 2**ᅟᅟ**Network 3** <br>
Parameters of *VisionTransformer*: <br>
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
Parameters of *AdamW*: <br>
*warmupSteps*     Number of starting steps with reduced learning rate   ⁢⁬⁬⁬                   ⁢⁢⁢100ᅟᅟᅟᅟᅟ200ᅟᅟᅟᅟᅟ 500<br>
*lr*                                                  The learning rate of the optimizer                                              0.01ᅟᅟᅟ              0.005ᅟᅟᅟ           0.001<br>
*weightDecay*           Amount of weight decay on weights of linear and patch embedding         0.001ᅟᅟᅟᅟ  0.003ᅟᅟᅟᅟ   0.01<br>
<br>
Parameters of *TrainNetwork*: <br>
*epochs*                                Number of epochs to train for                                              20ᅟᅟᅟᅟᅟ    30ᅟᅟ  ᅟᅟ           40<br>
*batchSize*                       Number of images to use in each update step                             100ᅟᅟᅟᅟᅟ100ᅟᅟᅟᅟᅟ 50<br>
*lrDecayStart*             The epoch to apply learning rate decay on                                10ᅟᅟᅟᅟᅟ     15ᅟᅟᅟᅟᅟ     20<br>
*augment*                         Whether to use data augmentation or not                               Falseᅟᅟᅟᅟ    Falseᅟᅟᅟᅟ   True<br>
