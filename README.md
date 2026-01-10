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
### Vision transformer structure: 
ᅟᅟ    **Patch embedding**ᅟᅟᅟSplit image into patches and project into tokens <br>
ᅟᅟᅟᅟᅟᅟ↓ <br>
ᅟ  ᅟ**Position encoding**ᅟᅟᅟEncode position information for each token <br>
ᅟᅟᅟᅟᅟᅟ↓ <br>
 **Transformer encoder block** ᅟMulti-head attention followed by MLP, both paired with layer norm and residual; repeated *numTrans* times <br>
ᅟᅟᅟᅟᅟᅟ↓ <br>
ᅟᅟ ᅟ **Layer norm** ᅟᅟᅟᅟᅟStablize logits (optional) <br>
ᅟᅟᅟᅟᅟᅟ↓ <br>
ᅟᅟᅟᅟ**Extract CLS**           ᅟᅟᅟGet classification encodings from first row of sequence (2nd) dimension<br>
ᅟᅟᅟᅟᅟᅟ↓ <br>
ᅟᅟᅟᅟ ᅟ **MLP**               ᅟᅟᅟᅟ Map CLS to class values <br>

### Networks and Variables

I assembled 3 different configurations of vision transformer using the layers I implemented. Network 1 and Network 2 are designed for classifying MNIST while Network 3 is configured for classifying CIFAR-10. For context, MNIST is a dataset of grayscale images of handwritten numbers 0 to 9, and CIFAR is a dataset of colored images of 10 types of objects, such as birds, planes, trucks, etc. <br>
Below is a table detailing some influential variables and each network's respective value for them. <br>

**Variable name**ᅟᅟ**Description**ᅟᅟᅟᅟᅟᅟᅟᅟᅟᅟᅟᅟᅟᅟᅟᅟᅟᅟᅟᅟᅟᅟᅟᅟᅟ   ᅟ**Network 1**ᅟᅟ**Network 2**ᅟᅟ**Network 3** <br>
Parameters of *VisionTransformer*: <br>
*patchSize*ᅟᅟ      ​Side length of a patch, must be divisible by image dimensions<br>
*numPatches*ᅟ     Number of square patches of size *patchSize* in one image<br>
*channels*ᅟᅟᅟ Number of channels of input, 1 for MNIST and 3 for CIFAR-10<br>
*modelDim*ᅟᅟ   Depth of embedding on each patch; must be dividible by *numHeads*<br>
*numHeads*ᅟᅟ  Number of attention heads in each transformer block<br>
*numTrans*ᅟᅟ     Number of layers of transformer encoders to stack<br>
*mlpWidth*ᅟᅟ     Hidden dimension of MLP in each transformer block<br>
*numClasses*ᅟ       Number of classes to classify samples into<br>
*activation*ᅟᅟ     Type of activation to use in MLP: ReLu or GELU<br>
*dropout*ᅟᅟᅟ    The percentage of MLP dropped; higher means more regularization<br>
*classifierLN*ᅟᅟ Whether a layer norm is placed before classification<br>
Parameters of *AdamW*: <br>
*warmupSteps*
*lr*
*weightDecay*
Parameters of *TrainNetwork*: <br>
*epochs*
*batchSize*
*lrDecayStart*
*augment*
