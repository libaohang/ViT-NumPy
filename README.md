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
