import numpy as np

def random_horizontal_flip(x, p=0.5):
    # x: (B, H, W, C)
    mask = np.random.rand(x.shape[0]) < p
    x_flipped = x.copy()
    x_flipped[mask] = x_flipped[mask, :, ::-1, :]
    return x_flipped

def random_crop(x, crop_size=32, padding=4):
    # x: (B, H, W, C)
    B, H, W, C = x.shape

    # pad with zeros 
    x_padded = np.pad(
        x,
        ((0, 0), (padding, padding), (padding, padding), (0, 0)),
        mode="constant"
    )

    cropped = np.empty((B, crop_size, crop_size, C), dtype=x.dtype)

    for i in range(B):
        top = np.random.randint(0, 2 * padding + 1)
        left = np.random.randint(0, 2 * padding + 1)
        cropped[i] = x_padded[i, top:top+crop_size, left:left+crop_size, :]

    return cropped

def augmentCIFAR10(x):
    x = random_horizontal_flip(x, p=0.5)
    x = random_crop(x, crop_size=32, padding=4)
    return x
