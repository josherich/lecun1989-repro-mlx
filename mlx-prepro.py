import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets

# -----------------------------------------------------------------------------

torch.manual_seed(1337)
np.random.seed(1337)

for split in {'train', 'test'}:

    data = datasets.MNIST('./data', train=split=='train', download=True)

    n = 7291 if split == 'train' else 2007
    rp = np.random.permutation(len(data))[:n]

    X = torch.full((n, 1, 16, 16), 0.0, dtype=torch.float32)
    Y = torch.full((n, 10), -1.0, dtype=torch.float32)
    for i, ix in enumerate(rp):
        I, yint = data[int(ix)]
        # PIL image -> numpy -> torch tensor -> [-1, 1] fp32
        xi = torch.from_numpy(np.array(I, dtype=np.float32)) / 127.5 - 1.0
        # add a fake batch dimension and a channel dimension of 1 or F.interpolate won't be happy
        xi = xi[None, None, ...]
        # resize to (16, 16) images with bilinear interpolation
        xi = F.interpolate(xi, (16, 16), mode='bilinear')
        X[i] = xi[0] # store

        # set the correct class to have target of +1.0
        Y[i, yint] = 1.0

    np.save(split + '1989x.npy', X.numpy())
    np.save(split + '1989y.npy', Y.numpy())
