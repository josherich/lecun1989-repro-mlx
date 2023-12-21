"""
Running this script eventually gives:
23
eval: split train. loss 4.073383e-03. error 0.62%. misses: 45
eval: split test . loss 2.838382e-02. error 4.09%. misses: 82
"""

import os
import json
import argparse

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from tensorboardX import SummaryWriter # pip install tensorboardX

# -----------------------------------------------------------------------------

class Net(nn.Module):
  """ 1989 LeCun ConvNet per description in the paper """

  def __init__(self):
    super().__init__()

    # initialization as described in the paper to my best ability, but it doesn't look right...
    winit = lambda fan_in, *shape: (mx.random.uniform(low=0, high=1, shape=shape) - 0.5) * 2 * 2.4 / fan_in**0.5
    macs = 0 # keep track of MACs (multiply accumulates)
    acts = 0 # keep track of number of activations

    # H1 layer parameters and their initialization
    self.H1w = winit(5*5*1, 12, 5, 5, 1)
    self.H1b = mx.zeros((8, 8, 12)) # presumably init to zero for biases
    # assert self.H1w.nelement() + self.H1b.nelement() == 1068
    macs += (5*5*1) * (8*8) * 12
    acts += (8*8) * 12

    # H2 layer parameters and their initialization
    """
    H2 neurons all connect to only 8 of the 12 input planes, with an unspecified pattern
    I am going to assume the most sensible block pattern where 4 planes at a time connect
    to differently overlapping groups of 8/12 input planes. We will implement this with 3
    separate convolutions that we concatenate the results of.
    """
    self.H2w = winit(5*5*8, 12, 5, 5, 8)
    self.H2b = mx.zeros((4, 4, 12)) # presumably init to zero for biases
    # assert self.H2w.nelement() + self.H2b.nelement() == 2592
    macs += (5*5*8) * (4*4) * 12
    acts += (4*4) * 12

    # H3 is a fully connected layer
    self.H3w = winit(4*4*12, 4*4*12, 30)
    self.H3b = mx.zeros(30)
    # assert self.H3w.nelement() + self.H3b.nelement() == 5790
    macs += (4*4*12) * 30
    acts += 30

    # output layer is also fully connected layer
    self.outw = winit(30, 30, 10)
    self.outb = -mx.ones(10) # 9/10 targets are -1, so makes sense to init slightly towards it
    # assert self.outw.nelement() + self.outb.nelement() == 310
    macs += 30 * 10
    acts += 10

    self.macs = macs
    self.acts = acts

  def __call__(self, x):

    x = mx.transpose(x, [0, 2, 3, 1]) # from (1, 1, 16, 16)
    # x has shape (1, 16, 16, 1)
    x = mx.pad(x, [(0,0), (2,2), (2,2), (0,0)], constant_values=-1.0) # pad by two using constant -1 for background
    x = mx.conv2d(x, self.H1w, stride=2) + self.H1b
    x = mx.tanh(x)

    # x is now shape (1, 8, 8, 12)
    x = mx.pad(x, [(0,0), (2,2), (2,2), (0,0)], constant_values=-1.0) # pad by two using constant -1 for background
    slice1 = mx.conv2d(x[:,:,:, 0:8], self.H2w[0:4], stride=2) # first 4 planes look at first 8 input planes
    slice2 = mx.conv2d(x[:,:,:, 4:12], self.H2w[4:8], stride=2) # next 4 planes look at last 8 input planes
    slice3 = mx.conv2d(mx.concatenate((x[:,:,:, 0:4], x[:,:,:, 8:12]), axis=3), self.H2w[8:12], stride=2) # last 4 planes are cross
    x = mx.concatenate((slice1, slice2, slice3), axis=3) + self.H2b
    x = mx.tanh(x)

    # x is now shape (1, 4, 4, 12)
    x = mx.flatten(x, start_axis=1) # (1, 4*4*12)
    x = mx.matmul(x, self.H3w) + self.H3b
    x = mx.tanh(x)

    # x is now shape (1, 30)
    x = mx.matmul(x, self.outw) + self.outb
    x = mx.tanh(x)

    # x is finally shape (1, 10)
    return x

# -----------------------------------------------------------------------------
def loss_fn(model, X, y):
  return mx.mean((y - model(X))**2)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train a 1989 LeCun ConvNet on digits")
    parser.add_argument('--learning-rate', '-l', type=float, default=0.03, help="SGD learning rate")
    parser.add_argument('--output-dir'   , '-o', type=str,   default='out/base', help="output directory for training logs")
    args = parser.parse_args()
    print(vars(args))

    # init rng
    mx.random.seed(1337)
    np.random.seed(1337)

    # set up logging
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    writer = SummaryWriter(args.output_dir)

    # init a model
    model = Net()
    print("model stats:")
    # print("# params:      ", sum(p.numel() for p in model.parameters())) # in paper total is 9,760
    print("# MACs:        ", model.macs)
    print("# activations: ", model.acts)

    # init data
    Xtr = mx.load('train1989x.npy')
    Ytr = mx.load('train1989y.npy')
    Xte = mx.load('test1989x.npy')
    Yte = mx.load('test1989y.npy')

    # init optimizer
    optimizer = optim.SGD(learning_rate=args.learning_rate)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    def eval_split(split):
        # eval the full train/test set, batched implementation for efficiency
        X, Y = (Xtr, Ytr) if split == 'train' else (Xte, Yte)
        Yhat = model(X)
        loss = mx.mean((Y - Yhat)**2)
        err = mx.mean((mx.argmax(Y, axis=1) != mx.argmax(Yhat, axis=1)).astype(mx.float32))
        print(f"eval: split {split:5s}. loss {loss.item():e}. error {err.item()*100:.2f}%. misses: {int(err.item()*Y.shape[0])}")
        writer.add_scalar(f'error/{split}', err.item()*100, pass_num)
        writer.add_scalar(f'loss/{split}', loss.item(), pass_num)

    # train
    for pass_num in range(23):

        # perform one epoch of training
        for step_num in range(Xtr.shape[0]):
            # fetch a single example into a batch of 1
            x, y = Xtr[step_num:step_num+1], Ytr[step_num:step_num+1]

            # forward the model and the loss
            loss, grads = loss_and_grad_fn(model, x, y)
            # calculate the gradient and update the parameters
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

        # after epoch epoch evaluate the train and test error / metrics
        print(pass_num + 1)
        eval_split('train')
        eval_split('test')

    # save final model to file
    model.save_weights(os.path.join(args.output_dir, 'model.npz'))
