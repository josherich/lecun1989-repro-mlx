import numpy as np

from typing import List, Callable
from tinygrad import Tensor, TinyJit, nn, GlobalCounters
from tinygrad.nn.state import get_parameters
from extra.datasets import fetch_mnist

DEVICE = 'METAL'

class Net:
  def __init__(self):
    winit = lambda fan_in, *shape: (Tensor.rand(*shape) - 0.5) * 2 * 2.4 / fan_in**0.5
    macs = 0
    acts = 0

    self.H1w = winit(5*5*1, 12, 1, 5, 5)
    self.H1b = Tensor.zeros(12, 8, 8)
    macs += 12*(8*8)*(5*5*1)
    acts += 12*(8*8)

    self.H2w = winit(5*5*8, 12, 8, 5, 5)
    self.H2b = Tensor.zeros(12, 4, 4)
    macs += 12*(4*4)*(5*5*8)
    acts += 12*(4*4)

    self.H3w = winit(4*4*12, 4*4*12, 30)
    self.H3b = Tensor.zeros(30)
    macs += 30*(4*4*12)
    acts += 30

    self.outw = winit(30, 30, 10)
    self.outb = -Tensor.ones(10)
    macs += 10*30
    acts += 10

    self.macs = macs
    self.acts = acts

  def __call__(self, x:Tensor) -> Tensor:
    x = x.pad(((0,0), (0,0), (2,2), (2,2)), -1.0)
    x = x.conv2d(self.H1w, stride=2) + self.H1b
    x = x.tanh()

    # x is now shape (1, 12, 8, 8)
    x = x.pad(((0,0), (0,0), (2,2), (2,2)), -1.0)
    slice1 = x[:, 0:8].conv2d(self.H2w[0:4], stride=2)
    slice2 = x[:, 4:12].conv2d(self.H2w[4:8], stride=2)
    slice3 = x[:, 0:4].cat(x[:, 8:12], dim=1).conv2d(self.H2w[8:12], stride=2)
    x = slice1.cat(slice2, slice3, dim=1) + self.H2b
    x = x.tanh()

    x = x.flatten(start_dim=1)
    x = x.matmul(self.H3w) + self.H3b
    x = x.tanh()

    x = x.matmul(self.outw) + self.outb
    x = x.tanh()

    return x

if __name__ == "__main__":
  Tensor.manual_seed(1337)
  np.random.seed(1337)

  model = Net()
  print("model stats:")
  print(f"# MACs: {model.macs}")
  print(f"# activations: {model.acts}")

  Xtr = np.load('train1989x.npy')
  Ytr = np.load('train1989y.npy')
  Xte = np.load('test1989x.npy')
  Yte = np.load('test1989y.npy')

  # skip args
  learning_rate = 0.03
  output_dir = 'out/base'

  parameters = get_parameters(model)
  optimizer = nn.optim.SGD(parameters, lr=learning_rate)

  def eval_split(split):
    # eval the full train/test set, batched implementation for efficiency
    X, Y = (Tensor(Xtr, device=DEVICE), Tensor(Ytr, device=DEVICE)) if split == 'train' else (Tensor(Xte, device=DEVICE), Tensor(Yte, device=DEVICE))
    Yhat = model(X)
    loss = ((Y - Yhat)**2).mean()
    err = ((Y.argmax(axis=1) != Yhat.argmax(axis=1)).float()).mean()
    print(f"eval: split {split:5s}. loss {loss.item():e}. error {err.item()*100:.2f}%. misses: {int(err.item()*Y.shape[0])}")

  @TinyJit
  def train_step(x: Tensor, y: Tensor) -> Tensor:
    with Tensor.train():
      yhat = model(x)
      loss = ((y - yhat)**2).mean()

      # calculate the gradient and update the parameters
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

  # train
  for pass_num in range(23):

    # perform one epoch of training
    for step_num in range(Xtr.shape[0]):
      # fetch a single example into a batch of 1
      x, y = Tensor(Xtr[step_num:step_num+1], device=DEVICE), Tensor(Ytr[step_num:step_num+1], device=DEVICE)

      train_step(x, y)

    # after epoch epoch evaluate the train and test error / metrics
    print(pass_num + 1)
    eval_split('train')
    eval_split('test')

  # save final model to file
  with open('./model.npy', 'wb') as f:
    for par in get_parameters(model):
      np.save(f, par.numpy())

