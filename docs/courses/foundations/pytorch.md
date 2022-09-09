---
template: lesson.html
title: PyTorch Fundamentals
description: Learn how to use the PyTorch machine learning framework.
keywords: pytorch, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/foundations.png
repository: https://github.com/GokuMohandas/Made-With-ML
notebook: https://github.com/GokuMohandas/Made-With-ML/blob/main/notebooks/05_PyTorch.ipynb
---

{% include "styles/lesson.md" %}

## Set up

We'll import PyTorch and set seeds for reproducibility. Note that PyTorch also required a seed since we will be generating random tensors.
```python linenums="1"
import numpy as np
import torch
```
```python linenums="1"
SEED = 1234
```
```python linenums="1"
# Set seed for reproducibility
np.random.seed(seed=SEED)
torch.manual_seed(SEED)
```
<pre class="output">
<torch._C.Generator at 0x7f2ddf6dfb10>
</pre>


## Basics
We'll first cover some basics with PyTorch such as creating tensors and converting from common data structures (lists, arrays, etc.) to tensors.
```python linenums="1"
# Creating a random tensor
x = torch.randn(2, 3) # normal distribution (rand(2,3) -> uniform distribution)
print(f"Type: {x.type()}")
print(f"Size: {x.shape}")
print(f"Values: \n{x}")
```
<pre class="output">
Type: torch.FloatTensor
Size: torch.Size([2, 3])
Values:
tensor([[ 0.0461,  0.4024, -1.0115],
        [ 0.2167, -0.6123,  0.5036]])
</pre>
```python linenums="1"
# Zero and Ones tensor
x = torch.zeros(2, 3)
print (x)
x = torch.ones(2, 3)
print (x)
```
<pre class="output">
tensor([[0., 0., 0.],
        [0., 0., 0.]])
tensor([[1., 1., 1.],
        [1., 1., 1.]])
</pre>
```python linenums="1"
# List → Tensor
x = torch.Tensor([[1, 2, 3],[4, 5, 6]])
print(f"Size: {x.shape}")
print(f"Values: \n{x}")
```
<pre class="output">
Size: torch.Size([2, 3])
Values:
tensor([[1., 2., 3.],
        [4., 5., 6.]])
</pre>
```python linenums="1"
# NumPy array → Tensor
x = torch.Tensor(np.random.rand(2, 3))
print(f"Size: {x.shape}")
print(f"Values: \n{x}")
```
<pre class="output">
Size: torch.Size([2, 3])
Values:
tensor([[0.1915, 0.6221, 0.4377],
        [0.7854, 0.7800, 0.2726]])
</pre>
```python linenums="1"
# Changing tensor type
x = torch.Tensor(3, 4)
print(f"Type: {x.type()}")
x = x.long()
print(f"Type: {x.type()}")
```
<pre class="output">
Type: torch.FloatTensor
Type: torch.LongTensor
</pre>

## Operations
Now we'll explore some basic operations with tensors.
```python linenums="1"
# Addition
x = torch.randn(2, 3)
y = torch.randn(2, 3)
z = x + y
print(f"Size: {z.shape}")
print(f"Values: \n{z}")
```
<pre class="output">
Size: torch.Size([2, 3])
Values:
tensor([[ 0.0761, -0.6775, -0.3988],
        [ 3.0633, -0.1589,  0.3514]])
</pre>
```python linenums="1"
# Dot product
x = torch.randn(2, 3)
y = torch.randn(3, 2)
z = torch.mm(x, y)
print(f"Size: {z.shape}")
print(f"Values: \n{z}")
```
<pre class="output">
Size: torch.Size([2, 2])
Values:
tensor([[ 1.0796, -0.0759],
        [ 1.2746, -0.5134]])
</pre>
```python linenums="1"
# Transpose
x = torch.randn(2, 3)
print(f"Size: {x.shape}")
print(f"Values: \n{x}")
y = torch.t(x)
print(f"Size: {y.shape}")
print(f"Values: \n{y}")
```
<pre class="output">
Size: torch.Size([2, 3])
Values:
tensor([[ 0.8042, -0.1383,  0.3196],
        [-1.0187, -1.3147,  2.5228]])
Size: torch.Size([3, 2])
Values:
tensor([[ 0.8042, -1.0187],
        [-0.1383, -1.3147],
        [ 0.3196,  2.5228]])
</pre>
```python linenums="1"
# Reshape
x = torch.randn(2, 3)
z = x.view(3, 2)
print(f"Size: {z.shape}")
print(f"Values: \n{z}")
```
<pre class="output">
Size: torch.Size([3, 2])
Values:
tensor([[ 0.4501,  0.2709],
        [-0.8087, -0.0217],
        [-1.0413,  0.0702]])
</pre>
```python linenums="1"
# Dangers of reshaping (unintended consequences)
x = torch.tensor([
    [[1,1,1,1], [2,2,2,2], [3,3,3,3]],
    [[10,10,10,10], [20,20,20,20], [30,30,30,30]]
])
print(f"Size: {x.shape}")
print(f"x: \n{x}\n")

a = x.view(x.size(1), -1)
print(f"\nSize: {a.shape}")
print(f"a: \n{a}\n")

b = x.transpose(0,1).contiguous()
print(f"\nSize: {b.shape}")
print(f"b: \n{b}\n")

c = b.view(b.size(0), -1)
print(f"\nSize: {c.shape}")
print(f"c: \n{c}")
```
<pre class="output">
Size: torch.Size([2, 3, 4])
x:
tensor([[[ 1,  1,  1,  1],
         [ 2,  2,  2,  2],
         [ 3,  3,  3,  3]],

        [[10, 10, 10, 10],
         [20, 20, 20, 20],
         [30, 30, 30, 30]]])


Size: torch.Size([3, 8])
a:
tensor([[ 1,  1,  1,  1,  2,  2,  2,  2],
        [ 3,  3,  3,  3, 10, 10, 10, 10],
        [20, 20, 20, 20, 30, 30, 30, 30]])


Size: torch.Size([3, 2, 4])
b:
tensor([[[ 1,  1,  1,  1],
         [10, 10, 10, 10]],

        [[ 2,  2,  2,  2],
         [20, 20, 20, 20]],

        [[ 3,  3,  3,  3],
         [30, 30, 30, 30]]])


Size: torch.Size([3, 8])
c:
tensor([[ 1,  1,  1,  1, 10, 10, 10, 10],
        [ 2,  2,  2,  2, 20, 20, 20, 20],
        [ 3,  3,  3,  3, 30, 30, 30, 30]])
</pre>
```python linenums="1"
# Dimensional operations
x = torch.randn(2, 3)
print(f"Values: \n{x}")
y = torch.sum(x, dim=0) # add each row's value for every column
print(f"Values: \n{y}")
z = torch.sum(x, dim=1) # add each columns's value for every row
print(f"Values: \n{z}")
```
<pre class="output">
Values:
tensor([[ 0.5797, -0.0599,  0.1816],
        [-0.6797, -0.2567, -1.8189]])
Values:
tensor([-0.1000, -0.3166, -1.6373])
Values:
tensor([ 0.7013, -2.7553])
</pre>


## Indexing
Now we'll look at how to extract, separate and join values from our tensors.
```python linenums="1"
x = torch.randn(3, 4)
print (f"x: \n{x}")
print (f"x[:1]: \n{x[:1]}")
print (f"x[:1, 1:3]: \n{x[:1, 1:3]}")
```
<pre class="output">
x:
tensor([[ 0.2111,  0.3372,  0.6638,  1.0397],
        [ 1.8434,  0.6588, -0.2349, -0.0306],
        [ 1.7462, -0.0722, -1.6794, -1.7010]])
x[:1]:
tensor([[0.2111, 0.3372, 0.6638, 1.0397]])
x[:1, 1:3]:
tensor([[0.3372, 0.6638]])
</pre>

## Slicing
```python linenums="1"
# Select with dimensional indices
x = torch.randn(2, 3)
print(f"Values: \n{x}")

col_indices = torch.LongTensor([0, 2])
chosen = torch.index_select(x, dim=1, index=col_indices) # values from column 0 & 2
print(f"Values: \n{chosen}")

row_indices = torch.LongTensor([0, 1])
col_indices = torch.LongTensor([0, 2])
chosen = x[row_indices, col_indices] # values from (0, 0) & (1, 2)
print(f"Values: \n{chosen}")
```
<pre class="output">
Values:
tensor([[ 0.6486,  1.7653,  1.0812],
        [ 1.2436,  0.8971, -0.0784]])
Values:
tensor([[ 0.6486,  1.0812],
        [ 1.2436, -0.0784]])
Values:
tensor([ 0.6486, -0.0784])
</pre>

## Joining

We can also combine our tensors via concatenation or stacking operations, which are consistent with [NumPy's joining functions'](numpy.md#joining){:target="_blank"} behaviors as well.

```python linenums="1"
x = torch.randn(2, 3)
print (x)
print (x.shape)
```
<pre class="output">
tensor([[-1.5944, -0.4218, -1.8219],
        [ 1.7446,  1.2058, -0.7753]])
torch.Size([2, 3])
</pre>

```python linenums="1"
# Concatenation
y = torch.cat([x, x], dim=0) # concat on a specified dimension
print (y)
print (y.shape)
```
<pre class="output">
tensor([[-1.5944, -0.4218, -1.8219],
        [ 1.7446,  1.2058, -0.7753],
        [-1.5944, -0.4218, -1.8219],
        [ 1.7446,  1.2058, -0.7753]])
torch.Size([4, 3])
</pre>

```python linenums="1"
# Stacking
z = torch.stack([x, x], dim=0) # stack on new dimension
print (z)
print (z.shape)
```
<pre class="output">
tensor([[[-1.5944, -0.4218, -1.8219],
         [ 1.7446,  1.2058, -0.7753]],

        [[-1.5944, -0.4218, -1.8219],
         [ 1.7446,  1.2058, -0.7753]]])
torch.Size([2, 2, 3])
</pre>

## Gradients
We can determine gradients (rate of change) of our tensors with respect to their constituents using gradient bookkeeping. The gradient is a vector that points in the direction of greatest increase of a function. We'll be using gradients in the [next lesson](linear-regression.md#gradients){:target="_blank"} to determine how to change our weights to affect a particular objective function (ex. loss).

$$ y = 3x + 2 $$

$$ z = \sum{y}/N $$

$$ \frac{\partial(z)}{\partial(x)} = \frac{\partial(z)}{\partial(y)} \frac{\partial(y)}{\partial(x)} = \frac{1}{N} * 3 = \frac{1}{12} * 3 = 0.25 $$

<pre class="output"></pre>
```python linenums="1"
# Tensors with gradient bookkeeping
x = torch.rand(3, 4, requires_grad=True)
y = 3*x + 2
z = y.mean()
z.backward() # z has to be scalar
print(f"x: \n{x}")
print(f"x.grad: \n{x.grad}")
```
<pre class="output">
x:
tensor([[0.7379, 0.0846, 0.4245, 0.9778],
        [0.6800, 0.3151, 0.3911, 0.8943],
        [0.6889, 0.8389, 0.1780, 0.6442]], requires_grad=True)
x.grad:
tensor([[0.2500, 0.2500, 0.2500, 0.2500],
        [0.2500, 0.2500, 0.2500, 0.2500],
        [0.2500, 0.2500, 0.2500, 0.2500]])
</pre>


## CUDA
We also load our tensors onto the GPU for parallelized computation using CUDA (a parallel computing platform and API from Nvidia).
```python linenums="1"
# Is CUDA available?
print (torch.cuda.is_available())
```
<pre class="output">
False
</pre>

If False (CUDA is not available), let's change that by following these steps: Go to *Runtime* > *Change runtime type* > Change *Hardware accelerator* to *GPU* > Click *Save*
```python linenums="1"
import torch
```
```python linenums="1"
# Is CUDA available now?
print (torch.cuda.is_available())
```
<pre class="output">
True
</pre>
```python linenums="1"
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print (device)
```
<pre class="output">
cuda
</pre>
```python linenums="1"
x = torch.rand(2,3)
print (x.is_cuda)
x = torch.rand(2,3).to(device) # Tensor is stored on the GPU
print (x.is_cuda)
```
<pre class="output">
False
True
</pre>


<!-- Citation -->
{% include "styles/cite.md" %}