---
template: lesson.html
title: NumPy for Machine Learning
description: Numerical analysis with the NumPy computing package.
keywords: numpy, python, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/ml_foundations.png
repository: https://github.com/GokuMohandas/madewithml
notebook: https://colab.research.google.com/github/GokuMohandas/madewithml/blob/main/notebooks/03_NumPy.ipynb
---

{% include "styles/lesson.md" %}

## Set up
First we'll import the NumPy package and set seeds for reproducability so that we can receive the exact same results every time.

```python linenums="1"
import numpy as np
```
```python linenums="1"
# Set seed for reproducibility
np.random.seed(seed=1234)
```
<pre class="output"></pre>

## Basics
<div class="ai-center-all">
    <img src="https://raw.githubusercontent.com/GokuMohandas/madewithml/main/images/basics/numpy/tensors.png" width="600">
</div>

```python linenums="1"
# Scalar
x = np.array(6)
print ("x: ", x)
print ("x ndim: ", x.ndim) # number of dimensions
print ("x shape:", x.shape) # dimensions
print ("x size: ", x.size) # size of elements
print ("x dtype: ", x.dtype) # data type
```
<pre class="output">
x:  6
x ndim:  0
x shape: ()
x size:  1
x dtype:  int64
</pre>
```python linenums="1"
# Vector
x = np.array([1.3 , 2.2 , 1.7])
print ("x: ", x)
print ("x ndim: ", x.ndim)
print ("x shape:", x.shape)
print ("x size: ", x.size)
print ("x dtype: ", x.dtype) # notice the float datatype
```
<pre class="output">
x:  [1.3 2.2 1.7]
x ndim:  1
x shape: (3,)
x size:  3
x dtype:  float64
</pre>
```python linenums="1"
# Matrix
x = np.array([[1,2], [3,4]])
print ("x:\n", x)
print ("x ndim: ", x.ndim)
print ("x shape:", x.shape)
print ("x size: ", x.size)
print ("x dtype: ", x.dtype)
```
<pre class="output">
x:
 [[1 2]
 [3 4]]
x ndim:  2
x shape: (2, 2)
x size:  4
x dtype:  int64
</pre>
```python linenums="1"
# 3-D Tensor
x = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
print ("x:\n", x)
print ("x ndim: ", x.ndim)
print ("x shape:", x.shape)
print ("x size: ", x.size)
print ("x dtype: ", x.dtype)
```
<pre class="output">
x:
 [[[1 2]
  [3 4]]

 [[5 6]
  [7 8]]]
x ndim:  3
x shape: (2, 2, 2)
x size:  8
x dtype:  int64
</pre>

NumPy also comes with several functions that allow us to create tensors quickly.
```python linenums="1"
# Functions
print ("np.zeros((2,2)):\n", np.zeros((2,2)))
print ("np.ones((2,2)):\n", np.ones((2,2)))
print ("np.eye((2)):\n", np.eye((2))) # identity matrix
print ("np.random.random((2,2)):\n", np.random.random((2,2)))
```
<pre class="output">
np.zeros((2,2)):
 [[0. 0.]
 [0. 0.]]
np.ones((2,2)):
 [[1. 1.]
 [1. 1.]]
np.eye((2)):
 [[1. 0.]
 [0. 1.]]
np.random.random((2,2)):
 [[0.19151945 0.62210877]
 [0.43772774 0.78535858]]
</pre>

## Indexing
We can extract specific values from our tensors using indexing.
!!! note
    Keep in mind that when indexing the row and column, indices start at `0`. And like indexing with lists, we can use negative indices as well (where `-1` is the last item).
<div class="ai-center-all">
    <img src="https://raw.githubusercontent.com/GokuMohandas/madewithml/main/images/basics/numpy/indexing.png" width="300">
</div>

```python linenums="1"
# Indexing
x = np.array([1, 2, 3])
print ("x: ", x)
print ("x[0]: ", x[0])
x[0] = 0
print ("x: ", x)
```
<pre class="output">
x:  [1 2 3]
x[0]:  1
x:  [0 2 3]
</pre>
```python linenums="1"
# Slicing
x = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print (x)
print ("x column 1: ", x[:, 1])
print ("x row 0: ", x[0, :])
print ("x rows 0,1 & cols 1,2: \n", x[0:2, 1:3])
```
<pre class="output">
[[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]]
x column 1:  [ 2  6 10]
x row 0:  [1 2 3 4]
x rows 0,1 & cols 1,2:
 [[2 3]
 [6 7]]
</pre>
```python linenums="1"
# Integer array indexing
print (x)
rows_to_get = np.array([0, 1, 2])
print ("rows_to_get: ", rows_to_get)
cols_to_get = np.array([0, 2, 1])
print ("cols_to_get: ", cols_to_get)
# Combine sequences above to get values to get
print ("indexed values: ", x[rows_to_get, cols_to_get]) # (0, 0), (1, 2), (2, 1)
```
<pre class="output">
[[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]]
rows_to_get:  [0 1 2]
cols_to_get:  [0 2 1]
indexed values:  [ 1  7 10]
</pre>
```python linenums="1"
# Boolean array indexing
x = np.array([[1, 2], [3, 4], [5, 6]])
print ("x:\n", x)
print ("x > 2:\n", x > 2)
print ("x[x > 2]:\n", x[x > 2])
```
<pre class="output">
x:
 [[1 2]
 [3 4]
 [5 6]]
x > 2:
 [[False False]
 [ True  True]
 [ True  True]]
x[x > 2]:
 [3 4 5 6]
</pre>

## Arithmetic
```python linenums="1"
# Basic math
x = np.array([[1,2], [3,4]], dtype=np.float64)
y = np.array([[1,2], [3,4]], dtype=np.float64)
print ("x + y:\n", np.add(x, y)) # or x + y
print ("x - y:\n", np.subtract(x, y)) # or x - y
print ("x * y:\n", np.multiply(x, y)) # or x * y
```
<pre class="output">
x + y:
 [[2. 4.]
 [6. 8.]]
x - y:
 [[0. 0.]
 [0. 0.]]
x * y:
 [[ 1.  4.]
 [ 9. 16.]]
</pre>

## Dot product
One of the most common NumPy operations we’ll use in machine learning is matrix multiplication using the dot product. We take the rows of our first matrix (2) and the columns of our second matrix (2) to determine the dot product, giving us an output of `[2 X 2]`. The only requirement is that the inside dimensions match, in this case the first matrix has 3 columns and the second matrix has 3 rows.
<div class="ai-center-all">
    <img src="https://raw.githubusercontent.com/GokuMohandas/madewithml/main/images/basics/numpy/dot.gif" width="450">
</div>

```python linenums="1"
# Dot product
a = np.array([[1,2,3], [4,5,6]], dtype=np.float64) # we can specify dtype
b = np.array([[7,8], [9,10], [11, 12]], dtype=np.float64)
c = a.dot(b)
print (f"{a.shape} · {b.shape} = {c.shape}")
print (c)
```
<pre class="output">
(2, 3) · (3, 2) = (2, 2)
[[ 58.  64.]
 [139. 154.]]
</pre>

## Axis operations
We can also do operations across a specific axis.
<div class="ai-center-all">
    <img src="https://raw.githubusercontent.com/GokuMohandas/madewithml/main/images/basics/numpy/axis.gif" width="450">
</div>

```python linenums="1"
# Sum across a dimension
x = np.array([[1,2],[3,4]])
print (x)
print ("sum all: ", np.sum(x)) # adds all elements
print ("sum axis=0: ", np.sum(x, axis=0)) # sum across rows
print ("sum axis=1: ", np.sum(x, axis=1)) # sum across columns
```
<pre class="output">
[[1 2]
 [3 4]]
sum all:  10
sum axis=0:  [4 6]
sum axis=1:  [3 7]
</pre>
```python linenums="1"
# Min/max
x = np.array([[1,2,3], [4,5,6]])
print ("min: ", x.min())
print ("max: ", x.max())
print ("min axis=0: ", x.min(axis=0))
print ("min axis=1: ", x.min(axis=1))
```
<pre class="output">
min:  1
max:  6
min axis=0:  [1 2 3]
min axis=1:  [1 4]
</pre>

## Broadcast
Here, we’re adding a vector with a scalar. Their dimensions aren’t compatible as is but how does NumPy still gives us the right result? This is where broadcasting comes in. The scalar is *broadcast* across the vector so that they have compatible shapes.
<div class="ai-center-all">
    <img src="https://raw.githubusercontent.com/GokuMohandas/madewithml/main/images/basics/numpy/broadcast.png" width="300">
</div>

```python linenums="1"
# Broadcasting
x = np.array([1,2]) # vector
y = np.array(3) # scalar
z = x + y
print ("z:\n", z)
```
<pre class="output">
z:
 [4 5]
</pre>


## Transpose
We often need to change the dimensions of our tensors for operations like the dot product. If we need to switch two dimensions, we can transpose
the tensor.
<div class="ai-center-all">
    <img src="https://raw.githubusercontent.com/GokuMohandas/madewithml/main/images/basics/numpy/transpose.png" width="400">
</div>

```python linenums="1"
# Transposing
x = np.array([[1,2,3], [4,5,6]])
print ("x:\n", x)
print ("x.shape: ", x.shape)
y = np.transpose(x, (1,0)) # flip dimensions at index 0 and 1
print ("y:\n", y)
print ("y.shape: ", y.shape)
```
<pre class="output">
x:
 [[1 2 3]
 [4 5 6]]
x.shape:  (2, 3)
y:
 [[1 4]
 [2 5]
 [3 6]]
y.shape:  (3, 2)
</pre>


## Reshape
Sometimes, we'll need to alter the dimensions of the matrix. Reshaping allows us to transform a tensor into different permissible shapes -- our reshaped tensor has the same amount of values in the tensor. (`1X6` = `2X3`). We can also use `-1` on a dimension and NumPy will infer the dimension based on our input tensor.

The way reshape works is by looking at each dimension of the new tensor and separating our original tensor into that many units. So here the dimension at index 0 of the new tensor is 2 so we divide our original tensor into 2 units, and each of those has 3 values.
<div class="ai-center-all">
    <img src="https://raw.githubusercontent.com/GokuMohandas/madewithml/main/images/basics/numpy/reshape.png" width="450">
</div>

```python linenums="1"
# Reshaping
x = np.array([[1,2,3,4,5,6]])
print (x)
print ("x.shape: ", x.shape)
y = np.reshape(x, (2, 3))
print ("y: \n", y)
print ("y.shape: ", y.shape)
z = np.reshape(x, (2, -1))
print ("z: \n", z)
print ("z.shape: ", z.shape)
```
<pre class="output">
[[1 2 3 4 5 6]]
x.shape:  (1, 6)
y:
 [[1 2 3]
 [4 5 6]]
y.shape:  (2, 3)
z:
 [[1 2 3]
 [4 5 6]]
z.shape:  (2, 3)
</pre>

### Unintended reshaping
Though reshaping is very convenient to manipulate tensors, we must be careful of their pitfalls as well. Let's look at the example below. Suppose we have `x`, which has the shape `[2 X 3 X 4]`.
<pre class="output">
[[[ 1  1  1  1]
  [ 2  2  2  2]
  [ 3  3  3  3]]
 [[10 10 10 10]
  [20 20 20 20]
  [30 30 30 30]]]
</pre>
We want to reshape x so that it has shape `[3 X 8]` which we'll get by moving the dimension at index 0 to become the dimension at index 1 and then combining the last two dimensions. But when we do this, we want our output

to look like:
<pre class="output">
[[ 1  1  1  1 10 10 10 10]
 [ 2  2  2  2 20 20 20 20]
 [ 3  3  3  3 30 30 30 30]]
</pre>
and not like:
<pre class="output">
[[ 1  1  1  1  2  2  2  2]
 [ 3  3  3  3 10 10 10 10]
 [20 20 20 20 30 30 30 30]]
</pre>
even though they both have the same shape `[3X8]`.

```python linenums="1"
x = np.array([[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]],
              [[10, 10, 10, 10], [20, 20, 20, 20], [30, 30, 30, 30]]])
print ("x:\n", x)
print ("x.shape: ", x.shape)
```
<pre class="output">
x:
 [[[ 1  1  1  1]
   [ 2  2  2  2]
   [ 3  3  3  3]]

 [[10 10 10 10]
  [20 20 20 20]
  [30 30 30 30]]]
x.shape:  (2, 3, 4)
</pre>

When we naively do a reshape, we get the right shape but the values are not what we're looking for.
<div class="ai-center-all">
    <img src="https://raw.githubusercontent.com/GokuMohandas/madewithml/main/images/basics/numpy/reshape_wrong.png" width="600">
</div>

```python linenums="1"
# Unintended reshaping
z_incorrect = np.reshape(x, (x.shape[1], -1))
print ("z_incorrect:\n", z_incorrect)
print ("z_incorrect.shape: ", z_incorrect.shape)
```
<pre class="output">
z_incorrect:
 [[ 1  1  1  1  2  2  2  2]
  [ 3  3  3  3 10 10 10 10]
  [20 20 20 20 30 30 30 30]]
z_incorrect.shape:  (3, 8)
</pre>

Instead, if we transpose the tensor and then do a reshape, we get our desired tensor. Transpose allows us to put our two vectors that we want to combine together and then we use reshape to join them together.

!!! note
    Always create a dummy example like this when you’re unsure about reshaping. Blindly going by the tensor shape can lead to lots of issues downstream.

<div class="ai-center-all">
    <img src="https://raw.githubusercontent.com/GokuMohandas/madewithml/main/images/basics/numpy/reshape_right.png" width="600">
</div>

```python linenums="1"
# Intended reshaping
y = np.transpose(x, (1,0,2))
print ("y:\n", y)
print ("y.shape: ", y.shape)
z_correct = np.reshape(y, (y.shape[0], -1))
print ("z_correct:\n", z_correct)
print ("z_correct.shape: ", z_correct.shape)
```
<pre class="output">
y:
 [[[ 1  1  1  1]
  [10 10 10 10]]

 [[ 2  2  2  2]
  [20 20 20 20]]

 [[ 3  3  3  3]
  [30 30 30 30]]]
y.shape:  (3, 2, 4)
z_correct:
 [[ 1  1  1  1 10 10 10 10]
  [ 2  2  2  2 20 20 20 20]
  [ 3  3  3  3 30 30 30 30]]
z_correct.shape:  (3, 8)
</pre>

## Expanding / reducing
We can also easily add and remove dimensions to our tensors and we'll want to do this to make tensors compatible for certain operations.

```python linenums="1"
# Adding dimensions
x = np.array([[1,2,3],[4,5,6]])
print ("x:\n", x)
print ("x.shape: ", x.shape)
y = np.expand_dims(x, 1) # expand dim 1
print ("y: \n", y)
print ("y.shape: ", y.shape)   # notice extra set of brackets are added
```
<pre class="output">
x:
 [[1 2 3]
  [4 5 6]]
x.shape:  (2, 3)
y:
 [[[1 2 3]]
  [[4 5 6]]]
y.shape:  (2, 1, 3)
</pre>

```python linenums="1"
# Removing dimensions
x = np.array([[[1,2,3]],[[4,5,6]]])
print ("x:\n", x)
print ("x.shape: ", x.shape)
y = np.squeeze(x, 1) # squeeze dim 1
print ("y: \n", y)
print ("y.shape: ", y.shape)  # notice extra set of brackets are gone
```
<pre class="output">
x:
 [[[1 2 3]]
  [[4 5 6]]]
x.shape:  (2, 1, 3)
y:
 [[1 2 3]
  [4 5 6]]
y.shape:  (2, 3)
</pre>


<!-- Citation -->
{% include "cite.md" %}