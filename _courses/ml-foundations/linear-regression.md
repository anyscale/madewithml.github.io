---
layout: page
title: Linear Regression · ML Foundations
description: Implement linear regression from scratch using NumPy and then using PyTorch.
image: /static/images/ml_foundations.png

course-url: /courses/ml-foundations/
next-lesson-url: /courses/ml-foundations/logistic-regression/
---

<!-- Header -->
<div class="row">
  <div class="col-md-8 col-6 mr-auto">
    <h1 class="page-title">{{ page.title | split: " · " | first }}</h1>
  </div>
  <div class="col-md-4 col-6">
    <div class="btn-group float-right mb-0" role="group">
      <a href="{{ page.course-url }}" class="btn btn-sm btn-outline-secondary"><i
          class="fas fa-sm fa-arrow-left mr-1"></i>Return to course</a>
    </div>
  </div>
</div>
<hr class="mt-0">

In this lesson, we'll learn about linear regression. We will understand the basic math behind it, implement it from scratch using NumPy and then in PyTorch.

- [Overview](#overview)
- [Generate data](#generate)
- [NumPy](#numpy)
- [PyTorch](#pytorch)
- [Regularization](#regularization)

> 📓 Follow along this lesson with the accompanying [notebook](https://colab.research.google.com/github/GokuMohandas/madewithml/blob/main/notebooks/06_Linear_Regression.ipynb){:target="_blank"}.

<h3 id="overview">Overview</h3>

Our goal is to learn a linear model $$ \hat{y} $$ that models $$ y $$ given $$ X $$ using weights $$ W $$ and bias $$ b $$:

$$ \hat{y} = XW + b $$

<div class="ai-center-all">
<table class="mathjax-table">
  <tbody>
    <tr>
      <td>$$ N $$</td>
      <td>$$ \text{total numbers of samples} $$</td>
    </tr>
    <tr>
      <th>$$ \hat{y} $$</th>
      <th>$$ \text{predictions} \in \mathbb{R}^{NX1} $$</th>
    </tr>
    <tr>
      <td>$$ X $$</td>
      <td>$$ \text{inputs} \in \mathbb{R}^{NXD} $$</td>
    </tr>
    <tr>
      <td>$$ W $$</td>
      <td>$$ \text{weights} \in \mathbb{R}^{DX1} $$</td>
    </tr>
    <tr>
      <td>$$ b $$</td>
      <td>$$ \text{bias} \in \mathbb{R}^{1} $$</td>
    </tr>
  </tbody>
</table>
</div>

- `Objective`:
    - Use inputs $$X$$ to predict the output $$\hat{y}$$ using a linear model. The model will be a line of best fit that minimizes the distance between the predicted (model's output) and target (ground truth) values. Training data $$(X, y)$$ is used to train the model and learn the weights $$W$$ using gradient descent.
- `Advantages`:
  - Computationally simple.
  - Highly interpretable.
  - Can account for continuous and categorical features.
- `Disadvantages`:
  - The model will perform well only when the data is linearly separable (for classification).
  - Usually not used for classification and only for regression.
- `Miscellaneous`:
    - You can also use linear regression for binary classification tasks where if the predicted continuous value is above a threshold, it belongs to a certain class. But we will cover better techniques for classification in future lessons and will focus on linear regression for continuous regression tasks only.


<h3 id="generate">Generate data</h3>

We're going to generate some simple dummy data to apply linear regression on. It's going to create roughly linear data (`y = 3.5X + noise`); the random noise is added to create realistic data that doesn't perfectly align in a line. Our goal is to have the model converge to a similar linear equation (there will be slight variance since we added some noise).
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```
```python
SEED = 1234
NUM_SAMPLES = 50
```
```python
# Set seed for reproducibility
np.random.seed(SEED)
```
```python
# Generate synthetic data
def generate_data(num_samples):
    """Generate dummy data for linear regression."""
    X = np.array(range(num_samples))
    random_noise = np.random.uniform(-10,20,size=num_samples)
    y = 3.5*X + random_noise # add some noise
    return X, y
```
```python
# Generate random (linear) data
X, y = generate_data(num_samples=NUM_SAMPLES)
data = np.vstack([X, y]).T
print (data[:5])
```
<pre class="output">
[[ 0.         -4.25441649]
 [ 1.         12.16326313]
 [ 2.         10.13183217]
 [ 3.         24.06075751]
 [ 4.         27.39927424]]
</pre>
```python
# Load into a Pandas DataFrame
df = pd.DataFrame(data, columns=['X', 'y'])
X = df[['X']].values
y = df[['y']].values
df.head()
```
<div class="output_subarea output_html rendered_html" style="margin-left: 10rem; margin-right: 10rem;"><div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>-4.254416</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>12.163263</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>10.131832</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.0</td>
      <td>24.060758</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.0</td>
      <td>27.399274</td>
    </tr>
  </tbody>
</table>
</div></div>

```python
# Scatter plot
plt.title("Generated data")
plt.scatter(x=df['X'], y=df['y'])
plt.show()
```
<div class="ai-center-all">
    <img width="350" src="data:image/png;base64,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">
</div>


<h3 id="numpy">NumPy</h3>
- [Split data](#split_numpy)
- [Standardize data](#standardize_numpy)
- [Weights](#weights_numpy)
- [Model](#model_numpy)
- [Loss](#loss_numpy)
- [Gradients](#gradients_numpy)
- [Update weights](#update_numpy)
- [Training](#training_numpy)
- [Evaluation](#evaluation_numpy)
- [Interpretability](#interpretability_numpy)

Now that we have our data prepared, we'll first implement linear regression using just NumPy. This will let us really understand the underlying operations.

<h4 id="split_numpy">Split data</h4>

Since our task is a regression task, we will randomly split our dataset into three sets: train, validation and test data splits.
- `train`: used to train our model.
- `val` : used to validate our model's performance during training.
- `test`: used to do an evaluation of our fully trained model.
> Be sure to check out our entire lesson focused on *properly* [splitting](https://madewithml.com/courses/applied-ml/splitting/){:target="_blank"} data in our [applied-ml](https://madewithml.com/courses/applied-ml/){:target="_blank"} course.

```python
TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15
```
```python
# Shuffle data
indices = list(range(NUM_SAMPLES))
np.random.shuffle(indices)
X = X[indices]
y = y[indices]
```
<pre class="output"></pre>
> Be careful not to shuffle X and y separately because then the inputs won't correspond to the outputs!

```python
# Split indices
train_start = 0
train_end = int(0.7*NUM_SAMPLES)
val_start = train_end
val_end = int((TRAIN_SIZE+VAL_SIZE)*NUM_SAMPLES)
test_start = val_end
```
```python
# Split data
X_train = X[train_start:train_end]
y_train = y[train_start:train_end]
X_val = X[val_start:val_end]
y_val = y[val_start:val_end]
X_test = X[test_start:]
y_test = y[test_start:]
print (f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print (f"X_val: {X_val.shape}, y_test: {y_val.shape}")
print (f"X_test: {X_test.shape}, y_test: {y_test.shape}")
```
<pre class="output">
X_train: (35, 1), y_train: (35, 1)
X_val: (7, 1), y_test: (7, 1)
X_test: (8, 1), y_test: (8, 1)
</pre>

<h4 id="standardize_numpy">Standardize data</h4>

We need to standardize our data (zero mean and unit variance) so a specific feature's magnitude doesn't affect how the model learns its weights.

$$ z = \frac{x_i - \mu}{\sigma} $$

<div class="ai-center-all">
<table class="mathjax-table">
  <tbody>
    <tr>
      <td>$$ z $$</td>
      <td>$$ \text{standardized value} $$</td>
    </tr>
    <tr>
      <th>$$ x_i $$</th>
      <th>$$ \text{inputs} $$</th>
    </tr>
    <tr>
      <td>$$ \mu $$</td>
      <td>$$ \text{mean} $$</td>
    </tr>
    <tr>
      <td>$$ \sigma $$</td>
      <td>$$ \text{standard deviation} $$</td>
    </tr>
  </tbody>
</table>
</div>

```python
def standardize_data(data, mean, std):
    return (data - mean)/std
```
```python
# Determine means and stds
X_mean = np.mean(X_train)
X_std = np.std(X_train)
y_mean = np.mean(y_train)
y_std = np.std(y_train)
```
<pre class="output"></pre>

We need to treat the validation and test sets as if they were hidden datasets. So we only use the train set to determine the mean and std to avoid biasing our training process.

```python
# Standardize
X_train = standardize_data(X_train, X_mean, X_std)
y_train = standardize_data(y_train, y_mean, y_std)
X_val = standardize_data(X_val, X_mean, X_std)
y_val = standardize_data(y_val, y_mean, y_std)
X_test = standardize_data(X_test, X_mean, X_std)
y_test = standardize_data(y_test, y_mean, y_std)
```
```python
# Check (means should be ~0 and std should be ~1)
# Check (means should be ~0 and std should be ~1)
print (f"mean: {np.mean(X_test, axis=0)[0]:.1f}, std: {np.std(X_test, axis=0)[0]:.1f}")
print (f"mean: {np.mean(y_test, axis=0)[0]:.1f}, std: {np.std(y_test, axis=0)[0]:.1f}")
```
<pre class="output">
mean: -0.4, std: 0.9
mean: -0.3, std: 1.0
</pre>

<h4 id="weights_numpy">Weights</h4>

Our goal is to learn a linear model $$ \hat{y} $$ that models $$ y $$ given $$ X $$ using weights $$ W $$ and bias $$ b $$ → $$ \hat{y} = XW + b $$

`Step 1`: Randomly initialize the model's weights $$ W $$.
```python
INPUT_DIM = X_train.shape[1] # X is 1-dimensional
OUTPUT_DIM = y_train.shape[1] # y is 1-dimensional
```
```python
# Initialize random weights
W = 0.01 * np.random.randn(INPUT_DIM, OUTPUT_DIM)
b = np.zeros((1, 1))
print (f"W: {W.shape}")
print (f"b: {b.shape}")
```
<pre class="output">
W: (1, 1)
b: (1, 1)
</pre>

<h4 id="model_numpy">Model</h4>

`Step 2`: Feed inputs $$X$$ into the model to receive the predictions $$ \hat{y} $$
```python
# Forward pass [NX1] · [1X1] = [NX1]
y_pred = np.dot(X_train, W) + b
print (f"y_pred: {y_pred.shape}")
```
<pre class="output">
y_pred: (35, 1)
</pre>

<h4 id="loss_numpy">Loss</h4>

`Step 3`: Compare the predictions $$\hat{y}$$ with the actual target values $$y$$ using the objective (cost) function to determine the loss $$J$$. A common objective function for linear regression is mean squarred error (MSE). This function calculates the difference between the predicted and target values and squares it.

$$ J(\theta) = MSE = \frac{1}{N} \sum_{i-1}^{N} (y_i - \hat{y}_i)^2 $$

```python
# Loss
N = len(y_train)
loss = (1/N) * np.sum((y_train - y_pred)**2)
print (f"loss: {loss:.2f}")
```
<pre class="output">
loss: 0.99
</pre>

<h4 id="gradients_numpy">Gradients</h4>

`Step 4`: Calculate the gradient of loss $$J(\theta)$$ w.r.t to the model weights.

$$ J(\theta) = \frac{1}{N} \sum_i (y_i - \hat{y}_i)^2  = \frac{1}{N}\sum_i (y_i - X_iW)^2 $$

$$ → \frac{\partial{J}}{\partial{W}} = -\frac{2}{N} \sum_i (y_i - X_iW) X_i = -\frac{2}{N} \sum_i (y_i - \hat{y}_i) X_i $$

$$ → \frac{\partial{J}}{\partial{b}} = -\frac{2}{N} \sum_i (y_i - X_iW)1 = -\frac{2}{N} \sum_i (y_i - \hat{y}_i)1 $$

```python
# Backpropagation
dW = -(2/N) * np.sum((y_train - y_pred) * X_train)
db = -(2/N) * np.sum((y_train - y_pred) * 1)
```

> The gradient is the derivative, or the rate of change of a function. It's a vector that points in the direction of greatest increase of a function. For example the gradient of our loss function ($$J$$) with respect to our weights ($$W$$) will tell us how to change $$W$$ so we can maximize $$J$$. However, we want to minimize our loss so we subtract the gradient from $$W$$.

<h4 id="update_numpy">Update weights</h4>

`Step 5`: Update the weights $$W$$ using a small learning rate $$\alpha$$.

$$ W = W - \alpha\frac{\partial{J}}{\partial{W}} $$

$$ b = b - \alpha\frac{\partial{J}}{\partial{b}} $$

```python
LEARNING_RATE = 1e-1
```
```python
# Update weights
W += -LEARNING_RATE * dW
b += -LEARNING_RATE * db
```

> The learning rate $$\alpha$$ is a way to control how much we update the weights by. If we choose a small learning rate, it may take a long time for our model to train. However, if we choose a large learning rate, we may overshoot and our training will never converge. The specific learning rate depends on our data and the type of models we use but it's typically good to explore in the range of $$[1e^{-8}, 1e^{-1}]$$. We'll explore learning rate update strategies in later lessons.

<h4 id="training_numpy">Training</h4>

` Step 6`: Repeat steps 2 - 5 to minimize the loss and train the model.
```python
NUM_EPOCHS = 100
```
```python
# Initialize random weights
W = 0.01 * np.random.randn(INPUT_DIM, OUTPUT_DIM)
b = np.zeros((1, ))

# Training loop
for epoch_num in range(NUM_EPOCHS):

    # Forward pass [NX1] · [1X1] = [NX1]
    y_pred = np.dot(X_train, W) + b

    # Loss
    loss = (1/len(y_train)) * np.sum((y_train - y_pred)**2)

    # Show progress
    if epoch_num%10 == 0:
        print (f"Epoch: {epoch_num}, loss: {loss:.3f}")

    # Backpropagation
    dW = -(2/N) * np.sum((y_train - y_pred) * X_train)
    db = -(2/N) * np.sum((y_train - y_pred) * 1)

    # Update weights
    W += -LEARNING_RATE * dW
    b += -LEARNING_RATE * db
```
<pre class="output">
Epoch: 0, loss: 0.990
Epoch: 10, loss: 0.039
Epoch: 20, loss: 0.028
Epoch: 30, loss: 0.028
Epoch: 40, loss: 0.028
Epoch: 50, loss: 0.028
Epoch: 60, loss: 0.028
Epoch: 70, loss: 0.028
Epoch: 80, loss: 0.028
Epoch: 90, loss: 0.028
</pre>

<h4 id="evaluation_numpy">Evaluation</h4>

Now we're ready to see how well our trained model will perform on our test (hold-out) data split. This will be our best measure on how well the model would perform on the real world, given that our dataset's distribution is close to unseen data.
```python
# Predictions
pred_train = W*X_train + b
pred_test = W*X_test + b
```
```python
# Train and test MSE
train_mse = np.mean((y_train - pred_train) ** 2)
test_mse = np.mean((y_test - pred_test) ** 2)
print (f"train_MSE: {train_mse:.2f}, test_MSE: {test_mse:.2f}")
```
<pre class="output">
train_MSE: 0.03, test_MSE: 0.01
</pre>

```python
# Figure size
plt.figure(figsize=(15,5))

# Plot train data
plt.subplot(1, 2, 1)
plt.title("Train")
plt.scatter(X_train, y_train, label='y_train')
plt.plot(X_train, pred_train, color='red', linewidth=1, linestyle='-', label='model')
plt.legend(loc='lower right')

# Plot test data
plt.subplot(1, 2, 2)
plt.title("Test")
plt.scatter(X_test, y_test, label='y_test')
plt.plot(X_test, pred_test, color='red', linewidth=1, linestyle='-', label='model')
plt.legend(loc='lower right')

# Show plots
plt.show()
```

<div class="ai-center-all">
    <img width="650" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA3IAAAE/CAYAAAADjvF6AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nOzdeXyU9bn///dFiBDXsLkkiGC1USxINKI1alXUqIdCwNZqbYV6KKc9X1vtkiPUU7VWBYr9tT1qW6lt1aNWq4ZUBY0LenAvSKiAmIoslQElAhHQQbN8fn/MJDOTZJJMZrlneT0fDx7cn3vuzH1NAlxc92cz55wAAAAAAJmjn9cBAAAAAABiQyEHAAAAABmGQg4AAAAAMgyFHAAAAABkGAo5AAAAAMgwFHIAAAAAkGEo5IA0Z2ZPmtk0r+MAAABA+qCQA5LAzPaE/Wo1M39Y+7JY3ss5d4Fz7p5kxQoAgBcSmSuD7/eCmc1IRqxAOurvdQBANnLO7d92bGYbJc1wzj3b8Toz6++ca05lbAAApIPe5koAXaNHDkghMzvTzDab2TVm9r6kP5vZIDN7wswazGxn8Hh42Ne0P2E0s+lm9pKZ3Rq8doOZXeDZBwIAIMHMrJ+ZzTKzd81su5n91cwGB18baGb3Bc83mtkyMzvEzG6WdLqk24M9erd7+ymA5KOQA1LvUEmDJR0haaYCfw//HGyPkOSX1F0COllSvaShkn4h6Y9mZskMGACAFPqepEpJX5JUJGmnpDuCr02TdJCkwyUNkfQdSX7n3LWSXpR0pXNuf+fclSmPGkgxCjkg9VolXe+c+9Q553fObXfOPeqc+8Q5t1vSzQokr2g2Oef+4JxrkXSPpMMkHZKCuAEASIXvSLrWObfZOfeppBskfcXM+ktqUqCAO8o51+Kce8M5t8vDWAHPMEcOSL0G59zetoaZ7SvpV5LOlzQoePoAM8sLFmsdvd924Jz7JNgZt38X1wEAkImOkLTQzFrDzrUo8NDyfxXojXvQzAol3adA0deU+jABb9EjB6Se69D+kaQSSSc75w6UdEbwPMMlAQC56D1JFzjnCsN+DXTO+ZxzTc65nznnRks6VdJESZcHv65jfgWyGoUc4L0DFJgX1xiczH29x/EAAOCl30u62cyOkCQzG2Zmk4PHZ5nZGDPLk7RLgaGWbT13H0g60ouAAS9QyAHe+7WkAkkfSnpN0lPehgMAgKd+I+kxSU+b2W4FcuPJwdcOlfSIAkXcWkn/p8Bwy7av+0pwVef/SW3IQOqZc/RCAwAAAEAmoUcOAAAAADIMhRwAAAAAZBgKOQAAAADIMBRyAAAAAJBhKOQAAAAAIMP09zqA7gwdOtSNHDnS6zAAAEn2xhtvfOicG+Z1HJmC/AgAuSNajkzrQm7kyJFavny512EAAJLMzDZ5HUMmIT8CQO6IliMZWgkAAAAAGYZCDgAAAAAyDIUcAAAAAGQYCjkAAAAAyDAUcgAAAACQYRJSyJnZn8xsm5mtjvL6mWb2kZmtDP66LhH3BQAAAIBclKjtB+6WdLuke7u55kXn3MQE3Q8AAAAAclZCCjnn3FIzG5mI9wIAZIaaOp/m19ZrS6NfRYUFqqooUWVpsddhAQCQE1I5R+6LZvYPM3vSzI5L4X0BAAlWU+fT7OpV8jX65ST5Gv2aXb1KNXU+r0MDACAnpKqQWyHpCOfc8ZJuk1QT7UIzm2lmy81seUNDQ4rCAwDEYn5tvfxNLRHn/E0tml9b71FEAACkEeekjRuTeouUFHLOuV3OuT3B48WS8s1saJRrFzjnypxzZcOGDUtFeACAGG1p9Md0HgCAnLF1q3ThhdKVVyb1Nikp5MzsUDOz4PH44H23p+LeAIDEKyos0BfeX6f1876soR/vjDgPAEDOWrhQe8ccrz9/NlRHH/ttlc9dkrRpBwlZ7MTM/iLpTElDzWyzpOsl5UuSc+73kr4i6btm1izJL+kS55xLxL0BACnmnGoe/amGLX9FkrSj4EBJUkF+nqoqSryMDAAAb+zeLV19tT5++jnNmDhLrx4ayIdtc8glJXxBsEStWnlpD6/frsD2BACATPb669Ipp2iYpBXX/1LfGzhOrtGvYlatBADkqldekb75Temss1Q543a947eIl9vmkKdlIQcAyHLOSaefLr38spSXJ330kU7Ybz+97HVcAAB4palJuvFG6Q9/kH7/e6myUutmLery0mTMIaeQA4Ac1+N+cC+/LJ12WuD43nsDTx0BAMhl9fXSN74hDRsmrVwpHXqopMBccV8XRVsy5pCnch85AECa6XY/uJYWySxUxH3yCUUcACC3OSf97ndSebn0rW9Jixa1F3GSVFVRooL8vIgvSdYccgo5AMhh0faD+2DWdVL/4KCNyspA4ipgRUoAQA57/31p4kTpj3+UXnpJ+s//DDzwDFNZWqw5U8eouLBAJqm4sEBzpo5JyhxyhlYCQA7rOGa/f0uz1t1aGTrR2CgddFCKowIAIM3U1Ejf+Y40Y4Z0/fVSfn7USytLi1Oy+BeFHADksPCx/D948T5d9cqDkqRnxp6lc/+xJDh/7o3o8+cAAMhmu3dLP/iB9Pzz0qOPBoZUpgmGVgJADquqKFGhmrVx3sT2Iu7Eqkf18d3/2/38OQAAst2rr0rjxkmtrYEFTdKoiJMo5AAgp1VeMVEr5wWGUi487iyVz3lOP730ZFWWFkedPze/tt6LUAEASI2mJum666QpU6T586U//Uk64ACvo+qEoZUAkIs+/ljaf/9Qe+dOTSks1JSwS6LteZOMvXAAAEgL//xnYFuBIUOkujrpsMO8jigqeuQAINeUlISKuBEjAitSFhZ2uizanjfJ2AsHAIBY1NT5VD53iUbNWqTyuUviH/bf2hrYTqC8XJo2TVq8OK2LOIkeOQDIHbt2Ra5AuWtXt0NFqipKNLt6VcTwymTthQMAQG+1zeFuy09tc7gl9W1Brmeflc49N3C8dq10zDGJCjWpKOQAIEsEVpis73qFyfB9bkaPltas6fH92r426nsCAOCB7uZwx5yjDjlE2rYtcLxjhzRoUIKiTD4KOQDIAtGeTu6z40NdeM640IUffyztu2+v3zdVe+EAANBbCZnD/e670lFHBY4nTpQefzwBkaUWc+QAIAt09XRy7U0XRBZxzsVUxAEAkI7insP99a+Hiri33srIIk6iRw4AMkZ3QyfDn0IesvtDvf7b6aEvjLEXDgCAdNbnOdy7d0sHHhg4HjBA2rs3iVEmH4UcAKSRaMVaTxO7iwoL5Gv0a+O8iRHvVz7nOb1MEQcAyCJ9msP9m99IV18dOH7sMenLX05BpMlFIQcAaaK7Yq2nid0/+3yezrkoVMSV/Kha/QoKNIcVJgEAWajXc7hbW6W8vFC7uTmyncGYIwcAaaK7Yq3bid1mOueiM9vPjbrmCQ0depDmTB3DQiVpwMzON7N6M1tnZrO6eH26mTWY2crgrxlexAkAWWfJklDR9vOfB+aKZ0kRJ9EjBwBpo7tirW3oZLhjtm3QU3/+XujE3r3SgAHakMwgERMzy5N0h6RzJW2WtMzMHnPOvdXh0oecc1emPEAAyFZFRdLWrYHjDNtWoLfokQOANNHdKlxVFSUqyA89Rdw4b2JkEedcYOI20s14Seucc+udc59JelDSZI9jAoDstWFDYO/UrVulCy4I5McsLOIkCjkASBsdizUptApXZWmx5kwdo4k76iMXNGlqCiQppKtiSe+FtTcHz3V0kZm9aWaPmNnhXb2Rmc00s+VmtryhoSEZsQJAZrv8cunIIwPHq1dLixd7G0+SMbQSANJET6twVZ4wXJXhX0ABly0el/QX59ynZvYfku6RdHbHi5xzCyQtkKSysjJ++ADQZs8e6YADAsd5eYEFTXIAhRwApJEuV+F68knpwgtD7SxacSsH+CSF97AND55r55zbHta8S9IvUhAXAGSH22+XvhecalBTI03OndHrFHIAkM7MItv0wmWaZZKONrNRChRwl0j6evgFZnaYcy44I1+TJK1NbYgAkIE6bivQ1CT1z63ShjlyAJCOqqsji7iWFoq4DOSca5Z0paRaBQq0vzrn1pjZjWY2KXjZ981sjZn9Q9L3JU33JloAyBC//W2oiLvqqkB+jLOIq6nzqXzuEo2atUjlc5eops7X8xd5LLfKVgCIUU2dL+qctaShFy6rOOcWS1rc4dx1YcezJc1OdVwAkJHCc+SmTdKIEXG/ZU2dT7OrV7Xv5epr9Gt29SpJSuv9WBPSI2dmfzKzbWa2OsrrZmb/E9wM9U0zOyER9wWAZGr7h93X6JdT6B/2pD2lu/32yATV2koRBwCAJC1bFpkjnUtIEScFFhlrK+La+JtaNL+2PiHvnyyJ6pG7W9Ltku6N8voFko4O/jpZ0u+CvwNA2uruH/aEP6FLci+cJz2LAAAkQniOXLw4sD9cAm1p9Md0Pl0kpEfOObdU0o5uLpks6V4X8JqkQjM7LBH3BoBkSck/7HPmdH7CmIQiLqU9iwAAJMKOHZ1zZIKLOEkqKiyI6Xy6SNViJ73dEBUA0kbS/2E3k37yk1A7ScMoM3XICAAgh02dKg0ZEjiePTupUw2qKkpUkB+5rU9Bfp6qKkqSds9ESLtVK81sppktN7PlDQ0NXocDIIcl7R/2mTOT3gsXLlOHjAAAclBrayBHLlwYaO/dK91yS1JvWVlarDlTx6i4sEAmqbiwQHOmjkn7KQipWrWyxw1R2zjnFkhaIEllZWXM8gfgmbZ/wBM6t8yDFSmLCgvk66JoS/chIwCAHHPnndJ3vhM4PuYYaW3qttWsLC1O+8Kto1QVco9JutLMHlRgkZOPwjY/BYC0lbB/2C++WHr44VA7hatRVlWURCyrLGXGkBEAQA4Jf9C5YYM0cqRnoWSKhBRyZvYXSWdKGmpmmyVdLylfkpxzv1dg/5wLJa2T9ImkbyXivgDQW56u2ujxvnBJ6VkEACARVqyQTjwx1GbbnV5LSCHnnLu0h9edpP+XiHsBQKw82+hzzBhpddj2mh4mp0wcMgIAyHLhDzqfeEL6t3/zLpYMlKqhlQDgmZTuB9fG4144AADSVmOjNGhQqE2O7JO0W7USABItpas2HnZYSlekBAAgo1x8caiI+/GPyZFxoEcOQNZL2aqN9MIBANA156R+YX1Ifr80cKB38WQBeuQAZL2kb/TZvz+9cAAARPPHP4aKuKOOCuRIiri40SMHIOslddXG8AJuwIDAxqUAACAgPE+uXy+NGuVdLFmGQg5ATkj4qo0HHCDt2RNq0wMHAEDIypVSaWmoTZ5MOIZWAkCszEJF3PjxJCcAAML16xcq4v72N/JkktAjBwC9xWImAABE99FHUmFhqE2eTCp65ACgN8KLuPPOIzkBABDu0ktDRdwPf0ieTAF65ACgO/TCAQAQHdsKeIYeOQCIJryImzAhpiKups6n8rlLNGrWIpXPXaKaOl8SAgQAwEN//nOoiDviCLYVSDF65ACgozh74WrqfJpdvUr+phZJkq/Rr9nVqyQFVs+sqfMlZysEAABSJTxXrlsnfe5z3sWSoyjkACBceGL6ylekhx+O+S3m19a3F3Ft/E0tml9bL0ndFnkAAKS1N9+Ujj8+1GbKgWcYWgkAUqCACy/inOtTESdJWxr9Uc/3VOQBAJC2BgwIFXHV1RRxHqOQA5DbnIss4GbOjDsxFRUWRD3fXZEHAEBa2rUrkCs/+yzQdk6aMsXbmEAhByCHmUWutOWcdOedcb9tVUWJCvLzIs4V5OepqqKk2yIPAIC0881vSgcdFDj+3vfohUsjzJEDkHs6LpX8X/8lzZuXsLdvm+sWbUGT8DlyUqjIAwAgbXTMlZ98IhXw0DGdUMgByChxr/iYon3hKkuLu4yrpyIPAADP3XuvNG1a4LioSPKxhU46opADkDF6Wta/W62tUl7YcMebb5Z+8pNkhdqtaEUeAACeC3/g+c9/Skcf7V0s6BaFHICM0d2Kj90WRinqhQMAIGOtXi2NGRNqkyvTHoudAMgYMa/42NQUWcTdfntCE1NNnU/lc5do1KxFKp+7RDV1DD0BAGSg/fYLFXEPP0wRlyHokQOQMYoKC+TromjrcsXHJPfCxTXMEwCAdLB7t3TggaF2a2vn/Im0RY8cgIzR3bL+7fbujUxC996blCeLbOwNAMhoV1wRKuL+8z8776uKtEePHICM0eOKjymcC8fG3gCAjNRxW4GPP5b23de7eNBnFHIAMkqXKz52HBpSXS1NmZLUOGIa5gkAQDq4/37pG98IHA8bJm3b5m08iAuFHIDM5tGKlFUVJWzsDQDIHOH58u23pRLyVaZjjhyAzLRjR2RSevrplK6yVVlarDlTx6i4sEAmqbiwQHOmjmGhEwBAennrrch86RxFXJZISI+cmZ0v6TeS8iTd5Zyb2+H16ZLmS2pbm/t259xdibg3gPRVU+eLPp8tHmmyLxwbewMA0tpBB0m7dgWOH3xQ+trXvI0HCRV3IWdmeZLukHSupM2SlpnZY865tzpc+pBz7sp47wcgMyRlef7335cOOyzUfuklqbw83lABAMgue/ZIBxwQarOtQFZKxNDK8ZLWOefWO+c+k/SgpMkJeF8AGSzhy/ObRRZxzlHEAQDQ0be/HSriZs5kW4EsloihlcWS3gtrb5Z0chfXXWRmZ0j6p6QfOOfe6+IamdlMSTMlacSIEQkID4AXErY8/6ZN0siRoXZdnTRuXN8DAwAgG3XcVmDPHmm//byLB0mXqsVOHpc00jk3VtIzku6JdqFzboFzrsw5VzZs2LAUhQcg0aItwx/T8vxmkUWccxRxAAB09OCDoSJu0KBAvqSIy3qJKOR8kg4Paw9XaFETSZJzbrtz7tNg8y5JJybgvgDSWFVFiQry8yLO9Xp5/vr6zsske7SgCQAAac1MuvTSwPFbbwVWdUZOSMTQymWSjjazUQoUcJdI+nr4BWZ2mHNua7A5SdLaBNwXQBprW9Ak5lUr02RFSgAA0trbb0vHHhtqky9zTtyFnHOu2cyulFSrwPYDf3LOrTGzGyUtd849Jun7ZjZJUrOkHZKmx3tfAKnT120EYlqef+VKqbQ01N64UTriiL4FDKSRXmzRM0DSvQqMVtku6WvOuY2pjhNABhk2TPrww8DxAw+EeuSQUxKyj5xzbrGkxR3OXRd2PFvS7ETcC0BqJWUbgY7ohUOW6uUWPf8uaadz7igzu0TSPEls9gSgs48/lvbfP9RmW4GclqrFTgBkqIRvIxDulVciE9DWrRRxyDa92aJnskKLgD0iaYIZ/zMD0MF3vxsq4q64gm0FkJgeOQDZK2HbCHRELxxyQ2+26Gm/Jjhd4SNJQyR9mJIIAaS3jtsK7NoVudk3chY9cgC6lZBtBMI9+2xkEbd9O0Uc0AtmNtPMlpvZ8oaGBq/DAZAKDz8cKuL23z+QLyniEEQhB6BbcW0j0JGZdO65obZz0uDBcUYIpLUet+gJv8bM+ks6SIFFTyKwzyqQY8ykiy8OHK9eLe3e7W08SDsUcgC6VVlarDlTx6i4sEAmqbiwQHOmjoltoZOamsheuF276IVDrmjfosfM9lFgi57HOlzzmKRpweOvSFriHH9BgJz1z39G5kznpOOO8y4epC3myAHoUUzbCHTEXDjksF5u0fNHSf9rZusU2KLnEu8iBuCpoqLAwl+SdO+90je/6W08SGsUcgCS49FHpa98JdT2+6WBA72LB/BIL7bo2Svpq6mOC0Aa+eQTab/9Qm22FUAvMLQSQOKZRRZxzlHEAQDQle99L1TEXX452wqg1+iRA5A4994rTZsWan/2mZSf7108AACks47zx1mREjGgkAOQGMyFAwCgd6qrpYsuChwPGCDt3ettPMhIDK0EEJ877ogs4lpaKOIAAIjGLFTEvfkmRRz6jB45AH3XoReufM5zqvrH1r6vcAkAQLZat046+uhQm4eeiBM9cgBiN2dORBE38r8e18hrnpCv0a/Z1atUU9dxv2MAAHLYiBGhIu7uuynikBD0yAGITYdeuJHXPBHR9je1aH5tPb1yAAD4/dK++4babCuABKJHDkDvzJ4dmXxaWzWqQxHXZkujv8e3q6nzqXzuEo2atUjlc5fQiwcAyC4/+EGoiLvsMrYVQMLRIwdAUqCwml9bry2NfhUVFqiqoiTUqxZlRcqiwgL5uijaigoLerzX7OpV8je1SFL7kExJ9OQBADJfeN5sbJQOOsi7WJC16JEDskhfe7naCitfo19OocJq83mTIpORcxHj+qsqSlSQnxfxXgX5eaqqKOn2fvNr69uLuDZtQzIBAMhYf/tbKG/26xfImRRxSBJ65IAsEU8vV1eF1dqbLoi8qIuJ2W3vG7UnL4poQy97MyQTAIC0FP7gs65OGjfOu1iQEyjkgCzRXS9XLIXVfQ9eq9M2/SP0Yg8ra1WWFsc8HLKvQzIBAEg769dLn/tcqM2KlEgRhlYCWSKeXq62AmrjvIkRRVz5nOcSE1wHfR2SCQBAWjnyyFARd9ddFHFIKXrkgCwRTy/XonuuUuHbq9vbI695QgX5eZqTpMKqr0MyAQBIC3v3SgVh+ZVtBeABCjkgS1RVlETMkZN62ctlpsKw5qhrnlBxCgqrvgzJBADAcz/+sfTLXwaOL75Yeughb+NBzqKQA7JEzL1cI0dKmzaF2sHhIBuSHCcAABkrvNdt506psDD6tUCSUcgBWaTXvVxR9oUDAABdePxxadKkUJu8iTRAIQd4pNsNuJOloCAwrr8NiQgAgO6FP/xcsUIqLfUuFiAMhRzggXj2fOuz8ETUr5/U0hL9WgAAct2GDYFVKdvw8BNpJiHbD5jZ+WZWb2brzGxWF68PMLOHgq+/bmYjE3FfIFN1t+dbwplFFnHOUcQBANCdkpJQEXfnnRRxSEtx98iZWZ6kOySdK2mzpGVm9phz7q2wy/5d0k7n3FFmdomkeZK+Fu+9gUwVz55vMQkv4A4+WPrgg8S+PwAA2eTTT6WBA0PtlpbAKBYgDSXiT+Z4Seucc+udc59JelDS5A7XTJZ0T/D4EUkTzNhsA7kr2t5uvdnzrVe66oWjiAMAILpZs0JF3NSpgdxJEYc0log/ncWS3gtrbw6e6/Ia51yzpI8kDUnAvYGMVFVRooL8vIhzvdrzrTfCC7jRoxkOAgBAT8ykefMCxzt2SI8+6m08QC+k3WMGM5tpZsvNbHlDQ4PX4QBJUVlarDlTx6i4sEAmqbiwQHOmjolvoZOueuHWrIk7VgAAstbixZ1z56BB3sUDxCARq1b6JB0e1h4ePNfVNZvNrL+kgyRt7+rNnHMLJC2QpLKyMroSkLV6vedbb4QnodNPl5YuTcz7AgCQrcJz57JlUlmZd7EAfZCIQm6ZpKPNbJQCBdslkr7e4ZrHJE2T9Kqkr0ha4hzjvYC4sbE3AACx2bRJGjky1CZ3IkPFPbQyOOftSkm1ktZK+qtzbo2Z3Whmk4KX/VHSEDNbJ+mHkjptUQAgRuFFXGUliQgAgJ584QuhIu63vyV3IqMlZENw59xiSYs7nLsu7HivpK8m4l5AzqMXDgCA2Hz2mTRgQKjNtgLIAvwJBjIJvXAAAMTmv/87VMRNmsS2AsgaCemRA5Bk9MIBABC78Py5fbs0eLB3sQAJxuMIIJ05F5mErrhCNSs2q3zuEo2atUjlc5eopq7jIrEAAOS42trO2wpQxCHL0CMHpKsueuFq6nyaXb1K/qYWSZKv0a/Z1askKXFbGQAAkMnC8+frr0vjx3sXC5BE9MgB6aa1NTIJ/ehH7UMp59fWtxdxbfxNLZpfW5/KCAEASD/vvde5F44iDlmMQg5IJ2ZSXl6o7Zx0663tzS2N/i6/LNp5AABywrhx0ogRgePbbmMuOXIChRyQDpqbI58i3nRTl0moqLCgyy+Pdh4AgKz22WeB/PmPfwTaLS3SlVd6GxOQIhRygNfMpPz8UNs56dpru7y0qqJEBfl5EecK8vNUVVGSzAgBAEg/118f2lbgggvYVgA5h8VOAK/s3SsVhPWk3XZbj08R2xY0mV9bry2NfhUVFqiqooSFTgAAuSV8FEtDgzR0qHexAB6hkAO8EMe+cJWlxRRuAIDc9Mwz0nnnhdrMhUMOo/8ZSKWPP44s4h5+mCQEAEBvmIWKuFdfJX8i59EjB6RKHL1wAADkrM2bpcMPD7XJn4AkeuSA5Nu5M7KIW7yYJAQAQG+UlYWKuF/9ivwJhKFHDkgmeuEAAIhdU5O0zz6hdnNz5D6rAOiRA5Ji27bIIm7pUoo4AAB64+c/DxVx554byJ8UcUAn9Mghp9TU+ZK/dD+9cAAkmdlgSQ9JGilpo6SLnXM7u7iuRdKqYPNfzrlJqYoRSDvhOXTbNmnYMO9iAdIcPXLIGTV1Ps2uXiVfo19Okq/Rr9nVq1RT50vMDd57LzIBLV9OEQfktlmSnnPOHS3puWC7K37n3LjgL4o45KYlSyJzqHMUcUAP6JFDzphfWy9/U0vEOX9Ti+bX1vfYK9djTx69cAA6myzpzODxPZJekHSNV8EAaSs8h770klRe7l0sQAahkEPO2NLoj+l8m7aevLYisK0nT5Iq9/tYKikJXbxmjTR6dGICBpDpDnHObQ0evy/pkCjXDTSz5ZKaJc11ztWkJDrAa1u2SMVhD0V5CArEhEIOOaOosEC+Loq2osKCbr8uWk9e5QnDIy8kAQE5x8yelXRoFy9dG95wzjkzi/aPxBHOOZ+ZHSlpiZmtcs6928W9ZkqaKUkjRoyIM3LAY1/8ovTaa4HjX/5S+uEPvY0HyEAUcsgZVRUlET1rklSQn6eqipJuvqpzj93oD9Zr8d3fD5145x3pqKMSGiuAzOCcOyfaa2b2gZkd5pzbamaHSdoW5T18wd/Xm9kLkkoldSrknHMLJC2QpLKyMp4cITM1N0v5+ZFtVqQE+oTFTpAzKkuLNWfqGBUXFsgkFRcWaM7UMT3Ojwvvsds4b2JkEeccRRyAaB6TNC14PE3S3zpeYGaDzGxA8HiopHJJb6UsQiCVbr45VMSddRbbCgBxokcOOaWytDjm7QaqKkp03+2P6pE/XdV+7szv36urp5+tykQHCCCbzJX0VzP7d6GvNHwAACAASURBVEmbJF0sSWZWJuk7zrkZko6VdKeZtSrwcHWuc45CDtknfEGT99+XDok2ZRRAb1HIAT2oPGF4RMFWPue55Ow/ByCrOOe2S5rQxfnlkmYEj1+RNCbFoQGp88ILgd63NswnBxKGQg6IZulS6UtfCrU//FAaMkQvexcRAACZI7wXbulS6fTTvYsFyEIUckBX2BcOAIC+ef996bDDQm1yKJAUcS12YmaDzewZM3sn+PugKNe1mNnK4K/H4rknkFRPPRVZxO3aRQICAKC3Tj89VMTNm0cOBZIo3h65WZKec87NNbNZwfY1XVznd86Ni/NeQHLRCwcAQN903FagqUnqz8AvIJni3X5gsqR7gsf3SCzihwz08MORRdwnn1DEAQDQW7/4RaiIO/30QA6liAOSLt6/ZYc457YGj9+XFG0t2YFmtlxSswJLK9fEeV8gMeiFAwCg78Lz6Nat0qGHehcLkGN67JEzs2fNbHUXvyaHX+ecc5Ki/S/4COdcmaSvS/q1mX2um/vNNLPlZra8oaEhls8C9N7dd0cmn08/pYgDAKC3XnwxMo86RxEHpFiPPXLOuXOivWZmH5jZYc65rWZ2mKRtUd7DF/x9vZm9IKlU0rtRrl0gaYEklZWV8T9rJB69cAAA9F14Hn3+eenMMz0LBchl8c6Re0zStODxNEl/63iBmQ0yswHB46GSyiW9Fed9gdj9z/9EJp/mZoo4AAB664MPOvfCUcQBnom3kJsr6Vwze0fSOcG2zKzMzO4KXnOspOVm9g9JzyswR45CDqllJl11VajtnJSX5108AABkkrPPDg2dvPlmHoQCaSCuxU6cc9slTeji/HJJM4LHr0gaE899gD676Sbppz8NtVtbOw+tjFNNnU/za+u1pdGvosICVVWUqLK0OKH3AADAEy0tkStQsq0AkDbi7ZED0pdZZBHnXFKKuNnVq+Rr9MtJ8jX6Nbt6lWrqfAm9DwAAKffLX4aKtlNOYVsBIM1QyCH7/PjHkQVba2vShoDMr62Xv6kl4py/qUXza+uTcj8AAFLCLJBPJcnnk1591dt4AHTCYxVklxSvSLml0R/TeQAA0trLL0unnRZqMxcOSFv0yCE7zJjReSWtFCSfosKCmM4DAJC2zEJF3HPPUcQBaY4eOWQ+D/eFq6oo0ezqVRHDKwvy81RVUZKyGAAAiEtDg3TwwaE2BRyQEeiRQ+a66CJPeuHCVZYWa87UMSouLJBJKi4s0JypY1i1EgCQGc49N1TE3XgjRRyQQeiRQ2bysBeuo8rSYgo3AEBm6bitwGefSfn53sUDIGb0yCGzXHKJ571wAABktF//OlTElZUF8ihFHJBx6JFD5kijXjgAADJSeC7dvFkqZkQJkKnokUP6q6igFw4AgHi8+mrnXEoRB2Q0CjmkrZo6XyDpPP20JOnTQUMo4AAAiJWZdOqpgeNnniGXAlmCoZVIS1vOvkCVzz/V3h55zRMqyM/TnDofC4sAANCFmjqf5tfWa0ujX0WFBbp2/DBdOGFs6AIKOCCr0COH9GOmomAR9/IRYzXymickSf6mFs2vrfcyMgAA0lJNnU+zq1fJ1+iXk3Tzgv8KFXHXXUcRB2QheuSQPs46S3rhhfZmWwEXbkujP4UBAQCQGebX1svf1CJzrdrwi0nt57/086f0f/9d4WFkAJKFQg7pIXwC9kUXqbzsP6UuiraiwoIUBgUAQGbY0ujXtDce18+evVOS9NbBo3Tht26T7Wn2ODIAyUIhB2+deKK0YkWoHRz6URUcIuJvaml/qSA/T1UVJamOEACAtLdh3sT24y9+98/aeuAwSTwABbIZhRwSouME66qKkp4XJQnvhfvWt6Q//am92fa1Mb8nAAC55O9/l04+ub0ZPi2BB6BAdqOQQ9xqOvSe+Rr9ml29SpK6LrxGjpQ2bQq1o0zAriwtpnADACCa8AeiTz2lmoO/oGIegAI5g0IOcWubYB2ubYXJTgkkPOlcdZX061+nIEIAALLIjh3SkCGhdvCBaKWiPEAFkJXYfgBxi7aSZMT5wsLIIs45ijgAAGI1aVKoiLv2WrYVAHIYPXKIW1FhgXzdrTAZXsD99KfSjTemKDIAALJEa6uUlxdqf/qptM8+3sUDwHP0yCFuVRUlKsjPizhXkJ+nl2dP6NwLRxEHAEBsfvvbUBF33HGBfEoRB+Q8euTQa9FWpuxqhcmXZ08IfeEvfiFVVXkUNQAAGSz8geimTdKIEd7FAiCtUMihV3pambK9oAtPOBJj9wEA6Ivly6WTTgq1yacAOmBoJXqlu5UpJQUSTHgR99vfknQAAOgLs1ARt3gx+RRAl+iRQ690uzKlB71wfdqAHACAdLZzpzR4cKhNAQegG3H1yJnZV81sjZm1mllZN9edb2b1ZrbOzGbFc094o30FynDOacO8iaH2PfekrIibXb1Kvka/nELDPGvqfEm/NwAASTF1aqiImzWLIg5Aj+LtkVstaaqkO6NdYGZ5ku6QdK6kzZKWmdljzrm34rw3UqiqoiRijtzG8AJOSmnCiWkDcgAA0lnHbQX27pUGDPAuHgAZI64eOefcWudcfQ+XjZe0zjm33jn3maQHJU2O577oXk2dT+Vzl2jUrEUqn7skIT1VlaXFmjN1jA4/cJ/IIu6RR1L+1LBXG5ADAJDu7rwzVMSVlATyKUUcgF5KxRy5YknvhbU3Szo5BffNST2tLhmPyhOGqzL8hEfDPnrcgBwAgHQXPr98wwZp5EjPQgGQmXrskTOzZ81sdRe/ktKrZmYzzWy5mS1vaGhIxi2yQrRetx5Xl+yL5ubIhOPxClrRNiCvqijxKCIAAHppxYrInOocRRyAPumxR845d06c9/BJOjysPTx4Ltr9FkhaIEllZWXM9O1Cd71uCR92mIb7wnW1ATmrVgIA0l54Tn38cWnixOjXAkAPUjG0cpmko81slAIF3CWSvp6C+2at7nrdEjbs8LPPIsfpv/66NH58X8JNivYNyAEgTZnZVyXdIOlYSeOdc8ujXHe+pN9IypN0l3NubsqCRGo0NkqDBoXaafBQFEDmi3f7gSlmtlnSFyUtMrPa4PkiM1ssSc65ZklXSqqVtFbSX51za+ILO7d11+uWkGGHZpFFnHNpVcQBQIZoW9l5abQLwlZ2vkDSaEmXmtno1ISHlLj44lAR9+MfU8QBSJi4euSccwslLezi/BZJF4a1F0taHM+9ENJdr1tcww79fmnffdubl39/gaZOvzBygRMAQK8459ZKknUcoh6pfWXn4LVtKzuzRU+mc07qF/a83O+XBg70Lh4AWScVQyuRYB33dJMie936NOyww380Rl7zhCRpWYJWvAQAdImVnbPRH/8ozZgROD7ySOndd72NB0BWopDLQAld7GP3bunAA9ubZ8/4vdYPGd7eZqNtAIjOzJ6VdGgXL13rnPtbgu81U9JMSRoxYkQi3xqJFP5g9N13A4UcACQBhVyGSshiHx164UZd84S6GrnPRtsA0LVUruzMqs5pbuVKqbQ01GYuHIAki2uxE2SonTsji7j16yXnoq5syUbbAJA07Ss7m9k+Cqzs/JjHMSFW/fqFiriaGoo4AClBIZdrzKTBg0Nt56RRoySx0TYAJBIrO+eAjz4K5NW2ws05afJkb2MCkDMo5HLFtm2RvXCbN3d6YlhZWqw5U8eouLBAJqm4sEBzpo5hfhwA9IFzbqFzbrhzboBz7hDnXEXw/BbnXMTKzs65zzvnPuecu9m7iBGTyy6TCgsDx1dfTS8cgJRjjlwu6Lj0dTfJho22AQDoRsdtBT75RCpgCgKA1KNHLptt2RJZxH3wAU8MAQDoq7vvDhVxI0YEcipFHACP0COXrWLohQMAAD0Iz6vvvCMddZR3sQCA6JHLPj5fZLLZsYMiDgCAvlq1KjKvOkcRByAt0COXTeiFAwAgcQYMkD77LHD86KPS1KnexgMAYSjkssHGje1bCEiS9uyR9tvPs3AAAMhou3ZJBx0UavNgFEAaYmhlpjOLLOKco4gDAKCvLr88VMRdeSVFHIC0RY9cpnrvvcCKWW38fmngQO/iAQAgk3XcVuDjj6V99/UuHgDoAT1ymcgsVMQdf3wg+VDEAQDQN4sWhYq4ww4L5FWKOABpjh65TLJhg3TkkaF2U5PUnx8hAAB9Fr5Q2KZNkaNdACCN0SOXKcxCRVx5eeBpIUUcAAB98+67nbcVoIgDkEEo5NJdfX1komlull56ybt4AADIdGecEdoL7plnWNAEQEaiSyedhRdw558vPfmkd7EAAJACNXU+za+t15ZGv4oKC1RVUaLK0uLEvPknn0Su7Nza2nkPVgDIEPTIpaPVqyMTS0sLRRwAIOvV1Pk0u3qVfI1+OUm+Rr9mV69STZ0v/je/8cZQETdnTqAXjiIOQAajRy7dhCeViy6SHnnEu1gAAEih+bX18je1RJzzN7Vofm1933vlOm4r8MknUkFBHFECQHqgRy5drFgRWcS1tlLEAQByypZGf0zne/TUU6EibsKEQFFHEQcgS9Ajlw7CC7jLL5fuuce7WAAA8EhRYYF8XRRtRYV9KL7Cc+uGDdLIkX0PDADSED1yXnr77c69cBRxAIAcVVVRooL8vIhzBfl5qqoo6f2bbNjQeVsBijgAWYhCzitm0rHHBo5vu41J1wCAnFdZWqw5U8eouLBAJqm4sEBzpo7p/fy4s84K7blaWxvTtgI1dT6Vz12iUbMWqXzuksQssAIAScTQylRbtUoaOzbUZu8aAADaVZYWx76wid8v7btvqB3jtgJtq2W2LbTStlpmWzwAkI7okUsls1AR94c/UMQBABCvm28OFXE33dSnES7drZYJAOkqrh45M/uqpBskHStpvHNueZTrNkraLalFUrNzriye+2acFSukE08MtSngAACITwK3FUj4apkAkALx9sitljRV0tJeXHuWc25czhVxZqEi7p57KOIAAIjX00+HirgvfSnubQWirYrZp9UyASBF4uqRc86tlSRjkY7OXn9dOuWUUJsCDgCA+IX/n2P9emnUqLjfsqqiJGKOnNSH1TIBIMVSNUfOSXrazN4ws5ndXWhmM81suZktb2hoSFF4CWYWKuIeeogiDgCAeG3c2HlbgQQUcVICVssEAA/02CNnZs9KOrSLl651zv2tl/c5zTnnM7ODJT1jZm8757ocjumcWyBpgSSVlZVlVgX00kvS6aeH2hRwAADE75xzpOeeCxw/+aR0/vkJv0WfVssEAA/1WMg5586J9ybOOV/w921mtlDSePVuXl3mCH9KuHChVFnZqy+rqfNpfm29tjT6VVRYoKqKEhIJAACStHdv5Ny3GLcVAIBslvShlWa2n5kd0HYs6TwFFknJDs8/33moRwxF3OzqVfI1+uUU2reGTUgBADlv7txQEfezn/VpWwEAyGbxbj8wRdJtkoZJWmRmK51zFWZWJOku59yFkg6RtDC4IEp/SQ84556KM+70EJ5QFi2SLrwwpi/vbt8aeuUAADkrPL9+/HHkZt8AAEnxr1q5UNLCLs5vkXRh8Hi9pOPjuU/aqa2NHJ/fx7lw7FsDAECY9eulz30ucDxhgvTss97GAwBpLFWrVmYPs1AR98wzcS1owr41AAAE/fd/h4q47dsp4gCgBxRyvfX4453nwp0T3zowVRUlKsjPizjHvjUAgJzy0UeB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q6Xf/U7y+3u1aBgAIHUYWgkAQBI45xZKWtjF+S2SLgwer5d0fIpD696NN0rXXx84bmiQhg71Nh4AQJco5AAAgLR7t3TggYHjK6+UbrvN23gAAN1iaCUAALnur38NFXFr1lDEAUAGoJADACBXtW0r8LWvSaeeKrW2SqNHex0VAKSljRs3tm/+3Re33HJLAqOhkAOAnBdPYjr11FMTHA1S5u9/l/r3lzZskGprpZdfZm84AOhGVhVyZjbfzN42szfNbKGZFUa57nwzqzezdWY2K557AkC2qqnzqXzuEo2atUjlc5eops6Xkvt2l5iam5u7/dpXXnklGSEh2ZqbpZNPlvbZJ7CtwHnneR0RAPRaovPlddddp1//+tft7WuvvVa/+c1vOl03a9Ysvfjiixo3bpx+9atfqaWlRVVVVTrppJM0duxY3XnnnZKkrVu36owzztC4ceP0hS98QS+++KJmzZolv9+vcePG6bLLLosr3jbxLnbyjKTZzrlmM5snabaka8IvMLM8SXdIOlfSZknLzOwx59xbcd4bALJGTZ1Ps6tXyd/UIknyNfo1u3qVJMW2eXOY6667ToMHD9bVV18tKZCYDj74YF111VUR182aNUtr167VuHHjNG3aNA0aNEjV1dXas2ePWlpatGjRIk2ePFk7d+5UU1OTbrrpJk2ePFmStP/++2vPnj164YUXdMMNN2jo0KFavXq1TjzxRN13330yenjSU//+gcVN9t/f60gAICbJyJdXXHGFpk6dqquvvlqtra168MEH9fe//73TdXPnztWtt96qJ554QpK0YMECHXTQQVq2bJk+/fRTlZeX67zzzlN1dbUqKip07bXXqqWlRZ988olOP/103X777Vq5cmUfP3lncRVyzrmnw5qvSfpKF5eNl7QuuMSyzOxBSZMlJbWQq6nzaX5tvbY0+lVUWKCqipI+/3ABINnm19a3J6U2/qYWza+tT3liuvvuu7VixQq9+eabGjx4sJqbm7Vw4UIdeOCB+vDDD3XKKado0qRJnYq0uro6rVmzRkVFRSovL9fLL7+s0047rU+xIwUSVMSRbwGkUjLy5ciRIzVkyBDV1dXpgw8+UGlpqYYMGdLj1z399NN688039cgjj0iSPvroI73zzjs66aSTdMUVV6ipqUmVlZUaN25cn+LqSSK3H7hC0kNdnC+W9F5Ye7OkkxN4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">
</div>

<h4 id="interpretability_numpy">Interpretability</h4>

Since we standardized our inputs and outputs, our weights were fit to those standardized values. So we need to unstandardize our weights so we can compare it to our true weight (3.5).

> Note that both X and y were standardized.

$$ \hat{y}_{scaled} = b_{scaled} + \sum_{j=1}^{k}{W_{scaled}}_j{x_{scaled}}_j $$

<div class="ai-center-all">
<table class="mathjax-table">
  <tbody>
    <tr>
      <td>$$ y_{scaled} $$</td>
      <td>$$ \frac{\hat{y} - \bar{y}}{\sigma_y} $$</td>
    </tr>
    <tr>
      <th>$$ x_{scaled} $$</th>
      <th>$$ \frac{x_j - \bar{x}_j}{\sigma_j} $$</th>
    </tr>
  </tbody>
</table>
</div>

$$\frac{\hat{y} - \bar{y}}{\sigma_y} = b_{scaled} + \sum_{j=1}^{k}{W_{scaled}}_j\frac{x_j - \bar{x}_j}{\sigma_j}$$

$$ \hat{y}_{scaled} = \frac{\hat{y}_{unscaled} - \bar{y}}{\sigma_y} = {b_{scaled}} + \sum_{j=1}^{k} {W_{scaled}}_j (\frac{x_j - \bar{x}_j}{\sigma_j}) $$

$$\hat{y}_{unscaled} = b_{scaled}\sigma_y + \bar{y} - \sum_{j=1}^{k} {W_{scaled}}_j(\frac{\sigma_y}{\sigma_j})\bar{x}_j + \sum_{j=1}^{k}{W_{scaled}}_j(\frac{\sigma_y}{\sigma_j})x_j $$

In the expression above, we can see the expression:

$$ \hat{y}_{unscaled} = W_{unscaled}x + b_{unscaled} $$

<div class="ai-center-all">
<table class="mathjax-table">
  <tbody>
    <tr>
      <td>$$ W_{unscaled} $$</td>
      <td>$$ \sum_{j=1}^{k}{W}_j(\frac{\sigma_y}{\sigma_j}) $$</td>
    </tr>
    <tr>
      <th>$$ b_{unscaled} $$</th>
      <th>$$ b_{scaled}\sigma_y + \bar{y} - \sum_{j=1}^{k} {W}_j(\frac{\sigma_y}{\sigma_j})\bar{x}_j $$</th>
    </tr>
  </tbody>
</table>
</div>

```python
# Unscaled weights
W_unscaled = W * (y_std/X_std)
b_unscaled = b * y_std + y_mean - np.sum(W_unscaled*X_mean)
print ("[actual] y = 3.5X + noise")
print (f"[model] y_hat = {W_unscaled[0][0]:.1f}X + {b_unscaled[0]:.1f}")
```
<pre class="output">
[actual] y = 3.5X + noise
[model] y_hat = 3.4X + 7.8
</pre>

<h3 id="pytorch">PyTorch</h3>
- [Split data](#split_pytorch)
- [Standardize data](#standardize_pytorch)
- [Weights](#weights_pytorch)
- [Model](#model_pytorch)
- [Loss](#loss_pytorch)
- [Optimizer](#optimizer)
- [Training](#training_pytorch)
- [Evaluation](#evaluation_pytorch)
- [Inference](#inference_pytorch)
- [Interpretability](#interpretability_pytorch)


Now that we've implemented linear regression with Numpy, let's do the same with PyTorch.
```python
import torch
```
```python
# Set seed for reproducibility
torch.manual_seed(SEED)
```
<pre class="output">
<torch._C.Generator at 0x7fbb75d12cf0>
</pre>

<h4 id="split_pytorch">Split data</h4>

This time, instead of splitting data using indices, let's use scikit-learn's built in [`train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split){:target="_blank"} function.
```python
from sklearn.model_selection import train_test_split
```
```python
TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15
```
```python
# Split (train)
X_train, X_, y_train, y_ = train_test_split(X, y, train_size=TRAIN_SIZE)
```
```python
print (f"train: {len(X_train)} ({(len(X_train) / len(X)):.2f})\n"
       f"remaining: {len(X_)} ({(len(X_) / len(X)):.2f})")
```
<pre class="output">
train: 35 (0.70)
remaining: 15 (0.30)
</pre>
```python
# Split (test)
X_val, X_test, y_val, y_test = train_test_split(
    X_, y_, train_size=0.5)
```
```python
print(f"train: {len(X_train)} ({len(X_train)/len(X):.2f})\n"
      f"val: {len(X_val)} ({len(X_val)/len(X):.2f})\n"
      f"test: {len(X_test)} ({len(X_test)/len(X):.2f})")
```
<pre class="output">
train: 35 (0.70)
val: 7 (0.14)
test: 8 (0.16)
</pre>

<h4 id="standardize_pytorch">Standardize data</h4>

This time we'll use scikit-learn's [`StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler) to standardize our data.

```python
from sklearn.preprocessing import StandardScaler
```
```python
# Standardize the data (mean=0, std=1) using training data
X_scaler = StandardScaler().fit(X_train)
y_scaler = StandardScaler().fit(y_train)
```
```python
# Apply scaler on training and test data
X_train = X_scaler.transform(X_train)
y_train = y_scaler.transform(y_train).ravel().reshape(-1, 1)
X_val = X_scaler.transform(X_val)
y_val = y_scaler.transform(y_val).ravel().reshape(-1, 1)
X_test = X_scaler.transform(X_test)
y_test = y_scaler.transform(y_test).ravel().reshape(-1, 1)
```
```python
# Check (means should be ~0 and std should be ~1)
print (f"mean: {np.mean(X_test, axis=0)[0]:.1f}, std: {np.std(X_test, axis=0)[0]:.1f}")
print (f"mean: {np.mean(y_test, axis=0)[0]:.1f}, std: {np.std(y_test, axis=0)[0]:.1f}")
```
<pre class="output">
mean: -0.3, std: 0.7
mean: -0.3, std: 0.6
</pre>

<h4 id="weights_pytorch">Weights</h4>

We will be using PyTorch's [Linear layers](https://pytorch.org/docs/stable/nn.html#linear-layers){:target="_blank"} in our MLP implementation. These layers will act as out weights (and biases).

$$ z = XW $$

```python
from torch import nn
```
```python
# Inputs
N = 3 # num samples
x = torch.randn(N, INPUT_DIM)
print (x.shape)
print (x.numpy())
```
<pre class="output">
torch.Size([3, 1])
[[ 0.04613046]
 [ 0.40240282]
 [-1.0115291 ]]
</pre>
```python
# Weights
m = nn.Linear(INPUT_DIM, OUTPUT_DIM)
print (m)
print (f"weights ({m.weight.shape}): {m.weight[0][0]}")
print (f"bias ({m.bias.shape}): {m.bias[0]}")
```
<pre class="output">
Linear(in_features=1, out_features=1, bias=True)
weights (torch.Size([1, 1])): 0.35
bias (torch.Size([1])): -0.34
</pre>
```python
# Forward pass
z = m(x)
print (z.shape)
print (z.detach().numpy())
```
<pre class="output">
torch.Size([3, 1])
[[-0.32104054]
 [-0.19719592]
 [-0.68869597]]
</pre>

<h4 id="model_pytorch">Model</h4>

$$ \hat{y} = XW + b $$

```python
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, x_in):
        y_pred = self.fc1(x_in)
        return y_pred
```
```python
# Initialize model
model = LinearRegression(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM)
print (model.named_parameters)
```
<pre class="output">
Model:
<bound method Module.named_parameters of LinearRegression(
  (fc1): Linear(in_features=1, out_features=1, bias=True)
)>
</pre>

<h4 id="loss_pytorch">Loss</h4>

This time we're using PyTorch's [loss functions](https://pytorch.org/docs/stable/nn.html#loss-functions){:target="_blank"}, specifically [`MSELoss`](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss){:target="_blank"}.

```python
loss_fn = nn.MSELoss()
y_pred = torch.Tensor([0., 0., 1., 1.])
y_true =  torch.Tensor([1., 1., 1., 0.])
loss = loss_fn(y_pred, y_true)
print('Loss: ', loss.numpy())
```
<pre class="output">
Loss:  0.75
</pre>

<h4 id="optimizer">Optimizer</h4>

When we implemented linear regression with just NumPy, we used batch gradient descent to update our weights. But there are actually many different gradient descent [optimization algorithms](https://pytorch.org/docs/stable/optim.html){:target="_blank"} to choose from and it depends on the situation. However, the [ADAM optimizer](https://pytorch.org/docs/stable/optim.html#torch.optim.Adam){:target="_blank"} has become a standard algorithm for most cases.

```python
from torch.optim import Adam
```
```python
# Optimizer
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
```

<h4 id="training_pytorch">Training</h4>

```python
# Convert data to tensors
X_train = torch.Tensor(X_train)
y_train = torch.Tensor(y_train)
X_val = torch.Tensor(X_val)
y_val = torch.Tensor(y_val)
X_test = torch.Tensor(X_test)
y_test = torch.Tensor(y_test)
```
```python
# Training
for epoch in range(NUM_EPOCHS):
    # Forward pass
    y_pred = model(X_train)

    # Loss
    loss = loss_fn(y_pred, y_train)

    # Zero all gradients
    optimizer.zero_grad()

    # Backward pass
    loss.backward()

    # Update weights
    optimizer.step()

    if epoch%20==0:
        print (f"Epoch: {epoch} | loss: {loss:.2f}")
```
<pre class="output">
Epoch: 0 | loss: 0.22
Epoch: 20 | loss: 0.03
Epoch: 40 | loss: 0.02
Epoch: 60 | loss: 0.02
Epoch: 80 | loss: 0.02
</pre>

<h4 id="evaluation_pytorch">Evaluation</h4>

Now we're ready to evaluate our trained model.

```python
# Predictions
pred_train = model(X_train)
pred_test = model(X_test)
```
```python
# Performance
train_error = loss_fn(pred_train, y_train)
test_error = loss_fn(pred_test, y_test)
print(f'train_error: {train_error:.2f}')
print(f'test_error: {test_error:.2f}')
```
<pre class="output">
train_error: 0.02
test_error: 0.01
</pre>

Since we only have one feature, it's easy to visually inspect the model.
```python
# Figure size
plt.figure(figsize=(15,5))

# Plot train data
plt.subplot(1, 2, 1)
plt.title("Train")
plt.scatter(X_train, y_train, label='y_train')
plt.plot(X_train, pred_train.detach().numpy(), color='red', linewidth=1, linestyle='-', label='model')
plt.legend(loc='lower right')

# Plot test data
plt.subplot(1, 2, 2)
plt.title("Test")
plt.scatter(X_test, y_test, label='y_test')
plt.plot(X_test, pred_test.detach().numpy(), color='red', linewidth=1, linestyle='-', label='model')
plt.legend(loc='lower right')

# Show plots
plt.show()
```

<div class="ai-center-all">
    <img width="650" src="data:image/png;base64,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cW/O7ZzTlClTNGvWrJS2E3li3TrpK1/xju+7T/qP//C3PQByjw/9ZS72lfn98divfy199FFMIU5izzcAhemoo47qmu7x1FNPaeDAgfrSl76ko48+Wvfcc48k6dFHH9WHH34oSTruuOM0b948vf/++5KkDz74QBs2bPCn8cguV10VCnEffkiIA5A3srGvTGpELusFhzbjwZ5vAArNlVdeqfPPP1+jR4/WF77wBd15552SpCuuuEJnnnmmRo4cqSOOOKKrkvCIESN09dVX64QTTlBHR4eKi4t14403av/99/fzbcBPW7aE1sJddBFr4QDknWzsK835NNUnFlVVVW7ZsmV+NwMAErJ69WoNHz7c72ZknWjfFzN7yTlX5VOTck4y/WN9Y0BzGtZoY2ubykpLVFtdkdwHmIsWSSed5B03NUkHH5z4cwEoSPSXIfH0kfk9IgcAALqkdL9U56QJE6Snn6agCQD4gH9xAQAoECnbL3XdOi+0Pf20dO+90urVhDgAyDD+1QUAoECkZL/U//qvyIIm3/lOCloGAIgXQQ4AgALR076oMe2XumWLt7fT5ZdLP/6xN7WytDTFLQQAxIogBwBAgaitrlBJcVHEtZj2S120KFSVsqlJ+v3v09RCAECsKHYCAECB6CxoEnPVSgqaAEDWIsgBAFBAYt4vdd260Fq4e+9lLRyAgrd+/Xo999xzOuussxL6+muvvVaXXnppytrDx2oAUODWr1+ve+65J6GvPeKII1LcGmQFCpoAwA6S6S8lL8ilEkEOALJEfWNA4+sWa+iMRzS+brHqGwMZed3eOqbt27f3+rXPPfdcOpoEv1DQBEAOSHV/efnll+v666/vOr/sssv0u9/9bof7ZsyYoWeeeUZjxozRddddp/b2dtXW1uqwww7T6NGjdeutt0qS3nnnHR199NEaM2aMDj74YD3zzDOaMWOG2traNGbMGJ199tlJtbcTUysBIAukdKPmoMsvv1x77rmnLr74Yklex7TXXnvpJz/5ScR9M2bM0OrVqzVmzBhNmTJF/fv31/z58/Xpp5+qvb1djzzyiCZNmqQPP/xQ27Zt09VXX61JkyZJknbbbTd9+umneuqpp3TllVdq4MCBWrFihb72ta/prrvukpkl+i1Bpi1aJJ10knf86qvSqFH+tgcAokhHf3n++edr8uTJuvjii9XR0aH77rtPL7744g731dXV6Te/+Y0efvhhSdLcuXO1xx57aOnSpdq6davGjx+vE044QfPnz1d1dbUuu+wytbe3a8uWLTrqqKN0ww03aPny5Qm+8x3lbZCrbwzEvpgbAHzW20bNme6Y7rjjDr388st69dVXteeee2r79u168MEH9aUvfUmbNm3S4YcfrokTJ+4Q0hobG7Vy5UqVlZVp/PjxWrJkiY488siE2o4MCi9oUlEhrVpFQRMAWSsd/eWQIUM0YMAANTY26r333lNlZaUGDBjQ59c99thjevXVVzVv3jxJ0kcffaTXX39dhx12mM4//3xt27ZNNTU1GjNmTELt6kteBrl0JHUASKeUbNTcTaIdkyQdf/zx2nPPPSVJzjldeumlevrpp7XTTjspEAjovffe0z777BPxNWPHjtXgwYMlSWPGjNH69esJctnuzTelAw7wjiloAiAHpKO/lKRp06bpjjvu0Lvvvqvzzz8/pq9xzukPf/iDqqurd3js6aef1iOPPKKpU6fqZz/7mc4999yk2hdNXn7k1ltSB4BslNRGzb3o7Jj+/Oc/x9wxSdIXO/cMk3T33XerpaVFL730kpYvX669995bn3322Q5f069fv67joqKiPtfXwWf/9V+hEEdBEwA5Il395emnn65FixZp6dKlUYOZJO2+++765JNPus6rq6t18803a9u2bZKkf/3rX/r3v/+tDRs2aO+999b3vvc9TZs2TS+//LIkqbi4uOveVMjLIJeupA4A6ZLwRs19SKRj6u6jjz7SXnvtpeLiYv3jH//Qhg0bkmoTssD27RQ0AZCT0tVf7rLLLjrmmGN0xhlnqKioKOo9o0ePVlFRkQ455BBdd911mjZtmkaMGKFDDz1UBx98sL7//e9r+/bteuqpp3TIIYeosrJS999/f9fa9OnTp2v06NEUO+lNWWmJAlFCW7JJHQDSJe6NmmPU2TGVlpbG1DFNnTpV/fv3j3j87LPP1mmnnaZRo0apqqpKBx10UFJtQhbYeWcvwAFAjklXf9nR0aHnn39ef//733u8p7i4WIsXL464du211+6wrcCUKVM0ZcqUHb5+9uzZmj17dlLtDJeXQa62uiJijZyUmqQOAOkU80bNcUi0Y5o6dWrX8cCBA/XPf/4z6td++umnkqQJEyZowoQJXddvuOGGxBsNAEAvUt1frlq1SqeeeqpOP/10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">
</div>

<h4 id="inference_pytorch">Inference</h4>

After training a model, we can use it to predict on new data.

```python
# Feed in your own inputs
sample_indices = [10, 15, 25]
X_infer = np.array(sample_indices, dtype=np.float32)
X_infer = torch.Tensor(X_scaler.transform(X_infer.reshape(-1, 1)))
```

Recall that we need to unstandardize our predictions.

$$ \hat{y}_{scaled} = \frac{\hat{y} - \mu_{\hat{y}}}{\sigma_{\hat{y}}} $$

$$ \hat{y} = \hat{y}_{scaled} * \sigma_{\hat{y}} + \mu_{\hat{y}} $$

```python
# Unstandardize predictions
pred_infer = model(X_infer).detach().numpy() * np.sqrt(y_scaler.var_) + y_scaler.mean_
for i, index in enumerate(sample_indices):
    print (f"{df.iloc[index]['y']:.2f} (actual) → {pred_infer[i][0]:.2f} (predicted)")
```
<pre class="output">
35.73 (actual) → 42.11 (predicted)
59.34 (actual) → 59.17 (predicted)
97.04 (actual) → 93.30 (predicted)
</pre>

<h4 id="interpretability_pytorch">Interpretability</h4>

Linear regression offers the great advantage of being highly interpretable. Each feature has a coefficient which signifies its importance/impact on the output variable y. We can interpret our coefficient as follows: by increasing X by 1 unit, we increase y by $$W$$ (~3.65) units.
```python
# Unstandardize coefficients
W = model.fc1.weight.data.numpy()[0][0]
b = model.fc1.bias.data.numpy()[0]
W_unscaled = W * (y_scaler.scale_/X_scaler.scale_)
b_unscaled = b * y_scaler.scale_ + y_scaler.mean_ - np.sum(W_unscaled*X_scaler.mean_)
print ("[actual] y = 3.5X + noise")
print (f"[model] y_hat = {W_unscaled[0]:.1f}X + {b_unscaled[0]:.1f}")
```
<pre class="output">
[actual] y = 3.5X + noise
[model] y_hat = 3.4X + 8.0
</pre>

<h3 id="regularization">Regularization</h3>

Regularization helps decrease overfitting. Below is `L2` regularization (ridge regression). There are many forms of regularization but they all work to reduce overfitting in our models. With `L2` regularization, we are penalizing the weights with large magnitudes by decaying them. Having certain weights with high magnitudes will lead to preferential bias with the inputs and we want the model to work with all the inputs and not just a select few. There are also other types of regularization like `L1` (lasso regression) which is useful for creating sparse models where some feature coefficients are zeroed out, or elastic which combines `L1` and `L2` penalties.
> Regularization is not just for linear regression. You can use it to regularize any model's weights including the ones we will look at in future lessons.

$$ J(\theta) = \frac{1}{2}\sum_{i}(X_iW - y_i)^2 + \frac{\lambda}{2}W^TW $$

$$ \frac{\partial{J}}{\partial{W}}  = X (\hat{y} - y) + \lambda W $$

$$ W = W - \alpha\frac{\partial{J}}{\partial{W}} $$

<div class="ai-center-all">
<table class="mathjax-table">
  <tbody>
    <tr>
      <td>$$ \lambda $$</td>
      <td>$$ \text{regularization coefficient} $$</td>
    </tr>
    <tr>
      <td>$$ \alpha $$</td>
      <td>$$ \text{learning rate} $$</td>
    </tr>
  </tbody>
</table>
</div>

In PyTorch, we can add L2 regularization by adjusting our optimizer. The Adam optimizer has a `weight_decay` parameter which to control the L2 penalty.

```python
L2_LAMBDA = 1e-2
```
```python
# Initialize model
model = LinearRegression(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM)
```
```python
# Optimizer (w/ L2 regularization)
optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_LAMBDA)
```
```python
# Training
for epoch in range(NUM_EPOCHS):
    # Forward pass
    y_pred = model(X_train)

    # Loss
    loss = loss_fn(y_pred, y_train)

    # Zero all gradients
    optimizer.zero_grad()

    # Backward pass
    loss.backward()

    # Update weights
    optimizer.step()

    if epoch%20==0:
        print (f"Epoch: {epoch} | loss: {loss:.2f}")
```
<pre class="output">
Epoch: 0 | loss: 2.20
Epoch: 20 | loss: 0.06
Epoch: 40 | loss: 0.03
Epoch: 60 | loss: 0.02
Epoch: 80 | loss: 0.02
</pre>
```python
# Predictions
pred_train = model(X_train)
pred_test = model(X_test)
```
```python
# Performance
train_error = loss_fn(pred_train, y_train)
test_error = loss_fn(pred_test, y_test)
print(f'train_error: {train_error:.2f}')
print(f'test_error: {test_error:.2f}')
```
<pre class="output">
train_error: 0.02
test_error: 0.01
</pre>


Regularization didn't make a difference in performance with this specific example because our data is generated from a perfect linear equation but for large realistic data, regularization can help our model generalize well.


<!-- Footer -->
<hr>
<div class="row mb-4">
  <div class="col-6 mr-auto">
    <a href="{{ page.course-url }}" class="btn btn-sm btn-outline-secondary"><i class="fas fa-sm fa-arrow-left mr-1"></i>Return to course</a>
  </div>
  <div class="col-6">
    <div class="float-right">
      <a href="{{ page.next-lesson-url }}" class="btn btn-sm btn-outline-secondary"><i class="fas fa-sm fa-arrow-right mr-1"></i>Next lesson</a>
    </div>
  </div>
</div>