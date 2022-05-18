---
template: lesson.html
title: Splitting a Dataset for Machine Learning
description: Appropriately splitting our dataset for training, validation and testing.
keywords: splitting, multiclass, multilabel, skmultilearn, data splits, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/mlops.png
repository: https://github.com/GokuMohandas/MLOps
notebook: https://colab.research.google.com/github/GokuMohandas/MLOps/blob/main/notebooks/tagifai.ipynb
---


{% include "styles/lesson.md" %}

## Intuition

To determine the efficacy of our models, we need to have an unbiased measuring approach. To do this, we split our dataset into `training`, `validation`, and `testing` data splits.

1. Use the training split to train the model.
  > Here the model will have access to both inputs and outputs to optimize its internal weights.
2. After each loop (epoch) of the training split, we will use the validation split to determine model performance.
  > Here the model will not use the outputs to optimize its weights but instead, we will use the performance to optimize training hyperparameters such as the learning rate, etc.
3. After training stops (epoch(s)), we will use the testing split to perform a one-time assessment of the model.
  > This is our best measure of how the model may behave on new, unseen data. Note that *training stops* when the performance improvement is not significant or any other stopping criteria that we may have specified.

!!! question "Creating proper data splits"
    What are the criteria we should focus on to ensure proper data splits?

    ??? quote "Show answer"

        - the dataset (and each data split) should be representative of data we will encounter
        - equal distributions of output values across all splits
        - shuffle your data if it's organized in a way that prevents input variance
        - avoid random shuffles if your task can suffer from data leaks (ex. `time-series`)

> We need to [clean](preprocessing.md) our data first before splitting, at least for the features that splitting depends on. So the process is more like: preprocessing (global, cleaning) → splitting → preprocessing (local, transformations).

## Label encoding
Before we split our dataset, we're going to encode our output labels where we'll be assigning each tag a unique index.

```python linenums="1"
import numpy as np
import random
```
```python linenums="1"
# Set seeds for reproducibility
seed = 42
np.random.seed(seed)
random.seed(seed)
```
```python linenums="1"
# Shuffle
df = df.sample(frac=1).reset_index(drop=True)
```

!!! question "Do we need to shuffle?"
    Why is it important that we shuffle our dataset?

    ??? quote "Show answer"
        We *need* to shuffle our data since our data is chronologically organized. The latest projects may have certain features or tags that are prevalent compared to earlier projects. If we don't shuffle before creating our data splits, then our model will only be trained on the earlier signals and fail to generalize. However, in other scenarios (ex. time-series forecasting), shuffling will lead do data leaks.

```python linenums="1"
# Get data
X = df.text.to_numpy()
y = df.tag
```

We'll be writing our own LabelEncoder which is based on scikit-learn's [implementation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html){:target="_blank"}.
```python linenums="1"
class LabelEncoder(object):
    """Label encoder for tag labels."""
    def __init__(self, class_to_index={}):
        self.class_to_index = class_to_index
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        self.classes = list(self.class_to_index.keys())

    def __len__(self):
        return len(self.class_to_index)

    def __str__(self):
        return f"<LabelEncoder(num_classes={len(self)})>"

    def fit(self, y):
        classes = np.unique(y)
        for i, class_ in enumerate(classes):
            self.class_to_index[class_] = i
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        self.classes = list(self.class_to_index.keys())
        return self

    def encode(self, y):
        encoded = np.zeros((len(y)), dtype=int)
        for i, item in enumerate(y):
            encoded[i] = self.class_to_index[item]
        return encoded

    def decode(self, y):
        classes = []
        for i, item in enumerate(y):
            classes.append(self.index_to_class[item])
        return classes

    def save(self, fp):
        with open(fp, "w") as fp:
            contents = {"class_to_index": self.class_to_index}
            json.dump(contents, fp, indent=4, sort_keys=False)

    @classmethod
    def load(cls, fp):
        with open(fp, "r") as fp:
            kwargs = json.load(fp=fp)
        return cls(**kwargs)
```

> If you're not familiar with the `@classmethod` decorator, learn more about it from our [Python lesson](../foundations/python.md#methods){:target="_blank"}.

```python linenums="1"
# Encode
label_encoder = LabelEncoder()
label_encoder.fit(y)
num_classes = len(label_encoder)
```
```python linenums="1"
label_encoder.class_to_index
```
<pre class="output">
{'computer-vision': 0,
 'mlops': 1,
 'natural-language-processing': 2,
 'other': 3}
</pre>
```python linenums="1"
label_encoder.index_to_class
```
<pre class="output">
{0: 'computer-vision',
 1: 'mlops',
 2: 'natural-language-processing',
 3: 'other'}
</pre>

```python linenums="1"
# Encode
label_encoder.encode(["computer-vision", "mlops", "mlops"])
```
<pre class="output">
array([0, 1, 1])
</pre>
```python linenums="1"
# Decode
label_encoder.decode(np.array([0, 1, 1]))
```
<pre class="output">
['computer-vision', 'mlops', 'mlops']
</pre>

```python linenums="1"
# Encode all our labels
y = label_encoder.encode(y)
print (y.shape)
```

## Naive split
It's time to split our dataset into three data splits for training, validation and testing.

```python linenums="1"
from sklearn.model_selection import train_test_split
```
```python linenums="1"
# Split sizes
train_size = 0.7
val_size = 0.15
test_size = 0.15
```

For our multi-class task (each input has one label), we want to ensure that each data split has similar class distributions. We can achieve this by specifying how to stratify the split by adding the [`stratify`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html){:target="_blank"} keyword argument.

```python linenums="1"
# Split (train)
X_train, X_, y_train, y_ = train_test_split(
    X, y, train_size=train_size, stratify=y)
```
```python linenums="1"
print (f"train: {len(X_train)} ({(len(X_train) / len(X)):.2f})\n"
       f"remaining: {len(X_)} ({(len(X_) / len(X)):.2f})")
```
<pre class="output">
train: 668 (0.70)
remaining: 287 (0.30)
</pre>
```python linenums="1"
# Split (test)
X_val, X_test, y_val, y_test = train_test_split(
    X_, y_, train_size=0.5, stratify=y_)
```
```python linenums="1"
print(f"train: {len(X_train)} ({len(X_train)/len(X):.2f})\n"
      f"val: {len(X_val)} ({len(X_val)/len(X):.2f})\n"
      f"test: {len(X_test)} ({len(X_test)/len(X):.2f})")
```
<pre class="output">
train: 668 (0.70)
val: 143 (0.15)
test: 144 (0.15)
</pre>
```python linenums="1"
# Get counts for each class
counts = {}
counts["train_counts"] = {tag: label_encoder.decode(y_train).count(tag) for tag in label_encoder.classes}
counts["val_counts"] = {tag: label_encoder.decode(y_val).count(tag) for tag in label_encoder.classes}
counts["test_counts"] = {tag: label_encoder.decode(y_test).count(tag) for tag in label_encoder.classes}
```
```python linenums="1"
# View distributions
pd.DataFrame({
    "train": counts["train_counts"],
    "val": counts["val_counts"],
    "test": counts["test_counts"]
}).T.fillna(0)
```
<div class="output_subarea output_html rendered_html"><div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>computer-vision</th>
      <th>mlops</th>
      <th>natural-language-processing</th>
      <th>other</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>train</th>
      <td>249</td>
      <td>55</td>
      <td>272</td>
      <td>92</td>
    </tr>
    <tr>
      <th>val</th>
      <td>53</td>
      <td>12</td>
      <td>58</td>
      <td>20</td>
    </tr>
    <tr>
      <th>test</th>
      <td>54</td>
      <td>12</td>
      <td>58</td>
      <td>20</td>
    </tr>
  </tbody>
</table>
</div></div>

It's hard to compare these because our train and test proportions are different. Let's see what the distribution looks like once we balance it out. What do we need to multiply our test ratio by so that we have the same amount as our train ratio?

$$ \alpha * N_{test} = N_{train} $$

$$ \alpha = \frac{N_{train}}{N_{test}} $$

```python linenums="1"
# Adjust counts across splits
for k in counts["val_counts"].keys():
    counts["val_counts"][k] = int(counts["val_counts"][k] * \
        (train_size/val_size))
for k in counts["test_counts"].keys():
    counts["test_counts"][k] = int(counts["test_counts"][k] * \
        (train_size/test_size))
```
```python linenums="1"
dist_df = pd.DataFrame({
    "train": counts["train_counts"],
    "val": counts["val_counts"],
    "test": counts["test_counts"]
}).T.fillna(0)
dist_df
```

<div class="output_subarea output_html rendered_html"><div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>computer-vision</th>
      <th>mlops</th>
      <th>natural-language-processing</th>
      <th>other</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>train</th>
      <td>249</td>
      <td>55</td>
      <td>272</td>
      <td>92</td>
    </tr>
    <tr>
      <th>val</th>
      <td>247</td>
      <td>56</td>
      <td>270</td>
      <td>93</td>
    </tr>
    <tr>
      <th>test</th>
      <td>252</td>
      <td>56</td>
      <td>270</td>
      <td>93</td>
    </tr>
  </tbody>
</table>
</div></div>

We can see how much deviance there is in our naive data splits by computing the standard deviation of each split's class counts from the mean (ideal split).

$$ \sigma = \sqrt{\frac{(x - \bar{x})^2}{N}} $$

```python linenums="1"
# Standard deviation
np.mean(np.std(dist_df.to_numpy(), axis=0))
```
<pre class="output">
0.9851056877051131
</pre>

```python linenums="1"
# Split DataFrames
train_df = pd.DataFrame({"text": X_train, "tags": label_encoder.decode(y_train)})
val_df = pd.DataFrame({"text": X_val, "tags": label_encoder.decode(y_val)})
test_df = pd.DataFrame({"text": X_test, "tags": label_encoder.decode(y_test)})
train_df.head()
```
<div class="output_subarea output_html rendered_html"><div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>laplacian pyramid reconstruction refinement se...</td>
      <td>computer-vision</td>
    </tr>
    <tr>
      <th>1</th>
      <td>extract stock sentiment news headlines project...</td>
      <td>natural-language-processing</td>
    </tr>
    <tr>
      <th>2</th>
      <td>big bad nlp database collection 400 nlp datase...</td>
      <td>natural-language-processing</td>
    </tr>
    <tr>
      <th>3</th>
      <td>job classification job classification done usi...</td>
      <td>natural-language-processing</td>
    </tr>
    <tr>
      <th>4</th>
      <td>optimizing mobiledet mobile deployments learn ...</td>
      <td>computer-vision</td>
    </tr>
  </tbody>
</table>
</div></div>

!!! tip "Multi-label classification"
    If we had a multi-label classification task, then we would've applied [iterative stratification](http://lpis.csd.auth.gr/publications/sechidis-ecmlpkdd-2011.pdf){:target="_blank"} via the [skmultilearn](http://scikit.ml/index.html){:target="_blank"} library, which essentially splits each input into subsets (where each label is considered individually) and then it distributes the samples starting with fewest "positive" samples and working up to the inputs that have the most labels.

    ```python
    from skmultilearn.model_selection import IterativeStratification
    def iterative_train_test_split(X, y, train_size):
        """Custom iterative train test split which
        'maintains balanced representation with respect
        to order-th label combinations.'
        """
        stratifier = IterativeStratification(
            n_splits=2, order=1, sample_distribution_per_fold=[1.0-train_size, train_size, ])
        train_indices, test_indices = next(stratifier.split(X, y))
        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]
        return X_train, X_test, y_train, y_test
    ```

    [Iterative stratification](http://scikit.ml/_modules/skmultilearn/model_selection/iterative_stratification.html#IterativeStratification){:target="_blank"} essentially creates splits while "trying to maintain balanced representation with respect to order-th label combinations". We used to an `order=1` for our iterative split which means we cared about providing representative distribution of each tag across the splits. But we can account for [higher-order](https://arxiv.org/abs/1704.08756){:target="_blank"} label relationships as well where we may care about the distribution of label combinations.

<!-- Citation -->
{% include "cite.md" %}
