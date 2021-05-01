---
template: lesson.html
title: Splitting a Dataset for Multilabel Classification
description: Appropriately splitting our dataset (multi-label) for training, validation and testing.
keywords: splitting, multilabel, skmultilearn, data splits, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/applied_ml.png
repository: https://github.com/GokuMohandas/MLOps
notebook: https://colab.research.google.com/github/GokuMohandas/MLOps/blob/main/notebooks/tagifai.ipynb
---


{% include "styles/lesson.md" %}

## Intuition

To determine the efficacy of our models, we need to have an unbiased measuring approach. To do this, we split our dataset into `training`, `validation`, and `testing` data splits. Here is the process:

1. Use the training split to train the model.
  > Here the model will have access to both inputs and outputs to optimize its internal weights.
2. After each loop (epoch) of the training split, we will use the validation split to determine model performance.
  > Here the model will not use the outputs to optimize its weights but instead, we will use the performance to optimize training hyperparameters such as the learning rate, etc.
3. After training stops (epoch(s)), we will use the testing split to perform a one-time assessment of the model.
  > This is our best measure of how the model may behave on new, unseen data. Note that *training stops* when the performance improvement is not significant or any other stopping criteria that we may have specified.

We need to ensure that our data is properly split so we can trust our evaluations. A few criteria are:

- the dataset (and each data split) should be representative of data we will encounter
- equal distributions of output values across all splits
- shuffle your data if it's organized in a way that prevents input variance
- avoid random shuffles if you task can suffer from data leaks (ex. `time-series`)

!!! note
    You need to [clean](preprocessing.md) your data first before splitting, at least for the features that splitting depends on. So the process is more like: preprocessing (global, cleaning) → splitting → preprocessing (local, transformations).

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

We *need* to shuffle our data since the latest projects are upfront and certain tags are trending now compared to a year ago. If we don't shuffle and create our data splits, then our model will only be trained on earlier tags and perform poorly on others.
```python linenums="1"
# Shuffle
df = df.sample(frac=1).reset_index(drop=True)
```
```python linenums="1"
# Get data
X = df.text.to_numpy()
y = df.tags
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
        classes = np.unique(list(itertools.chain.from_iterable(y)))
        for i, class_ in enumerate(classes):
            self.class_to_index[class_] = i
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        self.classes = list(self.class_to_index.keys())
        return self

    def encode(self, y):
        y_one_hot = np.zeros((len(y), len(self.class_to_index)), dtype=int)
        for i, item in enumerate(y):
            for class_ in item:
                y_one_hot[i][self.class_to_index[class_]] = 1
        return y_one_hot

    def decode(self, y):
        classes = []
        for i, item in enumerate(y):
            indices = np.where(item == 1)[0]
            classes.append([self.index_to_class[index] for index in indices])
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

!!! note
    If you're not familiar with the `@classmethod` decorator, learn more about it from our [Python lesson](../basics/python.md#methods){:target="_blank"}.

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
{'attention': 0,
 'autoencoders': 1,
 'computer-vision': 2,
 ...
 'transfer-learning': 31,
 'transformers': 32,
 'unsupervised-learning': 33,
 'wandb': 34}
</pre>

Since we're dealing with multilabel classification, we're going to convert our label indices into one-hot representation where each input's set of labels is represented by a binary array.

```python linenums="1"
# Sample
label_encoder.encode([["attention", "data-augmentation"]])
```
<pre class="output">
array([[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
</pre>
```python linenums="1"
# Encode all our labels
y = label_encoder.encode(y)
```

## Naive split
For traditional `multi-class` tasks (each input has one label), we want to ensure that each data split has similar class distributions. However, our task is `multi-label` classification (an input can have many labels) which complicates the stratification process.

First, we'll naively split our dataset randomly and show the large deviations between the (adjusted) class distributions across the splits. We'll use scikit-learn's [`train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html){:target="_blank"} function to do the splits.

```python linenums="1"
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection.measures import get_combination_wise_output_matrix
```
```python linenums="1"
# Split sizes
train_size = 0.7
val_size = 0.15
test_size = 0.15
```
```python linenums="1"
# Split (train)
X_train, X_, y_train, y_ = train_test_split(X, y, train_size=train_size)
```
```python linenums="1"
print (f"train: {len(X_train)} ({(len(X_train) / len(X)):.2f})\n"
       f"remaining: {len(X_)} ({(len(X_) / len(X)):.2f})")
```
<pre class="output">
train: 1010 (0.70)
remaining: 434 (0.30)
</pre>
```python linenums="1"
# Split (test)
X_val, X_test, y_val, y_test = train_test_split(
    X_, y_, train_size=0.5)
```
```python linenums="1"
print(f"train: {len(X_train)} ({len(X_train)/len(X):.2f})\n"
      f"val: {len(X_val)} ({len(X_val)/len(X):.2f})\n"
      f"test: {len(X_test)} ({len(X_test)/len(X):.2f})")
```
<pre class="output">
train: 1010 (0.70)
val: 217 (0.15)
test: 217 (0.15)
</pre>
```python linenums="1"
# Get counts for each class
counts = {}
counts["train_counts"] = Counter(str(combination) for row in get_combination_wise_output_matrix(
    y_train, order=1) for combination in row)
counts["val_counts"] = Counter(str(combination) for row in get_combination_wise_output_matrix(
    y_val, order=1) for combination in row)
counts["test_counts"] = Counter(str(combination) for row in get_combination_wise_output_matrix(
    y_test, order=1) for combination in row)
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
      <th>(15,)</th>
      <th>(19,)</th>
      <th>(33,)</th>
      <th>(21,)</th>
      <th>(32,)</th>
      <th>(14,)</th>
      <th>(2,)</th>
      <th>(20,)</th>
      <th>(24,)</th>
      <th>(5,)</th>
      <th>(16,)</th>
      <th>(9,)</th>
      <th>(22,)</th>
      <th>(12,)</th>
      <th>(23,)</th>
      <th>(0,)</th>
      <th>(25,)</th>
      <th>(34,)</th>
      <th>(28,)</th>
      <th>(10,)</th>
      <th>(26,)</th>
      <th>(27,)</th>
      <th>(13,)</th>
      <th>(17,)</th>
      <th>(3,)</th>
      <th>(1,)</th>
      <th>(7,)</th>
      <th>(11,)</th>
      <th>(18,)</th>
      <th>(4,)</th>
      <th>(6,)</th>
      <th>(29,)</th>
      <th>(8,)</th>
      <th>(31,)</th>
      <th>(30,)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>train</th>
      <td>314</td>
      <td>37</td>
      <td>26</td>
      <td>26</td>
      <td>145</td>
      <td>33</td>
      <td>274</td>
      <td>191</td>
      <td>41</td>
      <td>55</td>
      <td>26</td>
      <td>56</td>
      <td>34</td>
      <td>41</td>
      <td>40</td>
      <td>90</td>
      <td>44</td>
      <td>28</td>
      <td>136</td>
      <td>44</td>
      <td>30</td>
      <td>31</td>
      <td>63</td>
      <td>45</td>
      <td>64</td>
      <td>27</td>
      <td>50</td>
      <td>34</td>
      <td>21</td>
      <td>33</td>
      <td>24</td>
      <td>24</td>
      <td>32</td>
      <td>33</td>
      <td>23</td>
    </tr>
    <tr>
      <th>val</th>
      <td>58</td>
      <td>8</td>
      <td>4</td>
      <td>2</td>
      <td>29</td>
      <td>8</td>
      <td>53</td>
      <td>33</td>
      <td>7</td>
      <td>9</td>
      <td>4</td>
      <td>11</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
      <td>20</td>
      <td>7</td>
      <td>2</td>
      <td>42</td>
      <td>13</td>
      <td>12</td>
      <td>4</td>
      <td>14</td>
      <td>14</td>
      <td>17</td>
      <td>3</td>
      <td>7</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>5</td>
      <td>7</td>
      <td>10</td>
      <td>7</td>
      <td>4</td>
    </tr>
    <tr>
      <th>test</th>
      <td>57</td>
      <td>6</td>
      <td>9</td>
      <td>4</td>
      <td>22</td>
      <td>10</td>
      <td>61</td>
      <td>34</td>
      <td>9</td>
      <td>11</td>
      <td>3</td>
      <td>11</td>
      <td>6</td>
      <td>4</td>
      <td>8</td>
      <td>10</td>
      <td>9</td>
      <td>9</td>
      <td>35</td>
      <td>7</td>
      <td>6</td>
      <td>5</td>
      <td>16</td>
      <td>10</td>
      <td>25</td>
      <td>11</td>
      <td>16</td>
      <td>11</td>
      <td>6</td>
      <td>5</td>
      <td>5</td>
      <td>9</td>
      <td>9</td>
      <td>6</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div></div>

It's hard to compare these because our train and test proportions are different. Let's see what the distribution looks like once we balance it out. What do we need to multiply our test ratio by so that we have the same amount as our train ratio?

$$ \alpha * N_{test} = N_{train} $$

$$ \alpha = \frac{N_{train}}{N_{test}} $$


<div class="output_subarea output_html rendered_html"><div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>(15,)</th>
      <th>(19,)</th>
      <th>(33,)</th>
      <th>(21,)</th>
      <th>(32,)</th>
      <th>(14,)</th>
      <th>(2,)</th>
      <th>(20,)</th>
      <th>(24,)</th>
      <th>(5,)</th>
      <th>(16,)</th>
      <th>(9,)</th>
      <th>(22,)</th>
      <th>(12,)</th>
      <th>(23,)</th>
      <th>(0,)</th>
      <th>(25,)</th>
      <th>(34,)</th>
      <th>(28,)</th>
      <th>(10,)</th>
      <th>(26,)</th>
      <th>(27,)</th>
      <th>(13,)</th>
      <th>(17,)</th>
      <th>(3,)</th>
      <th>(1,)</th>
      <th>(7,)</th>
      <th>(11,)</th>
      <th>(18,)</th>
      <th>(4,)</th>
      <th>(6,)</th>
      <th>(29,)</th>
      <th>(8,)</th>
      <th>(31,)</th>
      <th>(30,)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>train</th>
      <td>314</td>
      <td>37</td>
      <td>26</td>
      <td>26</td>
      <td>145</td>
      <td>33</td>
      <td>274</td>
      <td>191</td>
      <td>41</td>
      <td>55</td>
      <td>26</td>
      <td>56</td>
      <td>34</td>
      <td>41</td>
      <td>40</td>
      <td>90</td>
      <td>44</td>
      <td>28</td>
      <td>136</td>
      <td>44</td>
      <td>30</td>
      <td>31</td>
      <td>63</td>
      <td>45</td>
      <td>64</td>
      <td>27</td>
      <td>50</td>
      <td>34</td>
      <td>21</td>
      <td>33</td>
      <td>24</td>
      <td>24</td>
      <td>32</td>
      <td>33</td>
      <td>23</td>
    </tr>
    <tr>
      <th>val</th>
      <td>270</td>
      <td>37</td>
      <td>18</td>
      <td>9</td>
      <td>135</td>
      <td>37</td>
      <td>247</td>
      <td>154</td>
      <td>32</td>
      <td>42</td>
      <td>18</td>
      <td>51</td>
      <td>42</td>
      <td>46</td>
      <td>51</td>
      <td>93</td>
      <td>32</td>
      <td>9</td>
      <td>196</td>
      <td>60</td>
      <td>56</td>
      <td>18</td>
      <td>65</td>
      <td>65</td>
      <td>79</td>
      <td>14</td>
      <td>32</td>
      <td>28</td>
      <td>14</td>
      <td>14</td>
      <td>23</td>
      <td>32</td>
      <td>46</td>
      <td>32</td>
      <td>18</td>
    </tr>
    <tr>
      <th>test</th>
      <td>266</td>
      <td>28</td>
      <td>42</td>
      <td>18</td>
      <td>102</td>
      <td>46</td>
      <td>284</td>
      <td>158</td>
      <td>42</td>
      <td>51</td>
      <td>14</td>
      <td>51</td>
      <td>28</td>
      <td>18</td>
      <td>37</td>
      <td>46</td>
      <td>42</td>
      <td>42</td>
      <td>163</td>
      <td>32</td>
      <td>28</td>
      <td>23</td>
      <td>74</td>
      <td>46</td>
      <td>116</td>
      <td>51</td>
      <td>74</td>
      <td>51</td>
      <td>28</td>
      <td>23</td>
      <td>23</td>
      <td>42</td>
      <td>42</td>
      <td>28</td>
      <td>32</td>
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
9.936725114942407
</pre>

!!! note
    For simple multiclass classification, you can specify how to stratify the split by adding the [`stratify`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html){:target="_blank"} keyword argument. But our task is multilabel classification, so we'll need to use other techniques to create even splits.


## Stratified split
Now we'll apply [iterative stratification](http://lpis.csd.auth.gr/publications/sechidis-ecmlpkdd-2011.pdf){:target="_blank"} via the [skmultilearn](http://scikit.ml/index.html){:target="_blank"} library, which essentially splits each input into subsets (where each label is considered individually) and then it distributes the samples starting with fewest "positive" samples and working up to the inputs that have the most labels.

```python linenums="1"
from skmultilearn.model_selection import IterativeStratification
```
```python linenums="1"
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
```python linenums="1"
# Get data
X = df.text.to_numpy()
y = df.tags
```
```python linenums="1"
# Binarize y
label_encoder = LabelEncoder()
label_encoder.fit(y)
y = label_encoder.encode(y)
```
```python linenums="1"
# Split
X_train, X_, y_train, y_ = iterative_train_test_split(
    X, y, train_size=train_size)
X_val, X_test, y_val, y_test = iterative_train_test_split(
    X_, y_, train_size=0.5)
```
```python linenums="1"
print(f"train: {len(X_train)} ({len(X_train)/len(X):.2f})\n"
      f"val: {len(X_val)} ({len(X_val)/len(X):.2f})\n"
      f"test: {len(X_test)} ({len(X_test)/len(X):.2f})")
```
<pre class="output">
train: 1000 (0.69)
val: 214 (0.15)
test: 230 (0.16)
</pre>

Let's see what the adjusted counts look like for these stratified data splits.
```python linenums="1"
# Get counts for each class
counts = {}
counts["train_counts"] = Counter(str(combination) for row in get_combination_wise_output_matrix(
    y_train, order=1) for combination in row)
counts["val_counts"] = Counter(str(combination) for row in get_combination_wise_output_matrix(
    y_val, order=1) for combination in row)
counts["test_counts"] = Counter(str(combination) for row in get_combination_wise_output_matrix(
    y_test, order=1) for combination in row)
```
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
      <th>(2,)</th>
      <th>(4,)</th>
      <th>(15,)</th>
      <th>(14,)</th>
      <th>(30,)</th>
      <th>(34,)</th>
      <th>(1,)</th>
      <th>(26,)</th>
      <th>(32,)</th>
      <th>(20,)</th>
      <th>(33,)</th>
      <th>(25,)</th>
      <th>(17,)</th>
      <th>(21,)</th>
      <th>(0,)</th>
      <th>(24,)</th>
      <th>(27,)</th>
      <th>(6,)</th>
      <th>(13,)</th>
      <th>(3,)</th>
      <th>(5,)</th>
      <th>(16,)</th>
      <th>(9,)</th>
      <th>(19,)</th>
      <th>(7,)</th>
      <th>(28,)</th>
      <th>(11,)</th>
      <th>(22,)</th>
      <th>(8,)</th>
      <th>(29,)</th>
      <th>(23,)</th>
      <th>(31,)</th>
      <th>(10,)</th>
      <th>(18,)</th>
      <th>(12,)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>train</th>
      <td>272.0</td>
      <td>29.0</td>
      <td>300.0</td>
      <td>36.0</td>
      <td>24.0</td>
      <td>27.0</td>
      <td>29.0</td>
      <td>32.0</td>
      <td>145.0</td>
      <td>181.0</td>
      <td>27.0</td>
      <td>42.0</td>
      <td>49.0</td>
      <td>27.0</td>
      <td>84.0</td>
      <td>40.0</td>
      <td>28.0</td>
      <td>24.0</td>
      <td>65.0</td>
      <td>74.0</td>
      <td>52.0</td>
      <td>19.0</td>
      <td>55.0</td>
      <td>36.0</td>
      <td>51.0</td>
      <td>149.0</td>
      <td>30.0</td>
      <td>34.0</td>
      <td>38.0</td>
      <td>26.0</td>
      <td>41.0</td>
      <td>32.0</td>
      <td>45.0</td>
      <td>21.0</td>
      <td>38.0</td>
    </tr>
    <tr>
      <th>val</th>
      <td>270.0</td>
      <td>32.0</td>
      <td>298.0</td>
      <td>28.0</td>
      <td>23.0</td>
      <td>28.0</td>
      <td>32.0</td>
      <td>37.0</td>
      <td>112.0</td>
      <td>177.0</td>
      <td>28.0</td>
      <td>42.0</td>
      <td>46.0</td>
      <td>0.0</td>
      <td>84.0</td>
      <td>51.0</td>
      <td>28.0</td>
      <td>23.0</td>
      <td>74.0</td>
      <td>74.0</td>
      <td>56.0</td>
      <td>18.0</td>
      <td>51.0</td>
      <td>32.0</td>
      <td>51.0</td>
      <td>149.0</td>
      <td>46.0</td>
      <td>32.0</td>
      <td>32.0</td>
      <td>32.0</td>
      <td>42.0</td>
      <td>32.0</td>
      <td>46.0</td>
      <td>23.0</td>
      <td>42.0</td>
    </tr>
    <tr>
      <th>test</th>
      <td>270.0</td>
      <td>23.0</td>
      <td>303.0</td>
      <td>42.0</td>
      <td>23.0</td>
      <td>28.0</td>
      <td>23.0</td>
      <td>37.0</td>
      <td>126.0</td>
      <td>182.0</td>
      <td>28.0</td>
      <td>42.0</td>
      <td>46.0</td>
      <td>23.0</td>
      <td>84.0</td>
      <td>28.0</td>
      <td>28.0</td>
      <td>23.0</td>
      <td>56.0</td>
      <td>74.0</td>
      <td>51.0</td>
      <td>46.0</td>
      <td>56.0</td>
      <td>37.0</td>
      <td>51.0</td>
      <td>149.0</td>
      <td>51.0</td>
      <td>37.0</td>
      <td>28.0</td>
      <td>32.0</td>
      <td>42.0</td>
      <td>32.0</td>
      <td>42.0</td>
      <td>18.0</td>
      <td>37.0</td>
    </tr>
  </tbody>
</table>
</div></div>
```python linenums="1"
dist_df = pd.DataFrame({
    "train": counts["train_counts"],
    "val": counts["val_counts"],
    "test": counts["test_counts"]
}).T.fillna(0)
```
```python linenums="1"
# Standard deviation
np.mean(np.std(dist_df.to_numpy(), axis=0))
```
<pre class="output">
3.142338654518357
</pre>

The standard deviation is much better but not 0 (perfect splits) because keep in mind that an input can have any combination of of classes yet each input can only belong in one of the data splits.

!!! note
    [Iterative stratification](http://scikit.ml/_modules/skmultilearn/model_selection/iterative_stratification.html#IterativeStratification){:target="_blank"} essentially creates splits while "trying to maintain balanced representation with respect to order-th label combinations". We used to an `order=1` for our iterative split which means we cared about providing representative distribution of each tag across the splits. But we can account for [higher-order](https://arxiv.org/abs/1704.08756){:target="_blank"} label relationships as well where we may care about the distribution of label combinations.

## Resources
- [How (and why) to create a good validation set](https://www.fast.ai/2017/11/13/validation-sets/){:target="_blank"}

<!-- Citation -->
{% include "cite.md" %}
