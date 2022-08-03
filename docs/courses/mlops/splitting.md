---
template: lesson.html
title: Splitting a Dataset for Machine Learning
description: Appropriately splitting our dataset for training, validation and testing.
keywords: splitting, multiclass, multilabel, skmultilearn, data splits, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/mlops.png
repository: https://github.com/GokuMohandas/mlops-course
notebook: https://github.com/GokuMohandas/mlops-course/blob/main/notebooks/tagifai.ipynb
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

## Naive split
We'll start by splitting our dataset into three data splits for training, validation and testing.

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
<div class="output_subarea output_html rendered_html ai-center-all"><div>
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

<div class="output_subarea output_html rendered_html ai-center-all"><div>
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
train_df = pd.DataFrame({"text": X_train, "tag": label_encoder.decode(y_train)})
val_df = pd.DataFrame({"text": X_val, "tag": label_encoder.decode(y_val)})
test_df = pd.DataFrame({"text": X_test, "tag": label_encoder.decode(y_test)})
train_df.head()
```
<div class="output_subarea output_html rendered_html ai-center-all"><div>
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
      <td>big bad nlp database collection 400 nlp datasets...</td>
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
