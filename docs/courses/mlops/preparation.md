---
template: lesson.html
title: Data Preparation
description: Preparing our dataset by ingesting and splitting it.
keywords: preparation, ingestion, splitting, mlops, machine learning
image: https://madewithml.com/static/images/mlops.png
repository: https://github.com/GokuMohandas/Made-With-ML
notebook: https://github.com/GokuMohandas/Made-With-ML/blob/main/notebooks/madewithml.ipynb
---

{% include "styles/lesson.md" %}

## Intuition

We'll start by first preparing our data by ingesting it from source and splitting it into training, validation and test data splits.

### Ingestion

Our data could reside in many different places (databases, files, etc.) and exist in different formats (CSV, JSON, Parquet, etc.). For our application, we'll load the data from a CSV file to a [Pandas DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html){:target="_blank"} using the [`read_csv`](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html){:target="_blank"} function.

> Here is a quick refresher on the [Pandas](../foundations/pandas.md){:target="_blank"} library.

```python linenums="1"
import pandas as pd
```

```python linenums="1"
# Data ingestion
DATASET_LOC = "https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/dataset.csv"
df = pd.read_csv(DATASET_LOC)
df.head()
```

<div class="output_subarea output_html rendered_html output_result" dir="auto"><div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align:right">
      <th></th>
      <th>id</th>
      <th>created_on</th>
      <th>title</th>
      <th>description</th>
      <th>tag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>2020-02-20 06:43:18</td>
      <td>Comparison between YOLO and RCNN on real world...</td>
      <td>Bringing theory to experiment is cool. We can ...</td>
      <td>computer-vision</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>2020-02-20 06:47:21</td>
      <td>Show, Infer &amp; Tell: Contextual Inference for C...</td>
      <td>The beauty of the work lies in the way it arch...</td>
      <td>computer-vision</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>2020-02-24 16:24:45</td>
      <td>Awesome Graph Classification</td>
      <td>A collection of important graph embedding, cla...</td>
      <td>other</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15</td>
      <td>2020-02-28 23:55:26</td>
      <td>Awesome Monte Carlo Tree Search</td>
      <td>A curated list of Monte Carlo tree search pape...</td>
      <td>other</td>
    </tr>
    <tr>
      <th>4</th>
      <td>25</td>
      <td>2020-03-07 23:04:31</td>
      <td>AttentionWalk</td>
      <td>A PyTorch Implementation of "Watch Your Step: ...</td>
      <td>other</td>
    </tr>
  </tbody>
</table>
</div></div>

> In our [data engineering lesson](data-engineering.md){:target="_blank"} we'll look at how to continually ingest data from more complex sources (ex. data warehouses)

### Splitting

Next, we need to split our training dataset into `train` and `val` data splits.

1. Use the `train` split to train the model.
  > Here the model will have access to both inputs (features) and outputs (labels) to optimize its internal weights.
2. After each iteration (epoch) through the training split, we will use the `val` split to determine the model's performance.
  > Here the model will not use the labels to optimize its weights but instead, we will use the validation performance to optimize training hyperparameters such as the learning rate, etc.
3. Finally, we will use a separate holdout [`test` dataset](https://github.com/GokuMohandas/Made-With-ML/blob/main/datasets/holdout.csv){:target="_blank"} to determine the model's performance after training.
  > This is our best measure of how the model may behave on new, unseen data that is from a similar distribution to our training dataset.

!!! tip
    For our application, we will have a [training dataset](https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/dataset.csv){:target="_blank"} to split into `train` and `val` splits and a **separate** [testing dataset](https://github.com/GokuMohandas/Made-With-ML/blob/main/datasets/holdout.csv){:target="_blank"} for the `test` set. While we could have one large dataset and split that into the three splits, it's a good idea to have a separate test dataset. Over time, our training data may grow and our test splits will look different every time. This will make it difficult to compare models against other models and against each other.

We can view the class counts in our dataset by using the [`pandas.DataFrame.value_counts`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.value_counts.html){:target="_blank"} function:

```python linenums="1"
from sklearn.model_selection import train_test_split
```

```python linenums="1"
# Value counts
df.tag.value_counts()
```

<pre class="output">
tag
natural-language-processing    310
computer-vision                285
other                          106
mlops                           63
Name: count, dtype: int64
</pre>

For our multi-class task (where each project has exactly one tag), we want to ensure that the data splits have similar class distributions. We can achieve this by specifying how to stratify the split by using the [`stratify`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html){:target="_blank"} keyword argument with sklearn's [`train_test_split()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html){:target="_blank"} function.

!!! question "Creating proper data splits"
    What are the criteria we should focus on to ensure proper data splits?

    ??? quote "Show answer"

        - the dataset (and each data split) should be representative of data we will encounter
        - equal distributions of output values across all splits
        - shuffle your data if it's organized in a way that prevents input variance
        - avoid random shuffles if your task can suffer from data leaks (ex. `time-series`)

```python linenums="1"
# Split dataset
test_size = 0.2
train_df, val_df = train_test_split(df, stratify=df.tag, test_size=test_size, random_state=1234)
```

How can we validate that our data splits have similar class distributions? We can view the frequency of each class in each split:

```python linenums="1"
# Train value counts
train_df.tag.value_counts()
```

<pre class="output">
tag
natural-language-processing    248
computer-vision                228
other                           85
mlops                           50
Name: count, dtype: int64
</pre>

Before we view our validation split's class counts, recall that our validation split is only `test_size` of the entire dataset. So we need to adjust the value counts so that we can compare it to the training split's class counts.


$$ \alpha * N_{test} = N_{train} $$

$$ N_{train} = 1 - N_{test} $$

$$ \alpha = \frac{N_{train}}{N_{test}} = \frac{1 - N_{test}}{N_{test}} $$

```python linenums="1"
# Validation (adjusted) value counts
val_df.tag.value_counts() * int((1-test_size) / test_size)
```

<pre class="output">
tag
natural-language-processing    248
computer-vision                228
other                           84
mlops                           52
Name: count, dtype: int64
</pre>

These adjusted counts looks very similar to our train split's counts. Now we're ready to [explore](exploratory-data-analysis.md){:target="_blank"} our dataset!

<!-- Course signup -->
{% include "templates/signup.md" %}

<!-- Citation -->
{% include "templates/cite.md" %}