---
template: lesson.html
title: Distributed Data Processing
description: Performing our data processing operations in a distributed manner.
keywords: distributed systems, scale, preprocessing, preparation, cleaning, feature engineering, filtering, transformations, mlops, machine learning
image: https://madewithml.com/static/images/mlops.png
repository: https://github.com/GokuMohandas/Made-With-ML
notebook: https://github.com/GokuMohandas/Made-With-ML/blob/main/notebooks/madewithml.ipynb
---

{% include "styles/lesson.md" %}

## Intuition

So far we've performed our data processing operations on a single machine. Our dataset was able to fit into a single Pandas DataFrame and we were able to perform our operations in a single Python process. But what if our dataset was too large to fit into a single machine? We would need to distribute our data processing operations across multiple machines. And with the increasing trend in ML for larger unstructured datasets and larger models (LLMs), we can quickly outgrow our single machine constraints and will need to go distributed.

!!! note
    Our dataset is intentionally small for this course so that we can quickly execute the code. But with our distributed set up in this lesson, we can easily switch to a mcuh larger dataset and the code will continue to execute perfectly. And if we add more compute resources, we can scale our data processing operations to be even faster with no changes to our code.

## Implementation

There are many frameworks for distributed computing, such as [Ray](https://docs.ray.io/en/latest/){:target="_blank"}, [Dask](https://www.dask.org/){:target="_blank"}, [Modin](https://github.com/modin-project/modin){:target="_blank"}, [Spark](https://spark.apache.org/){:target="_blank"}, etc. All of these are great options but for our application we want to choose a framework that is will allow us to scale our data processing operations with **minimal changes to our existing code** and **all in Python**. We also want to choose a framework that will integrate well when we want to distributed our downstream workloads (training, tuning, serving, etc.).

To address these needs, we'll be using Ray, a distributed computing framework that makes it easy to scale your Python applications. It's a general purpose framework that can be used for a variety of applications but we'll be using it for our [data processing](https://docs.ray.io/en/latest/data/data.html){:target="_blank"} operations first (and more later). And it also has great integrations with the previously mentioned distributed data processing frameworks ([Dask](https://docs.ray.io/en/latest/ray-more-libs/dask-on-ray.html){:target="_blank"}, [Modin](https://docs.ray.io/en/latest/ray-more-libs/modin/index.html){:target="_blank"}, [Spark](https://docs.ray.io/en/latest/ray-more-libs/raydp.html){:target="_blank"}).

<div class="ai-center-all">
  <img src="/static/images/mlops/ray/data.svg" width="700" alt="ray data">
</div>

### Setup

The only setup we have to do is set Ray to preserve order when acting on our data. This is important for ensuring reproducible and deterministic results.

```python linenums="1"
ray.data.DatasetContext.get_current().execution_options.preserve_order = True  # deterministic
```

### Ingestion

We'll start by ingesting our dataset. Ray has a range of [input/output functions](https://docs.ray.io/en/latest/data/api/input_output.html){:target="_blank"} that supports all major data formats and sources.

```python linenums="1"
# Data ingestion
ds = ray.data.read_csv(DATASET_LOC)
ds = ds.random_shuffle(seed=1234)
ds.take(1)
```

<pre class="output">
[{'id': 2166,
  'created_on': datetime.datetime(2020, 8, 17, 5, 19, 41),
  'title': 'Pix2Pix',
  'description': 'Tensorflow 2.0 Implementation of the paper Image-to-Image Translation using Conditional GANs by Philip Isola, Jun-Yan Zhu, Tinghui Zhou and Alexei A. Efros.',
  'tag': 'computer-vision'}]
</pre>

### Splitting

Next, we'll split our dataset into our training and validation splits. Ray has a built-in [`train_test_split`](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.train_test_split.html){:target="_blank"} function but we're using a [modified version](https://github.com/GokuMohandas/Made-With-ML/blob/main/madewithml/data.py){:target="_blank"} so that we can stratify our split based on the `tag` column.

```python linenums="1"
import sys
sys.path.append("..")
from madewithml.data import stratify_split
```

```python linenums="1"
# Split dataset
test_size = 0.2
train_ds, val_ds = stratify_split(ds, stratify="tag", test_size=test_size)
```

### Preprocessing

And finally, we're ready to preprocess our data splits. One of the advantages of using Ray is that we won't have to change anything to our original Pandas-based preprocessing function we implemented in the [previous lesson](preprocessing.md#best-practices){:target="_blank"}. Instead, we can use it directly with Ray's [`map_batches`](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.map_batches.html){:target="_blank"} utility to *map* our preprocessing function across *batches* in our data in a distributed manner.

```python linenums="1"
# Mapping
tags = train_ds.unique(column="tag")
class_to_index = {tag: i for i, tag in enumerate(tags)}
```

```python linenums="1"
# Distributed preprocessing
sample_ds = train_ds.map_batches(
  preprocess,
  fn_kwargs={"class_to_index": class_to_index},
  batch_format="pandas")
sample_ds.show(1)
```

<pre class="output">
{'ids': array([  102,  5800, 14982,  1422,  4958, 14982,   437,  3294,  3577,
       12574,  2747,  1262,  7222,   103,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0]), 'masks': array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0]), 'targets': 2}
</pre>

<!-- Course signup -->
{% include "templates/signup.md" %}

<!-- Citation -->
{% include "templates/cite.md" %}