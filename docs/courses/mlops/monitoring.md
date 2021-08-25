---
template: lesson.html
title: Monitoring ML Systems
description: Monitoring ML systems to identify and mitigate model performance decay stemming from drift.
keywords: monitoring, drift, data drift, concept drift, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/mlops.png
repository: https://github.com/GokuMohandas/MLOps
notebook: https://colab.research.google.com/github/GokuMohandas/MLOps/blob/main/notebooks/monitoring.ipynb
---

## Intuition

Even though we've trained and thoroughly evaluated our model, the real work begins. This is one of the fundamental differences between traditional software engineering and ML development. Traditionally, with rule based, deterministic, software, the brunt of the work occurs at the initial stage and once deployed, our system works as we've defined it. But with machine learning, we haven't explicitly defined how something works but allowed data to architect a probabilistic solution. This approach is subject to natural performance degradation over time, as well as unintended behavior, since the data the model sees will be different from what it has been trained on. This isn't something we should be trying to avoid but rather understand and mitigate as much as possible. In this lesson, we'll understand the short comings (delayed outcomes, too late, etc.) from attempting to capture performance degradation in order to motivate the need for [drift](#drift) detection.

!!! note
    Testing and monitoring share a lot of similarities, such as ensuring that certain expectations around data completeness, schema adherence, etc. are met. However, a key distinction is that monitoring involves comparing data live, streaming distributions from production to fixed/sliding reference (typically training data) distributions.

## System health

The first step to insure that our model is performing well is to ensure that the actual system is up and running as it should. This can include metrics specific to service requests such as latency, throughput, error rates, etc. as well as infrastructure utilization such as CPU/GPU utilization, memory, etc.

<div class="ai-center-all">
    <a href="https://miro.medium.com/max/2400/1*DQdiQupXSSd3fldg9eAQjA.jpeg" target="_blank"><img width="600" src="https://miro.medium.com/max/2400/1*DQdiQupXSSd3fldg9eAQjA.jpeg"></a>
</div>

Fortunately, most cloud providers and even orchestration layers will provide this insight into our system's health for free through a dashboard. In the event we don't, we can easily use [Grafana](https://grafana.com/){:target="_blank"}, [Datadog](https://www.datadoghq.com/){:target="_blank"}, etc. to ingest system performance metrics from logs to create a customized dashboard and set alerts.

## Performance

Unfortunately, just monitoring the system's health won't be enough to capture the underlying issues with our model. So, naturally, the next layer of metrics to monitor involves the model's performance. These could be quantitative evaluation metrics that we used during model evaluation (accuracy, precision, f1, etc.) but also key business metrics that the model influences (ROI, click rate, etc.).

It's usually never enough to just analyze the coarse-grained (rolling) performance metrics across the entire span of time since the model has been deployed. Instead, we should inspect performance across a period of time that's significant for our application (ex. daily). These fine-grained metrics might be more indicative of our system's health and we might be able to identify issues faster by not undermining them with historical data.

> All the code accompanying this lesson can be found in this [notebook](https://colab.research.google.com/github/GokuMohandas/MLOps/blob/main/notebooks/monitoring.ipynb){:target="_blank"}.

```python linenums="1"
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()
```
```python linenums="1"
# Generate data
hourly_f1 = list(np.random.randint(94, 98, 24*20)) + \
            list(np.random.randint(92, 96, 24*5)) + \
            list(np.random.randint(88, 96, 24*5)) + \
            list(np.random.randint(86, 92, 24*5))
```
```python linenums="1"
# Rolling f1
rolling_f1 = [np.mean(hourly_f1[:n]) for n in range(1, len(hourly_f1)+1)]
print (f"Average rolling f1 on the last day: {np.mean(rolling_f1[-24:]):.1f}")
```
<pre class="output">
Average rolling f1 on the last day: 93.7
</pre>
```python linenums="1"
# Window f1
window_size = 24
window_f1 = np.convolve(hourly_f1, np.ones(window_size)/window_size, mode="valid")
print (f"Average window f1 on the last day: {np.mean(window_f1[-24:]):.1f}")
```
<pre class="output">
Average window f1 on the last day: 88.8
</pre>
```python linenums="1"
plt.ylim([80, 100])
plt.hlines(y=90, xmin=0, xmax=len(hourly_f1), colors="blue", linestyles="dashed", label="threshold")
plt.plot(rolling_f1, label="rolling")
plt.plot(window_f1, label="window")
plt.legend()
```

<div class="ai-center-all">
    <img width="500" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/monitoring/performance_drift.png">
</div>

!!! note
    We may need to monitor metrics at various window sizes to catch performance degradation as soon as possible. Here we're monitoring the overall f1 but we can do the same for slices of data, individual classes, etc. For example, if we monitor the performance on a specific tag, we may be able to quickly catch new algorithms that were released for that tag (ex. new transformer architecture).

However, there are two main obstacles with this approach:

- **Delayed outcomes**: we may not always have the ground-truth outcomes available to determine the model's performance on production inputs. This is especially true if there is significant lag or annotation is required on the real-world data. To mitigate this, we could
    - devise a **proxy signal** that can help us *estimate* the model's performance. For example, in our tag prediction task, we could use the actual tags that an author attributes to a project as the intermediary labels until we have verified labels from an annotation pipeline.
    - use the statistical features of our inputs and predicted outputs as a indirect [measure](#measuring-drfit).
- **Too late**: if we wait to catch the model decay based on the performance, it may have already cause significant damage to downstream business pipelines that are dependent on it. We need to employ more fine-grained monitoring to identify the *sources* of model drift prior to actual performance degradation.

## Drift

We need to first understand the different types of issues that can cause our model's performance to decay (model drift). The best way to do this is to look at all the moving pieces of what we're trying to model and how each one can experience drift.

<center>

| Entity               | Description                              | Drift                                                               |
| :------------------- | :--------------------------------------- | :------------------------------------------------------------------ |
| $X$                  | inputs (features)                        | data drift     $\rightarrow P(X) \neq P_{ref}(X)$                 |
| $y$                  | outputs (ground-truth)                   | target drift   $\rightarrow P(y) \neq P_{ref}(y)$                 |
| $P(y \vert X)$       | actual relationship between $X$ and $y$  | concept drift  $\rightarrow P(y \vert X) \neq P_{ref}(y \vert X)$ |

</center>

### Data drift

Data drift, also known as feature drift or covariate shift, occurs when the distribution of the *production* data is different from the *training* data. The model is not equipped to deal with this drift in the feature space and so, it's predictions may not be reliable. The actual cause of drift can be attributed to natural changes in the real-world but also to systemic issues such as missing data, pipeline errors, schemas changes, etc. It's important to inspect the drifted data and trace it back along it's pipeline to identify when and where the drift was introduced.

!!! warning
    Besides just looking at the distribution of our input data, we also want to ensure that the workflows to retrieve and process our input data is the same during training and serving to avoid training-serving skew. However, we can skip this step if we retrieve our features from the same source location for both training and serving, ie. from a [feature store](feature-store.md){:target="_blank"}.

<div class="ai-center-all">
    <img width="700" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/monitoring/data_drift.png">
</div>
<div class="ai-center-all">
    <small>Data drift can occur in either continuous or categorical features.</small>
</div>

!!! note
    As data starts to drift, we may not yet notice significant decay in our model's performance, especially if the model is able to interpolate well. However, this is a great opportunity to [potentially](#solutions) retrain before the drift starts to impact performance.

### Target drift

Besides just the input data changing, as with data drift, we can also experience drift in our outcomes. This can be a shift in the distributions but also the removal or addition of new classes with categorical tasks. Though retraining can mitigate the performance decay caused target drift, it can often be avoided with proper inter-pipeline communication about new classes, schemas changes, etc.

### Concept drift

Besides the input and output data drifting, we can have the actual relationship between them drift as well. This concept drift renders our model ineffective because the patterns it learned to map between the original inputs and outputs are no longer relevant. Concept drift can be something that occurs in [various patterns](https://link.springer.com/article/10.1007/s11227-018-2674-1){:target="_blank"}:

<div class="ai-center-all">
    <img width="500" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/monitoring/concept_drift.png">
</div>

- gradually over a period of time
- abruptly as a result of an external event
- periodically as a result of recurring events

!!! note
    All the different types of drift we discussed can can occur simultaneously which can complicated identifying the sources of drift.

## Locating drift

Now that we've identified the different types of drift, we need to learn how to locate and how often to measure it. Here are the constraints we need to consider:

- **reference window**: the set of points to compare production data distributions with to identify drift.
- **target window**: the set of points to compare with the reference window to determine if drift has occurred.

Since we're dealing with online drift detection (ie. detecting drift in live production data as opposed to past batch data), we can employ either a [fixed or sliding window approach](https://onlinelibrary.wiley.com/doi/full/10.1002/widm.1381){:target="_blank"} to identify our set of points for comparison. Typically, the reference window is a fixed, recent subset of the training data while the target window slides over time.

[Scikit-multiflow](https://scikit-multiflow.github.io/){:target="_blank"} provides a toolkit for concept drift detection [techniques](https://scikit-multiflow.readthedocs.io/en/stable/api/api.html#module-skmultiflow.drift_detection){:target="_blank"} directly on streaming data. The package offers windowed, moving average functionality (including dynamic preprocessing) and even methods around concepts like [gradual concept drift](https://scikit-multiflow.readthedocs.io/en/stable/api/generated/skmultiflow.drift_detection.EDDM.html#skmultiflow-drift-detection-eddm){:target="_blank"}.

!!! note
    We can also compare across various window sizes simultaneously to ensure smaller cases of drift aren't averaged out by large window sizes.

## Measuring drift

Once have the window of points we wish to compare, we need to know how to compare them.

### Expectations

The first line of measurement can be rule-based such as validating [expectations](https://docs.greatexpectations.io/en/latest/reference/glossary_of_expectations.html){:target="_blank"} around missing values, data types, value ranges, etc. as we did in our [data testing lesson](testing.md#expectations){:target="_blank"}. These can be done with or without a reference window and using the [mostly argument](https://docs.greatexpectations.io/en/latest/reference/core_concepts/expectations/standard_arguments.html#mostly){:target="_blank"} for some level of tolerance.

```python linenums="1"
import great_expectations as ge
import pandas as pd
from tagifai import config, utils
```
```python linenums="1"
# Create DataFrame
features_fp = Path(config.DATA_DIR, "features.json")
features = utils.load_dict(filepath=features_fp)
df = ge.dataset.PandasDataset(features)
```
```python linenums="1"
# Simulated production data
prod_df = ge.dataset.PandasDataset([{"text": "hello"}, {"text": 0}, {"text": "world"}])
```
```python linenums="1"
# Expectation suite
df.expect_column_values_to_not_be_null(column="text")
df.expect_column_values_to_be_of_type(column="text", type_="str")
expectation_suite = df.get_expectation_suite()
```
```python linenums="1"
# Validate reference data
df.validate(expectation_suite=expectation_suite, only_return_failures=True)["statistics"]
```
<pre class="output">
{'evaluated_expectations': 2,
 'successful_expectations': 2,
 'unsuccessful_expectations': 0,
 'success_percent': 100.0}
</pre>
```python linenums="1"
# Validate production data
prod_df.validate(expectation_suite=expectation_suite, only_return_failures=True)["statistics"]
```
<pre class="output">
{'evaluated_expectations': 2,
 'successful_expectations': 1,
 'unsuccessful_expectations': 1,
 'success_percent': 50.0}
</pre>

### Univariate

Once we've validated our rule-based expectations, we need to quantitatively measure drift. Traditionally, in order to compare two different sets of points to see if they come from the same distribution, we use [two-sample hypothesis testing](https://en.wikipedia.org/wiki/Two-sample_hypothesis_testing){:target="_blank"} on the distance measured by a test.

#### Kolmogorov-Smirnov (KS) test

For example, a Kolmogorov-Smirnov (KS) test, which determines the maximum distance between two distribution's cumulative density functions.

> We want to monitor the distribution of the # of tags for a given input so we can capture issues around irregular inference behavior. Other univariate data we may want to monitor include # of tokens in text, % of unknown tokens in text, etc.

```python linenums="1"
from alibi_detect.cd import KSDrift
```

```python linenums="1"
# Reference
df["num_tags"] = df.tags.apply(lambda x: len(x))
reference = df["num_tags"][-400:-200].to_numpy()
```

```python linenums="1"
# Initialize drift detector
length_drift_detector = KSDrift(reference, p_val=0.01)
```

```python linenums="1"
# No drift
no_drift = df["num_tags"][-200:].to_numpy()
length_drift_detector.predict(no_drift, return_p_val=True, return_distance=True)
plt.hist(reference, alpha=0.75, label="reference")
plt.hist(no_drift, alpha=0.5, label="production")
plt.legend()
plt.show()
```

<pre class="output">
{'data': {'is_drift': 0,
  'distance': array([0.06], dtype=float32),
  'p_val': array([0.8428848], dtype=float32),
  'threshold': 0.01},
 'meta': {'name': 'KSDrift', 'detector_type': 'offline', 'data_type': None}}
</pre>

<div class="ai-center-all">
    <img width="500" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/monitoring/ks_no_drift.png">
</div>

```python linenums="1"
# Drift
drift = np.random.normal(10, 2, len(reference))
length_drift_detector.predict(drift, return_p_val=True, return_distance=True)
plt.hist(reference, alpha=0.75, label="reference")
plt.hist(drift, alpha=0.5, label="production")
plt.legend()
plt.show()
```

<pre class="output">
{'data': {'is_drift': 1,
  'distance': array([0.61], dtype=float32),
  'p_val': array([2.9666908e-36], dtype=float32),
  'threshold': 0.01},
 'meta': {'name': 'KSDrift', 'detector_type': 'offline', 'data_type': None}}
</pre>

<div class="ai-center-all">
    <img width="500" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/monitoring/ks_drift.png">
</div>

#### Chi-squared test

And similarly for categorical data (input features, targets, etc.) we can apply the [Pearson's chi-squared test](https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test){:target="_blank"} to determine if a frequency of events (1D) is consistent with a reference distribution.

> We're creating a categorical variable for the # of predicted tags but we could very very apply it to the tag distribution it self, individual tags (binary), slices of tags, etc.

```python linenums="1"
from alibi_detect.cd import ChiSquareDrift
```

```python linenums="1"
# Reference
df.tag_count = df.tags.apply(lambda x: "small" if len(x) <= 3 else ("medium" if len(x) <= 8 else "large"))
reference = df.tag_count[-400:-200].to_numpy()
plt.hist(reference, alpha=0.75, label="reference")
plt.legend()
target_drift_detector = ChiSquareDrift(reference, p_val=0.01)
```

```python linenums="1"
# No drift
no_drift = df.tag_count[-200:].to_numpy()
plt.hist(reference, alpha=0.75, label="reference")
plt.hist(no_drift, alpha=0.5, label="production")
plt.legend()
target_drift_detector.predict(no_drift, return_p_val=True, return_distance=True)
```

<pre class="output">
{'data': {'is_drift': 0,
  'distance': array([2.008658], dtype=float32),
  'p_val': array([0.36629033], dtype=float32),
  'threshold': 0.01},
 'meta': {'name': 'ChiSquareDrift',
  'detector_type': 'offline',
  'data_type': None}}
</pre>

<div class="ai-center-all">
    <img width="500" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/monitoring/chi_no_drift.png">
</div>

```python linenums="1"
# Drift
drift = np.array(["small"]*80 + ["medium"]*40 + ["large"]*80)
plt.hist(reference, alpha=0.75, label="reference")
plt.hist(drift, alpha=0.5, label="production")
plt.legend()
target_drift_detector.predict(drift, return_p_val=True, return_distance=True)
```

<pre class="output">
{'data': {'is_drift': 1,
  'distance': array([118.03355], dtype=float32),
  'p_val': array([2.3406739e-26], dtype=float32),
  'threshold': 0.01},
 'meta': {'name': 'ChiSquareDrift',
  'detector_type': 'offline',
  'data_type': None}}
</pre>

<div class="ai-center-all">
    <img width="500" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/monitoring/chi_drift.png">
</div>

### Multivariate

As we can see, measuring drift is fairly straightforward for univariate data but difficult for multivariate data. We'll summarize the reduce and measure approach outlined in the following paper: [Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift](https://arxiv.org/abs/1810.11953){:target="_blank"}.

<div class="ai-center-all">
    <img width="700" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/monitoring/failing_loudly.png">
</div>
<div class="ai-center-all mt-2">
    <small>Detecting drift as outlined in <a href="https://arxiv.org/abs/1810.11953" target="_blank">Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift</a></small>
</div>

We'll use the embedded inputs from our model as our multivariate data. We'll first apply dimensionality reduction and then conduct two different tests to detect drift. We'll start by establishing our reference window:

```python linenums="1"
import torch
import torch.nn as nn
from tagifai import data, main
```

```python linenums="1"
# Set device
device = utils.set_device(cuda=False)
```

```python linenums="1"
# Load model
run_id = open(Path(config.MODEL_DIR, "run_id.txt")).read()
artifacts = main.load_artifacts(run_id=run_id)
```

```python linenums="1"
# Retrieve artifacts
params = artifacts["params"]
label_encoder = artifacts["label_encoder"]
tokenizer = artifacts["tokenizer"]
embeddings_layer = artifacts["model"].embeddings
embedding_dim = embeddings_layer.embedding_dim
```

```python linenums="1"
def get_data_tensor(texts):
    preprocessed_texts = [data.preprocess(text, lower=params.lower, stem=params.stem) for text in texts]
    X = np.array(tokenizer.texts_to_sequences(preprocessed_texts), dtype="object")
    y_filler = np.zeros((len(X), len(label_encoder)))
    dataset = data.CNNTextDataset(X=X, y=y_filler, max_filter_size=int(params.max_filter_size))
    dataloader = dataset.create_dataloader(batch_size=len(texts))
    return next(iter(dataloader))[0]
```

```python linenums="1"
# Reference
reference = get_data_tensor(texts=df.text[-400:-200].to_list())
reference.shape
```

<pre class="output">
torch.Size([200, 186])
</pre>

!!! note
    We can't use encoded text because each character's categorical representation is arbitrary. However, the embedded text's representation does capture semantic meaning which makes it possible for us to detect drift on. With tabular data and images, we can use those numerical representation as is (can preprocess if needed) since the values are innately meaningful.

#### Dimensionality reduction

We first apply dimensionality reduction to the data before hypothesis testing. Popular options include:

- [Principle component analysis (PCA)](https://en.wikipedia.org/wiki/Principal_component_analysis){:target="_blank"}: orthogonal transformations that preserve the variability of the dataset.
- [Autoencoders (AE)](https://en.wikipedia.org/wiki/Autoencoder){:target="_blank"}: networks that consume the inputs and attempt to reconstruct it from an lower dimensional space while minimizing the error. These can either be trained or untrained (the Failing loudly paper recommends untrained).
- [Black box shift detectors (BBSD)](https://arxiv.org/abs/1802.03916){:target="_blank"}: the actual model trained on the training data can be used as a dimensionality reducer. We can either use the softmax outputs (multivariate) or the actual predictions (univariate).

```python linenums="1"
from functools import partial
from alibi_detect.cd.pytorch import preprocess_drift
```

```python linenums="1"
# Untrained autoencoder (UAE) reducer
enc_dim = 32
reducer = nn.Sequential(
    embeddings_layer,
    nn.AdaptiveAvgPool2d((1, embedding_dim)),
    nn.Flatten(),
    nn.Linear(embedding_dim, 256),
    nn.ReLU(),
    nn.Linear(256, enc_dim)
).to(device).eval()
```

```python linenums="1"
# Preprocessing with the reducer
preprocess_fn = partial(preprocess_drift, model=reducer, batch_size=params.batch_size)
```

#### Two-sample tests

The different dimensionality reduction techniques applied on multivariate data yield either 1D or multidimensional data and so different **statistical tests** are used to calculate drift:

- **[Maximum Mean Discrepancy (MMD)](https://jmlr.csail.mit.edu/papers/v13/gretton12a.html){:target="_blank"}**: a kernel-based approach that determines the distance between two distributions by computing the distance between the mean embeddings of the features from both distributions.

```python linenums="1"
from alibi_detect.cd import MMDDrift
```

```python linenums="1"
# Initialize drift detector
embeddings_mmd_drift_detector = MMDDrift(reference, backend="pytorch", p_val=.01, preprocess_fn=preprocess_fn)
```

```python linenums="1"
# No drift
no_drift = get_data_tensor(texts=df.text[-200:].to_list())
embeddings_mmd_drift_detector.predict(no_drift)
```

<pre class="output">
{'data': {'is_drift': 0,
  'distance': 0.0006961822509765625,
  'p_val': 0.2800000011920929,
  'threshold': 0.01,
  'distance_threshold': 0.008359015},
 'meta': {'name': 'MMDDriftTorch',
  'detector_type': 'offline',
  'data_type': None,
  'backend': 'pytorch'}}
</pre>

```python linenums="1"
# No drift (with benign injection)
texts = ["BERT " + text for text in df.text[-200:].to_list()]
drift = get_data_tensor(texts=texts)
embeddings_mmd_drift_detector.predict(drift)
```

<pre class="output">
{'data': {'is_drift': 0,
  'distance': 0.0030127763748168945,
  'p_val': 0.10000000149011612,
  'threshold': 0.01,
  'distance_threshold': 0.0038004518},
 'meta': {'name': 'MMDDriftTorch',
  'detector_type': 'offline',
  'data_type': None,
  'backend': 'pytorch'}}
</pre>

```python linenums="1"
# Drift
texts = ["UNK " + text for text in df.text[-200:].to_list()]
drift = get_data_tensor(texts=texts)
embeddings_mmd_drift_detector.predict(drift)
```

<pre class="output">
{'data': {'is_drift': 1,
  'distance': 0.009676754474639893,
  'p_val': 0.009999999776482582,
  'threshold': 0.01,
  'distance_threshold': 0.005815625},
 'meta': {'name': 'MMDDriftTorch',
  'detector_type': 'offline',
  'data_type': None,
  'backend': 'pytorch'}}
</pre>

<hr>

- **[Kolmogorov-Smirnov (KS) Test](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test){:target="_blank"} + [Bonferroni Correction](https://en.wikipedia.org/wiki/Bonferroni_correction){:target="_blank"}**: determines the maximum distance between two distribution's cumulative density functions. We can apply this on each dimension of the multidimensional data and then use the Bonferroni correction (conservative) or the [False discovery rate](https://en.wikipedia.org/wiki/False_discovery_rate){:target="_blank"} (FDR) correction to mitigate issues stemming from multiple comparisons.

```python linenums="1"
from alibi_detect.cd import KSDrift
```

```python linenums="1"
# Initialize drift detector
embeddings_ks_drift_detector = KSDrift(reference, p_val=.01, preprocess_fn=preprocess_fn, correction="bonferroni")
```

```python linenums="1"
# No drift
no_drift = get_data_tensor(texts=df.text[-200:].to_list())
embeddings_ks_drift_detector.predict(no_drift)
```

<pre class="output">
{'data': {'is_drift': 0,
  'distance': array([0.115, 0.08 , 0.09 , 0.07 , 0.11 , 0.15 , 0.155, 0.075, 0.085,
         0.08 , 0.075, 0.105, 0.14 , 0.09 , 0.065, 0.11 , 0.065, 0.09 ,
         0.085, 0.07 , 0.065, 0.12 , 0.115, 0.065, 0.105, 0.095, 0.115,
         0.17 , 0.095, 0.055, 0.08 , 0.05 ], dtype=float32),
  'p_val': array([0.13121547, 0.51821935, 0.37063202, 0.6846707 , 0.16496263,
         0.01983924, 0.01453765, 0.60035014, 0.44107372, 0.51821935,
         0.60035014, 0.20524779, 0.03582512, 0.37063202, 0.7671913 ,
         0.16496263, 0.7671913 , 0.37063202, 0.44107372, 0.6846707 ,
         0.7671913 , 0.10330375, 0.13121547, 0.7671913 , 0.20524779,
         0.30775842, 0.13121547, 0.00537641, 0.30775842, 0.9063568 ,
         0.51821935, 0.95321596], dtype=float32),
  'threshold': 0.0003125},
 'meta': {'name': 'KSDrift', 'detector_type': 'offline', 'data_type': None}}
</pre>

```python linenums="1"
# Drift
texts = ["UNK " + text for text in df.text[-200:].to_list()]
drift = get_data_tensor(texts=texts)
embeddings_ks_drift_detector.predict(drift)
```

<pre class="output">
{'data': {'is_drift': 1,
  'distance': array([0.17 , 0.125, 0.085, 0.09 , 0.14 , 0.175, 0.21 , 0.17 , 0.14 ,
         0.08 , 0.08 , 0.105, 0.205, 0.115, 0.065, 0.075, 0.055, 0.135,
         0.065, 0.08 , 0.105, 0.13 , 0.125, 0.095, 0.105, 0.17 , 0.13 ,
         0.26 , 0.115, 0.055, 0.095, 0.05 ], dtype=float32),
  'p_val': array([5.3764065e-03, 8.0500402e-02, 4.4107372e-01, 3.7063202e-01,
         3.5825118e-02, 3.7799652e-03, 2.3947345e-04, 5.3764065e-03,
         3.5825118e-02, 5.1821935e-01, 5.1821935e-01, 2.0524779e-01,
         3.6656143e-04, 1.3121547e-01, 7.6719129e-01, 6.0035014e-01,
         9.0635681e-01, 4.7406290e-02, 7.6719129e-01, 5.1821935e-01,
         2.0524779e-01, 6.2092341e-02, 8.0500402e-02, 3.0775842e-01,
         2.0524779e-01, 5.3764065e-03, 6.2092341e-02, 1.8819204e-06,
         1.3121547e-01, 9.0635681e-01, 3.0775842e-01, 9.5321596e-01],
        dtype=float32),
  'threshold': 0.0003125},
 'meta': {'name': 'KSDrift', 'detector_type': 'offline', 'data_type': None}}
</pre>

> Note that each feature (enc_dim=32) has a distance and an associated p-value.

We could repeat this process for tensor outputs at various layers in our model (embedding, conv layers, softmax, etc.). Just keep in mind that our outputs from the reducer need to be a 2D matrix so we may need to do additional preprocessing such as pooling 3D embedding tensors. [TorchDrift](https://torchdrift.org/) is another great package that offers a suite of reducers (PCA, AE, etc.) and drift detectors (MMD) to monitor for drift at any stage in our model.

!!! note
    Another interesting approach for detecting drift involves training a separate model that can distinguish between data from the reference and production distributions. If such a classifier can be trained that it performs better than random chance (0.5), confirmed with a [binomial test](https://en.wikipedia.org/wiki/Binomial_test){:target="_blank"}, then we have two statistically different distributions. This isn't a popular approach because it involves creating distinct datasets and the compute for training every time we want to measure drift with two windows of data.

## Outliers

With drift, we're comparing a window of production data with reference data as opposed to looking at any one specific data point. While each individual point may not be an anomaly or outlier, the group of points may cause a drift. The easiest way to illustrate this is to imagine feeding our live model the same input data point repeatedly. The actual data point may not have anomalous features but feeding it repeatedly will cause the feature distribution that the model is receiving to change and lead to drift.

!!! note
    When we identify outliers, we may want to let the end user know that the model's response may not be reliable. Additionally, we may want to remove the outliers from the next training set or further inspect them and upsample them in case they're early signs of what future distributions of incoming features will look like.

<div class="ai-center-all">
    <img width="600" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/monitoring/outliers.png">
</div>

Unfortunately, it's not very easy to detect outliers because it's hard to constitute the criteria for an outlier. Therefore the outlier detection task is typically unsupervised and requires a stochastic streaming algorithm to identify potential outliers. Luckily, there are several powerful libraries such as [PyOD](https://pyod.readthedocs.io/en/latest/){:target="_blank"}, [Alibi Detect](https://docs.seldon.io/projects/alibi-detect/en/latest/){:target="_blank"}, [WhyLogs](https://whylogs.readthedocs.io/en/latest/){:target="_blank"} (uses [Apache DataSketches](https://datasketches.apache.org/){:target="_blank"}), etc. that offer a suite of outlier detection functionality (largely for tabular and image data for now). We can use these packages with our [pipelines](pipelines.md){:target="_blank"} or even [Kafka](https://kafka.apache.org/){:target="_blank"} data streams to continuously monitor for outliers.

```python linenums="1"
from alibi_detect.od import OutlierVAE
X_train = (n_samples, n_features)
outlier_detector = OutlierVAE(
    threshold=0.05,
    encoder_net=encoder,
    decoder_net=decoder,
    latent_dim=512
)
outlier_detector.fit(X_train, epochs=50)
outlier_detector.infer_threshold(X, threshold_perc=95)  # infer from % outliers
preds = outlier_detector.predict(X, outlier_type="instance", outlier_perc=75)
```

!!! note
    Typically, outlier detection algorithms fit (ex. via reconstruction) to the training set to understand what normal data looks like and then we can use a threshold to predict outliers. If we have a small labeled dataset with outliers, we can empirically choose our threshold but if not, we can choose some reasonable tolerance.

!!! warning
    We shouldn't identify outliers by looking at a model’s predicted probabilities because they need to be [calibrated](https://arxiv.org/abs/1706.04599){:target="_blank"} before using them as reliable measures of confidence.

## Solutions

It's not enough to just be able to measure drift or identify outliers but to also be able to act on it. We want to be able to alert on drift, inspect it and then act on it.

### Alert

Once we've identified outliers and/or measured statistically significant drift, we need to a devise a workflow to notify stakeholders of the issues. A negative connotation with monitoring is fatigue stemming from false positive alerts. This can be mitigated by choosing the appropriate constraints (ex. alerting thresholds) based on what's important to our specific application. For example, thresholds could be:

- fixed values/range for situations where we're concretely aware of expected upper/lower bounds.
```python linenums="1"
if percentage_unk_tokens > 5%:
    trigger_alert()
```
- [forecasted](https://www.datadoghq.com/blog/forecasts-datadog/){;target="_blank"} thresholds dependent on previous inputs, time, etc.
```python linenums="1"
if current_f1 < forecast_f1(current_time):
    trigger_alert()
```
- appropriate p-values for different drift detectors (&darr; p-value = &uarr; confident that the distributions are different).
```python linenums="1"
from alibi_detect.cd import KSDrift
length_drift_detector = KSDrift(reference, p_val=0.01)
```

Once we have our carefully crafted alerting workflows in place, we can notify stakeholders as issues arise via email, [Slack](https://slack.com/){:target="_blank"}, [PageDuty](https://www.pagerduty.com/){:target="_blank"}, etc. The stakeholders can be of various levels (core engineers, managers, etc.) and they can subscribe to the alerts that are appropriate for them.

### Inspect

Once we receive an alert, we need to inspect it before acting on it. An alert needs several components in order for us to completely inspect it:

- specific alert that was triggered
- relevant metadata (time, inputs, outputs, etc.)
- thresholds / expectations that failed
- drift detection tests that were conducted
- data from reference and target windows
- [log](logging.md){:target="_blank"} records from the relevant window of time

```bash
# Sample alerting ticket
{
    "triggered_alerts": ["text_length_drift"],
    "threshold": 0.05,
    "measurement": "KSDrift",
    "distance": 0.86,
    "p_val": 0.03,
    "reference": [],
    "target": [],
    "logs": ...
}
```

With these pieces of information, we can work backwards from the alert towards identifying the root cause of the issue. **Root cause analysis (RCA)** is an important first step when it comes to monitoring because we want to prevent the same issue from impacting our system again. Often times, many alerts are triggered but they maybe all actually be caused by the same underlying issue. In this case, we'd want to intelligently trigger just one alert that pinpoints the core issue. For example, let's say we receive an alert that our overall user satisfaction ratings are reducing but we also receive another alert that our North American users also have low satisfaction ratings. Here's the system would automatically assess for drift in user satisfaction ratings across many different slices and aggregations to discover that only users in a specific area are experiencing the issue but because it's a popular user base, it ends up triggering all aggregate downstream alerts as well!

### Act

There are many different ways we can act to drift based on the situation. An initial impulse may be to retrain our model on the new data but it may not always solve the underlying issue.

- ensure all data expectations have passed.
- confirm no data schema changes.
- retrain the model on the new shifted dataset.
- move the reference window to more recent data or give it more weight.
- determine if outliers are potentially valid data points.

## Production

Since detecting drift and outliers can involve compute intensive operations, we need a solution that can execute serverless workloads on top of our event data streams (ex. [Kafka](https://kafka.apache.org/){:target="_blank"}). Typically these solutions will ingest payloads (ex. model's inputs and outputs) and can trigger monitoring workloads. This allows us to segregate the resources for monitoring from our actual ML application and scale them as needed.

<div class="ai-center-all">
    <img width="600" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/monitoring/serverless.png">
</div>

When it actually comes to implementing a monitoring system, we have several options, ranging from fully managed to from-scratch. Several popular managed solutions are [Fiddler](https://www.fiddler.ai/ml-monitoring){:target="_blank"}, [Arize](https://arize.com/){:target="_blank"}, [Arthur](https://www.arthur.ai/){:target="_blank"}, [Mona](https://www.monalabs.io/){:target="_blank"}, [WhyLogs](https://whylogs.readthedocs.io/en/latest/){:target="_blank"}, etc., all of which allow us to create custom monitoring views, trigger alerts, etc. There are even several great open-source solutions such as [Gantry](https://gantry.io/){:target="_blank"}, [TorchDrift](https://torchdrift.org/){:target="_blank"}, [WhyLabs](https://whylabs.ai/){:target="_blank"}, [EvidentlyAI](https://evidentlyai.com/){:target="_blank"}, etc.

We'll often notice that monitoring solutions are offered as part of the larger deployment option such as [TensorFlow Extended (TFX)](https://www.tensorflow.org/tfx){:target="_blank"}, [TorchServe](https://pytorch.org/serve/){:target="_blank"}, [Sagemaker](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor.html){:target="_blank"}, etc. And if we're already working with Kubernetes, we could use [KNative](https://knative.dev/){:target="_blank"} or [Kubeless](https://kubeless.io/){:target="_blank"} for serverless workload management. But we could also use a higher level framework such as [KFServing](https://www.kubeflow.org/docs/components/kfserving/){:target="_blank"} or [Seldon core](https://docs.seldon.io/projects/seldon-core/en/v0.4.0/#){:target="_blank"} that natively use a serverless framework like KNative.

!!! note
    Learn about how the monitoring workflows connect to the our overall ML systems in our [pipeline lesson](pipelines.md#monitoring). Monitoring offers a stream of signals that our update policy engine consumes to decide what to do next (continue, warrant an inspection, retrain the model on new data, rollback to a previous model version, etc.).

## References
- [An overview of unsupervised drift detection methods](https://onlinelibrary.wiley.com/doi/full/10.1002/widm.1381){:target="_blank"}
- [Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift](https://arxiv.org/abs/1810.11953){:target="_blank"}
- [Monitoring and explainability of models in production](https://arxiv.org/abs/2007.06299){:target="_blank"}
- [A simple solution for monitoring ML systems](https://www.jeremyjordan.me/ml-monitoring/){:target="_blank"}
- [Detecting and Correcting for Label Shift with Black Box Predictors](https://arxiv.org/abs/1802.03916){:target="_blank"}
- [Outlier and anomaly pattern detection on data streams](https://link.springer.com/article/10.1007/s11227-018-2674-1){:target="_blank"}