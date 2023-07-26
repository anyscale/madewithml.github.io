---
template: lesson.html
title: Evaluating Machine Learning Models
description: Evaluating ML models by assessing overall, per-class and slice performances.
keywords: evaluation, mlops, machine learning
image: https://madewithml.com/static/images/mlops.png
repository: https://github.com/GokuMohandas/Made-With-ML
notebook: https://github.com/GokuMohandas/Made-With-ML/blob/main/notebooks/madewithml.ipynb
---

{% include "styles/lesson.md" %}

## Intuition

Evaluation is an integral part of modeling and it's one that's often glossed over. We'll often find evaluation to involve simply computing the accuracy or other global metrics but for many real work applications, a much more nuanced evaluation process is required. However, before evaluating our model, we always want to:

- be clear about what metrics we are prioritizing
- be careful not to over optimize on any one metric because it may mean you're compromising something else

## Setup

Let's start by setting up our metrics dictionary that we'll fill in as we go along and all the data we'll need for evaluation: grounds truth labels (`y_test`, predicted labels (`y_pred`) and predicted probabilities (`y_prob`).

```python linenums="1"
# Metrics
metrics = {"overall": {}, "class": {}}
```
```python linenums="1"
# y_test
preprocessor = predictor.get_preprocessor()
preprocessed_ds = preprocessor.transform(test_ds)
values = preprocessed_ds.select_columns(cols=["targets"]).take_all()
y_test = np.stack([item["targets"] for item in values])
```
```python linenums="1"
# y_pred
test_df = test_ds.to_pandas()
z = predictor.predict(data=test_df)["predictions"]  # adds text column (in-memory)
y_pred = np.stack(z).argmax(1)
```
```python linenums="1"
# y_prob
y_prob = torch.tensor(np.stack(z)).softmax(dim=1).numpy()
print (np.shape(y_test))
print (np.shape(y_prob))
```
```python linenums="1"
# Add columns (for convenience)
test_df = test_ds.to_pandas()
test_df["text"] = test_df["title"] + " " + test_df["description"]
test_df["prediction"] = test_df.index.map(lambda i: preprocessor.index_to_class[y_pred[i]])
test_df.head()
```

<div class="output_subarea output_html rendered_html output_result" dir="auto"><div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>created_on</th>
      <th>title</th>
      <th>description</th>
      <th>tag</th>
      <th>text</th>
      <th>prediction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19</td>
      <td>2020-03-03 13:54:31</td>
      <td>Diffusion to Vector</td>
      <td>Reference implementation of Diffusion2Vec (Com...</td>
      <td>other</td>
      <td>Diffusion to Vector Reference implementation o...</td>
      <td>other</td>
    </tr>
    <tr>
      <th>1</th>
      <td>26</td>
      <td>2020-03-07 23:11:58</td>
      <td>Graph Wavelet Neural Network</td>
      <td>A PyTorch implementation of "Graph Wavelet Neu...</td>
      <td>other</td>
      <td>Graph Wavelet Neural Network A PyTorch impleme...</td>
      <td>other</td>
    </tr>
    <tr>
      <th>2</th>
      <td>44</td>
      <td>2020-03-08 00:32:58</td>
      <td>Capsule Graph Neural Network</td>
      <td>A PyTorch implementation of "Capsule Graph Neu...</td>
      <td>other</td>
      <td>Capsule Graph Neural Network A PyTorch impleme...</td>
      <td>other</td>
    </tr>
    <tr>
      <th>3</th>
      <td>80</td>
      <td>2020-03-20 05:59:32</td>
      <td>NeRF: Neural Radiance Fields</td>
      <td>Representing scenes as neural radiance fields ...</td>
      <td>computer-vision</td>
      <td>NeRF: Neural Radiance Fields Representing scen...</td>
      <td>computer-vision</td>
    </tr>
    <tr>
      <th>4</th>
      <td>84</td>
      <td>2020-03-20 15:18:43</td>
      <td>Mention Classifier</td>
      <td>Category prediction model\r\nThis repo contain...</td>
      <td>natural-language-processing</td>
      <td>Mention Classifier Category prediction model\r...</td>
      <td>natural-language-processing</td>
    </tr>
  </tbody>
</table>
</div></div>

## Coarse-grained

While we were developing our models, our evaluation process involved computing the coarse-grained metrics such as overall precision, recall and f1 metrics.

- **True positives (TP)**: we correctly predicted class X.
- **False positives (FP)**: we incorrectly predicted class X but it was another class.
- **True negatives (TN)**: we correctly predicted that it's wasn't the class X.
- **False negatives (FN)**: we incorrectly predicted that it wasn't the class X but it was.

$$ \text{precision} = \frac{TP}{TP + FP} $$

$$ \text{recall} = \frac{TP}{TP + FN} $$

$$ \text{f1} = \frac{2 * precision * recall}{precision + recall} $$

<div class="ai-center-all">
    <img src="/static/images/mlops/evaluation/metrics.png" width="600" alt="metrics">
</div>

```python linenums="1"
from sklearn.metrics import precision_recall_fscore_support
```
```python linenums="1"
# Overall metrics
overall_metrics = precision_recall_fscore_support(y_test, y_pred, average="weighted")
metrics["overall"]["precision"] = overall_metrics[0]
metrics["overall"]["recall"] = overall_metrics[1]
metrics["overall"]["f1"] = overall_metrics[2]
metrics["overall"]["num_samples"] = np.float64(len(y_test))
print (json.dumps(metrics["overall"], indent=4))
```
<pre class="output">
{
    "precision": 0.916248340770615,
    "recall": 0.9109947643979057,
    "f1": 0.9110623702438432,
    "num_samples": 191.0
}
</pre>

!!! note
    The [precision_recall_fscore_support()](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html){:target="_blank"} function from scikit-learn has an input parameter called `average` which has the following options below. We'll be using the different averaging methods for different metric granularities.

    - `None`: metrics are calculated for each unique class.
    - `binary`: used for binary classification tasks where the `pos_label` is specified.
    - `micro`: metrics are calculated using global TP, FP, and FN.
    - `macro`: per-class metrics which are averaged without accounting for class imbalance.
    - `weighted`: per-class metrics which are averaged by accounting for class imbalance.
    - `samples`: metrics are calculated at the per-sample level.

## Fine-grained

Inspecting these coarse-grained, overall metrics is a start but we can go deeper by evaluating the same fine-grained metrics at the categorical feature levels.

```python linenums="1"
from collections import OrderedDict
```
```python linenums="1"
# Per-class metrics
class_metrics = precision_recall_fscore_support(y_test, y_pred, average=None)
for i, _class in enumerate(preprocessor.class_to_index):
    metrics["class"][_class] = {
        "precision": class_metrics[0][i],
        "recall": class_metrics[1][i],
        "f1": class_metrics[2][i],
        "num_samples": np.float64(class_metrics[3][i]),
    }
```
```python linenums="1"
# Metrics for a specific class
tag="natural-language-processing"
print (json.dumps(metrics["class"][tag], indent=2))
```
<pre class="output">
{
  "precision": 0.9036144578313253,
  "recall": 0.9615384615384616,
  "f1": 0.9316770186335404,
  "num_samples": 78.0
}
</pre>

```python linenums="1"
# Sorted tags
sorted_tags_by_f1 = OrderedDict(sorted(
        metrics["class"].items(), key=lambda tag: tag[1]["f1"], reverse=True))
for item in sorted_tags_by_f1.items():
    print (json.dumps(item, indent=2))
```
<pre class="output">
[
  "natural-language-processing",
  {
    "precision": 0.9036144578313253,
    "recall": 0.9615384615384616,
    "f1": 0.9316770186335404,
    "num_samples": 78.0
  }
]
[
  "computer-vision",
  {
    "precision": 0.9838709677419355,
    "recall": 0.8591549295774648,
    "f1": 0.9172932330827067,
    "num_samples": 71.0
  }
]
[
  "other",
  {
    "precision": 0.8333333333333334,
    "recall": 0.9615384615384616,
    "f1": 0.8928571428571429,
    "num_samples": 26.0
  }
]
[
  "mlops",
  {
    "precision": 0.8125,
    "recall": 0.8125,
    "f1": 0.8125,
    "num_samples": 16.0
  }
]
</pre>

## Confusion matrix

Besides just inspecting the metrics for each class, we can also identify the true positives, false positives and false negatives. Each of these will give us insight about our model beyond what the metrics can provide.

- **True positives (TP)**: learn about where our model performs well.
- **False positives (FP)**: potentially identify samples which may need to be relabeled.
- False negatives (FN): identify the model's less performant areas to oversample later.

> It's a good to have our FP/FN samples feed back into our annotation pipelines in the event we want to fix their labels and have those changes be reflected everywhere.

```python linenums="1"
# TP, FP, FN samples
tag = "natural-language-processing"
index = preprocessor.class_to_index[tag]
tp, fp, fn = [], [], []
for i, true in enumerate(y_test):
    pred = y_pred[i]
    if index==true==pred:
        tp.append(i)
    elif index!=true and index==pred:
        fp.append(i)
    elif index==true and index!=pred:
        fn.append(i)
```
```python linenums="1"
print (tp)
print (fp)
print (fn)
```
<pre class="output">
[4, 9, 12, 17, 19, 23, 25, 26, 29, 30, 31, 32, 33, 34, 42, 47, 49, 50, 54, 56, 65, 66, 68, 71, 75, 76, 77, 78, 79, 82, 92, 94, 95, 97, 99, 101, 109, 113, 114, 118, 120, 122, 126, 128, 129, 130, 131, 133, 134, 135, 138, 139, 140, 141, 142, 144, 148, 149, 152, 159, 160, 161, 163, 166, 170, 172, 173, 174, 177, 179, 183, 184, 187, 189, 190]
[41, 44, 73, 102, 110, 150, 154, 165]
[16, 112, 115]
</pre>

```python linenums="1"
# Samples
num_samples = 3
cm = [(tp, "True positives"), (fp, "False positives"), (fn, "False negatives")]
for item in cm:
    if len(item[0]):
        print (f"\n=== {item[1]} ===")
        for index in item[0][:num_samples]:
            print (f"{test_df.iloc[index].text}")
            print (f"    true: {test_df.tag[index]}")
            print (f"    pred: {test_df.prediction[index]}\n")
```
<pre class="output">
=== True positives ===
Mention Classifier Category prediction model
This repo contains AllenNLP model for prediction of Named Entity categories by its mentions.
    true: natural-language-processing
    pred: natural-language-processing

Finetune: Scikit-learn Style Model Finetuning for NLP Finetune is a library that allows users to leverage state-of-the-art pretrained NLP models for a wide variety of downstream tasks.
    true: natural-language-processing
    pred: natural-language-processing

Finetuning Transformers with JAX + Haiku Walking through a port of the RoBERTa pre-trained model to JAX + Haiku, then fine-tuning the model to solve a downstream task.
    true: natural-language-processing
    pred: natural-language-processing


=== False positives ===
How Docker Can Help You Become A More Effective Data Scientist A look at Docker from the perspective of a data scientist.
    true: mlops
    pred: natural-language-processing

Transfer Learning & Fine-Tuning With Keras Your 100% up-to-date guide to transfer learning & fine-tuning with Keras.
    true: computer-vision
    pred: natural-language-processing

Exploratory Data Analysis on MS COCO Style Datasets A Simple Toolkit to do exploratory data analysis on MS COCO style formatted datasets.
    true: computer-vision
    pred: natural-language-processing


=== False negatives ===
The Unreasonable Effectiveness of Recurrent Neural Networks A close look at how RNNs are able to perform so well.
    true: natural-language-processing
    pred: other

Machine Learning Projects  This Repo contains projects done by me while learning the basics. All the familiar types of regression, classification, and clustering methods have been used.
    true: natural-language-processing
    pred: other

BERT Distillation with Catalyst How to distill BERT with Catalyst.
    true: natural-language-processing
    pred: mlops

</pre>

!!! tip
    It's a really good idea to do this kind of analysis using our rule-based approach to catch really obvious labeling errors.

## Confidence learning

While the confusion-matrix sample analysis was a coarse-grained process, we can also use fine-grained confidence based approaches to identify potentially mislabeled samples. Here we’re going to focus on the specific labeling quality as opposed to the final model predictions.

Simple confidence based techniques include identifying samples whose:

- **Categorical**
    - prediction is incorrect (also indicate TN, FP, FN)
    - confidence score for the correct class is below a threshold
    - confidence score for an incorrect class is above a threshold
    - standard deviation of confidence scores over top N samples is low
    - different predictions from same model using different parameters

- **Continuous**
    - difference between predicted and ground-truth values is above some %

```python linenums="1"
# Tag to inspect
tag = "natural-language-processing"
index = class_to_index[tag]
indices = np.where(y_test==index)[0]
```

```python linenums="1"
# Confidence score for the correct class is below a threshold
low_confidence = []
min_threshold = 0.5
for i in indices:
    prob = y_prob[i][index]
    if prob <= 0.5:
        low_confidence.append({
            "text": f"{test_df.iloc[i].text}",
            "true": test_df.tag[i],
            "pred": test_df.prediction[i],
            "prob": prob})
```

```python linenums="1"
low_confidence[0:3]
```

<pre class="output">
[{'text': 'The Unreasonable Effectiveness of Recurrent Neural Networks A close look at how RNNs are able to perform so well.',
  'true': 'natural-language-processing',
  'pred': 'other',
  'prob': 0.0023471832},
 {'text': 'Machine Learning Projects  This Repo contains projects done by me while learning the basics. All the familiar types of regression, classification, and clustering methods have been used.',
  'true': 'natural-language-processing',
  'pred': 'other',
  'prob': 0.0027675298},
 {'text': 'BERT Distillation with Catalyst How to distill BERT with Catalyst.',
  'true': 'natural-language-processing',
  'pred': 'mlops',
  'prob': 0.37908182}]
</pre>

But these are fairly crude techniques because neural networks are easily [overconfident](https://arxiv.org/abs/1706.04599){:target="_blank"} and so their confidences cannot be used without calibrating them.

<div class="ai-center-all">
    <img src="/static/images/mlops/evaluation/calibration.png" width="400" alt="accuracy vs. confidence">
</div>
<div class="ai-center-all mt-1">
  <small>Modern (large) neural networks result in higher accuracies but are over confident.<br><a href="https://arxiv.org/abs/1706.04599" target="_blank">On Calibration of Modern Neural Networks</a></small>
</div>

* **Assumption**: *“the probability associated with the predicted class label should reflect its ground truth correctness likelihood.”*
* **Reality**: *“modern (large) neural networks are no longer well-calibrated”*
* **Solution**: apply temperature scaling (extension of [Platt scaling](https://en.wikipedia.org/wiki/Platt_scaling){:target="_blank"}) on model outputs

Recent work on [confident learning](https://arxiv.org/abs/1911.00068){:target="_blank"} ([cleanlab](https://github.com/cleanlab/cleanlab){:target="_blank"}) focuses on identifying noisy labels (with calibration), which can then be properly relabeled and used for training.

```python linenums="1"
import cleanlab
from cleanlab.filter import find_label_issues
```
```python linenums="1"
# Find label issues
label_issues = find_label_issues(labels=y_test, pred_probs=y_prob, return_indices_ranked_by="self_confidence")
test_df.iloc[label_issues].drop(columns=["text"]).head()
```

<div class="output_subarea output_html rendered_html output_result" dir="auto"><div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>created_on</th>
      <th>title</th>
      <th>description</th>
      <th>tag</th>
      <th>prediction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>165</th>
      <td>2137</td>
      <td>2020-08-13 02:10:03</td>
      <td>Unpopular Opinion - Data Scientists Should Be ...</td>
      <td>I believe data scientists can be more effectiv...</td>
      <td>mlops</td>
      <td>natural-language-processing</td>
    </tr>
    <tr>
      <th>154</th>
      <td>1976</td>
      <td>2020-07-27 14:12:03</td>
      <td>Close-Domain fine-tuning for table detection</td>
      <td>In this project, we show the benefits of using...</td>
      <td>computer-vision</td>
      <td>natural-language-processing</td>
    </tr>
    <tr>
      <th>16</th>
      <td>264</td>
      <td>2020-04-06 21:33:32</td>
      <td>The Unreasonable Effectiveness of Recurrent Ne...</td>
      <td>A close look at how RNNs are able to perform s...</td>
      <td>natural-language-processing</td>
      <td>other</td>
    </tr>
    <tr>
      <th>103</th>
      <td>1459</td>
      <td>2020-06-16 03:06:10</td>
      <td>SuperGlue: Learning Feature Matching with Grap...</td>
      <td>SuperGlue, a neural network that matches two s...</td>
      <td>other</td>
      <td>computer-vision</td>
    </tr>
    <tr>
      <th>112</th>
      <td>1524</td>
      <td>2020-06-20 10:42:25</td>
      <td>Machine Learning Projects</td>
      <td>This Repo contains projects done by me while l...</td>
      <td>natural-language-processing</td>
      <td>other</td>
    </tr>
  </tbody>
</table>
</div></div>

Not all of these are necessarily labeling errors but situations where the predicted probabilities were not so confident. Therefore, it will be useful to attach the predicted outcomes along side results. This way, we can know if we need to relabel, upsample, etc. as mitigation strategies to improve our performance.

> The operations in this section can be applied to entire labeled dataset to discover labeling errors via confidence learning.

## Slicing

Just inspecting the overall and class metrics isn't enough to deploy our new version to production. There may be key slices of our dataset that we need to do really well on:

- Target / predicted classes (+ combinations)
- Features (explicit and implicit)
- Metadata (timestamps, sources, etc.)
- Priority slices / experience (minority groups, large users, etc.)

An easy way to create and evaluate slices is to define slicing functions.

```python linenums="1"
from snorkel.slicing import PandasSFApplier
from snorkel.slicing import slice_dataframe
from snorkel.slicing import slicing_function
```

```python linenums="1"
@slicing_function()
def nlp_llm(x):
    """NLP projects that use LLMs."""
    nlp_project = "natural-language-processing" in x.tag
    llm_terms = ["transformer", "llm", "bert"]
    llm_project = any(s.lower() in x.text.lower() for s in llm_terms)
    return (nlp_project and llm_project)
```
```python linenums="1"
@slicing_function()
def short_text(x):
    """Projects with short titles and descriptions."""
    return len(x.text.split()) < 8  # less than 8 words
```

Here we're using Snorkel's [`slicing_function`](https://snorkel.readthedocs.io/en/latest/packages/_autosummary/slicing/snorkel.slicing.slicing_function.html){:target="blank"} to create our different slices. We can visualize our slices by applying this slicing function to a relevant DataFrame using [`slice_dataframe`](https://snorkel.readthedocs.io/en/latest/packages/_autosummary/slicing/snorkel.slicing.slice_dataframe.html){:target="_blank"}.

```python linenums="1"
nlp_llm_df = slice_dataframe(test_df, nlp_llm)
nlp_llm_df[["text", "tag"]].head()
```

<div class="output_subarea output_html rendered_html ai-center-all"><div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>tag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12</th>
      <td>Finetuning Transformers with JAX + Haiku Walki...</td>
      <td>natural-language-processing</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Question Answering with a Fine-Tuned BERT What...</td>
      <td>natural-language-processing</td>
    </tr>
    <tr>
      <th>29</th>
      <td>BertViz Tool for visualizing attention in the ...</td>
      <td>natural-language-processing</td>
    </tr>
    <tr>
      <th>30</th>
      <td>The Transformer Family This post presents how ...</td>
      <td>natural-language-processing</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Pruning Bert to Accelerate Inference After pre...</td>
      <td>natural-language-processing</td>
    </tr>
  </tbody>
</table>
</div></div>

```python linenums="1"
short_text_df = slice_dataframe(test_df, short_text)
short_text_df[["text", "tag"]].head()
```

<div class="output_subarea output_html rendered_html ai-center-all"><div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>tag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>75</th>
      <td>NLPAug Data augmentation for NLP</td>
      <td>natural-language-processing</td>
    </tr>
    <tr>
      <th>123</th>
      <td>Offline Reinforcement Learning Challenges, alg...</td>
      <td>other</td>
    </tr>
    <tr>
      <th>127</th>
      <td>Image Classifier Pure JavaScript Image Classifier</td>
      <td>computer-vision</td>
    </tr>
    <tr>
      <th>132</th>
      <td>imgaug Image augmentation for machine learning...</td>
      <td>computer-vision</td>
    </tr>
    <tr>
      <th>140</th>
      <td>QSVM Quantum SVM for sentiment analysis</td>
      <td>natural-language-processing</td>
    </tr>
  </tbody>
</table>
</div></div>

We can define even more slicing functions and create a slices record array using the [`PandasSFApplier`](https://snorkel.readthedocs.io/en/latest/packages/_autosummary/slicing/snorkel.slicing.PandasSFApplier.html){:target="_blank"}. The slices array has N (# of data points) items and each item has S (# of slicing functions) items, indicating whether that data point is part of that slice. Think of this record array as a masking layer for each slicing function on our data.

```python linenums="1"
# Slices
slicing_functions = [nlp_llm, short_text]
applier = PandasSFApplier(slicing_functions)
slices = applier.apply(test_df)
slices
```

<pre class="output">
rec.array([(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
           (1, 0) (0, 0) (0, 1) (0, 0) (0, 0) (1, 0) (0, 0) (0, 0) (0, 1) (0, 0)
           ...
           (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 1),
           (0, 0), (0, 0)],
    dtype=[('nlp_cnn', '&lt;i8'), ('short_text', '&lt;i8')])
</pre>

To calculate metrics for our slices, we could use [snorkel.analysis.Scorer](https://snorkel.readthedocs.io/en/latest/packages/_autosummary/analysis/snorkel.analysis.Scorer.html){:target="_blank"} but we've implemented a version that will work for multiclass or multilabel scenarios.

```python linenums="1"
# Score slices
metrics["slices"] = {}
for slice_name in slices.dtype.names:
    mask = slices[slice_name].astype(bool)
    if sum(mask):
        slice_metrics = precision_recall_fscore_support(
            y_test[mask], y_pred[mask], average="micro"
        )
        metrics["slices"][slice_name] = {}
        metrics["slices"][slice_name]["precision"] = slice_metrics[0]
        metrics["slices"][slice_name]["recall"] = slice_metrics[1]
        metrics["slices"][slice_name]["f1"] = slice_metrics[2]
        metrics["slices"][slice_name]["num_samples"] = len(y_test[mask])
```

```python linenums="1"
print(json.dumps(metrics["slices"], indent=2))
```

<pre class="output">
{
  "nlp_llm": {
    "precision": 0.9642857142857143,
    "recall": 0.9642857142857143,
    "f1": 0.9642857142857143,
    "num_samples": 28
  },
  "short_text": {
    "precision": 1.0,
    "recall": 1.0,
    "f1": 1.0,
    "num_samples": 7
  }
}
</pre>

Slicing can help identify sources of *bias* in our data. For example, our model has most likely learned to associated algorithms with certain applications such as CNNs used for computer vision or transformers used for NLP projects. However, these algorithms are not being applied beyond their initial use cases. We’d need ensure that our model learns to focus on the application over algorithm. This could be learned with:

- enough data (new or oversampling incorrect predictions)
- masking the algorithm (using text matching heuristics)

## Interpretability

Besides just comparing predicted outputs with ground truth values, we can also inspect the inputs to our models. What aspects of the input are more influential towards the prediction? If the focus is not on the relevant features of our input, then we need to explore if there is a hidden pattern we're missing or if our model has learned to overfit on the incorrect features. We can use techniques such as [SHAP](https://github.com/slundberg/shap){:target="_blank"} (SHapley Additive exPlanations) or [LIME](https://github.com/marcotcr/lime){:target="_blank"} (Local Interpretable Model-agnostic Explanations) to inspect feature importance. On a high level, these techniques learn which features have the most signal by assessing the performance in their absence. These inspections can be performed on a global level (ex. per-class) or on a local level (ex. single prediction).

```python linenums="1"
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline
```

[`LimeTextExplainer.explain_instance`](https://lime-ml.readthedocs.io/en/latest/lime.html#lime.lime_text.LimeTextExplainer.explain_instance){:target="_blank"} function requires a `classifier_fn` that takes in a list of strings and outputs the predicted probabilities.

```python linenums="1"
def classifier_fn(texts):
    df = pd.DataFrame({"title": texts, "description": "", "tag": "other"})
    z = predictor.predict(data=df)["predictions"]
    y_prob = torch.tensor(np.stack(z)).softmax(dim=1).numpy()
    return y_prob
```
```python linenums="1"
# Explain instance
text = "Using pretrained convolutional neural networks for object detection."
explainer = LimeTextExplainer(class_names=list(class_to_index.keys()))
explainer.explain_instance(text, classifier_fn=classifier_fn, top_labels=1).show_in_notebook(text=True)
```

<div class="ai-center-all">
    <img src="/static/images/mlops/evaluation/lime.png" width="1000" alt="LIME for ml interpretability">
</div>

> We can also use model-specific approaches to interpretability we we did in our [embeddings lesson](../foundations/embeddings.md#interpretability){:target="_blank"} to identify the most influential n-grams in our text.


## Behavioral testing

Besides just looking at metrics, we also want to conduct some behavioral sanity tests. Behavioral testing is the process of testing input data and expected outputs while treating the model as a black box. They don't necessarily have to be adversarial in nature but more along the types of perturbations we'll see in the real world once our model is deployed. A landmark paper on this topic is [Beyond Accuracy: Behavioral Testing of NLP Models with CheckList](https://arxiv.org/abs/2005.04118){:target="_blank"} which breaks down behavioral testing into three types of tests:

- `#!js invariance`: Changes should not affect outputs.
```python linenums="1"
# INVariance via verb injection (changes should not affect outputs)
tokens = ["revolutionized", "disrupted"]
texts = [f"Transformers applied to NLP have {token} the ML field." for token in tokens]
[preprocessor.index_to_class[y_prob.argmax()] for y_prob in classifier_fn(texts=texts)]
```
<pre class="output">
['natural-language-processing', 'natural-language-processing']
</pre>
- `#!js directional`: Change should affect outputs.
```python linenums="1"
# DIRectional expectations (changes with known outputs)
tokens = ["text classification", "image classification"]
texts = [f"ML applied to {token}." for token in tokens]
[preprocessor.index_to_class[y_prob.argmax()] for y_prob in classifier_fn(texts=texts)]
```
<pre class="output">
['natural-language-processing', 'computer-vision']
</pre>
- `#!js minimum functionality`: Simple combination of inputs and expected outputs.
```python linenums="1"
# Minimum Functionality Tests (simple input/output pairs)
tokens = ["natural language processing", "mlops"]
texts = [f"{token} is the next big wave in machine learning." for token in tokens]
[preprocessor.index_to_class[y_prob.argmax()] for y_prob in classifier_fn(texts=texts)]
```
<pre class="output">
['natural-language-processing', 'mlops']
</pre>

> We'll learn how to systematically create tests in our [testing lesson](testing.md#behavioral-testing){:target="_blank"}.

## Online evaluation

Once we've evaluated our model's ability to perform on a static dataset we can run several types of **online evaluation** techniques to determine performance on actual production data. It can be performed using labels or, in the event we don't readily have labels, [proxy signals](monitoring.md#performance){:target="_blank"}.

- manually label a subset of incoming data to evaluate periodically.
- asking the initial set of users viewing a newly categorized content if it's correctly classified.
- allow users to report misclassified content by our model.

And there are many different experimentation strategies we can use to measure real-time performance before committing to replace our existing version of the system.

### AB tests
AB testing involves sending production traffic to our current system (control group) and the new version (treatment group) and measuring if there is a statistical difference between the values for two metrics. There are several common issues with AB testing such as accounting for different sources of bias, such as the novelty effect of showing some users the new system. We also need to ensure that the same users continue to interact with the same systems so we can compare the results without contamination.

<div class="ai-center-all">
    <img width="500" src="/static/images/mlops/systems-design/ab.png" alt="ab tests">
</div>

> In many cases, if we're simply trying to compare the different versions for a certain metric, AB testing can take while before we reach statical significance since traffic is evenly split between the different groups. In this scenario, [multi-armed bandits](https://en.wikipedia.org/wiki/Multi-armed_bandit){:target="_blank"} will be a better approach since they continuously assign traffic to the better performing version.

### Canary tests
Canary tests involve sending most of the production traffic to the currently deployed system but sending traffic from a small cohort of users to the new system we're trying to evaluate. Again we need to make sure that the same users continue to interact with the same system as we gradually roll out the new system.

<div class="ai-center-all">
    <img width="500" src="/static/images/mlops/systems-design/canary.png" alt="canary deployment">
</div>

### Shadow tests
Shadow testing involves sending the same production traffic to the different systems. We don't have to worry about system contamination and it's very safe compared to the previous approaches since the new system's results are not served. However, we do need to ensure that we're replicating as much of the production system as possible so we can catch issues that are unique to production early on. But overall, shadow testing is easy to monitor, validate operational consistency, etc.

<div class="ai-center-all">
    <img width="500" src="/static/images/mlops/systems-design/shadow.png" alt="shadow deployment">
</div>

!!! question "What can go wrong?"
    If shadow tests allow us to test our updated system without having to actually serve the new results, why doesn't everyone adopt it?

    ??? quote "Show answer"
        With shadow deployment, we'll miss out on any live feedback signals (explicit/implicit) from our users since users are not directly interacting with the product using our new version.

        We also need to ensure that we're replicating as much of the production system as possible so we can catch issues that are unique to production early on. This is rarely possible because, while your ML system may be a standalone microservice, it ultimately interacts with an intricate production environment that has *many* dependencies.

## Capability vs. alignment

We've seen the many different metrics that we'll want to calculate when it comes to evaluating our model but not all metrics mean the same thing. And this becomes very important when it comes to choosing the "*best*" model(s).

- **capability**: the ability of our model to perform a task, measured by the objective function we optimize for (ex. log loss)
- **alignment**: desired behavior of our model, measure by metrics that are not differentiable or don't account for misclassifications and probability differences (ex. accuracy, precision, recall, etc.)

While capability (ex. loss) and alignment (ex. accuracy) metrics may seem to be aligned, their differences can indicate issues in our data:

- ↓ accuracy, ↑ loss = large errors on lots of data (worst case)
- ↓ accuracy, ↓ loss = small errors on lots of data, distributions are close but tipped towards misclassifications (misaligned)
- ↑ accuracy, ↑ loss = large errors on some data (incorrect predictions have very skewed distributions)
- ↑ accuracy, ↓ loss = no/few errors on some data (best case)


## Resources

- [Model Evaluation, Model Selection, and Algorithm Selection in Machine Learning](https://arxiv.org/abs/1811.12808){:target="_blank"}
- [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599){:target="_blank"}
- [Confident Learning: Estimating Uncertainty in Dataset Labels](https://arxiv.org/abs/1911.00068){:target="_blank"}
- [Automated Data Slicing for Model Validation](https://arxiv.org/abs/1807.06068){:target="_blank"}
- [SliceLine: Fast, Linear-Algebra-based Slice Finding for ML Model Debugging](https://mboehm7.github.io/resources/sigmod2021b_sliceline.pdf){:target="_blank"}
- [Distributionally Robust Neural Networks for Group Shifts](https://arxiv.org/abs/1911.08731){:target="_blank"}
- [No Subclass Left Behind: Fine-Grained Robustness in Coarse-Grained Classification Problems](https://arxiv.org/abs/2011.12945){:target="_blank"}
- [Model Patching: Closing the Subgroup Performance Gap with Data Augmentation](https://arxiv.org/abs/2008.06775){:target="_blank"}

<!-- Course signup -->
{% include "templates/signup.md" %}

<!-- Citation -->
{% include "templates/cite.md" %}