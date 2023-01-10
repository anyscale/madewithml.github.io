---
template: lesson.html
title: Evaluating Machine Learning Models
description: Evaluating ML models by assessing overall, per-class and slice performances.
keywords: evaluation, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/mlops.png
repository: https://github.com/GokuMohandas/mlops-course
notebook: https://github.com/GokuMohandas/mlops-course/blob/main/notebooks/tagifai.ipynb
---

{% include "styles/lesson.md" %}

## Intuition

Evaluation is an integral part of modeling and it's one that's often glossed over. We'll often find evaluation to involve simply computing the accuracy or other global metrics but for many real work applications, a much more nuanced evaluation process is required. However, before evaluating our model, we always want to:

- be clear about what metrics we are prioritizing
- be careful not to over optimize on any one metric because it may mean you're compromising something else

```python linenums="1"
# Metrics
metrics = {"overall": {}, "class": {}}
```
```python linenums="1"
# Data to evaluate
other_index = label_encoder.class_to_index["other"]
y_prob = model.predict_proba(X_test)
y_pred = custom_predict(y_prob=y_prob, threshold=threshold, index=other_index)
```

## Coarse-grained

While we were iteratively developing our baselines, our evaluation process involved computing the coarse-grained metrics such as overall precision, recall and f1 metrics.

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
    "precision": 0.8990934378802025,
    "recall": 0.8194444444444444,
    "f1": 0.838280325954406,
    "num_samples": 144.0
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
for i, _class in enumerate(label_encoder.classes):
    metrics["class"][_class] = {
        "precision": class_metrics[0][i],
        "recall": class_metrics[1][i],
        "f1": class_metrics[2][i],
        "num_samples": np.float64(class_metrics[3][i]),
    }
```
```python linenums="1"
# Metrics for a specific class
tag = "natural-language-processing"
print (json.dumps(metrics["class"][tag], indent=2))
```
<pre class="output">
{
  "precision": 0.9803921568627451,
  "recall": 0.8620689655172413,
  "f1": 0.9174311926605505,
  "num_samples": 58.0
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
    "precision": 0.9803921568627451,
    "recall": 0.8620689655172413,
    "f1": 0.9174311926605505,
    "num_samples": 58.0
  }
]
[
  "mlops",
  {
    "precision": 0.9090909090909091,
    "recall": 0.8333333333333334,
    "f1": 0.8695652173913043,
    "num_samples": 12.0
  }
]
[
  "computer-vision",
  {
    "precision": 0.975,
    "recall": 0.7222222222222222,
    "f1": 0.8297872340425532,
    "num_samples": 54.0
  }
]
[
  "other",
  {
    "precision": 0.4523809523809524,
    "recall": 0.95,
    "f1": 0.6129032258064516,
    "num_samples": 20.0
  }
]
</pre>

> Due to our custom predict function, we're able to achieve high precision for the categories except for `other`. Based on our [product design](design.md#metrics){:target="_blank"}, we decided that it's more important to be precise about our explicit ML categories (nlp, cv, and mlops) and that we would have a manual labeling workflow to recall any misclassifications in the `other` category. Overtime, our model will become better in this category as well.

## Confusion matrix

Besides just inspecting the metrics for each class, we can also identify the true positives, false positives and false negatives. Each of these will give us insight about our model beyond what the metrics can provide.

- **True positives (TP)**: learn about where our model performs well.
- **False positives (FP)**: potentially identify samples which may need to be relabeled.
- False negatives (FN): identify the model's less performant areas to oversample later.

> It's a good to have our FP/FN samples feed back into our annotation pipelines in the event we want to fix their labels and have those changes be reflected everywhere.

```python linenums="1"
# TP, FP, FN samples
tag = "mlops"
index = label_encoder.class_to_index[tag]
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
[1, 3, 4, 41, 47, 77, 94, 127, 138]
[14, 88]
[30, 71, 106]
</pre>

```python linenums="1"
index = tp[0]
print (X_test_raw[index])
print (f"true: {label_encoder.decode([y_test[index]])[0]}")
print (f"pred: {label_encoder.decode([y_pred[index]])[0]}")
```
<pre class="output">
pytest pytest framework makes easy write small tests yet scales support complex functional testing
true: mlops
pred: mlops
</pre>

```python linenums="1"
# Samples
num_samples = 3
cm = [(tp, "True positives"), (fp, "False positives"), (fn, "False negatives")]
for item in cm:
    if len(item[0]):
        print (f"\n=== {item[1]} ===")
        for index in item[0][:num_samples]:
            print (f"  {X_test_raw[index]}")
            print (f"    true: {label_encoder.decode([y_test[index]])[0]}")
            print (f"    pred: {label_encoder.decode([y_pred[index]])[0]}\n")
```
<pre class="output">
=== True positives ===
  pytest pytest framework makes easy write small tests yet scales support complex functional testing
    true: mlops
    pred: mlops

  test machine learning code systems minimal examples testing machine learning correct implementation expected learned behavior model performance
    true: mlops
    pred: mlops

  continuous machine learning cml cml helps organize mlops infrastructure top traditional software engineering stack instead creating separate ai platforms
    true: mlops
    pred: mlops


=== False positives ===
  paint machine learning web app allows create landscape painting style bob ross using deep learning model served using spell model server
    true: computer-vision
    pred: mlops


=== False negatives ===
  hidden technical debt machine learning systems using software engineering framework technical debt find common incur massive ongoing maintenance costs real world ml systems
    true: mlops
    pred: other

  neptune ai lightweight experiment management tool fits workflow
    true: mlops
    pred: other

</pre>

!!! tip
    It's a really good idea to do this kind of analysis using our rule-based approach to catch really obvious labeling errors.

## Confident learning

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
# y
y_prob = model.predict_proba(X_test)
print (np.shape(y_test))
print (np.shape(y_prob))
```

```python linenums="1"
# Used to show raw text
test_df = pd.DataFrame({"text": X_test_raw, "tag": label_encoder.decode(y_test)})
```

```python linenums="1"
# Tag to inspect
tag = "mlops"
index = label_encoder.class_to_index[tag]
indices = np.where(y_test==index)[0]
```

```python linenums="1"
# Confidence score for the correct class is below a threshold
low_confidence = []
min_threshold = 0.5
for i in indices:
    prob = y_prob[i][index]
    if prob <= 0.5:
        low_confidence.append({"text": test_df.text[i],
                               "true": label_encoder.index_to_class[y_test[i]],
                               "pred": label_encoder.index_to_class[y_pred[i]],
                               "prob": prob})
```

```python linenums="1"
low_confidence[0:5]
```

<pre class="output">
[{'pred': 'other',
  'prob': 0.41281721056332804,
  'text': 'neptune ai lightweight experiment management tool fits workflow',
  'true': 'mlops'}]
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

```bash
pip install cleanlab==1.0.1 -q
```
```python linenums="1"
import cleanlab
from cleanlab.pruning import get_noise_indices
```
```python linenums="1"
# Determine potential labeling errors
label_error_indices = get_noise_indices(
            s=y_test,
            psx=y_prob,
            sorted_index_method="self_confidence",
            verbose=0)
```

Not all of these are necessarily labeling errors but situations where the predicted probabilities were not so confident. Therefore, it will be useful to attach the predicted outcomes along side results. This way, we can know if we need to relabel, upsample, etc. as mitigation strategies to improve our performance.

```python linenums="1"
num_samples = 5
for index in label_error_indices[:num_samples]:
    print ("text:", test_df.iloc[index].text)
    print ("true:", test_df.iloc[index].tag)
    print ("pred:", label_encoder.decode([y_pred[index]])[0])
    print ()
```

<pre class="output">
text: module 2 convolutional neural networks cs231n lecture 5 move fully connected neural networks convolutional neural networks
true: computer-vision
pred: other
</pre>

> The operations in this section can be applied to entire labeled dataset to discover labeling errors via confidence learning.

## Manual slices

Just inspecting the overall and class metrics isn't enough to deploy our new version to production. There may be key slices of our dataset that we need to do really well on:

- Target / predicted classes (+ combinations)
- Features (explicit and implicit)
- Metadata (timestamps, sources, etc.)
- Priority slices / experience (minority groups, large customers, etc.)

An easy way to create and evaluate slices is to define slicing functions.

```bash
pip install snorkel==0.9.8 -q
```

```python linenums="1"
from snorkel.slicing import PandasSFApplier
from snorkel.slicing import slice_dataframe
from snorkel.slicing import slicing_function
```

```python linenums="1"
@slicing_function()
def nlp_cnn(x):
    """NLP Projects that use convolution."""
    nlp_projects = "natural-language-processing" in x.tag
    convolution_projects = "CNN" in x.text or "convolution" in x.text
    return (nlp_projects and convolution_projects)
```
```python linenums="1"
@slicing_function()
def short_text(x):
    """Projects with short titles and descriptions."""
    return len(x.text.split()) < 8  # less than 8 words
```

Here we're using Snorkel's [`slicing_function`](https://snorkel.readthedocs.io/en/latest/packages/_autosummary/slicing/snorkel.slicing.slicing_function.html){:target="blank"} to create our different slices. We can visualize our slices by applying this slicing function to a relevant DataFrame using [`slice_dataframe`](https://snorkel.readthedocs.io/en/latest/packages/_autosummary/slicing/snorkel.slicing.slice_dataframe.html){:target="_blank"}.

```python linenums="1"
nlp_cnn_df = slice_dataframe(test_df, nlp_cnn)
nlp_cnn_df[["text", "tag"]].head()
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
      <th>126</th>
      <td>understanding convolutional neural networks nl...</td>
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
      <th>tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>44</th>
      <td>flashtext extract keywords sentence replace ke...</td>
      <td>natural-language-processing</td>
    </tr>
    <tr>
      <th>62</th>
      <td>tudatasets collection benchmark datasets graph...</td>
      <td>other</td>
    </tr>
    <tr>
      <th>70</th>
      <td>tsfresh automatic extraction relevant features...</td>
      <td>other</td>
    </tr>
    <tr>
      <th>88</th>
      <td>axcell automatic extraction results machine le...</td>
      <td>computer-vision</td>
    </tr>
  </tbody>
</table>
</div></div>

We can define even more slicing functions and create a slices record array using the [`PandasSFApplier`](https://snorkel.readthedocs.io/en/latest/packages/_autosummary/slicing/snorkel.slicing.PandasSFApplier.html){:target="_blank"}. The slices array has N (# of data points) items and each item has S (# of slicing functions) items, indicating whether that data point is part of that slice. Think of this record array as a masking layer for each slicing function on our data.

```python linenums="1"
# Slices
slicing_functions = [nlp_cnn, short_text]
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
  "nlp_cnn": {
    "precision": 1.0,
    "recall": 1.0,
    "f1": 1.0,
    "num_samples": 1
  },
  "short_text": {
    "precision": 0.8,
    "recall": 0.8,
    "f1": 0.8000000000000002,
    "num_samples": 5
  }
}
</pre>

Slicing can help identify sources of *bias* in our data. For example, our model has most likely learned to associated algorithms with certain applications such as CNNs used for computer vision or transformers used for NLP projects. However, these algorithms are not being applied beyond their initial use cases. We’d need ensure that our model learns to focus on the application over algorithm. This could be learned with:

- enough data (new or oversampling incorrect predictions)
- masking the algorithm (using text matching heuristics)

## Generated slices

Manually creating slices is a massive improvement towards identifying problem subsets in our dataset compared to coarse-grained evaluation but what if there are problematic slices of our dataset that we failed to identify? [SliceLine](https://mboehm7.github.io/resources/sigmod2021b_sliceline.pdf){:target="_blank"} is a recent work that uses a linear-algebra and pruning based technique to identify large slices (specify minimum slice size) that result in meaningful errors from the forward pass. Without pruning, automatic slice identification becomes computationally intensive because it involves enumerating through many combinations of data points to identify the slices. But with this technique, we can discover hidden underperforming subsets in our dataset that we weren’t explicitly looking for!

<div class="ai-center-all">
    <img src="/static/images/mlops/evaluation/slicefinder.png" width="1000" alt="slicefinder GUI">
</div>
<div class="ai-center-all mt-1">
  <small>SliceFinder GUI<br><a href="https://arxiv.org/abs/1807.06068" target="_blank">Automated Data Slicing for Model Validation</a></small>
</div>

### Hidden stratification

What if the features to generate slices on are implicit/hidden?

<div class="ai-center-all">
    <img src="/static/images/mlops/evaluation/subgroups.png" width="1000" alt="Subgroup examples">
</div>
<div class="ai-center-all mt-1">
  <small><a href="https://arxiv.org/abs/1911.08731" target="_blank">Distributionally Robust Neural Networks for Group Shifts</a></small>
</div>

To address this, there are recent [clustering-based techniques](https://arxiv.org/abs/2011.12945){:target="_blank"} to identify these hidden slices and improve the system.

1. Estimate implicit subclass labels via unsupervised clustering
2. Train new more robust model using these clusters

<div class="ai-center-all">
    <img src="/static/images/mlops/evaluation/clustering.png" width="1000" alt="Identifying subgroups via clustering and training on them.">
</div>
<div class="ai-center-all mt-1">
  <small><a href="https://arxiv.org/abs/2011.12945" target="_blank">No Subclass Left Behind: Fine-Grained Robustness in Coarse-Grained Classification Problems</a></small>
</div>

### Model patching

Another recent work on [model patching](https://arxiv.org/abs/2008.06775){:target="_blank"} takes this another step further by learning how to transform between subgroups so we can train models on the augmented data:

1. Learn subgroups
2. Learn transformations (ex. [CycleGAN](https://junyanz.github.io/CycleGAN/){:target="_blank"}) needed to go from one subgroup to another under the same superclass (label)
3. Augment data with artificially introduced subgroup features
4. Train new robust model on augmented data

<div class="ai-center-all">
    <img src="/static/images/mlops/evaluation/model_patching.png" width="400" alt="Using learned subgroup transformations to augment data.">
</div>
<div class="ai-center-all mt-1">
  <small><a href="https://arxiv.org/abs/2008.06775" target="_blank">Model Patching: Closing the Subgroup Performance Gap with Data Augmentation</a></small>
</div>

## Interpretability

Besides just comparing predicted outputs with ground truth values, we can also inspect the inputs to our models. What aspects of the input are more influential towards the prediction? If the focus is not on the relevant features of our input, then we need to explore if there is a hidden pattern we're missing or if our model has learned to overfit on the incorrect features. We can use techniques such as [SHAP](https://github.com/slundberg/shap){:target="_blank"} (SHapley Additive exPlanations) or [LIME](https://github.com/marcotcr/lime){:target="_blank"} (Local Interpretable Model-agnostic Explanations) to inspect feature importance. On a high level, these techniques learn which features have the most signal by assessing the performance in their absence. These inspections can be performed on a global level (ex. per-class) or on a local level (ex. single prediction).

```bash
pip install lime==0.2.0.1 -q
```
```python linenums="1"
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline
```

It's easier to use LIME with scikit-learn [pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html){:target="_blank"} so we'll combine our vectorizer and model into one construct.

```python linenums="1"
# Create pipeline
pipe = make_pipeline(vectorizer, model)
```
```python linenums="1"
# Explain instance
text = "Using pretrained convolutional neural networks for object detection."
explainer = LimeTextExplainer(class_names=label_encoder.classes)
explainer.explain_instance(text, classifier_fn=pipe.predict_proba, top_labels=1).show_in_notebook(text=True)
```

<div class="ai-center-all">
    <img src="/static/images/mlops/evaluation/lime.png" width="1000" alt="LIME for ml interpretability">
</div>

> We can also use model-specific approaches to interpretability we we did in our [embeddings lesson](../foundations/embeddings.md#interpretability){:target="_blank"} to identify the most influential n-grams in our text.

## Counterfactuals

Another way to evaluate our systems is to identify counterfactuals -- data with similar features that belongs to another class (classification) or above a certain difference (regression). These points allow us to evaluate model sensitivity to certain features and feature values that may be signs of overfitting. A great tool to identify and probe for counterfactuals (also great for slicing and fairness metrics) is the [What-if tool](https://pair-code.github.io/what-if-tool/){:target="_blank"}.

<div class="ai-center-all">
    <img src="https://4.bp.blogspot.com/-hnqXfHQvl5I/W5b3f-hk0yI/AAAAAAAADUc/hBOXtobPdAUQ5aAG_xOwYf8AWp8YbL-kQCLcBGAs/s640/image2.gif" width="500" alt="Identifying counterfactuals using the What-if tool">
</div>
<div class="ai-center-all mt-2">
  <small><a href="https://ai.googleblog.com/2018/09/the-what-if-tool-code-free-probing-of.html" target="_blank">Identifying counterfactuals using the What-if tool</a></small>
</div>

> For our task, this can involve projects that use algorithms are typically reserved for a certain application area (such as CNNs for computer vision or transformers for NLP).

## Behavioral testing

Besides just looking at metrics, we also want to conduct some behavioral sanity tests. Behavioral testing is the process of testing input data and expected outputs while treating the model as a black box. They don't necessarily have to be adversarial in nature but more along the types of perturbations we'll see in the real world once our model is deployed. A landmark paper on this topic is [Beyond Accuracy: Behavioral Testing of NLP Models with CheckList](https://arxiv.org/abs/2005.04118){:target="_blank"} which breaks down behavioral testing into three types of tests:

- `#!js invariance`: Changes should not affect outputs.
```python linenums="1"
# INVariance via verb injection (changes should not affect outputs)
tokens = ["revolutionized", "disrupted"]
texts = [f"Transformers applied to NLP have {token} the ML field." for token in tokens]
predict_tag(texts=texts)
```
<pre class="output">
['natural-language-processing', 'natural-language-processing']
</pre>
- `#!js directional`: Change should affect outputs.
```python linenums="1"
# DIRectional expectations (changes with known outputs)
tokens = ["text classification", "image classification"]
texts = [f"ML applied to {token}." for token in tokens]
predict_tag(texts=texts)
```
<pre class="output">
['natural-language-processing', 'computer-vision']
</pre>
- `#!js minimum functionality`: Simple combination of inputs and expected outputs.
```python linenums="1"
# Minimum Functionality Tests (simple input/output pairs)
tokens = ["natural language processing", "mlops"]
texts = [f"{token} is the next big wave in machine learning." for token in tokens]
predict_tag(texts=texts)
```
<pre class="output">
['natural-language-processing', 'mlops']
</pre>

> We'll learn how to systematically create tests in our [testing lesson](testing.md#behavioral-testing){:target="_blank"}.

## Evaluating evaluations

How can we know if our models and systems are performing better over time? Unfortunately, depending on how often we retrain or how quickly our dataset grows, it won't always be a simple decision where all metrics/slices are performing better than the previous version. In these scenarios, it's important to know what our main priorities are and where we can have some leeway:

- What criteria are most important?
- What criteria can/cannot regress?
- How much of a regression can be tolerated?

```python linenums="1"
assert precision > prev_precision  # most important, cannot regress
assert recall >= best_prev_recall - 0.03  # recall cannot regress > 3%
assert metrics["class"]["nlp"]["f1"] > prev_nlp_f1  # priority class
assert metrics["slices"]["class"]["nlp_cnn"]["f1"] > prev_nlp_cnn_f1  # priority slice
```

And as we develop these criteria over time, we can systematically enforce them via [CI/CD workflows](cicd.md){:target="_blank"} to decrease the manual time in between system updates.

!!! question "Seems straightforward, doesn't it?"
    With all these different evaluation methods, how can we choose "the best" version of our model if some versions are better for some evaluation criteria?

    ??? quote "Show answer"
        The team needs to agree on what evaluation criteria are most important and what is the minimum performance required for each one. This will allow us to filter amongst all the different solutions by removing ones that don't satisfy all the minimum requirements and ranking amongst the remaining by which ones perform the best for the highest priority criteria.

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

## Model CI

An effective way to evaluate our systems is to encapsulate them as a collection (suite) and use them for [continuous integration](cicd.md){:target="_blank"}. We would continue to add to our evaluation suites and they would be executed whenever we are experimenting with changes to our system (new models, data, etc.). Often, problematic slices of data identified during [monitoring](monitoring.md){:target="blank"} are often added to the evaluation test suite to avoid repeating the same regressions in the future.

## Capability vs. alignment

We've seen the many different metrics that we'll want to calculate when it comes to evaluating our model but not all metrics mean the same thing. And this becomes very important when it comes to choosing the "*best*" model(s).

- **capability**: the ability of our model to perform a task, measured by the objective function we optimize for (ex. log loss)
- **alignment**: desired behavior of our model, measure by metrics that are not differentiable or don't account for misclassifications and probability differences (ex. accuracy, precision, recall, etc.)

While our capability (ex. loss) and alignment (ex. accuracy) metrics seem to be aligned, their differences can indicate issues in our system:

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
{% include "templates/course-signup.md" %}

<!-- Citation -->
{% include "templates/cite.md" %}