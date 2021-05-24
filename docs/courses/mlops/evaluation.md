---
template: lesson.html
title: Evaluating ML Models
description: Evaluating ML models by assessing overall, per-class and slice performances.
keywords: evaluation, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/mlops.png
repository: https://github.com/GokuMohandas/MLOps
video: https://www.youtube.com/watch?v=AwajdDVR_C4
---

{% include "styles/lesson.md" %}

## Intuition

Evaluation is an integral part of modeling and it's one that's often glossed over. We'll often find evaluation to involve simply computing the accuracy or other global metrics but for many real work applications, a much more nuanced evaluation process is required. However, before evaluating our model, we always want to:

- be clear about what metrics we are prioritizing
- be careful not to over optimize on any one metric

## Overall

While we were iteratively developing our baselines, our evaluation process involved computing the overall precision, recall and f1 metrics.

```python linenums="1"
# Evaluate
metrics = precision_recall_fscore_support(y_test, y_pred, average="weighted")
performance = {"precision": metrics[0], "recall": metrics[1], "f1": metrics[2]}
print (json.dumps(performance, indent=2))
```
<pre class="output">
{
  "precision": 0.7839452388425872,
  "recall": 0.6081370449678801,
  "f1": 0.6677329148413014
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

Inspecting these overall metrics is a start but we can go deeper by evaluating the same metrics at the per-class level.

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
tag = "transformers"
print (json.dumps(metrics["class"][tag], indent=2))
```
<pre class="output">
{
  "precision": 0.6785714285714286,
  "recall": 0.6785714285714286,
  "f1": 0.6785714285714286,
  "num_samples": 28.0
}
</pre>

## Confusion matrix sample analysis

Besides just inspecting the metrics for each class, we can also identify the true positives, false positives and false negatives. Each of these will give us insight about our model beyond what the metrics can provide.

- True positives: learn about where our model performs well.
- False positives: potentially identify samples which may need to be relabeled.
- False negatives: identify the model's less performant areas to oversample later.

!!! note
    It's a good to have our FP/FN samples feed back into our annotation pipelines in the event we want to fix their labels and have those changes be reflected everywhere.

```python linenums="1"
# TP, FP, FN samples
index = label_encoder.class_to_index[tag]
tp, fp, fn = [], [], []
for i in range(len(y_test)):
    true = y_test[i][index]
    pred = y_pred[i][index]
    if true and pred:
        tp.append(i)
    elif not true and pred:
        fp.append(i)
    elif true and not pred:
        fn.append(i)
```
```python linenums="1"
print (tp)
print (fp)
print (fn)
```
<pre class="output">
[12, 24, 31, 49, 54, 58, 65, 89, 104, 122, 127, 148, 149, 152, 165, 172, 184, 198, 201]
[46, 73, 85, 92, 119, 130, 144, 169, 211]
[22, 27, 36, 66, 90, 135, 154, 200, 212]
</pre>

```python linenums="1"
index = tp[0]
print (X_test[index])
print (f"true: {label_encoder.decode([y_test[index]])[0]}")
print (f"pred: {label_encoder.decode([y_pred[index]])[0]}\n")
```
<pre class="output">
haystack neural question answering scale transformers scale question answering search
true: ['huggingface', 'natural-language-processing', 'question-answering', 'transformers']
pred: ['huggingface', 'natural-language-processing', 'question-answering', 'transformers']
</pre>

```python linenums="1"
# Sorted tags
sorted_tags_by_f1 = OrderedDict(sorted(
        metrics["class"].items(), key=lambda tag: tag[1]["f1"], reverse=True))
```
```python linenums="1"
# Samples
num_samples = 3
if len(tp):
    print ("\n=== True positives ===")
    for i in tp[:num_samples]:
        print (f"  {X_test[i]}")
        print (f"    true: {label_encoder.decode([y_test[i]])[0]}")
        print (f"    pred: {label_encoder.decode([y_pred[i]])[0]}\n")
if len(fp):
    print ("=== False positives === ")
    for i in fp[:num_samples]:
        print (f"  {X_test[i]}")
        print (f"    true: {label_encoder.decode([y_test[i]])[0]}")
        print (f"    pred: {label_encoder.decode([y_pred[i]])[0]}\n")
if len(fn):
    print ("=== False negatives ===")
    for i in fn[:num_samples]:
        print (f"  {X_test[i]}")
        print (f"    true: {label_encoder.decode([y_test[i]])[0]}")
        print (f"    pred: {label_encoder.decode([y_pred[i]])[0]}\n")
```
<pre class="output">
=== True positives ===
  haystack neural question answering scale transformers scale question answering search
    true: ['huggingface', 'natural-language-processing', 'question-answering', 'transformers']
    pred: ['huggingface', 'natural-language-processing', 'question-answering', 'transformers']

  build sota conversational ai transfer learning train dialog agent leveraging transfer learning openai gpt gpt 2 transformer language model
    true: ['language-modeling', 'natural-language-processing', 'transfer-learning', 'transformers']
    pred: ['attention', 'huggingface', 'language-modeling', 'natural-language-processing', 'transfer-learning', 'transformers']

  ways compress bert post list briefly taxonomize papers seen compressing bert
    true: ['attention', 'natural-language-processing', 'pretraining', 'transformers']
    pred: ['attention', 'natural-language-processing', 'transformers']

=== False positives ===
  shakespeare meets googleflax application rnns flax character level language model
    true: ['natural-language-processing']
    pred: ['attention', 'language-modeling', 'natural-language-processing', 'transformers']

  help read text summarization using flask huggingface text summarization translation questions answers generation using huggingface deployed using flask streamlit detailed guide github
    true: ['huggingface', 'natural-language-processing']
    pred: ['huggingface', 'natural-language-processing', 'transformers']

  zero shot text classification generative language models overview text generation approach zero shot text classification gpt 2
    true: ['natural-language-processing']
    pred: ['language-modeling', 'natural-language-processing', 'transformers']

=== False negatives ===
  electra explaining new self supervised task language representation learning electra uses replace token detection
    true: ['attention', 'generative-adversarial-networks', 'language-modeling', 'natural-language-processing', 'representation-learning', 'transformers']
    pred: ['self-supervised-learning']

  scitldr extreme summarization scientific documents new automatic summarization task high source compression requiring expert background knowledge complex language understanding
    true: ['natural-language-processing', 'transformers']
    pred: ['natural-language-processing']

  gpt3 works visualizations animations compilation threads explaining gpt3
    true: ['natural-language-processing', 'transformers']
    pred: ['natural-language-processing']
</pre>

!!! note
    It's a really good idea to do this kind of analysis using our rule-based approach to catch really obvious labeling errors.


## Slices

Just inspecting the overall and class metrics isn't enough to deploy our new version to production. There may be key slices of our dataset that we expect to do really well on (ie. minority groups, large customers, etc.) and we need to ensure that their metrics are also improving. An easy way to create and evaluate slices is to define slicing functions.

```python linenums="1"
from snorkel.slicing import PandasSFApplier
from snorkel.slicing import slice_dataframe
from snorkel.slicing import slicing_function
```

```python linenums="1"
@slicing_function()
def cv_transformers(x):
    """Projects with the `computer-vision` and `transformers` tags."""
    return all(tag in x.tags for tag in ["computer-vision", "transformers"])
```

```python linenums="1"
@slicing_function()
def short_text(x):
    """Projects with short titles and descriptions."""
    return len(x.text.split()) < 7  # less than 7 words
```

Here we're using Snorkel's [`slicing_function`](https://snorkel.readthedocs.io/en/v0.9.3/packages/_autosummary/slicing/snorkel.slicing.slicing_function.html){:target="blank"} to create our different slices. We can visualize our slices by applying this slicing function to a relevant DataFrame using [`slice_dataframe`](https://snorkel.readthedocs.io/en/v0.9.3/packages/_autosummary/slicing/snorkel.slicing.slice_dataframe.html){:target="_blank"}.

```python linenums="1"
short_text_df = slice_dataframe(test_df, short_text)
short_text_df[["text", "tags"]].head()
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
      <th>6</th>
      <td>autokeras automl library deep learning</td>
      <td>[keras]</td>
    </tr>
    <tr>
      <th>50</th>
      <td>tune sklearn scikit learn api raytune</td>
      <td>[scikit-learn]</td>
    </tr>
    <tr>
      <th>146</th>
      <td>g simclr tensorflow implementation g simclr</td>
      <td>[computer-vision, self-supervised-learning, te...</td>
    </tr>
  </tbody>
</table>
</div></div>


We can define even more slicing functions and create a slices record array using the [`PandasSFApplier`](https://snorkel.readthedocs.io/en/v0.9.6/packages/_autosummary/slicing/snorkel.slicing.PandasSFApplier.html){:target="_blank"}. The slices array has N (# of data points) items and each item has S (# of slicing functions) items, indicating whether that data point is part of that slice. Think of this record array as a masking layer for each slicing function on our data.

```python linenums="1"
# Slices
slicing_functions = [cv_transformers, short_text]
applier = PandasSFApplier(slicing_functions)
slices = applier.apply(test_df)
print (slices)
```

<pre class="output">
[(0, 0) (0, 1) (0, 0) (1, 0) (0, 0) (0, 0) (0, 0) (0, 0) (0, 0) (0, 0)
 (1, 0) (0, 0) (0, 1) (0, 0) (0, 0) (1, 0) (0, 0) (0, 0) (0, 1) (0, 0)
 ...
 (0, 0) (1, 0) (0, 0) (0, 0) (0, 0) (0, 0) (0, 0) (0, 0) (0, 0) (0, 0)
 (0, 0) (0, 0) (1, 0) (0, 0) (0, 0) (0, 0) (1, 0)]
</pre>

If our task was multiclass instead of multilabel, we could've used [snorkel.analysis.Scorer](https://snorkel.readthedocs.io/en/v0.9.1/packages/_autosummary/analysis/snorkel.analysis.Scorer.html){:target="_blank"} to retrieve our slice metrics. But we've implemented a naive version for our multilabel task based on it.

```python linenums="1"
# Score slices
metrics["slices"] = {}
metrics["slices"]["class"] = {}
for slice_name in slices.dtype.names:
    mask = slices[slice_name].astype(bool)
    if sum(mask):
        slice_metrics = precision_recall_fscore_support(
            y_test[mask], y_pred[mask], average="micro"
        )
        metrics["slices"]["class"][slice_name] = {}
        metrics["slices"]["class"][slice_name]["precision"] = slice_metrics[0]
        metrics["slices"]["class"][slice_name]["recall"] = slice_metrics[1]
        metrics["slices"]["class"][slice_name]["f1"] = slice_metrics[2]
        metrics["slices"]["class"][slice_name]["num_samples"] = len(y_true[mask])
```
```python linenums="1"
# Weighted overall slice metrics
metrics["slices"]["overall"] = {}
for metric in ["precision", "recall", "f1"]:
    metrics["slices"]["overall"][metric] = np.mean(
        list(
            itertools.chain.from_iterable(
                [
                    [metrics["slices"]["class"][slice_name][metric]]
                    * metrics["slices"]["class"][slice_name]["num_samples"]
                    for slice_name in metrics["slices"]["class"]
                ]
            )
        )
    )
```

```python linenums="1"
print(json.dumps(metrics["slices"], indent=2))
```

<pre class="output">
{
  "class": {
    "cv_transformers": {
      "precision": 1.0,
      "recall": 0.75,
      "f1": 0.8571428571428571,
      "num_samples": 1
    },
    "short_text": {
      "precision": 0.6666666666666666,
      "recall": 0.8,
      "f1": 0.7272727272727272,
      "num_samples": 3
    }
  },
  "overall": {
    "precision": 0.7499999999999999,
    "recall": 0.7875000000000001,
    "f1": 0.7597402597402596
  }
}
</pre>

## Extensions

We've explored user generated slices but there is currently quite a bit of research on automatically generated slices and overall model robustness. A notable toolkit is the [Robustness Gym](https://arxiv.org/abs/2101.04840){:target="_blank"} which programmatically builds slices, performs adversarial attacks, rule-based data augmentation, benchmarking, reporting and much more.

<div class="ai-center-all">
    <img width="600" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/testing/gym.png">
</div>
<div class="ai-center-all mt-2">
    <a href="https://arxiv.org/abs/2101.04840" target="_blank">Robustness Gym slice builders</a>
</div>

Instead of passively observing slice performance, we could try and improve them. Usually, a slice may exhibit poor performance when there are too few samples and so a natural approach is to oversample. However, these methods change the underlying data distribution and can cause issues with overall / other slices. It's also not scalable to train a separate model for each unique slice and combine them via [Mixture of Experts (MoE)](https://www.cs.toronto.edu/~hinton/csc321/notes/lec15.pdf){:target="_blank"}. To combat all of these technical challenges and more, the Snorkel team introduced the [Slice Residual Attention Modules (SRAMs)](https://arxiv.org/abs/1909.06349){:target="_blank"}, which can sit on any backbone architecture (ie. our CNN feature extractor) and learn slice-aware representations for the class predictions.

<div class="ai-center-all">
    <img width="600" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/testing/sram.png">
</div>
<div class="ai-center-all mt-3">
    <a href="https://arxiv.org/abs/1909.06349" target="_blank">Slice Residual Attention Modules (SRAMs)</a>
</div>

!!! note
    In our [testing lesson](testing.md){:target="_blank"}, we'll cover another way to evaluate our model known as [behavioral testing](testing.md#behavioral-testing){:target="_blank"}, which we'll also include as part of performance report.

## Resources

- [Slice-based Learning: A Programming Model for Residual Learning in Critical Data Slices](https://papers.nips.cc/paper/2019/file/351869bde8b9d6ad1e3090bd173f600d-Paper.pdf){:target="_blank"}
- [Robustness Gym: Unifying the NLP Evaluation Landscape](https://arxiv.org/abs/2101.04840){:target="_blank"}


<!-- Citation -->
{% include "cite.md" %}