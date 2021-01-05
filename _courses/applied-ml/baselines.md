---
layout: page
title: Baselines · Applied ML
description: Motivating the use of baselines for iterative modeling.
image: /static/images/applied_ml.png
redirect_from: /courses/applied-ml-in-production/baselines/

course-url: /courses/applied-ml/
next-lesson-url: /courses/applied-ml/experiment-tracking/
---

<!-- Header -->
<div class="row">
  <div class="col-md-8 col-6 mr-auto">
    <h1 class="page-title">{{ page.title | split: " · " | first }}</h1>
  </div>
  <div class="col-md-4 col-6">
    <div class="btn-group float-right mb-0" role="group">
      <a href="{% link index.md %}" class="btn btn-sm btn-outline-secondary"><i
          class="fas fa-sm fa-arrow-left mr-1"></i>Return home</a>
    </div>
  </div>
</div>
<hr class="mt-0">

- [Intuition](#intuition)
- [Application](#application)
  - [Random](#random)
  - [Rule-based](#rule-based)
  - [Simple ML](#simple-ml)
  - [CNN w/ embeddings](#cnn)
  - [RNN w/ embeddings](#rnn)
  - [Transformers w/ contextual embeddings](#transformers)
  - [Tradeoffs](#tradeoffs)

<h3 id="intuition"><u>Intuition</u></h3>

**What are they?** Baselines are simple benchmarks which pave the way for iterative development.

**Why do we need them?**
- Rapid experimentation via hyperparameter tuning thanks to low model complexity.
- Discovery of data issues, false assumptions, bugs in code, etc. since model itself is not complex.
- [Pareto's principle](https://en.wikipedia.org/wiki/Pareto_principle){:target="_blank"}: we can achieve decent performance with minimal initial effort.

**How do we use them?**
1. Start with the simplest possible baseline to compare subsequent development with. This is often a random (chance) model.
2. Develop a rule-based approach (when possible) using IFTT, auxiliary data, etc.
3. Slowly add complexity by *addressing* limitations and *motivating* representations and model architectures.
4. Weigh *tradeoffs* (performance, latency, size, etc.) between performant baselines.
5. Revisit and iterate on baselines as your dataset grows.

> 🔄 You can also baseline on your dataset. Instead of using a fixed dataset and iterating on the models, choose a good baseline and iterate on the dataset.
- remove or fix data samples (FP, FN)
- prepare and transform features
- expand or consolidate classes
- auxiliary datasets

**Tradeoffs to consider**

When choosing what model architecture(s) to proceed with, there are a few important aspects to consider:
- `performance`: consider overall and fine-grained (ex. per-class) performance.
- `latency`: how quickly does your model respond for inference.
- `size`: how large is your model and can you support it's storage.
- `compute`: how much will it cost ($, carbon footprint, etc.) to train your model?
- `interpretability`: does your model need to explain its predictions?
- `bias checks`: does your model pass key bias checks?
- `🕓 to develop`: how long do you have to develop the first version?
- `🕓 to retrain`: how long does it take to retrain your model? This is very important to consider if you need to retrain often.
- `maintenance overhead`: who and what will be required to maintain your model versions because the real work with ML begins after deploying v1. You can't just hand it off to your site reliability team to maintain it like many teams do with traditional software.

<h3 id="application"><u>Application</u></h3>

> <i class="fab fa-github ai-color-black mr-1"></i>: Code for this lesson can be found here: [applied-ml/tagifai.ipynb](https://colab.research.google.com/github/GokuMohandas/applied-ml/blob/main/notebooks/tagifai.ipynb){:target="_blank"}

Each application's baseline trajectory varies based on the task and motivations. For our application, we're going to follow this path:
1. [Random](#random)
2. [Rule-based](#rule-based)
3. [Simple ML](#simple-ml)
4. [CNN w/ embeddings](#cnn)
5. [RNN w/ embeddings](#rnn)
6. [Transformers w/ contextual embeddings](#transformers)

We'll motivate the need for slowly adding complexity from both the *representation* (ex. embeddings) and *architecture* (ex. CNNs) views, as well as address the limitation at each step of the way.

> If you're unfamiliar with of the concepts here, be sure to check out the [GokuMohandas/madewithml](https://github.com/GokuMohandas/madewithml){:target="_blank"} (🔥 Among [top ML repos](https://github.com/topics/deep-learning){:target="_blank"} on GitHub).


We'll first set up some functions that we'll be using across the different baseline experiments.
```python
from sklearn.metrics import precision_recall_fscore_support
import torch
```
```python
def set_seeds(seed=1234):
    """Set seeds for reproducability."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # multi-GPU
```
```python
def get_data_splits(df, train_size=0.7):
    """"""
    # Get data
    X = df.text.to_numpy()
    y = df.tags

    # Binarize y
    label_encoder = LabelEncoder()
    label_encoder.fit(y)
    y = label_encoder.encode(y)

    # Split
    X_train, X_, y_train, y_ = iterative_train_test_split(
        X, y, train_size=train_size)
    X_val, X_test, y_val, y_test = iterative_train_test_split(
        X_, y_, train_size=0.5)

    return X_train, X_val, X_test, y_train, y_val, y_test, label_encoder
```
```python
def get_performance(y_true, y_pred, classes):
    """Per-class performance metrics."""
    # Get metrics
    performance = {'overall': {}, 'class': {}}
    metrics = precision_recall_fscore_support(y_true, y_pred)

    # Overall performance
    performance['overall']['precision'] = np.mean(metrics[0])
    performance['overall']['recall'] = np.mean(metrics[1])
    performance['overall']['f1'] = np.mean(metrics[2])
    performance['overall']['num_samples'] = np.float64(np.sum(metrics[3]))

    # Per-class performance
    for i in range(len(classes)):
        performance['class'][classes[i]] = {
            "precision": metrics[0][i],
            "recall": metrics[1][i],
            "f1": metrics[2][i],
            "num_samples": np.float64(metrics[3][i])
        }

    return performance
```

<hr>

<h3 id="random">Random</h3>

<u><i>motivation</i></u>: We want to know what random (chance) performance looks like. All of our efforts should be well above this.

```python
# Set seeds
set_seeds()
```
```python
# Get data splits
preprocessed_df = df.copy()
preprocessed_df.text = preprocessed_df.text.apply(preprocess, lower=True, stem=True)
X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = get_data_splits(preprocessed_df)
print (f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print (f"X_val: {X_val.shape}, y_val: {y_val.shape}")
print (f"X_test: {X_test.shape}, y_test: {y_test.shape}")
```
<pre class="output">
X_train: (1000,), y_train: (1000, 35)
X_val: (227,), y_val: (227, 35)
X_test: (217,), y_test: (217, 35)
</pre>
```python
# Label encoder
print (label_encoder)
print (label_encoder.classes)
```
<pre class="output">
<LabelEncoder(num_classes=35)>
['attention', 'autoencoders', 'computer-vision', 'convolutional-neural-networks', 'data-augmentation', 'embeddings', 'flask', 'generative-adversarial-networks', 'graph-neural-networks', 'graphs', 'huggingface', 'image-classification', 'interpretability', 'keras', 'language-modeling', 'natural-language-processing', 'node-classification', 'object-detection', 'pretraining', 'production', 'pytorch', 'question-answering', 'regression', 'reinforcement-learning', 'representation-learning', 'scikit-learn', 'segmentation', 'self-supervised-learning', 'tensorflow', 'tensorflow-js', 'time-series', 'transfer-learning', 'transformers', 'unsupervised-learning', 'wandb']
</pre>
```python
# Generate random predictions
y_pred = np.random.randint(low=0, high=2, size=(len(y_test), len(label_encoder.classes)))
print (y_pred.shape)
print (y_pred[0:5])
```
<pre class="output">
(217, 35)
[[0 0 1 1 1 0 0 0 1 1 0 0 1 1 1 0 1 1 1 1 1 1 0 0 1 0 1 1 0 1 1 0 0 0 1]
 [0 1 0 0 0 0 0 1 1 1 1 1 0 1 1 1 1 0 1 1 1 1 1 0 1 1 0 1 0 0 0 0 1 1 1]
 [1 1 0 1 0 0 0 1 1 1 0 0 1 1 1 0 0 1 0 0 1 1 1 1 1 0 1 1 0 0 1 0 0 1 1]
 [0 1 1 0 1 1 0 0 1 1 0 1 1 0 1 1 1 0 0 0 1 1 1 0 1 1 0 0 0 1 0 0 1 1 0]
 [0 0 1 1 1 0 1 1 0 1 0 1 1 0 1 1 1 0 0 1 0 1 1 1 1 1 1 1 0 0 0 0 1 1 0]]
</pre>
```python
# Evaluate
performance = get_performance(
    y_true=y_test, y_pred=y_pred, classes=label_encoder.classes)
print (json.dumps(performance['overall'], indent=2))
```
<pre class="output">
{
  "precision": 0.0662941602216066,
  "recall": 0.5065299488415251,
  "f1": 0.10819194263879019,
  "num_samples": 480.0
}
</pre>

We made the assumption that there is an equal probability for whether an input has a tag or not but this isn't true. Let's use the **train split** to figure out what the true probability is.

```python
# Percentage of 1s (tag presence)
tag_p = np.sum(np.sum(y_train)) / (len(y_train) * len(label_encoder.classes))
print (tag_p)
```
<pre class="output">
0.06291428571428571
</pre>
```python
# Generate weighted random predictions
y_pred = np.random.choice(
    np.arange(0, 2), size=(len(y_test), len(label_encoder.classes)),
    p=[1-tag_p, tag_p])
```
```python
# Validate percentage
np.sum(np.sum(y_pred)) / (len(y_pred) * len(label_encoder.classes))
```
<pre class="output">
0.06240947992100066
</pre>
```python
# Evaluate
performance = get_performance(
    y_true=y_test, y_pred=y_pred, classes=label_encoder.classes)
print (json.dumps(performance['overall'], indent=2))
```
<pre class="output">
{
  "precision": 0.060484184552507536,
  "recall": 0.053727634571230636,
  "f1": 0.048704498064854516,
  "num_samples": 480.0
}
</pre>

<u><i>limitations</i></u>: we didn't use the tokens in our input to affect our predictions so nothing was learned.

<hr>

<h3 id="rule-based">Rule-based</h3>
- [Unstemmed](#unstemmed)
- [Stemmed](#stemmed)
- [Evaluation](#evaluation)
- [Inference](#inference)

<u><i>motivation</i></u>: we want to use signals in our inputs (along with domain expertise and auxiliary data) to determine the labels.

```python
# Set seeds
set_seeds()
```

<h4 id="unstemmed">Unstemmed</h4>

```python
# Get data splits
preprocessed_df = df.copy()
preprocessed_df.text = preprocessed_df.text.apply(preprocess, lower=True)
X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = get_data_splits(preprocessed_df)
```
```python
# Restrict to relevant tags
print (len(tags_dict))
tags_dict = {tag: tags_dict[tag] for tag in label_encoder.classes}
print (len(tags_dict))
```
<pre class="output">
400
35
</pre>
```python
# Map aliases
aliases = {}
for tag, values in tags_dict.items():
    aliases[preprocess(tag)] = tag
    for alias in values['aliases']:
        aliases[preprocess(alias)] = tag
aliases
```
<pre class="output">
{'ae': 'autoencoders',
 'attention': 'attention',
 'autoencoders': 'autoencoders',
 'cnn': 'convolutional-neural-networks',
 'computer vision': 'computer-vision',
 ...
 'unsupervised learning': 'unsupervised-learning',
 'vision': 'computer-vision',
 'wandb': 'wandb',
 'weights biases': 'wandb'}
</pre>
```python
def get_classes(text, aliases, tags_dict):
    """If a token matches an alias,
    then add the corresponding tag
    class (and parent tags if any)."""
    classes = []
    for alias, tag in aliases.items():
        if alias in text:
            classes.append(tag)
            for parent in tags_dict[tag]["parents"]:
                classes.append(parent)
    return list(set(classes))
```
```python
# Sample
text = "This project extends gans for data augmentation specifically for object detection tasks."
get_classes(text=preprocess(text), aliases=aliases, tags_dict=tags_dict)
```
<pre class="output">
['object-detection',
 'data-augmentation',
 'generative-adversarial-networks',
 'computer-vision']
</pre>
```python
# Prediction
y_pred = []
for text in X_test:
    classes = get_classes(text, aliases, tags_dict)
    y_pred.append(classes)
```
```python
# Encode labels
y_pred = label_encoder.encode(y_pred)
```
```python
# Evaluate
performance = get_performance(y_true=y_test, y_pred=y_pred, classes=label_encoder.classes)
print (json.dumps(performance['overall'], indent=4))
```
<pre class="output">
{
    "precision": 0.8527917293434535,
    "recall": 0.38066760941576216,
    "f1": 0.48975323243320396,
    "num_samples": 480.0
}
</pre>
```python
# Inspection
tag = "transformers"
print (json.dumps(performance["class"][tag], indent=2))
```
<pre class="output">
{
  "precision": 1.0,
  "recall": 0.32,
  "f1": 0.48484848484848486,
  "num_samples": 25.0
}
</pre>

<h4 id="stemmed">Stemmed</h4>

Before we do a more involved analysis, let's see if we can do better. We're looking for exact matches with the aliases which isn't always perfect, for example:
```python
print (aliases[preprocess('gan')])
# print (aliases[preprocess('gans')]) # this won't find any match
print (aliases[preprocess('generative adversarial networks')])
# print (aliases[preprocess('generative adversarial network')]) # this won't find any match
```
<pre class="output">
generative-adversarial-networks
generative-adversarial-networks
</pre>
We don't want to keep adding explicit rules but we can use [stemming](https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html){:target="_blank"} to represent different forms of a word uniformly, for example:
```python
print (porter.stem("democracy"))
print (porter.stem("democracies"))
```
<pre class="output">
democraci
democraci
</pre>
So let's now stem our aliases as well as the tokens in our input text and then look for matches.
```python
# Get data splits
preprocessed_df = df.copy()
preprocessed_df.text = preprocessed_df.text.apply(preprocess, lower=True, stem=True)
X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = get_data_splits(preprocessed_df)
```
```python
# Map aliases
aliases = {}
for tag, values in tags_dict.items():
    aliases[preprocess(tag, stem=True)] = tag
    for alias in values['aliases']:
        aliases[preprocess(alias, stem=True)] = tag
aliases
```
<pre class="output">
{'ae': 'autoencoders',
 'attent': 'attention',
 'autoencod': 'autoencoders',
 'cnn': 'convolutional-neural-networks',
 'comput vision': 'computer-vision',
 ...
 'vision': 'computer-vision',
 'wandb': 'wandb',
 'weight bias': 'wandb'}
</pre>
```python
# Checks (we will write proper tests soon)
print (aliases[preprocess('gan', stem=True)])
print (aliases[preprocess('gans', stem=True)])
print (aliases[preprocess('generative adversarial network', stem=True)])
print (aliases[preprocess('generative adversarial networks', stem=True)])
```
<pre class="output">
generative-adversarial-networks
generative-adversarial-networks
generative-adversarial-networks
generative-adversarial-networks
</pre>
> We'll write proper tests for all of these functions when we move our code to Python scripts.
```python
# Sample
text = "This project extends gans for data augmentation specifically for object detection tasks."
get_classes(text=preprocess(text, stem=True), aliases=aliases, tags_dict=tags_dict)
```
<pre class="output">
['object-detection',
 'data-augmentation',
 'generative-adversarial-networks',
 'computer-vision']
</pre>
```python
# Prediction
y_pred = []
for text in X_test:
    classes = get_classes(text, aliases, tags_dict)
    y_pred.append(classes)
```
```python
# Encode labels
y_pred = label_encoder.encode(y_pred)
```

<h4 id="evaluation">Evaluation</h4>

```python
# Evaluate
performance = get_performance(y_true=y_test, y_pred=y_pred, classes=label_encoder.classes)
print (json.dumps(performance['overall'], indent=4))
```
<pre class="output">
{
    "precision": 0.8405837971552256,
    "recall": 0.48656350456551384,
    "f1": 0.5794244643481148,
    "num_samples": 473.0
}
</pre>
```python
# Inspection
tag = "transformers"
print (json.dumps(performance["class"][tag], indent=2))
```
<pre class="output">
{
  "precision": 0.9285714285714286,
  "recall": 0.48148148148148145,
  "f1": 0.6341463414634146,
  "num_samples": 27.0
}
</pre>
```python
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
```python
print (tp)
print (fp)
print (fn)
```
<pre class="output">
[1, 14, 15, 28, 46, 54, 94, 160, 165, 169, 190, 194, 199]
[49]
[4, 18, 61, 63, 72, 75, 89, 99, 137, 141, 142, 163, 174, 206]
</pre>
```python
index = tp[0]
print (X_test[index])
print (f"true: {label_encoder.decode([y_test[index]])[0]}")
print (f"pred: {label_encoder.decode([y_pred[index]])[0]}\n")
```
<pre class="output">
insight project insight design creat nlp servic code base front end gui streamlit backend server fastapi usag transform
true: ['attention', 'huggingface', 'natural-language-processing', 'pytorch', 'transfer-learning', 'transformers']
pred: ['natural-language-processing', 'transformers']
</pre>
```python
# Sorted tags
sorted_tags_by_f1 = OrderedDict(sorted(
        performance['class'].items(), key=lambda tag: tag[1]['f1'], reverse=True))
```
```python
@widgets.interact(tag=list(sorted_tags_by_f1.keys()))
def display_tag_analysis(tag='transformers'):
    # Performance
    print (json.dumps(performance["class"][tag], indent=2))

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

    # Samples
    num_samples = 3
    if len(tp):
        print ("\n=== True positives ===\n")
        for i in tp[:num_samples]:
            print (f"  {X_test[i]}")
            print (f"    true: {label_encoder.decode([y_test[i]])[0]}")
            print (f"    pred: {label_encoder.decode([y_pred[i]])[0]}\n")
    if len(fp):
        print ("=== False positives ===\n")
        for i in fp[:num_samples]:
            print (f"  {X_test[i]}")
            print (f"    true: {label_encoder.decode([y_test[i]])[0]}")
            print (f"    pred: {label_encoder.decode([y_pred[i]])[0]}\n")
    if len(fn):
        print ("=== False negatives ===\n")
        for i in fn[:num_samples]:
            print (f"  {X_test[i]}")
            print (f"    true: {label_encoder.decode([y_test[i]])[0]}")
            print (f"    pred: {label_encoder.decode([y_pred[i]])[0]}\n")
```
This is the output for the `transformers` tag:
<pre class="output">
{
  "precision": 0.9285714285714286,
  "recall": 0.48148148148148145,
  "f1": 0.6341463414634146,
  "num_samples": 27.0
}

=== True positives ===

  insight project insight design creat nlp servic code base front end gui streamlit backend server fastapi usag transform
    true: ['attention', 'huggingface', 'natural-language-processing', 'pytorch', 'transfer-learning', 'transformers']
    pred: ['natural-language-processing', 'transformers']

  hyperparamet optim transform guid basic grid search optim fact hyperparamet choos signific impact final model perform
    true: ['natural-language-processing', 'transformers']
    pred: ['natural-language-processing', 'transformers']

  transform neural network architectur explain time explain transform work look easi explan exactli right
    true: ['attention', 'natural-language-processing', 'transformers']
    pred: ['natural-language-processing', 'transformers']

=== False positives ===

  multi target albument mani imag mani mask bound box key point transform sync
    true: ['computer-vision', 'data-augmentation']
    pred: ['natural-language-processing', 'transformers']

=== False negatives ===

  size fill blank multi mask fill roberta size fill blank condit text fill idea fill miss word sentenc probabl choic word
    true: ['attention', 'huggingface', 'language-modeling', 'natural-language-processing', 'transformers']
    pred: []

  gpt3 work visual anim compil thread explain gpt3
    true: ['natural-language-processing', 'transformers']
    pred: []

  tinybert tinybert 7 5x smaller 9 4x faster infer bert base achiev competit perform task natur languag understand
    true: ['attention', 'natural-language-processing', 'transformers']
    pred: []
</pre>

> You can use false positives/negatives to discover potential errors in annotation. This can be especially useful when analyzing FP/FNs from rule-based approaches.

Though we achieved decent precision, the recall is quite low. This is because rule-based approaches can yield labels with high certainty when there is an absolute condition match but it fails to generalize or learn implicit patterns.

<h4 id="inference">Inference</h4>

```python
# Infer
text = "Transfer learning with transformers for self-supervised learning"
print (preprocess(text, stem=True))
get_classes(text=preprocess(text, stem=True), aliases=aliases, tags_dict=tags_dict)
```
<pre class="output">
transfer learn transform self supervis learn
['self-supervised-learning',
 'transfer-learning',
 'transformers',
 'natural-language-processing']
</pre>
Now let's see what happens when we replace the word *transformers* with *BERT*. Sure we can add this as an alias but we can't keep doing this. This is where it makes sense to learn from the data as opposed to creating explicit rules.
```python
# Infer
text = "Transfer learning with BERT for self-supervised learning"
print (preprocess(text, stem=True))
get_classes(text=preprocess(text, stem=True), aliases=aliases, tags_dict=tags_dict)
```
<pre class="output">
transfer learn bert self supervis learn
['self-supervised-learning', 'transfer-learning']
</pre>


<u><i>limitations</i></u>: we failed to generalize or learn any implicit patterns to predict the labels because we treat the tokens in our input as isolated entities.

> We would ideally spend more time tuning our model because it's so simple and quick to train. This approach also applies to all the other models we'll look at as well.

<hr>

<h4 id="simple-ml">Simple ML</h4>
<u><i>motivation</i></u>:
- *representation*: use term frequency-inverse document frequency [(TF-IDF)](https://en.wikipedia.org/wiki/Tf%E2%80%93idf){:target="_blank"} to capture the significance of a token to a particular input with respect to all the inputs, as opposed to treating the words in our input text as isolated tokens.
- *architecture*: we want our model to meaningfully extract the encoded signal to predict the output labels.

So far we've treated the words in our input text as isolated tokens and we haven't really captured any meaning between tokens. Let's use term frequency–inverse document frequency (**TF-IDF**) to capture the significance of a token to a particular input with respect to all the inputs.

$$ w_{i, j} = \text{tf}_{i, j} * log(\frac{N}{\text{df}_i}) $$

<div class="ai-center-all">
<table class="mathjax-table">
  <tbody>
    <tr>
      <td>$$ w_{i, j} $$</td>
      <td>$$ \text{tf-idf weight for term i in document j} $$</td>
    </tr>
    <tr>
      <th>$$ \text{tf}_{i, j} $$</th>
      <th>$$ \text{# of times term i appear in document j} $$</th>
    </tr>
    <tr>
      <td>$$ N $$</td>
      <td>$$ \text{total # of documents} $$</td>
    </tr>
    <tr>
      <td>$$ \text{df}_i $$</td>
      <td>$$ \text{# of documents with token i} $$</td>
    </tr>
  </tbody>
</table>
</div>


```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
```
```python
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import MultiLabelBinarizer
```
```python
# Set seeds
set_seeds()
```
```python
# Get data splits
preprocessed_df = df.copy()
preprocessed_df.text = preprocessed_df.text.apply(preprocess, lower=True, stem=True)
X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = get_data_splits(preprocessed_df)
```
```python
# Tf-idf
vectorizer = TfidfVectorizer()
print (X_train[0])
X_train = vectorizer.fit_transform(X_train)
X_val = vectorizer.transform(X_val)
X_test = vectorizer.transform(X_test)
print (X_train.shape)
print (X_train[0]) # scipy.sparse.csr_matrix
```
<pre class="output">
albument fast imag augment librari easi use wrapper around librari
(1000, 2654)
  (0, 190)	0.34307733697679055
  (0, 2630)	0.3991510203964918
  (0, 2522)	0.14859192074955896
  (0, 728)	0.29210630687446
  (0, 1356)	0.4515371929370289
  (0, 217)	0.2870036535570893
  (0, 1157)	0.18851186612963625
  (0, 876)	0.31431481238098835
  (0, 118)	0.44156912440424356
</pre>
```python
def fit_and_evaluate(model):
    """Fit and evaluate each model."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    performance = get_performance(
        y_true=y_test, y_pred=y_pred, classes=list(label_encoder.classes))
    return performance['overall']
```
```python
# Models
performance = {}
performance['logistic-regression'] = fit_and_evaluate(OneVsRestClassifier(
    LogisticRegression(), n_jobs=1))
performance['k-nearest-neighbors'] = fit_and_evaluate(
    KNeighborsClassifier())
performance['random-forest'] = fit_and_evaluate(
    RandomForestClassifier(n_jobs=-1))
performance['gradient-boosting-machine'] = fit_and_evaluate(OneVsRestClassifier(
    GradientBoostingClassifier()))
performance['support-vector-machine'] = fit_and_evaluate(OneVsRestClassifier(
    LinearSVC(), n_jobs=-1))
print (json.dumps(performance, indent=2))
```
<pre class="output">
{
  "logistic-regression": {
    "precision": 0.3563624338624338,
    "recall": 0.0858365150175495,
    "f1": 0.13067443826527078,
    "num_samples": 480.0
  },
  "k-nearest-neighbors": {
    "precision": 0.6172562358276645,
    "recall": 0.3213868500136974,
    "f1": 0.400741288236766,
    "num_samples": 480.0
  },
  "random-forest": {
    "precision": 0.5851306333244963,
    "recall": 0.21548369514995133,
    "f1": 0.29582560665419344,
    "num_samples": 480.0
  },
  "gradient-boosting-machine": {
    "precision": 0.7104917071723794,
    "recall": 0.5106819976684509,
    "f1": 0.575225354377256,
    "num_samples": 480.0
  },
  "support-vector-machine": {
    "precision": 0.8059313061625735,
    "recall": 0.40445445906037036,
    "f1": 0.5164548230244397,
    "num_samples": 480.0
  }
}
</pre>

<u><i>limitations</i></u>:
- *representation*: TF-IDF representations don't encapsulate much signal beyond frequency but we require more fine-grained token representations.
- *architecture*: we want to develop models that can use better represented encodings in a more contextual manner.


<hr>

<h3 id="cnn">CNN w/ Embeddings</h3>
- [Set up](#setup_cnn)
- [Tokenizer](#tokenizer_cnn)
- [Data imbalance](#imbalance_cnn)
- [Datasets](#datasets_cnn)
- [Model](#model_cnn)
- [Training](#training_cnn)
- [Evaluation](#evaluation_cnn)
- [Inference](#inference_cnn)


<u><i>motivation</i></u>:
- *representation*: we want to have more robust (split tokens to characters) and meaningful [embeddings]({% link _courses/ml-foundations/embeddings.md %}){:target="_blank"} representations for our input tokens.
- *architecture*: we want to process our encoded inputs using [convolution (CNN)]({% link _courses/ml-foundations/convolutional-neural-networks.md %}){:target="_blank"} filters that can learn to analyze windows of embedded tokens to extract meaningful signal.


<h4 id="setup_cnn">Set up</h4>

We'll set up the task by setting seeds for reproducability, creating our data splits abd setting the device.
```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
```
```python
# Set seeds
set_seeds()
```
```python
# Get data splits
preprocessed_df = df.copy()
preprocessed_df.text = preprocessed_df.text.apply(preprocess, lower=True)
X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = get_data_splits(preprocessed_df)
X_test_raw = X_test
```
```python
# Set device
cuda = True
device = torch.device('cuda' if (
    torch.cuda.is_available() and cuda) else 'cpu')
torch.set_default_tensor_type('torch.FloatTensor')
if device.type == 'cuda':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
print (device)
```
<pre class="output">
cuda
</pre>

<h4 id="tokenizer_cnn">Tokenizer</h4>

We're going to tokenize our input text as character tokens so we can be robust to spelling errors and learn to generalize across tags. (ex. learning that RoBERTa, or any other future BERT based archiecture, warrants same tag as BERT).

<div class="ai-center-all">
    <img width="500" src="https://raw.githubusercontent.com/GokuMohandas/madewithml/main/images/ml-foundations/cnn/inputs.png">
</div>

```python
class Tokenizer(object):
    def __init__(self, char_level, num_tokens=None,
                 pad_token='<PAD>', oov_token='<UNK>',
                 token_to_index=None):
        self.char_level = char_level
        self.separator = '' if self.char_level else ' '
        if num_tokens: num_tokens -= 2 # pad + unk tokens
        self.num_tokens = num_tokens
        self.oov_token = oov_token
        if not token_to_index:
            token_to_index = {'<PAD>': 0, '<UNK>': 1}
        self.token_to_index = token_to_index
        self.index_to_token = {v: k for k, v in self.token_to_index.items()}

    def __len__(self):
        return len(self.token_to_index)

    def __str__(self):
        return f"<Tokenizer(num_tokens={len(self)})>"

    def fit_on_texts(self, texts):
        if self.char_level:
            all_tokens = [token for text in texts for token in text]
        if not self.char_level:
            all_tokens = [token for text in texts for token in text.split(' ')]
        counts = Counter(all_tokens).most_common(self.num_tokens)
        self.min_token_freq = counts[-1][1]
        for token, count in counts:
            index = len(self)
            self.token_to_index[token] = index
            self.index_to_token[index] = token
        return self

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            if not self.char_level:
                text = text.split(' ')
            sequence = []
            for token in text:
                sequence.append(self.token_to_index.get(
                    token, self.token_to_index[self.oov_token]))
            sequences.append(np.asarray(sequence))
        return sequences

    def sequences_to_texts(self, sequences):
        texts = []
        for sequence in sequences:
            text = []
            for index in sequence:
                text.append(self.index_to_token.get(index, self.oov_token))
            texts.append(self.separator.join([token for token in text]))
        return texts

    def save(self, fp):
        with open(fp, 'w') as fp:
            contents = {
                'char_level': self.char_level,
                'oov_token': self.oov_token,
                'token_to_index': self.token_to_index
            }
            json.dump(contents, fp, indent=4, sort_keys=False)

    @classmethod
    def load(cls, fp):
        with open(fp, 'r') as fp:
            kwargs = json.load(fp=fp)
        return cls(**kwargs)
```
```python
# Tokenize
char_level = True
tokenizer = Tokenizer(char_level=char_level)
tokenizer.fit_on_texts(texts=X_train)
vocab_size = len(tokenizer)
print (tokenizer)
```
<pre class="output">
&lt;Tokenizer(num_tokens=39)&gt;
</pre>
```python
tokenizer.token_to_index
```
<pre class="output">
{' ': 2,
 '0': 30,
 '1': 31,
 '2': 26,
 ...
 '&lt;PAD&gt;': 0,
 '&lt;UNK&gt;': 1,
 ...
 'x': 25,
 'y': 21,
 'z': 27}
</pre>
```python
# Convert texts to sequences of indices
X_train = np.array(tokenizer.texts_to_sequences(X_train))
X_val = np.array(tokenizer.texts_to_sequences(X_val))
X_test = np.array(tokenizer.texts_to_sequences(X_test))
preprocessed_text = tokenizer.sequences_to_texts([X_train[0]])[0]
print ("Text to indices:\n"
    f"  (preprocessed) → {preprocessed_text}\n"
    f"  (tokenized) → {X_train[0]}")
```
<pre class="output">
Text to indices:
  (preprocessed) → albumentations fast image augmentation library easy use wrapper around libraries
  (tokenized) → [ 7 11 20 17 16  3  5  6  7  6  4 10  5  9  2 19  7  9  6  2  4 16  7 14
  3  2  7 17 14 16  3  5  6  7  6  4 10  5  2 11  4 20  8  7  8 21  2  3
  7  9 21  2 17  9  3  2 23  8  7 13 13  3  8  2  7  8 10 17  5 15  2 11
  4 20  8  7  8  4  3  9]
</pre>

<h4 id="imbalance_cnn">Data imbalance</h4>

We'll factor class weights in our objective function ([binary cross entropy with logits](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html){:target="_blank"}) to help with class imbalance. There are many other techniques such as over sampling from underrepresented classes, undersampling, etc. but we'll cover these in a separate unit lesson on data imbalance.

```python
# Class weights
counts = np.bincount([label_encoder.class_to_index[class_] for class_ in all_tags])
class_weights = {i: 1.0/count for i, count in enumerate(counts)}
print (f"class counts: {counts},\nclass weights: {class_weights}")
```
<pre class="output">
class counts: [120  41 388 106  41  75  34  73  51  78  64  51  55  93  51 429  33  69
  30  51 258  32  49  59  57  60  48  40 213  40  34  46 196  39  39],
class weights: {0: 0.008333333333333333, 1: 0.024390243902439025, 2: 0.002577319587628866, 3: 0.009433962264150943, 4: 0.024390243902439025, 5: 0.013333333333333334, 6: 0.029411764705882353, 7: 0.0136986301369863, 8: 0.0196078431372549, 9: 0.01282051282051282, 10: 0.015625, 11: 0.0196078431372549, 12: 0.01818181818181818, 13: 0.010752688172043012, 14: 0.0196078431372549, 15: 0.002331002331002331, 16: 0.030303030303030304, 17: 0.014492753623188406, 18: 0.03333333333333333, 19: 0.0196078431372549, 20: 0.003875968992248062, 21: 0.03125, 22: 0.02040816326530612, 23: 0.01694915254237288, 24: 0.017543859649122806, 25: 0.016666666666666666, 26: 0.020833333333333332, 27: 0.025, 28: 0.004694835680751174, 29: 0.025, 30: 0.029411764705882353, 31: 0.021739130434782608, 32: 0.00510204081632653, 33: 0.02564102564102564, 34: 0.02564102564102564}
</pre>

<h4 id="datasets_cnn">Datasets</h4>

We're going to place our data into a [`Dataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset){:target="_blank"} and use a [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader){:target="_blank"} to efficiently create batches for training and evaluation.

```python
def pad_sequences(sequences, max_seq_len=0):
    """Pad sequences to max length in sequence."""
    max_seq_len = max(max_seq_len, max(len(sequence) for sequence in sequences))
    padded_sequences = np.zeros((len(sequences), max_seq_len))
    for i, sequence in enumerate(sequences):
        padded_sequences[i][:len(sequence)] = sequence
    return padded_sequences
```
```python
class CNNTextDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, max_filter_size):
        self.X = X
        self.y = y
        self.max_filter_size = max_filter_size

    def __len__(self):
        return len(self.y)

    def __str__(self):
        return f"<Dataset(N={len(self)})>"

    def __getitem__(self, index):
        X = self.X[index]
        y = self.y[index]
        return [X, y]

    def collate_fn(self, batch):
        """Processing on a batch."""
        # Get inputs
        X = np.array(batch, dtype=object)[:, 0]
        y = np.stack(np.array(batch, dtype=object)[:, 1], axis=0)

        # Pad inputs
        X = pad_sequences(sequences=X, max_seq_len=self.max_filter_size)

        # Cast
        X = torch.LongTensor(X.astype(np.int32))
        y = torch.FloatTensor(y.astype(np.int32))

        return X, y

    def create_dataloader(self, batch_size, shuffle=False, drop_last=False):
        return torch.utils.data.DataLoader(
            dataset=self,
            batch_size=batch_size,
            collate_fn=self.collate_fn,
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=True)
```
```python
# Create datasets
filter_sizes = list(range(1, 11))
batch_size = 64
train_dataset = CNNTextDataset(
    X=X_train, y=y_train, max_filter_size=max(filter_sizes))
val_dataset = CNNTextDataset(
    X=X_val, y=y_val, max_filter_size=max(filter_sizes))
test_dataset = CNNTextDataset(
    X=X_test, y=y_test, max_filter_size=max(filter_sizes))
print ("Data splits:\n"
    f"  Train dataset:{train_dataset.__str__()}\n"
    f"  Val dataset: {val_dataset.__str__()}\n"
    f"  Test dataset: {test_dataset.__str__()}\n"
    "Sample point:\n"
    f"  X: {train_dataset[0][0]}\n"
    f"  y: {train_dataset[0][1]}")
```
<pre class="output">
Data splits:
  Train dataset:<Dataset(N=1000)>
  Val dataset: <Dataset(N=227)>
  Test dataset: <Dataset(N=217)>
Sample point:
  X: [ 7 11 20 17 16  3  5  6  7  6  4 10  5  9  2 19  7  9  6  2  4 16  7 14
  3  2  7 17 14 16  3  5  6  7  6  4 10  5  2 11  4 20  8  7  8 21  2  3
  7  9 21  2 17  9  3  2 23  8  7 13 13  3  8  2  7  8 10 17  5 15  2 11
  4 20  8  7  8  4  3  9]
  y: [0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
</pre>
```python
# Create dataloaders
train_dataloader = train_dataset.create_dataloader(
    batch_size=batch_size)
val_dataloader = val_dataset.create_dataloader(
    batch_size=batch_size)
test_dataloader = test_dataset.create_dataloader(
    batch_size=batch_size)
batch_X, batch_y = next(iter(train_dataloader))
print ("Sample batch:\n"
    f"  X: {list(batch_X.size())}\n"
    f"  y: {list(batch_y.size())}")
```
<pre class="output">
Sample batch:
  X: [64, 186]
  y: [64, 35]
</pre>

<h4 id="model_cnn">Model</h4>

We'll be using a convolutional neural network on top of our embedded tokens to extract meaningful spatial signal. This time, we'll be using many filter widths to act as n-gram feature extractors. If you're not familiar with CNNs be sure to check out the [CNN lesson](https://madewithml.com/courses/ml-foundations/convolutional-neural-networks/){:target="_blank"} where we walkthrough every component of the architecture.

<div class="ai-center-all">
    <img width="500" src="https://raw.githubusercontent.com/GokuMohandas/madewithml/main/images/ml-foundations/cnn/convolution.gif">
</div>

Let's visualize the model's forward pass.

1. We'll first tokenize our inputs (`batch_size`, `max_seq_len`).
2. Then we'll embed our tokenized inputs (`batch_size`, `max_seq_len`, `embedding_dim`).
3. We'll apply convolution via filters (`filter_size`, `vocab_size`, `num_filters`) followed by batch normalization. Our filters act as character level n-gram detecors. We have three different filter sizes (2, 3 and 4) and they will act as bi-gram, tri-gram and 4-gram feature extractors, respectivelyy.
4. We'll apply 1D global max pooling which will extract the most relevant information from the feature maps for making the decision.
5. We feed the pool outputs to a fully-connected (FC) layer (with dropout).
6. We use one more FC layer with softmax to derive class probabilities.

<div class="ai-center-all">
    <img width="1000" src="https://raw.githubusercontent.com/GokuMohandas/madewithml/main/images/ml-foundations/embeddings/model.png">
</div>

```python
# Arguments
embedding_dim = 128
num_filters = 128
hidden_dim = 128
dropout_p = 0.5
```
```python
class CNN(nn.Module):
    def __init__(self, embedding_dim, vocab_size, num_filters, filter_sizes,
                 hidden_dim, dropout_p, num_classes, padding_idx=0):
        super(CNN, self).__init__()

        # Initialize embeddings
        self.embeddings = nn.Embedding(
                embedding_dim=embedding_dim, num_embeddings=vocab_size,
                padding_idx=padding_idx)

        # Conv weights
        self.filter_sizes = filter_sizes
        self.conv = nn.ModuleList(
            [nn.Conv1d(in_channels=embedding_dim,
                       out_channels=num_filters,
                       kernel_size=f) for f in filter_sizes])

        # FC weights
        self.dropout = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(num_filters*len(filter_sizes), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, inputs, channel_first=False):

        # Embed
        x_in, = inputs
        x_in = self.embeddings(x_in)
        if not channel_first:
            x_in = x_in.transpose(1, 2)  # (N, channels, sequence length)

        z = []
        max_seq_len = x_in.shape[2]
        for i, f in enumerate(self.filter_sizes):

            # `SAME` padding
            padding_left = int(
                (self.conv[i].stride[0]*(max_seq_len-1) - max_seq_len + self.filter_sizes[i])/2)
            padding_right = int(math.ceil(
                (self.conv[i].stride[0]*(max_seq_len-1) - max_seq_len + self.filter_sizes[i])/2))

            # Conv
            _z = self.conv[i](F.pad(x_in, (padding_left, padding_right)))

            # Pool
            _z = F.max_pool1d(_z, _z.size(2)).squeeze(2)
            z.append(_z)

        # Concat outputs
        z = torch.cat(z, 1)

        # FC
        z = self.fc1(z)
        z = self.dropout(z)
        z = self.fc2(z)

        return z
```
<pre class="output">
</pre>

* **VALID**: no padding, the filters only use the "valid" values in the input. If the filter cannot reach all the input values (filters go left to right), the extra values on the right are dropped.
* **SAME**: adds padding evenly to the right (preferred) and left sides of the input so that all values in the input are processed.

<div class="ai-center-all">
    <img width="500" src="https://raw.githubusercontent.com/GokuMohandas/madewithml/main/images/ml-foundations/cnn/padding.png">
</div>

We're add `SAME` padding so that the convolutional outputs are the same shape as our inputs. The amount of padding for the SAME padding can be determined using the same equation. We want out output to have the same width as our input, so we solve for P:

$$ \frac{W-F+2P}{S} + 1 = W $$

$$ P = \frac{S(W-1) - W + F}{2} $$

If $$P$$ is not a whole number, we round up (using `math.ceil`) and place the extra padding on the right side.

```python
# Initialize model
model = CNN(
    embedding_dim=embedding_dim, vocab_size=vocab_size,
    num_filters=num_filters, filter_sizes=filter_sizes,
    hidden_dim=hidden_dim, dropout_p=dropout_p,
    num_classes=num_classes)
model = model.to(device)
print (model.named_parameters)
```
<pre class="output">
bound method Module.named_parameters of CNN(
  (embeddings): Embedding(39, 128, padding_idx=0)
  (conv): ModuleList(
    (0): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
    (1): Conv1d(128, 128, kernel_size=(2,), stride=(1,))
    (2): Conv1d(128, 128, kernel_size=(3,), stride=(1,))
    (3): Conv1d(128, 128, kernel_size=(4,), stride=(1,))
    (4): Conv1d(128, 128, kernel_size=(5,), stride=(1,))
    (5): Conv1d(128, 128, kernel_size=(6,), stride=(1,))
    (6): Conv1d(128, 128, kernel_size=(7,), stride=(1,))
    (7): Conv1d(128, 128, kernel_size=(8,), stride=(1,))
    (8): Conv1d(128, 128, kernel_size=(9,), stride=(1,))
    (9): Conv1d(128, 128, kernel_size=(10,), stride=(1,))
  )
  (dropout): Dropout(p=0.5, inplace=False)
  (fc1): Linear(in_features=1280, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=35, bias=True)
)
</pre>

<h4 id="training_cnn">Training</h4>

We'll define a Trainer object which we will use for training, validation and inference.

```python
# Arguments
lr = 2e-4
num_epochs = 200
patience = 10
```
```python
class Trainer(object):
    def __init__(self, model, device, loss_fn=None, optimizer=None, scheduler=None):

        # Set params
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train_step(self, dataloader):
        """Train step."""
        # Set model to train mode
        self.model.train()
        loss = 0.0

        # Iterate over train batches
        for i, batch in enumerate(dataloader):

            # Step
            batch = [item.to(self.device) for item in batch]  # Set device
            inputs, targets = batch[:-1], batch[-1]
            self.optimizer.zero_grad()  # Reset gradients
            z = self.model(inputs)  # Forward pass
            J = self.loss_fn(z, targets)  # Define loss
            J.backward()  # Backward pass
            self.optimizer.step()  # Update weights

            # Cumulative Metrics
            loss += (J.detach().item() - loss) / (i + 1)

        return loss

    def eval_step(self, dataloader):
        """Validation or test step."""
        # Set model to eval mode
        self.model.eval()
        loss = 0.0
        y_trues, y_probs = [], []

        # Iterate over val batches
        with torch.no_grad():
            for i, batch in enumerate(dataloader):

                # Step
                batch = [item.to(self.device) for item in batch]  # Set device
                inputs, y_true = batch[:-1], batch[-1]
                z = self.model(inputs)  # Forward pass
                J = self.loss_fn(z, y_true).item()

                # Cumulative Metrics
                loss += (J - loss) / (i + 1)

                # Store outputs
                y_prob = torch.sigmoid(z).cpu().numpy()
                y_probs.extend(y_prob)
                y_trues.extend(y_true.cpu().numpy())

        return loss, np.vstack(y_trues), np.vstack(y_probs)

    def predict_step(self, dataloader):
        """Prediction step."""
        # Set model to eval mode
        self.model.eval()
        y_probs = []

        # Iterate over val batches
        with torch.no_grad():
            for i, batch in enumerate(dataloader):

                # Forward pass w/ inputs
                inputs, targets = batch[:-1], batch[-1]
                y_prob = self.model(inputs)

                # Store outputs
                y_probs.extend(y_prob)

        return np.vstack(y_probs)

    def train(self, num_epochs, patience, train_dataloader, val_dataloader):
        best_val_loss = np.inf
        for epoch in range(num_epochs):
            # Steps
            train_loss = self.train_step(dataloader=train_dataloader)
            val_loss, _, _ = self.eval_step(dataloader=val_dataloader)
            self.scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = self.model
                _patience = patience  # reset _patience
            else:
                _patience -= 1
            if not _patience:  # 0
                print("Stopping early!")
                break

            # Logging
            print(
                f"Epoch: {epoch+1} | "
                f"train_loss: {train_loss:.5f}, "
                f"val_loss: {val_loss:.5f}, "
                f"lr: {self.optimizer.param_groups[0]['lr']:.2E}, "
                f"_patience: {_patience}"
            )
        return best_model
```
```python
# Define loss
class_weights_tensor = torch.Tensor(np.array(list(class_weights.values())))
loss = nn.BCEWithLogitsLoss(weight=class_weights_tensor)
```
```python
# Define optimizer & scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=5)
```
```python
# Trainer module
trainer = Trainer(
    model=model, device=device, loss_fn=loss_fn,
    optimizer=optimizer, scheduler=scheduler)
```
```python
# Train
best_model = trainer.train(
    num_epochs, patience, train_dataloader, val_dataloader)
```
<pre class="output">
Epoch: 1 | train_loss: 0.00624, val_loss: 0.00285, lr: 2.00E-04, _patience: 10
Epoch: 2 | train_loss: 0.00401, val_loss: 0.00283, lr: 2.00E-04, _patience: 10
Epoch: 3 | train_loss: 0.00362, val_loss: 0.00266, lr: 2.00E-04, _patience: 10
Epoch: 4 | train_loss: 0.00332, val_loss: 0.00263, lr: 2.00E-04, _patience: 10
...
Epoch: 49 | train_loss: 0.00061, val_loss: 0.00149, lr: 2.00E-05, _patience: 4
Epoch: 50 | train_loss: 0.00055, val_loss: 0.00159, lr: 2.00E-05, _patience: 3
Epoch: 51 | train_loss: 0.00056, val_loss: 0.00152, lr: 2.00E-05, _patience: 2
Epoch: 52 | train_loss: 0.00057, val_loss: 0.00156, lr: 2.00E-05, _patience: 1
Stopping early!
</pre>

<h4 id="evaluation_cnn">Evaluation</h4>

```python
from pathlib import Path
from sklearn.metrics import precision_recall_curve
```
```python
# Threshold-PR curve
train_loss, y_true, y_prob = trainer.eval_step(dataloader=train_dataloader)
precisions, recalls, thresholds = precision_recall_curve(y_true.ravel(), y_prob.ravel())
plt.plot(thresholds, precisions[:-1], "r--", label="Precision")
plt.plot(thresholds, recalls[:-1], "b-", label="Recall")
plt.ylabel("Performance")
plt.xlabel("Threshold")
plt.legend(loc='best')
```
<div class="ai-center-all">
    <img width="400" src="data:image/png;base64,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bDnw9kGDbPvatftvO0xNmjRh0aJFjB8/niZNmtCmTRsyMzPZtWsXXbt2pWHDhtStW5e+ffvu+Z6nn36aOnXqcNppp/HAAw8AcP311zN48GAAHnjgATIzM6lbty733nsvAI899hjPPfccALNmzeLMM8+kbt26XHLJJawPLYh67rnncv/999OoUSNOPvlkvvrqq8M+HudcbNxyiw0ffe+92L5vUgwfjSc5OTmMGjWKFi1aAPDNN98wd+5catSoQb9+/ShRogTTpk1j+/btnH322VxwwQV89913DB06lClTppCRkcFvv/2212uuW7eOIUOG8N133yEibNiwYb/3vfbaa+nVqxdNmzalW7duPP7447z44ot7Ypo6dSojR47k8ccfP2Rzk3MuOMcfb1VJR4yA0L9vTCRnIhg//uDbMjIOvb1MmUNvP4itW7dSr149wK4IOnTowMSJE2nUqNGe8fmffvops2fP3vMpPzs7mx9++IHPP/+cG264gYyMDABKlSq112uXKFGCokWL0qFDBy6++GIuvvjivbZnZ2ezYcMGmjZtCsB1113HFVdcsWf7pZdeCsDpp5/O0qVLD/vYnHOx85e/WCJYsQIqV47NeyZnIghAbh/Bvo477rg991WVXr16ceGFF+61z+jRow/52oULF2bq1KmMGTOGwYMH89JLLzF27NiwYzvmmGMA69A+kr4K51zshD7P8cUX8Pe/x+Y9k7OPIE5deOGF9O7dm52htem+//57Nm/eTPPmzRk4cOCekUb7Ng1t2rSJ7OxsWrVqxQsvvMC333671/YSJUpQsmTJPe3/b7zxxp6rA+dcYjntNGsi6tXLhpPGgl8RxNBNN93E0qVLadCgAapK2bJl+eijj2jRogWzZs0iKyuL9PR0WrVqxVNPPbXn+zZu3Ejbtm3Ztm0bqsrzzz+/32u/9tprdOrUiS1btnDCCScwcODAWB6acy5C0tLgrrvg0Udh2TIbUhptorFKORGSlZWl06dP3+u5BQsWUKtWrYAiSg7+M3QufsyZA3Xr2mDG666LzGuKyAxVzTrQNm8acs65OJOZCcWLw8SJsXk/TwTOORdn0tLg/PNh1KjY9BN4InDOuTh00UWwfDnMmxf99/JE4Jxzceicc+xrLJqHPBE451wcOukkm986YUL038sTgXPOxSERuPBCGDkSoj0P1BNBhOQvQ926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">
</div>

```python
# Determining the best threshold
def find_best_threshold(y_true, y_prob):
    """Find the best threshold for maximum F1."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = (2 * precisions * recalls) / (precisions + recalls)
    return thresholds[np.argmax(f1s)]
```
```python
# Best threshold for f1
threshold = find_best_threshold(y_true.ravel(), y_prob.ravel())
threshold
```
<pre class="output">
0.23890994
</pre>
```python
# Determine predictions using threshold
test_loss, y_true, y_prob = trainer.eval_step(dataloader=test_dataloader)
y_pred = np.array([np.where(prob >= threshold, 1, 0) for prob in y_prob])
```
```python
# Evaluate
performance = get_performance(
    y_true=y_test, y_pred=y_pred, classes=label_encoder.classes)
print (json.dumps(performance['overall'], indent=2))
```
<pre class="output">
{
  "precision": 0.8134201912838206,
  "recall": 0.5244273766053323,
  "f1": 0.6134741297877828,
  "num_samples": 480.0
}
</pre>

We can do the same type of inspection as with the rule-based baseline. This is the output for the `transformers` tag:
<pre class="output">
{
  "precision": 0.782608695652174,
  "recall": 0.72,
  "f1": 0.7499999999999999,
  "num_samples": 25.0
}

=== True positives ===

  insight project insight designed create nlp service code base front end gui streamlit backend server fastapi usage transformers
    true: ['attention', 'huggingface', 'natural-language-processing', 'pytorch', 'transfer-learning', 'transformers']
    pred: ['natural-language-processing', 'transformers']

  transformer neural network architecture explained time explain transformers work looking easy explanation exactly right
    true: ['attention', 'natural-language-processing', 'transformers']
    pred: ['natural-language-processing', 'transformers']

  multi task training hugging face transformers nlp recipe multi task training transformers trainer nlp datasets
    true: ['huggingface', 'natural-language-processing', 'transformers']
    pred: ['huggingface', 'natural-language-processing', 'transformers']

=== False positives ===

  evaluation metrics language modeling article focus traditional intrinsic metrics extremely useful process training language model
    true: ['language-modeling', 'natural-language-processing']
    pred: ['language-modeling', 'natural-language-processing', 'transformers']

  multi target albumentations many images many masks bounding boxes key points transform sync
    true: ['computer-vision', 'data-augmentation']
    pred: ['computer-vision', 'natural-language-processing', 'transformers']

  lda2vec tools interpreting natural language lda2vec model tries mix best parts word2vec lda single framework
    true: ['embeddings', 'interpretability', 'natural-language-processing']
    pred: ['natural-language-processing', 'transformers']

=== False negatives ===

  sized fill blank multi mask filling roberta sized fill blank conditional text filling idea filling missing words sentence probable choice words
    true: ['attention', 'huggingface', 'language-modeling', 'natural-language-processing', 'transformers']
    pred: ['natural-language-processing']

  gpt3 works visualizations animations compilation threads explaining gpt3
    true: ['natural-language-processing', 'transformers']
    pred: ['interpretability', 'natural-language-processing']

  multimodal meme classification uniter given state art results various image text related problems project aims finetuning uniter solve hateful memes challenge
    true: ['attention', 'computer-vision', 'image-classification', 'natural-language-processing', 'transformers']
    pred: ['computer-vision']
</pre>
```python
# Save artifacts
dir = Path("cnn")
dir.mkdir(parents=True, exist_ok=True)
tokenizer.save(fp=Path(dir, 'tokenzier.json'))
label_encoder.save(fp=Path(dir, 'label_encoder.json'))
torch.save(best_model.state_dict(), Path(dir, 'model.pt'))
with open(Path(dir, 'performance.json'), "w") as fp:
    json.dump(performance, indent=2, sort_keys=False, fp=fp)
```

<h4 id="inference_cnn">Inference</h4>

```python
# Load artifacts
device = torch.device("cpu")
tokenizer = Tokenizer.load(fp=Path(dir, 'tokenzier.json'))
label_encoder = LabelEncoder.load(fp=Path(dir, 'label_encoder.json'))
model = CNN(
    embedding_dim=embedding_dim, vocab_size=vocab_size,
    num_filters=num_filters, filter_sizes=filter_sizes,
    hidden_dim=hidden_dim, dropout_p=dropout_p,
    num_classes=num_classes)
model.load_state_dict(torch.load(Path(dir, 'model.pt'), map_location=device))
model.to(device)
```
<pre class="output">
CNN(
  (embeddings): Embedding(39, 128, padding_idx=0)
  (conv): ModuleList(
    (0): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
    (1): Conv1d(128, 128, kernel_size=(2,), stride=(1,))
    (2): Conv1d(128, 128, kernel_size=(3,), stride=(1,))
    (3): Conv1d(128, 128, kernel_size=(4,), stride=(1,))
    (4): Conv1d(128, 128, kernel_size=(5,), stride=(1,))
    (5): Conv1d(128, 128, kernel_size=(6,), stride=(1,))
    (6): Conv1d(128, 128, kernel_size=(7,), stride=(1,))
    (7): Conv1d(128, 128, kernel_size=(8,), stride=(1,))
    (8): Conv1d(128, 128, kernel_size=(9,), stride=(1,))
    (9): Conv1d(128, 128, kernel_size=(10,), stride=(1,))
  )
  (dropout): Dropout(p=0.5, inplace=False)
  (fc1): Linear(in_features=1280, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=35, bias=True)
)
</pre>
```python
# Initialize trainer
trainer = Trainer(model=model, device=device)
```
```python
# Dataloader
text = "Transfer learning with BERT for self-supervised learning"
X = np.array(tokenizer.texts_to_sequences([preprocess(text)]))
y_filler = label_encoder.encode([np.array([label_encoder.classes[0]]*len(X))])
dataset = CNNTextDataset(
    X=X, y=y_filler, max_filter_size=max(filter_sizes))
dataloader = dataset.create_dataloader(
    batch_size=batch_size)
```
```python
# Inference
y_prob = trainer.predict_step(dataloader)
y_pred = np.array([np.where(prob >= threshold, 1, 0) for prob in y_prob])
label_encoder.decode(y_pred)
```
<pre class="output">
[['attention',
  'natural-language-processing',
  'self-supervised-learning',
  'transfer-learning',
  'transformers']]
</pre>

<u><i>limitations</i></u>:
- *representation*: embeddings are not contextual.
- *architecture*: extracting signal from encoded inputs is limited by filter widths.

> Since we're dealing with simple architectures and fast training times, it's a good opportunity to explore tuning and experiment with k-fold cross validation to properly reach any conclusions about performance.

<hr>

<h3 id="rnn">RNN w/ Embeddings</h3>
- [Set up](#setup_rnn)
- [Tokenizer](#tokenizer_rnn)
- [Data imbalance](#imbalance_rnn)
- [Datasets](#datasets_rnn)
- [Model](#model_rnn)
- [Training](#training_rnn)
- [Evaluation](#evaluation_rnn)
- [Inference](#inference_rnn)

<u><i>motivation</i></u>: let's see if processing our embedded tokens in a sequential fashion using [recurrent neural networks (RNNs)]({% link _courses/ml-foundations/recurrent-neural-networks.md %}){:target="_blank"} can yield better performance.

<h4 id="setup_rnn">Set up</h4>

```python
# Set seeds
set_seeds()
```
```python
# Get data splits
preprocessed_df = df.copy()
preprocessed_df.text = preprocessed_df.text.apply(preprocess, lower=True)
X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = get_data_splits(preprocessed_df)
X_test_raw = X_test
```
```python
# Set device
cuda = True
device = torch.device('cuda' if (
    torch.cuda.is_available() and cuda) else 'cpu')
torch.set_default_tensor_type('torch.FloatTensor')
if device.type == 'cuda':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
print (device)
```

<h4 id="tokenizer_rnn">Tokenizer</h4>

```python
# Tokenize
char_level = True
tokenizer = Tokenizer(char_level=char_level)
tokenizer.fit_on_texts(texts=X_train)
vocab_size = len(tokenizer)
print ("X tokenizer:\n"
    f"  {tokenizer}")
```
<pre class="output">
X tokenizer:
  &lt;Tokenizer(num_tokens=39)&gt;
</pre>
```python
tokenizer.token_to_index
```
<pre class="output">
{' ': 2,
 '0': 30,
 '1': 31,
 '2': 26,
 ...
 '&lt;PAD&gt;': 0,
 '&lt;UNK&gt;': 1,
 ...
 'x': 25,
 'y': 21,
 'z': 27}
</pre>
```python
# Convert texts to sequences of indices
X_train = np.array(tokenizer.texts_to_sequences(X_train))
X_val = np.array(tokenizer.texts_to_sequences(X_val))
X_test = np.array(tokenizer.texts_to_sequences(X_test))
preprocessed_text = tokenizer.sequences_to_texts([X_train[0]])[0]
print ("Text to indices:\n"
    f"  (preprocessed) → {preprocessed_text}\n"
    f"  (tokenized) → {X_train[0]}")
```
<pre class="output">
Text to indices:
  (preprocessed) → albumentations fast image augmentation library easy use wrapper around libraries
  (tokenized) → [ 7 11 20 17 16  3  5  6  7  6  4 10  5  9  2 19  7  9  6  2  4 16  7 14
  3  2  7 17 14 16  3  5  6  7  6  4 10  5  2 11  4 20  8  7  8 21  2  3
  7  9 21  2 17  9  3  2 23  8  7 13 13  3  8  2  7  8 10 17  5 15  2 11
  4 20  8  7  8  4  3  9]
</pre>

<h4 id="imbalance_rnn">Data imbalance</h4>

We'll factor class weights in our objective function ([binary cross entropy with logits](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html){:target="_blank"}) to help with class imbalance. There are many other techniques such as over sampling from underrepresented classes, undersampling, etc. but we'll cover these in a separate unit lesson on data imbalance.

```python
# Class weights
counts = np.bincount([label_encoder.class_to_index[class_] for class_ in all_tags])
class_weights = {i: 1.0/count for i, count in enumerate(counts)}
print (f"class counts: {counts},\nclass weights: {class_weights}")
```
<pre class="output">
class counts: [120  41 388 106  41  75  34  73  51  78  64  51  55  93  51 429  33  69
  30  51 258  32  49  59  57  60  48  40 213  40  34  46 196  39  39],
class weights: {0: 0.008333333333333333, 1: 0.024390243902439025, 2: 0.002577319587628866, 3: 0.009433962264150943, 4: 0.024390243902439025, 5: 0.013333333333333334, 6: 0.029411764705882353, 7: 0.0136986301369863, 8: 0.0196078431372549, 9: 0.01282051282051282, 10: 0.015625, 11: 0.0196078431372549, 12: 0.01818181818181818, 13: 0.010752688172043012, 14: 0.0196078431372549, 15: 0.002331002331002331, 16: 0.030303030303030304, 17: 0.014492753623188406, 18: 0.03333333333333333, 19: 0.0196078431372549, 20: 0.003875968992248062, 21: 0.03125, 22: 0.02040816326530612, 23: 0.01694915254237288, 24: 0.017543859649122806, 25: 0.016666666666666666, 26: 0.020833333333333332, 27: 0.025, 28: 0.004694835680751174, 29: 0.025, 30: 0.029411764705882353, 31: 0.021739130434782608, 32: 0.00510204081632653, 33: 0.02564102564102564, 34: 0.02564102564102564}
</pre>

<h4 id="datasets_rnn">Datasets</h4>

```python
class RNNTextDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __str__(self):
        return f"<Dataset(N={len(self)})>"

    def __getitem__(self, index):
        X = self.X[index]
        y = self.y[index]
        return [X, len(X), y]

    def collate_fn(self, batch):
        """Processing on a batch."""
        # Get inputs
        X = np.array(batch, dtype=object)[:, 0]
        seq_lens = np.array(batch, dtype=object)[:, 1]
        y = np.stack(np.array(batch, dtype=object)[:, 2], axis=0)

        # Pad inputs
        X = pad_sequences(sequences=X)

        # Cast
        X = torch.LongTensor(X.astype(np.int32))
        seq_lens = torch.LongTensor(seq_lens.astype(np.int32))
        y = torch.FloatTensor(y.astype(np.int32))

        return X, seq_lens, y

    def create_dataloader(self, batch_size, shuffle=False, drop_last=False):
        return torch.utils.data.DataLoader(
            dataset=self,
            batch_size=batch_size,
            collate_fn=self.collate_fn,
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=True)
```
```python
# Create datasets
batch_size = 64
train_dataset = RNNTextDataset(
    X=X_train, y=y_train)
val_dataset = RNNTextDataset(
    X=X_val, y=y_val)
test_dataset = RNNTextDataset(
    X=X_test, y=y_test)
print ("Data splits:\n"
    f"  Train dataset:{train_dataset.__str__()}\n"
    f"  Val dataset: {val_dataset.__str__()}\n"
    f"  Test dataset: {test_dataset.__str__()}\n"
    "Sample point:\n"
    f"  X: {train_dataset[0][0]}\n"
    f"  seq_len: {train_dataset[0][1]}\n"
    f"  y: {train_dataset[0][2]}")
```
<pre class="output">
Data splits:
  Train dataset:<Dataset(N=1000)>
  Val dataset: <Dataset(N=227)>
  Test dataset: <Dataset(N=217)>
Sample point:
  X: [ 7 11 20 17 16  3  5  6  7  6  4 10  5  9  2 19  7  9  6  2  4 16  7 14
  3  2  7 17 14 16  3  5  6  7  6  4 10  5  2 11  4 20  8  7  8 21  2  3
  7  9 21  2 17  9  3  2 23  8  7 13 13  3  8  2  7  8 10 17  5 15  2 11
  4 20  8  7  8  4  3  9]
  seq_len: 80
  y: [0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
</pre>
```python
# Create dataloaders
train_dataloader = train_dataset.create_dataloader(
    batch_size=batch_size)
val_dataloader = val_dataset.create_dataloader(
    batch_size=batch_size)
test_dataloader = test_dataset.create_dataloader(
    batch_size=batch_size)
batch_X, batch_seq_lens, batch_y = next(iter(train_dataloader))
print (batch_X.shape)
print ("Sample batch:\n"
    f"  X: {list(batch_X.size())}\n"
    f"  seq_lens: {list(batch_seq_lens.size())}\n"
    f"  y: {list(batch_y.size())}")
```
<pre class="output">
torch.Size([64, 186])
Sample batch:
  X: [64, 186]
  seq_lens: [64]
  y: [64, 35]
</pre>

<h4 id="model_rnn">Model</h4>

We'll be using a recurrent neural network to process our embedded tokens one at a time (sequentially). If you're not familiar with RNNs be sure to check out the [RNN lesson]({% link _courses/ml-foundations/recurrent-neural-networks.md %}){:target="_blank"} *where* we walkthrough every component of the architecture.

<div class="ai-center-all">
    <img width="500" src="https://raw.githubusercontent.com/GokuMohandas/madewithml/main/images/ml-foundations/rnn/vanilla.png">
</div>

$$ \text{RNN forward pass for a single time step } X_t $$:

$$ h_t = tanh(W_{hh}h_{t-1} + W_{xh}X_t+b_h) $$

<div class="ai-center-all">
<table class="mathjax-table">
  <tbody>
    <tr>
      <td>$$ N $$</td>
      <td>$$ \text{batch size} $$</td>
    </tr>
    <tr>
      <td>$$ E $$</td>
      <td>$$ \text{embedding dim} $$</td>
    </tr>
    <tr>
      <td>$$ H $$</td>
      <td>$$ \text{number of hidden units} $$</td>
    </tr>
    <tr>
      <td>$$ W_{hh} $$</td>
      <td>$$ \text{RNN weights} \in \mathbb{R}^{HXH} $$</td>
    </tr>
    <tr>
      <th>$$ h_{t-1} $$</th>
      <th>$$ \text{previous timestep's hidden state} \in in \mathbb{R}^{NXH} $$</th>
    </tr>
    <tr>
      <td>$$ W_{xh} $$</td>
      <td>$$ \text{input weights} \in \mathbb{R}^{EXH} $$</td>
    </tr>
    <tr>
      <td>$$ X_t $$</td>
      <td>$$ \text{input at time step } t \in \mathbb{R}^{NXE} $$</td>
    </tr>
    <tr>
      <td>$$ b_h $$</td>
      <td>$$ \text{hidden units bias} \in \mathbb{R}^{HX1} $$</td>
    </tr>
    <tr>
      <td>$$ h_t $$</td>
      <td>$$ \text{output from RNN for timestep } t $$</td>
    </tr>
  </tbody>
</table>
</div>

```python
# Arguments
embedding_dim = 128
rnn_hidden_dim = 128
hidden_dim = 128
dropout_p = 0.5
```
```python
def gather_last_relevant_hidden(hiddens, seq_lens):
    """Extract and collect the last relevant
    hidden state based on the sequence length."""
    seq_lens = seq_lens.long().detach().cpu().numpy() - 1
    out = []
    for batch_index, column_index in enumerate(seq_lens):
        out.append(hiddens[batch_index, column_index])
    return torch.stack(out)
```
```python
class RNN(nn.Module):
    def __init__(self, embedding_dim, vocab_size, rnn_hidden_dim,
                 hidden_dim, dropout_p, num_classes, padding_idx=0):
        super(RNN, self).__init__()

        # Initialize embeddings
        self.embeddings = nn.Embedding(embedding_dim=embedding_dim,
                                       num_embeddings=vocab_size,
                                       padding_idx=padding_idx)

        # RNN
        self.rnn = nn.GRU(embedding_dim, rnn_hidden_dim,
                          batch_first=True, bidirectional=True)

        # FC weights
        self.dropout = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(rnn_hidden_dim*2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, inputs):
        # Inputs
        x_in, seq_lens = inputs

        # Embed
        x_in = self.embeddings(x_in)

        # Rnn outputs
        out, h_n = self.rnn(x_in)
        z = gather_last_relevant_hidden(hiddens=out, seq_lens=seq_lens)

        # FC layers
        z = self.fc1(z)
        z = self.dropout(z)
        z = self.fc2(z)

        return z
```
```python
# Initialize model
model = RNN(
    embedding_dim=embedding_dim, vocab_size=vocab_size,
    rnn_hidden_dim=rnn_hidden_dim, hidden_dim=hidden_dim,
    dropout_p=dropout_p, num_classes=num_classes)
model = model.to(device)
print (model.named_parameters)
```
<pre class="output">
bound method Module.named_parameters of RNN(
  (embeddings): Embedding(39, 128, padding_idx=0)
  (rnn): GRU(128, 128, batch_first=True, bidirectional=True)
  (dropout): Dropout(p=0.5, inplace=False)
  (fc1): Linear(in_features=256, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=35, bias=True)
)
</pre>

<h4 id="training_rnn">Training</h4>

```python
# Arguments
lr = 2e-3
num_epochs = 200
patience = 10
```
```python
# Define loss
class_weights_tensor = torch.Tensor(np.array(list(class_weights.values())))
loss = nn.BCEWithLogitsLoss(weight=class_weights_tensor)
```
```python
# Define optimizer & scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=5)
```
```python
# Trainer module
trainer = Trainer(
    model=model, device=device, loss_fn=loss_fn,
    optimizer=optimizer, scheduler=scheduler)
```
```python
# Train
best_model = trainer.train(
    num_epochs, patience, train_dataloader, val_dataloader)
```
<pre class="output">
Epoch: 1 | train_loss: 0.00612, val_loss: 0.00328, lr: 2.00E-03, _patience: 10
Epoch: 2 | train_loss: 0.00325, val_loss: 0.00276, lr: 2.00E-03, _patience: 10
Epoch: 3 | train_loss: 0.00299, val_loss: 0.00267, lr: 2.00E-03, _patience: 10
Epoch: 4 | train_loss: 0.00287, val_loss: 0.00261, lr: 2.00E-03, _patience: 10
...
Epoch: 27 | train_loss: 0.00167, val_loss: 0.00250, lr: 2.00E-04, _patience: 4
Epoch: 28 | train_loss: 0.00160, val_loss: 0.00252, lr: 2.00E-04, _patience: 3
Epoch: 29 | train_loss: 0.00154, val_loss: 0.00250, lr: 2.00E-04, _patience: 2
Epoch: 30 | train_loss: 0.00153, val_loss: 0.00250, lr: 2.00E-04, _patience: 1
Stopping early!
</pre>

<h4 id="evaluation_rnn">Evaluation</h4>

```python
# Threshold-PR curve
train_loss, y_true, y_prob = trainer.eval_step(dataloader=train_dataloader)
precisions, recalls, thresholds = precision_recall_curve(y_true.ravel(), y_prob.ravel())
plt.plot(thresholds, precisions[:-1], "r--", label="Precision")
plt.plot(thresholds, recalls[:-1], "b-", label="Recall")
plt.ylabel("Performance")
plt.xlabel("Threshold")
plt.legend(loc='best')
```
<div class="ai-center-all">
    <img width="400" src="data:image/png;base64,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o0YN2rVrx65duwBo3LgxPXr0oF69elSuXJkff/wxIOfo7/TR3uh8/6rAVLSQ3E+AN+0Yk3vO6eoy0DrVvqboa6/prNF+/eCBB2DoUJs5akJEVjWubr1V17gcOKBT4E509916+e9/4ZZbjj82a5bfb52SksK0adNo0UKr3yxatIjff/+dhIQEhg4dStGiRVmwYAGHDx/myiuvpFmzZqxcuZIvvviC+fPnEx8fz99//33cayYnJzNp0iRWrlyJiLA7i8kZnTp1YtCgQTRq1IhevXrx/PPP8+abbx6L6ZdffmHq1Kk8//zzp+1u8pe/LYJbgCbANufcPUBNoGie393kP+eOdQkxcuSxu2NjoW9faNAA3n8frrsOrKyTiVYHDx6kVq1aJCYmUq5cOe69914A6tWrd2xa5rfffsuoUaOoVasW9evXJzk5mdWrVzN9+nTuuece4uPjASh+QnmWokWLEhcXx7333svEiROPPS7dnj172L17N40aNQKgc+fOzJ49+9jxm266CYA6deqwYcOGgJyvv51cB51zaSKSIiJnAzs4vnyECXXOwbXXQpcuul7g7LOhwPHfA2JjtdjoM89A//5acXraNLjkEo9iNgZO/w0+Pv70x0uWzFELIF3hwoVZsmTJSfefmWl/buccgwYNonnz5sc95ptvvjntaxcsWJBffvmF77//nvHjx/P2228zY8YMv2M744wzAB3QTklJ8ft5p5OTonPFgGHojKFFwNyARGDyx4MP6h/E6NG6UUGBrP/rCxbU7qHPPoPkZGjaFP78M39DNSYcNG/enHfffZejR48C8Mcff7B//36aNm3KiBEjjs00OrFraN++fezZs4eWLVvyxhtv8Ouvvx53vGjRopxzzjnH+v9Hjx59rHUQLH61CJxz6QVnhojI18DZzrmlwQvLBMzu3cfvKObnvpW33KKNhubNda1Bnz7w+OOnzB/GRJ377ruPDRs2cNlll+Gco1SpUnz++ee0aNGCJUuWkJiYSKFChWjZsiUvv/zyseft3buXtm3bcujQIZxzDBgw4KTX/vDDD3nggQc4cOAAF198MSNGjAjqufg9fVREanByraGJwQnr1Gz6aA59/LEOCsfGajXRwoVz9PQFC6B7d5g3Dxo1gmHDdM2ZMcESCtNHw11Qpo+KyHBgOHAzcIPv0jpvoZqg27sX7rhD+3mOHMlx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">
</div>

```python
# Best threshold for f1
threshold = find_best_threshold(y_true.ravel(), y_prob.ravel())
threshold
```
<pre class="output">
0.22973001
</pre>
```python
# Determine predictions using threshold
test_loss, y_true, y_prob = trainer.eval_step(dataloader=test_dataloader)
y_pred = np.array([np.where(prob >= threshold, 1, 0) for prob in y_prob])
```
```python
# Evaluate
performance = get_performance(
    y_true=y_test, y_pred=y_pred, classes=label_encoder.classes)
print (json.dumps(performance['overall'], indent=2))
```
<pre class="output">
{
  "precision": 0.3170755112080674,
  "recall": 0.20761471963996597,
  "f1": 0.22826804744644114,
  "num_samples": 480.0
}
</pre>

<h4 id="inference_rnn">Inference</h4>

> Detailed inspection and inference in the [notebook](https://colab.research.google.com/github/GokuMohandas/applied-ml/blob/main/notebooks/tagifai.ipynb){:target="_blank"}.

<u><i>limitations</i></u>: since we're using character embeddings our encoded sequences are quite long (>100), the RNNs may potentially be suffering from memory issues. We also can't process our tokens in parallel because we're restricted by sequential processing.

> Don't be afraid to experiment with stacking models if they're able to extract unique signal from your encoded data, for example applying CNNs on the outputs from the RNN (outputs from all tokens, not just last relevant one).

<hr>

<h3 id="transformers">Transformers w/ Contextual Embeddings</h3>
- [Set up](#setup_transformer)
- [Tokenizer](#tokenizer_transformer)
- [Data imbalance](#imbalance_transformer)
- [Datasets](#datasets_transformer)
- [Model](#model_transformer)
- [Training](#training_transformer)
- [Evaluation](#evaluation_transformer)
- [Inference](#inference_transformer)

<u><i>motivation</i></u>:
- *representation*: we want better representation for our input tokens via contextual embeddings where the token representation is based on the specific neighboring tokens. We can also use sub-word tokens, as opposed to character or word tokens, since they can hold more meaningful representations for many of our keywords, prefixes, suffixes, etc. without having to use filters with specific widths.
- *architecture*: we want to use [Transformers](https://www.youtube.com/watch?v=LwV7LKunDbs){:target="_blank"} to attend (in parallel) to all the tokens in our input, as opposed to being limited by filter spans (CNNs) or memory issues from sequential processing (RNNs).

<div class="ai-center-all">
    <img width="450" src="https://miro.medium.com/max/2880/1*BHzGVskWGS_3jEcYYi6miQ.png">
</div>
<div class="ai-center-all">
  <small>Transformer base architecture [<a href="https://miro.medium.com/max/2880/1*BHzGVskWGS_3jEcYYi6miQ.png" target="_blank">source</a>]</small>
</div>

<h4 id="setup_transformer">Set up</h4>

```python
# Set seeds
set_seeds()
```
```python
# Get data splits
preprocessed_df = df.copy()
preprocessed_df.text = preprocessed_df.text.apply(preprocess, lower=True)
X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = get_data_splits(preprocessed_df)
X_test_raw = X_test
```
```python
# Set device
cuda = True
device = torch.device('cuda' if (
    torch.cuda.is_available() and cuda) else 'cpu')
torch.set_default_tensor_type('torch.FloatTensor')
if device.type == 'cuda':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
print (device)
```

<h4 id="tokenizer_transformer">Tokenizer</h4>

We'll be using the [BertTokenizer](https://huggingface.co/transformers/model_doc/bert.html#berttokenizer){:target="_blank"} to tokenize our input text in to sub-word tokens.
```python
from transformers import DistilBertTokenizer
from transformers import BertTokenizer
```
```python
# Load tokenizer and model
# tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
vocab_size = len(tokenizer)
print (vocab_size)
```
<pre class="output">
Downloading: 100%
228k/228k [00:00<00:00, 1.09MB/s]

31090
</pre>
```python
# Tokenize inputs
encoded_input = tokenizer(X_train.tolist(), return_tensors='pt', padding=True)
X_train_ids = encoded_input['input_ids']
X_train_masks = encoded_input['attention_mask']
print (X_train_ids.shape, X_train_masks.shape)
encoded_input = tokenizer(X_val.tolist(), return_tensors='pt', padding=True)
X_val_ids = encoded_input['input_ids']
X_val_masks = encoded_input['attention_mask']
print (X_val_ids.shape, X_val_masks.shape)
encoded_input = tokenizer(X_test.tolist(), return_tensors='pt', padding=True)
X_test_ids = encoded_input['input_ids']
X_test_masks = encoded_input['attention_mask']
print (X_test_ids.shape, X_test_masks.shape)
```
<pre class="output">
torch.Size([1000, 41]) torch.Size([1000, 41])
torch.Size([227, 38]) torch.Size([227, 38])
torch.Size([217, 38]) torch.Size([217, 38])
</pre>
```python
# Decode
print (f"{X_train_ids[0]}\n{tokenizer.decode(X_train_ids[0])}")
```
<pre class="output">
tensor([  102,  6160,  1923,   288,  3254,  1572, 18205,  5560,  4578,   626,
        23474,   291,  2715, 10558,   103,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0])
[CLS] albumentations fast image augmentation library easy use wrapper around libraries [SEP] [PAD] [PAD] ...
</pre>
```python
# Sub-word tokens
print (tokenizer.convert_ids_to_tokens(ids=X_train_ids[0]))
```
<pre class="output">
['[CLS]', 'alb', '##ument', '##ations', 'fast', 'image', 'augmentation', 'library', 'easy', 'use', 'wrap', '##per', 'around', 'libraries', '[SEP]', '[PAD]', ...]
</pre>

<h4 id="imbalance_transformer">Data imbalance</h4>

```python
# Class weights
counts = np.bincount([label_encoder.class_to_index[class_] for class_ in all_tags])
class_weights = {i: 1.0/count for i, count in enumerate(counts)}
print ("class counts:\n"
    f"  {counts}\n\n"
    "class weights:\n"
    f"  {class_weights}")
```
<pre class="output">
class counts:
  [120  41 388 106  41  75  34  73  51  78  64  51  55  93  51 429  33  69
  30  51 258  32  49  59  57  60  48  40 213  40  34  46 196  39  39]

class weights:
  {0: 0.008333333333333333, 1: 0.024390243902439025, 2: 0.002577319587628866, 3: 0.009433962264150943, 4: 0.024390243902439025, 5: 0.013333333333333334, 6: 0.029411764705882353, 7: 0.0136986301369863, 8: 0.0196078431372549, 9: 0.01282051282051282, 10: 0.015625, 11: 0.0196078431372549, 12: 0.01818181818181818, 13: 0.010752688172043012, 14: 0.0196078431372549, 15: 0.002331002331002331, 16: 0.030303030303030304, 17: 0.014492753623188406, 18: 0.03333333333333333, 19: 0.0196078431372549, 20: 0.003875968992248062, 21: 0.03125, 22: 0.02040816326530612, 23: 0.01694915254237288, 24: 0.017543859649122806, 25: 0.016666666666666666, 26: 0.020833333333333332, 27: 0.025, 28: 0.004694835680751174, 29: 0.025, 30: 0.029411764705882353, 31: 0.021739130434782608, 32: 0.00510204081632653, 33: 0.02564102564102564, 34: 0.02564102564102564}
</pre>

<h4 id="datasets_transformer">Datasets</h4>

```python
class TransformerTextDataset(torch.utils.data.Dataset):
    def __init__(self, ids, masks, targets):
        self.ids = ids
        self.masks = masks
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __str__(self):
        return f"<Dataset(N={len(self)})>"

    def __getitem__(self, index):
        ids = torch.tensor(self.ids[index], dtype=torch.long)
        masks = torch.tensor(self.masks[index], dtype=torch.long)
        targets = torch.FloatTensor(self.targets[index])
        return ids, masks, targets

    def create_dataloader(self, batch_size, shuffle=False, drop_last=False):
        return torch.utils.data.DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=False)
```
```python
# Create datasets
train_dataset = TransformerTextDataset(ids=X_train_ids, masks=X_train_masks, targets=y_train)
val_dataset = TransformerTextDataset(ids=X_val_ids, masks=X_val_masks, targets=y_val)
test_dataset = TransformerTextDataset(ids=X_test_ids, masks=X_test_masks, targets=y_test)
print ("Data splits:\n"
    f"  Train dataset:{train_dataset.__str__()}\n"
    f"  Val dataset: {val_dataset.__str__()}\n"
    f"  Test dataset: {test_dataset.__str__()}\n"
    "Sample point:\n"
    f"  ids: {train_dataset[0][0]}\n"
    f"  masks: {train_dataset[0][1]}\n"
    f"  targets: {train_dataset[0][2]}")
```
<pre class="output">
Data splits:
  Train dataset:<Dataset(N=1000)>
  Val dataset: <Dataset(N=227)>
  Test dataset: <Dataset(N=217)>
Sample point:
  ids: tensor([  102,  6160,  1923,   288,  3254,  1572, 18205,  5560,  4578,   626,
        23474,   291,  2715, 10558,   103,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0])
  masks: tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
  targets: tensor([0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       device='cpu')
</pre>
```python
# Create dataloaders
batch_size = 64
train_dataloader = train_dataset.create_dataloader(
    batch_size=batch_size)
val_dataloader = val_dataset.create_dataloader(
    batch_size=batch_size)
test_dataloader = test_dataset.create_dataloader(
    batch_size=batch_size)
batch = next(iter(train_dataloader))
print ("Sample batch:\n"
    f"  ids: {batch[0].size()}\n"
    f"  masks: {batch[1].size()}\n"
    f"  targets: {batch[2].size()}")
```
<pre class="output">
Sample batch:
  ids: torch.Size([64, 41])
  masks: torch.Size([64, 41])
  targets: torch.Size([64, 35])
</pre>

<h4 id="model_transformer">Model</h4>

We're going to use a pretrained [BertModel](https://huggingface.co/transformers/model_doc/bert.html#bertmodel) to act as a feature extractor. We'll only use the encoder to receive sequential and pooled outputs (`is_decoder=False` is default).

```python
from transformers import BertModel
```
```python
# transformer = BertModel.from_pretrained("distilbert-base-uncased")
# embedding_dim = transformer.config.dim
transformer = BertModel.from_pretrained("allenai/scibert_scivocab_uncased")
embedding_dim = transformer.config.hidden_size
```
```python
class Transformer(nn.Module):
    def __init__(self, transformer, dropout_p, embedding_dim, num_classes):
        super(Transformer, self).__init__()
        self.transformer = transformer
        self.dropout = torch.nn.Dropout(dropout_p)
        self.fc1 = torch.nn.Linear(embedding_dim, num_classes)

    def forward(self, inputs):
        ids, masks = inputs
        seq, pool = self.transformer(input_ids=ids, attention_mask=masks)
        z = self.dropout(pool)
        z = self.fc1(z)
        return z
```

> We decided to work with the pooled output, but we could have just as easily worked with the sequential output (encoder representation for each sub-token) and applied a CNN (or other decoder options) on top of it.

```python
# Initialize model
dropout_p = 0.5
model = Transformer(
    transformer=transformer, dropout_p=dropout_p,
    embedding_dim=embedding_dim, num_classes=num_classes)
model = model.to(device)
print (model.named_parameters)
```
<pre class="output">
bound method Module.named_parameters of Transformer(
  (transformer): BertModel(
    (embeddings): BertEmbeddings(
      (word_embeddings): Embedding(31090, 768, padding_idx=0)
      (position_embeddings): Embedding(512, 768)
      (token_type_embeddings): Embedding(2, 768)
      (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (encoder): BertEncoder(
      (layer): ModuleList(
        (0): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (1): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        ...
        11 more BertLayers
        ...
      )
    )
    (pooler): BertPooler(
      (dense): Linear(in_features=768, out_features=768, bias=True)
      (activation): Tanh()
    )
  )
  (dropout): Dropout(p=0.5, inplace=False)
  (fc1): Linear(in_features=768, out_features=35, bias=True)
)
</pre>

<h4 id="training_transformer">Training</h4>

```python
# Arguments
lr = 1e-4
num_epochs = 200
patience = 10
```
```python
# Define loss
class_weights_tensor = torch.Tensor(np.array(list(class_weights.values())))
loss = nn.BCEWithLogitsLoss(weight=class_weights_tensor)
```
```python
# Define optimizer & scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=5)
```
```python
# Trainer module
trainer = Trainer(
    model=model, device=device, loss_fn=loss_fn,
    optimizer=optimizer, scheduler=scheduler)
```
```python
# Train
best_model = trainer.train(
    num_epochs, patience, train_dataloader, val_dataloader)
```
<pre class="output">
Epoch: 1 | train_loss: 0.00647, val_loss: 0.00354, lr: 1.00E-04, _patience: 10
Epoch: 2 | train_loss: 0.00331, val_loss: 0.00280, lr: 1.00E-04, _patience: 10
Epoch: 3 | train_loss: 0.00295, val_loss: 0.00272, lr: 1.00E-04, _patience: 10
Epoch: 4 | train_loss: 0.00291, val_loss: 0.00271, lr: 1.00E-04, _patience: 10
...
Epoch: 43 | train_loss: 0.00039, val_loss: 0.00130, lr: 1.00E-06, _patience: 4
Epoch: 44 | train_loss: 0.00038, val_loss: 0.00130, lr: 1.00E-06, _patience: 3
Epoch: 45 | train_loss: 0.00038, val_loss: 0.00130, lr: 1.00E-06, _patience: 2
Epoch: 46 | train_loss: 0.00038, val_loss: 0.00130, lr: 1.00E-06, _patience: 1
Stopping early!
</pre>

<h4 id="evaluation_transformer">Evaluation</h4>

```python
# Threshold-PR curve
train_loss, y_true, y_prob = trainer.eval_step(dataloader=train_dataloader)
precisions, recalls, thresholds = precision_recall_curve(y_true.ravel(), y_prob.ravel())
plt.plot(thresholds, precisions[:-1], "r--", label="Precision")
plt.plot(thresholds, recalls[:-1], "b-", label="Recall")
plt.ylabel("Performance")
plt.xlabel("Threshold")
plt.legend(loc='best')
```
<div class="ai-center-all">
    <img width="400" src="data:image/png;base64,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ns6GDRv2ec3169czbtw4vvvuO0SETZs27fe+l156KU8//TQdO3bk7rvv5t577+XJJ5/8PaZZs2bxwQcfcO+99xba3OSci6xrr4WLL4b586FVq6CjMYmZCAor85eeXvj2qlUPqkzgjh07aNmyJWBnBFdeeSXTp0+nXbt2v4/P//jjj5k/f/7v3/I3b97M999/z8SJE7niiitIT08HoHLlyvu8dsWKFUlNTeXKK6/knHPO4Zxzztln++bNm9m0aRMdO3YE4LLLLuPCCy/8fft5550HQJs2bVi+fHmxj805V3JOO82up0zxRJBwcvoI8itXrtzvt1WVp59+mq5du+6zz4QJEwp97dKlSzNr1iwmTZrE2LFjeeaZZ5g8eXLYsR122GGAdWgfTF+Fc67k1KoFjRrBhAlw001BR2MSs48gRnXt2pVhw4axZ88eAJYsWcK2bdvo0qULL7744u8jjfI3DW3dupXNmzfTo0cPnnjiCeblG3JQsWJFjjjiiN/b/1999dXfzw6cc7GnWzeYOhV27Ag6EuNnBFF01VVXsXz5clq3bo2qUq1aNd5++226devG3LlzycjIoGzZsvTo0YMHH3zw9+dt2bKFXr16sXPnTlSVxx9/fL/Xfvnll7nmmmvYvn079erV48UXX4zmoTnniqFrV3jySVvCMl8DQSBEVYOOoVgyMjI0MzNzn8cWLVpE48aNA4ooMfjP0Lno2bEDKleGSy6B556LznuKyGxVzShomzcNOedclKWlQb9+8J//QL7vtYHwROCccwF49FGoWdMSwpYtwcbiicA55wJQpQq8/rpVI73mmmBj8UTgnHMBOeUUqz/0+usQ5BQfTwTOOReg66+H0qVhwIDgyk54InDOuQDVrm2l0D75BAoYGR4VnghKSN4y1Oeee26B9YAORZ06dVi3bh0A5cuXL9HXds4F66qroE8fuP12KKBAQcR5IighectQV65cmaFDhwYdknMuTojAs8/aAveDBkG0p3cl3MziG28s+YzasqXNAgzXiSeeyPz58wH44Ycf+Mtf/sLatWtJT0/n+eefp1GjRvz6669cc801LFu2DIBhw4Zx0kkn0bt3b1auXMnOnTu54YYbGDBgQMkejHMuJlWrBv/8p3UejxgBV14ZvfdOuEQQtOzsbCZNmsSVod/igAEDGD58OA0aNGDmzJlcd911TJ48mYED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">
</div>

```python
# Best threshold for f1
threshold = find_best_threshold(y_true.ravel(), y_prob.ravel())
threshold
```
<pre class="output">
0.34790307
</pre>
```python
# Determine predictions using threshold
test_loss, y_true, y_prob = trainer.eval_step(dataloader=test_dataloader)
y_pred = np.array([np.where(prob >= threshold, 1, 0) for prob in y_prob])
```
```python
# Evaluate
performance = get_performance(
    y_true=y_test, y_pred=y_pred, classes=label_encoder.classes)
print (json.dumps(performance['overall'], indent=2))
```
<pre class="output">
{
  "precision": 0.7524809959244634,
  "recall": 0.5251264830544388,
  "f1": 0.5904032248915119,
  "num_samples": 480.0
}
</pre>

<h4 id="inference_transformer">Inference</h4>

> Detailed inspection, inference and visualization of attention heads in the [notebook](https://colab.research.google.com/github/GokuMohandas/applied-ml/blob/main/notebooks/tagifai.ipynb){:target="_blank"}.

<u><i>limitations</i></u>: transformers can be quite large and we'll have to weigh tradeoffs before deciding on a model.

<hr>

<h3 id="tradeoffs">Tradeoffs</h3>
We're going to go with the embeddings via CNN approach and optimize it because performance is quite similar to the contextualized embeddings via transformers approach but at much lower cost.

```python
# Performance
with open(Path("cnn", "performance.json"), "r") as fp:
    cnn_performance = json.load(fp)
with open(Path("transformers", "performance.json"), "r") as fp:
    transformers_performance = json.load(fp)
print (f'CNN: f1 = {cnn_performance["overall"]["f1"]}')
print (f'Transformer: f1 = {transformers_performance["overall"]["f1"]}')
```
<pre class="output">
CNN: f1 = 0.6119912020434568
Transformer: f1 = 0.5904032248915119
</pre>

This was just one run on one split so you'll want to experiment with k-fold cross validation to properly reach any conclusions about performance. Also make sure you take the time to tune these baselines since their training periods are quite fast (we can achieve f1 of 0.7 with just a bit of tuning for both CNN / Transformers). We'll cover hyperparameter tuning in a few lessons so you can replicate the process here on your own time. We should also benchmark on other important metrics as we iterate, not just precision and recall.

```python
# Size
print (f'CNN: {Path("cnn", "model.pt").stat().st_size/1000000:.1f} MB')
print (f'Transformer: {Path("transformers", "model.pt").stat().st_size/1000000:.1f} MB')
```
<pre class="output">
CNN: 4.3 MB
Transformer: 439.9 MB
</pre>

We'll consider other tradeoffs such as maintenance overhead, bias test passes, etc. as we develop.

> Interpretability was not one of requirements but note that we could've tweaked model outputs to deliver it. For example, since we used SAME padding for our CNN, we can use the activation scores to extract influential n-grams. Similarly, we could have used self-attention weights from our Transformer encoder to find influential sub-tokens.


<h3><u>Resources</u></h3>
- [Backing off towards simplicity - why baselines need more love](https://smerity.com/articles/2017/baselines_need_love.html){:target="_blank"}
- [Model Evaluation, Model Selection, and Algorithm Selection in Machine Learning](https://arxiv.org/abs/1811.12808){:target="_blank"}


<!-- Footer -->
<hr>
<div class="row mb-4">
  <div class="col-6 mr-auto">
    <a href="{% link index.md %}" class="btn btn-sm btn-outline-secondary"><i class="fas fa-sm fa-arrow-left mr-1"></i>Return home</a>
  </div>
  <div class="col-6">
    <div class="float-right">
      <a href="{{ page.next-lesson-url }}" class="btn btn-sm btn-outline-secondary"><i class="fas fa-sm fa-arrow-right mr-1"></i>Next lesson</a>
    </div>
  </div>
</div>