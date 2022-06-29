---
template: lesson.html
title: Modeling Baselines
description: Motivating the use of baselines for iterative modeling.
keywords: baselines, modeling, pytorch, transformers, huggingface, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/mlops.png
repository: https://github.com/GokuMohandas/mlops-course
notebook: https://colab.research.google.com/github/GokuMohandas/mlops-course/blob/main/notebooks/tagifai.ipynb
---

{% include "styles/lesson.md" %}

## Intuition

Baselines are simple benchmarks which pave the way for iterative development:

- Rapid experimentation via hyperparameter tuning thanks to low model complexity.
- Discovery of data issues, false assumptions, bugs in code, etc. since model itself is not complex.
- [Pareto's principle](https://en.wikipedia.org/wiki/Pareto_principle){:target="_blank"}: we can achieve decent performance with minimal initial effort.

## Process

Here is the high level approach to establishing baselines:

1. Start with the simplest possible baseline to compare subsequent development with. This is often a random (chance) model.
2. Develop a rule-based approach (when possible) using IFTTT, auxiliary data, etc.
3. Slowly add complexity by *addressing* limitations and *motivating* representations and model architectures.
4. Weigh *tradeoffs* (performance, latency, size, etc.) between performant baselines.
5. Revisit and iterate on baselines as your dataset grows.

!!! question "Tradeoffs to consider"

    When choosing what model architecture(s) to proceed with, what are important tradeoffs to consider? And how can we prioritize them?

    ??? quote "Show answer"

        Prioritization of these tradeoffs depends on your context.

        - `#!js performance`: consider coarse-grained and fine-grained (ex. per-class) performance.
        - `#!js latency`: how quickly does your model respond for inference.
        - `#!js size`: how large is your model and can you support it's storage.
        - `#!js compute`: how much will it cost ($, carbon footprint, etc.) to train your model?
        - `#!js interpretability`: does your model need to explain its predictions?
        - `#!js bias checks`: does your model pass key bias checks?
        - `#!js time to develop`: how long do you have to develop the first version?
        - `#!js time to retrain`: how long does it take to retrain your model? This is very important to consider if you need to retrain often.
        - `#!js maintenance overhead`: who and what will be required to maintain your model versions because the real work with ML begins after deploying v1. You can't just hand it off to your site reliability team to maintain it like many teams do with traditional software.

!!! tip "Iterate on the data"
    We can also baseline on your dataset. Instead of using a fixed dataset and iterating on the models, choose a good baseline and iterate on the dataset:

    - remove or fix data samples (false positives & negatives)
    - prepare and transform features
    - expand or consolidate classes
    - incorporate auxiliary datasets
    - identify unique slices to boost

## Distributed training

All the training we need to do for our application happens on one worker with one accelerator (CPU/GPU), however, we'll want to consider distributed training for very large models or when dealing with large datasets. Distributed training can involve:

- **data parallelism**: workers received different slices of the larger dataset.
    - *synchronous training* uses [AllReduce](https://tech.preferred.jp/en/blog/technologies-behind-distributed-deep-learning-allreduce/#:~:text=AllReduce%20is%20an%20operation%20that,of%20length%20N%20called%20A_p.){:target="_blank"} to aggregate gradients and update all the workers weights at the end of each batch (synchronous).
    - *asynchronous training* uses a universal parameter server to update weights as each worker trains on its slice of data (asynchronous).
- **model parallelism**: all workers use the same dataset but the model is split amongst them (more difficult to implement compared to data parallelism because it's difficult to isolate and combine signal from backpropagation).

There are lots of options for applying distributed training such as with PyTorch's [distributed package](https://pytorch.org/tutorials/beginner/dist_overview.html){:target="_blank"}, [Ray](https://ray.io/){:target="_blank"}, [Horovd](https://horovod.ai/){:target="_blank"}, etc.

## Optimization

Distributed training strategies are great for when our data or models are too large for training but what about when our models are too large to deploy? The following model compression techniques are commonly used to make large models fit within existing infrastructure:

- [**Pruning**](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html){:target="_blank"}: remove weights (unstructured) or entire channels (structured) to reduce the size of the network. The objective is to preserve the model’s performance while increasing its sparsity.
- [**Quantization**](https://pytorch.org/docs/stable/torch.quantization.html){:target="_blank"}: reduce the memory footprint of the weights by reducing their precision (ex. 32 bit to 8 bit). We may loose some precision but it shouldn’t affect performance too much.
- [**Distillation**](https://arxiv.org/abs/2011.14691){:target="_blank"}: training smaller networks to “mimic” larger networks by having it reproduce the larger network’s layers’ outputs.

<div class="ai-center-all">
    <img width="750" src="/static/images/mlops/baselines/kd.png">
</div>
<div class="ai-center-all">
    <small>Distilling the knowledge in a neural network [<a href="https://nni.readthedocs.io/en/latest/TrialExample/KDExample.html" target="_blank">source</a>]</small>
</div>

## Baselines

Each application's baseline trajectory varies based on the task. For our application, we're going to follow this path:

1. [Random](#random)
2. [Rule-based](#rule-based)
3. [Simple ML](#simple-ml)

We'll motivate the need for slowly adding complexity to both the **representation** (ex. text vectorization) and **architecture** (ex. logistic regression), as well as address the limitations at each step of the way.

> If you're unfamiliar with of the modeling concepts here, be sure to check out the [Foundations lessons](https://madewithml.com/#foundations){:target="_blank"}.

!!! note
    The specific model we use is irrelevant for this MLOps course since the main focus is on all the components required to put a model in production and maintain it. So feel free to choose any model as we continue to the other lessons after this notebook.

We'll first set up some functions that we'll be using across the different baseline experiments.
```python linenums="1"
import random
```
```python linenums="1"
def set_seeds(seed=42):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
```
```python linenums="1"
def preprocess(df, lower, stem):
    """Preprocess the data."""
    df["text"] = df.title + " " + df.description  # feature engineering
    df.text = df.text.apply(clean_text, lower=lower, stem=stem)  # clean text
    return df
```
```python linenums="1"
def get_data_splits(X, y, train_size=0.7):
    """Generate balanced data splits."""
    X_train, X_, y_train, y_ = train_test_split(
        X, y, train_size=train_size, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_, y_, train_size=0.5, stratify=y_)
    return X_train, X_val, X_test, y_train, y_val, y_test
```

Our dataset is small so we'll train using the whole dataset but for larger datasets, we should always test on a small subset (after shuffling when necessary) so we aren't wasting time on compute.

```python linenums="1"
df = df.sample(frac=1).reset_index(drop=True)  # shuffle
df = df[: num_samples]  # None = all samples
```

!!! question "Do we need to shuffle?"
    Why is it important that we shuffle our dataset?

    ??? quote "Show answer"
        We *need* to shuffle our data since our data is chronologically organized. The latest projects may have certain features or tags that are prevalent compared to earlier projects. If we don't shuffle before creating our data splits, then our model will only be trained on the earlier signals and fail to generalize. However, in other scenarios (ex. time-series forecasting), shuffling will lead do data leaks.

### Random
<u><i>motivation</i></u>: We want to know what random (chance) performance looks like. All of our efforts should be well above this baseline.

```python linenums="1"
from sklearn.metrics import precision_recall_fscore_support
```
```python linenums="1"
# Setup
set_seeds()
df = pd.DataFrame(json.load(open("labeled_projects.json", "r")))
df = df.sample(frac=1).reset_index(drop=True)
df = preprocess(df, lower=True, stem=False)
label_encoder = LabelEncoder().fit(df.tag)
X_train, X_val, X_test, y_train, y_val, y_test = \
    get_data_splits(X=df.text.to_numpy(), y=label_encoder.encode(df.tag))
```
```python linenums="1"
# Label encoder
print (label_encoder)
print (label_encoder.classes)
```
<pre class="output">
&lt;LabelEncoder(num_classes=4)&gt;
['computer-vision', 'mlops', 'natural-language-processing', 'other']
</pre>

```python linenums="1"
# Generate random predictions
y_pred = np.random.randint(low=0, high=len(label_encoder), size=len(y_test))
print (y_pred.shape)
print (y_pred[0:5])
```
<pre class="output">
(144,)
[0 0 0 1 3]
</pre>

```python linenums="1"
# Evaluate
metrics = precision_recall_fscore_support(y_test, y_pred, average="weighted")
performance = {"precision": metrics[0], "recall": metrics[1], "f1": metrics[2]}
print (json.dumps(performance, indent=2))
```
<pre class="output">
{
  "precision": 0.31684880006233446,
  "recall": 0.2361111111111111,
  "f1": 0.2531624273393283
}
</pre>

We made the assumption that there is an equal probability for every class. Let's use the train split to figure out what the true probability is.

```python linenums="1"
# Class frequencies
p = [Counter(y_test)[index]/len(y_test) for index in range(len(label_encoder))]
p
```
<pre class="output">
[0.375, 0.08333333333333333, 0.4027777777777778, 0.1388888888888889]
</pre>
```python linenums="1"
# Generate weighted random predictions
y_pred = np.random.choice(a=range(len(label_encoder)), size=len(y_test), p=p)
```
```python linenums="1"
# Evaluate
metrics = precision_recall_fscore_support(y_test, y_pred, average="weighted")
performance = {"precision": metrics[0], "recall": metrics[1], "f1": metrics[2]}
print (json.dumps(performance, indent=2))
```
<pre class="output">
{
  "precision": 0.316412540257649,
  "recall": 0.3263888888888889,
  "f1": 0.31950372012322
}
</pre>

<u><i>limitations</i></u>: we didn't use the tokens in our input to affect our predictions so nothing was learned.

### Rule-based
<u><i>motivation</i></u>: we want to use signals in our inputs (along with domain expertise and auxiliary data) to determine the labels.

```python linenums="1"
# Setup
set_seeds()
df = pd.DataFrame(json.load(open("labeled_projects.json", "r")))
df = df.sample(frac=1).reset_index(drop=True)
df = preprocess(df, lower=True, stem=False)
label_encoder = LabelEncoder().fit(df.tag)
X_train, X_val, X_test, y_train, y_val, y_test = \
    get_data_splits(X=df.text.to_numpy(), y=label_encoder.encode(df.tag))
```
```python linenums="1"
# Restrict to relevant tags
print (len(tags_dict))
tags_dict = {tag: tags_dict[tag] for tag in label_encoder.classes if tag != "other"}
print (len(tags_dict))
```
<pre class="output">
4
3
</pre>

```python linenums="1"
# Map aliases
aliases = {}
for tag, values in tags_dict.items():
    aliases[clean_text(tag)] = tag
    for alias in values["aliases"]:
        aliases[clean_text(alias)] = tag
aliases
```
<pre class="output">
{'computer vision': 'computer-vision',
 'cv': 'computer-vision',
 'mlops': 'mlops',
 'natural language processing': 'natural-language-processing',
 'nlp': 'natural-language-processing',
 'nlproc': 'natural-language-processing',
 'production': 'mlops',
 'vision': 'computer-vision'}
</pre>

```python linenums="1"
def get_tag(text, aliases, tags_dict):
    """If a token matches an alias,
    then add the corresponding tag class."""
    for alias, tag in aliases.items():
        if alias in text:
            return tag
    return None
```

```python linenums="1"
# Sample
text = "A pretrained model hub for popular nlp models."
get_tag(text=clean_text(text), aliases=aliases, tags_dict=tags_dict)
```
<pre class="output">
'natural-language-processing'
</pre>

```python linenums="1"
# Prediction
tags = []
for text in X_test:
    tag = get_tag(text, aliases, tags_dict)
    tags.append(tag)
```
```python linenums="1"
# Encode labels
y_pred = [label_encoder.class_to_index[tag] if tag is not None else -1 for tag in tags]
```
```python linenums="1"
# Evaluate
metrics = precision_recall_fscore_support(y_test, y_pred, average="weighted")
performance = {"precision": metrics[0], "recall": metrics[1], "f1": metrics[2]}
print (json.dumps(performance, indent=2))
```
<pre class="output">
{
  "precision": 0.8322649572649572,
  "recall": 0.16666666666666666,
  "f1": 0.2766019343060607
}
</pre>

!!! question "Why is recall so low?"

    How come our precision is high but our recall is so low?

    ??? quote "Show answer"
        Only relying on the aliases can prove catastrophic when those particular aliases aren't used in our input signals. To improve this, we can build a bag of words of related terms. For example, mapping terms such as `text classification` and `named entity recognition` to the `natural-language-processing` tag but building this is a non-trivial task. Not to mention, we'll need to keep updating these rules as the data landscape matures.

```python linenums="1"
# Pitfalls
text = "Transfer learning with transformers for text classification."
print (get_tag(text=clean_text(text), aliases=aliases, tags_dict=tags_dict))
```
<pre class="output">
None
</pre>

!!! tip
    We could also use [stemming](https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html){:target="_blank"} to further refine our rule-based process:

    ```python linenums="1"
    from nltk.stem import PorterStemmer
    stemmer = PorterStemmer()
    print (stemmer.stem("democracy"))
    print (stemmer.stem("democracies"))
    ```
    <pre class="output">
    democraci
    democraci
    </pre>

    But these rule-based approaches can only yield labels with high certainty when there is an absolute condition match so it's best not to spend too much more effort on this approach.

<u><i>limitations</i></u>: we failed to generalize or learn any implicit patterns to predict the labels because we treat the tokens in our input as isolated entities.

### Vectorization
<u><i>motivation</i></u>:

- *representation*: use term frequency-inverse document frequency [(TF-IDF)](https://en.wikipedia.org/wiki/Tf%E2%80%93idf){:target="_blank"} to capture the significance of a token to a particular input with respect to all the inputs, as opposed to treating the words in our input text as isolated tokens.
- *architecture*: we want our model to meaningfully extract the encoded signal to predict the output labels.

So far we've treated the words in our input text as isolated tokens and we haven't really captured any meaning between tokens. Let's use TF-IDF (via Scikit-learn's [`TfidfVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html){:target="_blank"}) to capture the significance of a token to a particular input with respect to all the inputs.

$$ w_{i, j} = \text{tf}_{i, j} * log(\frac{N}{\text{df}_i}) $$

<center>

| Variable    | Description                          |
| :---------- | :----------------------------------- |
| $w_{i, j}$         | tf-idf weight for term $i$ in document $j$  |
| $\text{tf}_{i, j}$ | # of times term $i$ appear in document $j$  |
| $N$                | total # of documents$                       |
| $\text{df}_i$      | # of documents with token $i$               |

</center>


```python linenums="1"
from sklearn.feature_extraction.text import TfidfVectorizer
```
```python linenums="1"
# Setup
set_seeds()
df = pd.DataFrame(json.load(open("labeled_projects.json", "r")))
df = df.sample(frac=1).reset_index(drop=True)
df = preprocess(df, lower=True, stem=False)
label_encoder = LabelEncoder().fit(df.tag)
X_train, X_val, X_test, y_train, y_val, y_test = \
    get_data_splits(X=df.text.to_numpy(), y=label_encoder.encode(df.tag))
```
```python linenums="1"
# Saving raw X_test to compare with later
X_test_raw = X_test
```
```python linenums="1"
# Tf-idf
vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2,7))  # char n-grams
print (X_train[0])
X_train = vectorizer.fit_transform(X_train)
X_val = vectorizer.transform(X_val)
X_test = vectorizer.transform(X_test)
print (X_train.shape)  # scipy.sparse.csr_matrix
```
<pre class="output">
tao large scale benchmark tracking object diverse dataset tracking object tao consisting 2 907 high resolution videos captured diverse environments half minute long
(668, 99664)
</pre>
```python linenums="1"
# Class weights
counts = np.bincount(y_train)
class_weights = {i: 1.0/count for i, count in enumerate(counts)}
print (f"class counts: {counts},\nclass weights: {class_weights}")
```
<pre class="output">
class counts: [249  55 272  92],
class weights: {0: 0.004016064257028112, 1: 0.01818181818181818, 2: 0.003676470588235294, 3: 0.010869565217391304}
</pre>

### Data imbalance

With our datasets, we may often notice a data imbalance problem where a range of continuous values (regression) or certain classes (classification) may have insufficient amounts of data to learn from. This becomes a major issue when training because the model will learn to generalize to the data available and perform poorly on regions where the data is sparse. There are several techniques to mitigate data imbalance, including [resampling](https://github.com/scikit-learn-contrib/imbalanced-learn){:target="_blank"}, incorporating class weights, [augmentation](augmentation.md){:target="_blank"}, etc. Though the ideal solution is to collect more data for the minority classes!

> We'll use the [imblearn package](https://imbalanced-learn.org/stable/){:target="_blank"} to ensure that we oversample our minority classes to be equal to the majority class (tag with most samples).

```bash
pip install imbalanced-learn==0.8.1 -q
```
```python linenums="1"
from imblearn.over_sampling import RandomOverSampler
```
```python linenums="1"
# Oversample (training set)
oversample = RandomOverSampler(sampling_strategy="all")
X_over, y_over = oversample.fit_resample(X_train, y_train)
```

!!! warning
    It's important that we applied sampling only on the train split so we don't introduce data leaks with the other data splits.

```python linenums="1"
# Class weights
counts = np.bincount(y_over)
class_weights = {i: 1.0/count for i, count in enumerate(counts)}
print (f"class counts: {counts},\nclass weights: {class_weights}")
```
<pre class="output">
class counts: [272 272 272 272],
class weights: {0: 0.003676470588235294, 1: 0.003676470588235294, 2: 0.003676470588235294, 3: 0.003676470588235294}
</pre>

<u><i>limitations</i></u>:

- **representation**: TF-IDF representations don't encapsulate much signal beyond frequency but we require more fine-grained token representations.
- **architecture**: we want to develop models that can use better represented encodings in a more contextual manner.

### Machine learning

We're going to use a stochastic gradient descent classifier ([SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html){:target="_blank"}) as our model. We're going to use log loss so that it's effectively [logistic regression](https://madewithml.com/courses/foundations/logistic-regression/){:target="_blank"} with SGD.

> We're doing this because we want to have more control over the training process (epochs) and not use scikit-learn's default second order optimization methods (ex. [LGBFS](https://en.wikipedia.org/wiki/Limited-memory_BFGS){:target="_blank"}) for logistic regression.

```python linenums="1"
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss, precision_recall_fscore_support
```
```python linenums="1"
# Initialize model
model = SGDClassifier(
    loss="log", penalty="l2", alpha=1e-4, max_iter=1,
    learning_rate="constant", eta0=1e-1, power_t=0.1,
    warm_start=True)
```
```python linenums="1"
# Train model
num_epochs = 100
for epoch in range(num_epochs):
    # Training
    model.fit(X_over, y_over)

    # Evaluation
    train_loss = log_loss(y_train, model.predict_proba(X_train))
    val_loss = log_loss(y_val, model.predict_proba(X_val))

    if not epoch%10:
        print(
            f"Epoch: {epoch:02d} | "
            f"train_loss: {train_loss:.5f}, "
            f"val_loss: {val_loss:.5f}"
        )
```
<pre class="output">
Epoch: 00 | train_loss: 1.16930, val_loss: 1.21451
Epoch: 10 | train_loss: 0.46116, val_loss: 0.65903
Epoch: 20 | train_loss: 0.31565, val_loss: 0.56018
Epoch: 30 | train_loss: 0.25207, val_loss: 0.51967
Epoch: 40 | train_loss: 0.21740, val_loss: 0.49822
Epoch: 50 | train_loss: 0.19615, val_loss: 0.48529
Epoch: 60 | train_loss: 0.18249, val_loss: 0.47708
Epoch: 70 | train_loss: 0.17330, val_loss: 0.47158
Epoch: 80 | train_loss: 0.16671, val_loss: 0.46765
Epoch: 90 | train_loss: 0.16197, val_loss: 0.46488
</pre>

We could further optimize our training pipeline with functionality such as [early stopping](https://madewithml.com/courses/foundations/utilities/#early-stopping){:target="_blank"} where we would use our validation set that we created. But we want to keep this model-agnostic course simplified during the modeling stage 😉

!!! warning
    The [SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html){:target="_blank"} has an `early_stopping` flag where you can specify a portion of the training set to be use for validation. Why would this be a bad idea in our case? Because we already applied oversampling in our training set and so we would be introduce data leaks if we did this.

```python linenums="1"
# Evaluate
y_pred = model.predict(X_test)
metrics = precision_recall_fscore_support(y_test, y_pred, average="weighted")
performance = {"precision": metrics[0], "recall": metrics[1], "f1": metrics[2]}
print (json.dumps(performance, indent=2))
```
<pre class="output">
{
  "precision": 0.8753577441077441,
  "recall": 0.8680555555555556,
  "f1": 0.8654096949533866
}
</pre>

!!! tip
    Scikit-learn has a concept called [pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html){:target="_blank"} which allows us to combine transformations and training steps into one callable function.

    We can create a pipeline from scratch:

    ```python linenums="1"
    # Create pipeline from scratch
    from sklearn.pipeline import Pipeline
    steps = (("tfidf", TfidfVectorizer()), ("model", SGDClassifier()))
    pipe = Pipeline(steps)
    pipe.fit(X_train, y_train)
    ```

    or make one with trained components:

    ```python linenums="1"
    # Make pipeline from existing components
    from sklearn.pipeline import make_pipeline
    pipe = make_pipeline(vectorizer, model)
    ```

<u><i>limitations</i></u>:

- *representation*: TF-IDF representations don't encapsulate much signal beyond frequency but we require more fine-grained token representations that can account for the significance of the token itself ([embeddings](https://madewithml.com/courses/foundations/embeddings/)).
- *architecture*: we want to develop models that can use better represented encodings in a more contextual manner.

```python linenums="1"
# Inference (with tokens similar to training data)
text = "Transfer learning with transformers for text classification."
y_pred = model.predict(vectorizer.transform([text]))
label_encoder.decode(y_pred)
```
<pre class="output">
['natural-language-processing']
</pre>
```python linenums="1"
# Probabilities
y_prob = model.predict_proba(vectorizer.transform([text]))
{tag:y_prob[0][i] for i, tag in enumerate(label_encoder.classes)}
```
<pre class="output">
{'computer-vision': 0.023672281234089494,
 'mlops': 0.004158589896756235,
 'natural-language-processing': 0.9621906411391856,
 'other': 0.009978487729968667}
</pre>
```python linenums="1"
# Inference (with tokens not similar to training data)
text = "Interpretability methods for explaining model behavior."
y_pred = model.predict(vectorizer.transform([text]))
label_encoder.decode(y_pred)
```
<pre class="output">
['natural-language-processing']
</pre>
```python linenums="1"
# Probabilities
y_prob = model.predict_proba(vectorizer.transform([text]))
{tag:y_prob[0][i] for i, tag in enumerate(label_encoder.classes)}
```
<pre class="output">
{'computer-vision': 0.13150802188532523,
 'mlops': 0.11198040241517894,
 'natural-language-processing': 0.584025872986128,
 'other': 0.17248570271336786}
</pre>

We're going to create a custom predict function where if the majority class is not above a certain softmax score, then we predict the `other` class. In our [objectives](purpose.md#objective){:target="_blank"}, we decided that precision is really important for us and that we can leverage the labeling and QA workflows to improve the recall during subsequent manual inspection.

!!! warning
    Our models can suffer from overconfidence so applying this limitation may not be as effective as we'd imagine, especially for larger neural networks. See the [confident learning](evaluation.md#confident-learning){:target="_blank"} section of the [evaluation lesson](evaluation.md){:target="_blank"} for more information.

```python linenums="1"
# Determine first quantile softmax score for the correct class (on validation split)
y_pred = model.predict(X_val)
y_prob = model.predict_proba(X_val)
threshold = np.quantile([y_prob[i][j] for i, j in enumerate(y_pred)], q=0.25)  # Q1
threshold
```

<pre class="output">
0.6742890218960005
</pre>

!!! warning
    It's very important that we do this on our validation split so we aren't inflating the value using the train split or leaking information prior to evaluation on the test split.

```python linenums="1"
# Custom predict function
def custom_predict(y_prob, threshold, index):
    """Custom predict function that defaults
    to an index if conditions are not met."""
    y_pred = [np.argmax(p) if max(p) > threshold else index for p in y_prob]
    return np.array(y_pred)
```

```python linenums="1"
def predict_tag(texts):
    y_prob = model.predict_proba(vectorizer.transform(texts))
    other_index = label_encoder.class_to_index["other"]
    y_pred = custom_predict(y_prob=y_prob, threshold=threshold, index=other_index)
    return label_encoder.decode(y_pred)
```

```python linenums="1"
# Inference (with tokens not similar to training data)
text = "Interpretability methods for explaining model behavior."
predict_tag(texts=[text])
```
<pre class="output">
['other']
</pre>

```python linenums="1"
# Evaluate
y_prob = model.predict_proba(X_test)
y_pred = custom_predict(y_prob=y_prob, threshold=threshold, index=other_index)
metrics = precision_recall_fscore_support(y_test, y_pred, average="weighted")
performance = {"precision": metrics[0], "recall": metrics[1], "f1": metrics[2]}
print (json.dumps(performance, indent=2))
```
<pre class="output">
{
  "precision": 0.9116161616161617,
  "recall": 0.7569444444444444,
  "f1": 0.7929971988795519
}
</pre>

!!! tip
    We could've even used per-class thresholds, especially since we have some data imbalance which can impact how confident the model is regarding some classes.

    ```python linenums="1"
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)
    class_thresholds = {}
    for index in range(len(label_encoder.classes)):
        class_thresholds[index] = np.mean(
            [y_prob[i][index] for i in np.where(y_pred==index)[0]])
    ```

> This MLOps course is actually model-agnostic (as long as it produces probability distributions) so feel free to use more complex representations ([embeddings](https://madewithml.com/courses/foundations/embeddings/){:target="_blank"}) with more sophisticated architectures ([CNNs](https://madewithml.com/courses/foundations/convolutional-neural-networks/){:target="_blank"}, [transformers](https://madewithml.com/courses/foundations/transformers/){:target="_blank"}, etc.). We're going to use this basic logistic regression model throughout the rest of the lessons because it's easy, fast and actually has comparable performance (<10% f1 diff compared to state-of-the-art pretrained transformers).

<!-- Citation -->
{% include "cite.md" %}