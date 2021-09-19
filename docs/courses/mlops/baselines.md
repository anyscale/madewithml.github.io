---
template: lesson.html
title: Modeling Baselines
description: Motivating the use of baselines for iterative modeling.
keywords: baselines, modeling, pytorch, transformers, huggingface, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/mlops.png
repository: https://github.com/GokuMohandas/MLOps
notebook: https://colab.research.google.com/github/GokuMohandas/MLOps/blob/main/notebooks/tagifai.ipynb
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
2. Develop a rule-based approach (when possible) using IFTT, auxiliary data, etc.
3. Slowly add complexity by *addressing* limitations and *motivating* representations and model architectures.
4. Weigh *tradeoffs* (performance, latency, size, etc.) between performant baselines.
5. Revisit and iterate on baselines as your dataset grows.

!!! note
    You can also baseline on your dataset. Instead of using a fixed dataset and iterating on the models, choose a good baseline and iterate on the dataset:

    - remove or fix data samples (FP, FN)
    - prepare and transform features
    - expand or consolidate classes
    - incorporate auxiliary datasets
    - identify unique slices to improve / upsample

## Tradeoffs

When choosing what model architecture(s) to proceed with, there are a few important aspects to consider:

- `#!js performance`: consider coarse-grained and fine-grained (ex. per-class) performance.
- `#!js latency`: how quickly does your model respond for inference.
- `#!js size`: how large is your model and can you support it's storage.
- `#!js compute`: how much will it cost ($, carbon footprint, etc.) to train your model?
- `#!js interpretability`: does your model need to explain its predictions?
- `#!js bias checks`: does your model pass key bias checks?
- `#!js time to develop`: how long do you have to develop the first version?
- `#!js time to retrain`: how long does it take to retrain your model? This is very important to consider if you need to retrain often.
- `#!js maintenance overhead`: who and what will be required to maintain your model versions because the real work with ML begins after deploying v1. You can't just hand it off to your site reliability team to maintain it like many teams do with traditional software.

## Application

Each application's baseline trajectory varies based on the task and motivations. For our application, we're going to follow this path:

1. [Random](#random)
2. [Rule-based](#rule-based)
3. [Simple ML](#simple-ml)
4. [CNN w/ embeddings](#cnn)
5. [RNN w/ embeddings](#rnn)
6. [Transformers w/ contextual embeddings](#transformers)

We'll motivate the need for slowly adding complexity from both the **representation** (ex. embeddings) and **architecture** (ex. CNNs) views, as well as address the limitation at each step of the way.

!!! note
    If you're unfamiliar with of the concepts here, be sure to check out the [GokuMohandas/MadeWithML](https://github.com/GokuMohandas/MadeWithML){:target="_blank"} (🔥 Among [top ML repos](https://github.com/topics/deep-learning){:target="_blank"} on GitHub).


We'll first set up some functions that we'll be using across the different baseline experiments.
```python linenums="1"
from sklearn.metrics import precision_recall_fscore_support
import torch
```
```python linenums="1"
def set_seeds(seed=1234):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # multi-GPU
```
```python linenums="1"
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
We'll define a Trainer object which we will use for training, validation and inference.
```python linenums="1"
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
        with torch.inference_mode():
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
        with torch.inference_mode():
            for i, batch in enumerate(dataloader):

                # Forward pass w/ inputs
                inputs, targets = batch[:-1], batch[-1]
                z = self.model(inputs)

                # Store outputs
                y_prob = torch.sigmoid(z).cpu().numpy()
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

!!! note
    Our dataset is small so we'll train using the whole dataset but for larger datasets, we should always test on a small subset  (after shuffling when necessary) so we aren't wasting time on compute. Here's how you can easily do this:
    ```python linenums="1"
    # Shuffling since projects are chronologically organized
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)

    # Subset
    if num_samples:
        df = df[:num_samples]
    ```

<hr>

## Random
<u><i>motivation</i></u>: We want to know what random (chance) performance looks like. All of our efforts should be well above this.

```python linenums="1"
# Set seeds
set_seeds()
```
```python linenums="1"
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
```python linenums="1"
# Label encoder
print (label_encoder)
print (label_encoder.classes)
```
<pre class="output">
&lt;LabelEncoder(num_classes=35)&gt;
['attention', 'autoencoders', 'computer-vision', 'convolutional-neural-networks', 'data-augmentation', 'embeddings', 'flask', 'generative-adversarial-networks', 'graph-neural-networks', 'graphs', 'huggingface', 'image-classification', 'interpretability', 'keras', 'language-modeling', 'natural-language-processing', 'node-classification', 'object-detection', 'pretraining', 'production', 'pytorch', 'question-answering', 'regression', 'reinforcement-learning', 'representation-learning', 'scikit-learn', 'segmentation', 'self-supervised-learning', 'tensorflow', 'tensorflow-js', 'time-series', 'transfer-learning', 'transformers', 'unsupervised-learning', 'wandb']
</pre>
```python linenums="1"
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
```python linenums="1"
# Evaluate
metrics = precision_recall_fscore_support(y_test, y_pred, average="weighted")
performance = {"precision": metrics[0], "recall": metrics[1], "f1": metrics[2]}
print (json.dumps(performance, indent=2))
```
<pre class="output">
{
  "precision": 0.12590604458654545,
  "recall": 0.5203426124197003,
  "f1": 0.18469743862395557
}
</pre>

We made the assumption that there is an equal probability for whether an input has a tag or not but this isn't true. Let's use the **train split** to figure out what the true probability is.

```python linenums="1"
# Percentage of 1s (tag presence)
tag_p = np.sum(np.sum(y_train)) / (len(y_train) * len(label_encoder.classes))
print (tag_p)
```
<pre class="output">
0.06291428571428571
</pre>
```python linenums="1"
# Generate weighted random predictions
y_pred = np.random.choice(
    np.arange(0, 2), size=(len(y_test), len(label_encoder.classes)),
    p=[1-tag_p, tag_p])
```
```python linenums="1"
# Validate percentage
np.sum(np.sum(y_pred)) / (len(y_pred) * len(label_encoder.classes))
```
<pre class="output">
0.06240947992100066
</pre>
```python linenums="1"
# Evaluate
metrics = precision_recall_fscore_support(y_test, y_pred, average="weighted")
performance = {"precision": metrics[0], "recall": metrics[1], "f1": metrics[2]}
print (json.dumps(performance, indent=2))
```
<pre class="output">
{
  "precision": 0.1121905967477629,
  "recall": 0.047109207708779445,
  "f1": 0.05309836327850377
}
</pre>

<u><i>limitations</i></u>: we didn't use the tokens in our input to affect our predictions so nothing was learned.

<hr>

## Rule-based
<u><i>motivation</i></u>: we want to use signals in our inputs (along with domain expertise and auxiliary data) to determine the labels.

```python linenums="1"
# Set seeds
set_seeds()
```

### Unstemmed

```python linenums="1"
# Get data splits
preprocessed_df = df.copy()
preprocessed_df.text = preprocessed_df.text.apply(preprocess, lower=True)
X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = get_data_splits(preprocessed_df)
```
```python linenums="1"
# Restrict to relevant tags
print (len(tags_dict))
tags_dict = {tag: tags_dict[tag] for tag in label_encoder.classes}
print (len(tags_dict))
```
<pre class="output">
400
35
</pre>
```python linenums="1"
# Map aliases
aliases = {}
for tag, values in tags_dict.items():
    aliases[preprocess(tag)] = tag
    for alias in values["aliases"]:
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
```python linenums="1"
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
```python linenums="1"
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
```python linenums="1"
# Prediction
y_pred = []
for text in X_test:
    classes = get_classes(text, aliases, tags_dict)
    y_pred.append(classes)
```
```python linenums="1"
# Encode labels
y_pred = label_encoder.encode(y_pred)
```
```python linenums="1"
# Evaluate
metrics = precision_recall_fscore_support(y_test, y_pred, average="weighted")
performance = {"precision": metrics[0], "recall": metrics[1], "f1": metrics[2]}
print (json.dumps(performance, indent=2))
```
<pre class="output">
{
    "precision": 0.8527917293434535,
    "recall": 0.38066760941576216,
    "f1": 0.48975323243320396,
    "num_samples": 480.0
}
</pre>
```python linenums="1"
# Inspection
tag = "transformers"
print (json.dumps(performance["class"][tag], indent=2))
```
<pre class="output">
{
  "precision": 0.886542414851697,
  "recall": 0.430406852248394,
  "f1": 0.556927275918014
}
</pre>

### Stemmed

We're looking for exact matches with the aliases which isn't always perfect, for example:
```python linenums="1"
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
```python linenums="1"
print (porter.stem("democracy"))
print (porter.stem("democracies"))
```
<pre class="output">
democraci
democraci
</pre>
So let's now stem our aliases as well as the tokens in our input text and then look for matches.
```python linenums="1"
# Get data splits
preprocessed_df = df.copy()
preprocessed_df.text = preprocessed_df.text.apply(preprocess, lower=True, stem=True)
X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = get_data_splits(preprocessed_df)
```
```python linenums="1"
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
```python linenums="1"
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
> We'll write [proper tests](testing.md){:target="_blank"} for all of these functions when we move our code to Python scripts.
```python linenums="1"
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
```python linenums="1"
# Prediction
y_pred = []
for text in X_test:
    classes = get_classes(text, aliases, tags_dict)
    y_pred.append(classes)
```
```python linenums="1"
# Encode labels
y_pred = label_encoder.encode(y_pred)
```

### Evaluation
We can look at overall and per-class performance on our test set.

!!! note
    When considering overall and per-class performance across different models, we should be aware of [Simpson's paradox](https://en.wikipedia.org/wiki/Simpson%27s_paradox){:target="_blank"} where a model can perform better on every class subset but not overall.

```python linenums="1"
# Evaluate
metrics = precision_recall_fscore_support(y_test, y_pred, average="weighted")
performance = {"precision": metrics[0], "recall": metrics[1], "f1": metrics[2]}
print (json.dumps(performance, indent=2))
```
<pre class="output">
{
  "precision": 0.907266867724384,
  "recall": 0.485838779956427,
  "f1": 0.6120705676738784
}
</pre>

Though we achieved decent precision, the recall is quite low. This is because rule-based approaches can yield labels with high certainty when there is an absolute condition match but it fails to generalize or learn implicit patterns.

### Inference

```python linenums="1"
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
```python linenums="1"
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

!!! note
    We would ideally spend more time tuning our model because it's so simple and quick to train. This approach also applies to all the other models we'll look at as well.

<hr>

## Simple ML
<u><i>motivation</i></u>:

- **representation**: use term frequency-inverse document frequency [(TF-IDF)](https://en.wikipedia.org/wiki/Tf%E2%80%93idf){:target="_blank"} to capture the significance of a token to a particular input with respect to all the inputs, as opposed to treating the words in our input text as isolated tokens.
- **architecture**: we want our model to meaningfully extract the encoded signal to predict the output labels.

So far we've treated the words in our input text as isolated tokens and we haven't really captured any meaning between tokens. Let's use term frequency–inverse document frequency (**TF-IDF**) to capture the significance of a token to a particular input with respect to all the inputs.

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
```
```python linenums="1"
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import MultiLabelBinarizer
```
```python linenums="1"
# Set seeds
set_seeds()
```
```python linenums="1"
# Get data splits
preprocessed_df = df.copy()
preprocessed_df.text = preprocessed_df.text.apply(preprocess, lower=True, stem=True)
X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = get_data_splits(preprocessed_df)
```
```python linenums="1"
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
```python linenums="1"
def fit_and_evaluate(model):
    """Fit and evaluate each model."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = precision_recall_fscore_support(y_test, y_pred, average="weighted")
    return {"precision": metrics[0], "recall": metrics[1], "f1": metrics[2]}
```
```python linenums="1"
# Models
performance = {}
performance["logistic-regression"] = fit_and_evaluate(OneVsRestClassifier(
    LogisticRegression(), n_jobs=1))
performance["k-nearest-neighbors"] = fit_and_evaluate(
    KNeighborsClassifier())
performance["random-forest"] = fit_and_evaluate(
    RandomForestClassifier(n_jobs=-1))
performance["gradient-boosting-machine"] = fit_and_evaluate(OneVsRestClassifier(
    GradientBoostingClassifier()))
performance["support-vector-machine"] = fit_and_evaluate(OneVsRestClassifier(
    LinearSVC(), n_jobs=-1))
print (json.dumps(performance, indent=2))
```
<pre class="output">
{
  "logistic-regression": {
    "precision": 0.633369022127052,
    "recall": 0.21841541755888652,
    "f1": 0.3064204603390899
  },
  "k-nearest-neighbors": {
    "precision": 0.7410281119097024,
    "recall": 0.47109207708779444,
    "f1": 0.5559182508714337
  },
  "random-forest": {
    "precision": 0.7722866712160075,
    "recall": 0.38329764453961457,
    "f1": 0.4852512297132596
  },
  "gradient-boosting-machine": {
    "precision": 0.8503271303309295,
    "recall": 0.6167023554603854,
    "f1": 0.7045318461336975
  },
  "support-vector-machine": {
    "precision": 0.8938397993500261,
    "recall": 0.5460385438972163,
    "f1": 0.6527334570244009
  }
}
</pre>

<u><i>limitations</i></u>:

- **representation**: TF-IDF representations don't encapsulate much signal beyond frequency but we require more fine-grained token representations.
- **architecture**: we want to develop models that can use better represented encodings in a more contextual manner.

## Distributed training

All the training we need to do for our application happens on one worker with one accelerator (GPU), however, we'll want to consider distributed training for very large models or when dealing with large datasets. Distributed training can involve:

- **data parallelism**: workers received different slices of the larger dataset.
    - *synchronous training* uses [AllReduce](https://tech.preferred.jp/en/blog/technologies-behind-distributed-deep-learning-allreduce/#:~:text=AllReduce%20is%20an%20operation%20that,of%20length%20N%20called%20A_p.){:target="_blank"} to aggregate gradients and update all the workers weights at the end of each batch (synchronous).
    - *asynchronous training* uses a universal parameter server to update weights as each worker trains on its slice of data (asynchronous).
- **model parallelism**: all workers use the same dataset but the model is split amongst them (more difficult to implement compared to data parallelism).

There are lots of options for applying distributed training such as with PyTorch's [distributed package](https://pytorch.org/tutorials/beginner/dist_overview.html){:target="_blank"}, [Ray](https://ray.io/){:target="_blank"}, [Horovd](https://horovod.ai/){:target="_blank"}, etc.

<hr>

## CNN w/ Embeddings

<u><i>motivation</i></u>:

- **representation**: we want to have more robust (split tokens to characters) and meaningful [embeddings](../foundations/embeddings.md){:target="_blank"} representations for our input tokens.
- **architecture**: we want to process our encoded inputs using [convolution (CNN)](../foundations/convolutional-neural-networks.md){:target="_blank"} filters that can learn to analyze windows of embedded tokens to extract meaningful signal.


### Set up

We'll set up the task by setting seeds for reproducibility, creating our data splits abd setting the device.
```python linenums="1"
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
```
```python linenums="1"
# Set seeds
set_seeds()
```
```python linenums="1"
# Get data splits
preprocessed_df = df.copy()
preprocessed_df.text = preprocessed_df.text.apply(preprocess, lower=True)
X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = get_data_splits(preprocessed_df)
X_test_raw = X_test  # use for later
```
```python linenums="1"
# Split DataFrames
train_df = pd.DataFrame({"text": X_train, "tags": label_encoder.decode(y_train)})
val_df = pd.DataFrame({"text": X_val, "tags": label_encoder.decode(y_val)})
test_df = pd.DataFrame({"text": X_test, "tags": label_encoder.decode(y_test)})
```
```python linenums="1"
# Set device
cuda = True
device = torch.device("cuda" if (
    torch.cuda.is_available() and cuda) else "cpu")
torch.set_default_tensor_type("torch.FloatTensor")
if device.type == "cuda":
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
print (device)
```
<pre class="output">
cuda
</pre>

### Tokenizer

We're going to tokenize our input text as character tokens so we can be robust to spelling errors and learn to generalize across tags. (ex. learning that RoBERTa, or any other future BERT based archiecture, warrants same tag as BERT).

<div class="ai-center-all">
    <img width="500" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/foundations/cnn/inputs.png">
</div>

```python linenums="1"
class Tokenizer(object):
    def __init__(self, char_level, num_tokens=None,
                 pad_token="<PAD>", oov_token="<UNK>",
                 token_to_index=None):
        self.char_level = char_level
        self.separator = '' if self.char_level else ' '
        if num_tokens: num_tokens -= 2 # pad + unk tokens
        self.num_tokens = num_tokens
        self.pad_token = pad_token
        self.oov_token = oov_token
        if not token_to_index:
            token_to_index = {pad_token: 0, oov_token: 1}
        self.token_to_index = token_to_index
        self.index_to_token = {v: k for k, v in self.token_to_index.items()}

    def __len__(self):
        return len(self.token_to_index)

    def __str__(self):
        return f"<Tokenizer(num_tokens={len(self)})>"

    def fit_on_texts(self, texts):
        if not self.char_level:
            texts = [text.split(" ") for text in texts]
        all_tokens = [token for text in texts for token in text]
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
        with open(fp, "w") as fp:
            contents = {
                "char_level": self.char_level,
                "oov_token": self.oov_token,
                "token_to_index": self.token_to_index
            }
            json.dump(contents, fp, indent=4, sort_keys=False)

    @classmethod
    def load(cls, fp):
        with open(fp, "r") as fp:
            kwargs = json.load(fp=fp)
        return cls(**kwargs)
```
```python linenums="1"
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
```python linenums="1"
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
```python linenums="1"
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
  (preprocessed) → hugging face achieved 2x performance boost qa question answering distilbert node js
  (tokenized) → [18 17 15 15  4  5 15  2 19  7 12  3  2  7 12 18  4  3 22  3 14  2 26 25
  2 13  3  8 19 10  8 16  7  5 12  3  2 20 10 10  9  6  2 30  7  2 30 17
  3  9  6  4 10  5  2  7  5  9 23  3  8  4  5 15  2 14  4  9  6  4 11 20
  3  8  6  2  5 10 14  3  2 28  9]
</pre>

### Data imbalance

We'll factor class weights in our objective function ([binary cross entropy with logits](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html){:target="_blank"}) to help with [class imbalance](labeling.md#data-imbalance){:target="_blank"}. There are many other techniques such as over sampling from underrepresented classes, undersampling, etc. but we'll cover these in a separate unit lesson on data imbalance.

```python linenums="1"
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

### Datasets

We're going to place our data into a [`Dataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset){:target="_blank"} and use a [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader){:target="_blank"} to efficiently create batches for training and evaluation.

```python linenums="1"
def pad_sequences(sequences, max_seq_len=0):
    """Pad sequences to max length in sequence."""
    max_seq_len = max(max_seq_len, max(len(sequence) for sequence in sequences))
    padded_sequences = np.zeros((len(sequences), max_seq_len))
    for i, sequence in enumerate(sequences):
        padded_sequences[i][:len(sequence)] = sequence
    return padded_sequences
```
```python linenums="1"
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
        batch = np.array(batch, dtype=object)
        X = batch[:, 0]
        y = np.stack(batch[:, 1], axis=0)

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
```python linenums="1"
# Create datasets
filter_sizes = list(range(1, 11))
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
  Train dataset: &lt;Dataset(N=1000)&gt;
  Val dataset: &lt;Dataset(N=227)&gt;
  Test dataset: &lt;Dataset(N=217)&gt;
Sample point:
  X: [ 7 11 20 17 16  3  5  6  7  6  4 10  5  9  2 19  7  9  6  2  4 16  7 14
  3  2  7 17 14 16  3  5  6  7  6  4 10  5  2 11  4 20  8  7  8 21  2  3
  7  9 21  2 17  9  3  2 23  8  7 13 13  3  8  2  7  8 10 17  5 15  2 11
  4 20  8  7  8  4  3  9]
  y: [0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
</pre>
```python linenums="1"
# Create dataloaders
batch_size = 64
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

### Model

We'll be using a convolutional neural network on top of our embedded tokens to extract meaningful spatial signal. This time, we'll be using many filter widths to act as n-gram feature extractors. If you're not familiar with CNNs be sure to check out the [CNN lesson](https://madewithml.com/courses/foundations/convolutional-neural-networks/){:target="_blank"} where we walkthrough every component of the architecture.

<div class="ai-center-all">
    <img width="500" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/foundations/cnn/convolution.gif">
</div>

Let's visualize the model's forward pass.

1. We'll first tokenize our inputs (`batch_size`, `max_seq_len`).
2. Then we'll embed our tokenized inputs (`batch_size`, `max_seq_len`, `embedding_dim`).
3. We'll apply convolution via filters (`filter_size`, `vocab_size`, `num_filters`) followed by batch normalization. Our filters act as character level n-gram detecors. We have three different filter sizes (2, 3 and 4) and they will act as bi-gram, tri-gram and 4-gram feature extractors, respectivelyy.
4. We'll apply 1D global max pooling which will extract the most relevant information from the feature maps for making the decision.
5. We feed the pool outputs to a fully-connected (FC) layer (with dropout).
6. We use one more FC layer with softmax to derive class probabilities.

<div class="ai-center-all">
    <img width="1000" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/foundations/embeddings/model.png">
</div>

```python linenums="1"
# Arguments
embedding_dim = 128
num_filters = 128
hidden_dim = 128
dropout_p = 0.5
```
```python linenums="1"
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

- **VALID**: no padding, the filters only use the "valid" values in the input. If the filter cannot reach all the input values (filters go left to right), the extra values on the right are dropped.
- **SAME**: adds padding evenly to the right (preferred) and left sides of the input so that all values in the input are processed.

<div class="ai-center-all">
    <img width="500" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/foundations/cnn/padding.png">
</div>

We're add `SAME` padding so that the convolutional outputs are the same shape as our inputs. The amount of padding for the SAME padding can be determined using the same equation. We want out output to have the same width as our input, so we solve for P:

$$ \frac{W-F+2P}{S} + 1 = W $$

$$ P = \frac{S(W-1) - W + F}{2} $$

If $P$ is not a whole number, we round up (using `math.ceil`) and place the extra padding on the right side.

```python linenums="1"
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
&lt;bound method Module.named_parameters of CNN(
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
)&gt;
</pre>

### Training

```python linenums="1"
# Arguments
lr = 2e-4
num_epochs = 100
patience = 10
```
```python linenums="1"
# Define loss
class_weights_tensor = torch.Tensor(np.array(list(class_weights.values())))
loss_fn = nn.BCEWithLogitsLoss(weight=class_weights_tensor)
```
```python linenums="1"
# Define optimizer & scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.1, patience=5)
```
```python linenums="1"
# Trainer module
trainer = Trainer(
    model=model, device=device, loss_fn=loss_fn,
    optimizer=optimizer, scheduler=scheduler)
```
```python linenums="1"
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

### Evaluation

```python linenums="1"
from pathlib import Path
from sklearn.metrics import precision_recall_curve
```
```python linenums="1"
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

```python linenums="1"
# Determining the best threshold
def find_best_threshold(y_true, y_prob):
    """Find the best threshold for maximum F1."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = (2 * precisions * recalls) / (precisions + recalls)
    return thresholds[np.argmax(f1s)]
```

```python linenums="1"
# Best threshold for f1
threshold = find_best_threshold(y_true.ravel(), y_prob.ravel())
threshold
```
<pre class="output">
0.23890994
</pre>

!!! question "How can we do better?"
    How can we improve on our process of identifying and using the appropriate threshold?

    ??? quote "Show answer"
        - [x] Plot PR curves for all classes (not just overall) to ensure a certain global threshold doesn't deliver very poor performance for any particular class
        - [x] Determine different thresholds for different classes and use them during inference

```python linenums="1"
# Determine predictions using threshold
test_loss, y_true, y_prob = trainer.eval_step(dataloader=test_dataloader)
y_pred = np.array([np.where(prob >= threshold, 1, 0) for prob in y_prob])
```
```python linenums="1"
# Evaluate
metrics = precision_recall_fscore_support(y_test, y_pred, average="weighted")
performance = {"precision": metrics[0], "recall": metrics[1], "f1": metrics[2]}
print (json.dumps(performance, indent=2))
```
<pre class="output">
{
  "precision": 0.795787334399384,
  "recall": 0.5944206008583691,
  "f1": 0.6612833723992106
}
</pre>

```python linenums="1"
# Save artifacts
dir = Path("cnn")
dir.mkdir(parents=True, exist_ok=True)
tokenizer.save(fp=Path(dir, "tokenzier.json"))
label_encoder.save(fp=Path(dir, "label_encoder.json"))
torch.save(best_model.state_dict(), Path(dir, "model.pt"))
with open(Path(dir, "performance.json"), "w") as fp:
    json.dump(performance, indent=2, sort_keys=False, fp=fp)
```

### Inference

```python linenums="1"
# Load artifacts
device = torch.device("cpu")
tokenizer = Tokenizer.load(fp=Path(dir, "tokenzier.json"))
label_encoder = LabelEncoder.load(fp=Path(dir, "label_encoder.json"))
model = CNN(
    embedding_dim=embedding_dim, vocab_size=vocab_size,
    num_filters=num_filters, filter_sizes=filter_sizes,
    hidden_dim=hidden_dim, dropout_p=dropout_p,
    num_classes=num_classes)
model.load_state_dict(torch.load(Path(dir, "model.pt"), map_location=device))
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
```python linenums="1"
# Initialize trainer
trainer = Trainer(model=model, device=device)
```
```python linenums="1"
# Dataloader
text = "Transfer learning with BERT for self-supervised learning"
X = np.array(tokenizer.texts_to_sequences([preprocess(text)]))
y_filler = label_encoder.encode([np.array([label_encoder.classes[0]]*len(X))])
dataset = CNNTextDataset(
    X=X, y=y_filler, max_filter_size=max(filter_sizes))
dataloader = dataset.create_dataloader(
    batch_size=batch_size)
```
```python linenums="1"
# Inference
y_prob = trainer.predict_step(dataloader)
y_pred = np.array([np.where(prob >= threshold, 1, 0) for prob in y_prob])
label_encoder.decode(y_pred)
```
<pre class="output">
[['natural-language-processing',
  'self-supervised-learning',
  'transfer-learning',
  'transformers']]
</pre>

<u><i>limitations</i></u>:

- **representation**: embeddings are not contextual.
- **architecture**: extracting signal from encoded inputs is limited by filter widths.

!!! note
    Since we're dealing with simple architectures and fast training times, it's a good opportunity to explore tuning and experiment with k-fold cross validation to properly reach any conclusions about performance.

<hr>

## Tradeoffs
We're going to go with the embeddings via CNN approach and optimize it because performance is quite similar to the contextualized embeddings via transformers approach but at much lower cost.

```python linenums="1"
# Performance
with open(Path("cnn", "performance.json"), "r") as fp:
    cnn_performance = json.load(fp)
print (f'CNN: f1 = {cnn_performance["f1"]}')
```
<pre class="output">
CNN: f1 = 0.6612833723992106
</pre>

This was just one run on one split so you'll want to experiment with k-fold cross validation to properly reach any conclusions about performance. Also make sure you take the time to tune these baselines since their training periods are quite fast (we can achieve f1 of 0.7 with just a bit of tuning for both CNN / Transformers). We'll cover hyperparameter tuning in a few lessons so you can replicate the process here on your own time. We should also benchmark on other important metrics as we iterate, not just precision and recall.

```python linenums="1"
# Size
print (f'CNN: {Path("cnn", "model.pt").stat().st_size/1000000:.1f} MB')
```
<pre class="output">
CNN: 4.3 MB
</pre>

We'll consider other tradeoffs such as maintenance overhead, behavioral test performances, etc. as we develop.

!!! note
    Interpretability was not one of requirements but note that we could've tweaked model outputs to deliver it. For example, since we used SAME padding for our CNN, we can use the activation scores to extract influential n-grams.


## Resources

- [Backing off towards simplicity - why baselines need more love](https://smerity.com/articles/2017/baselines_need_love.html){:target="_blank"}
- [Model Evaluation, Model Selection, and Algorithm Selection in Machine Learning](https://arxiv.org/abs/1811.12808){:target="_blank"}


<!-- Citation -->
{% include "cite.md" %}