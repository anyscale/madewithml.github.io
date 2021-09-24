---
template: lesson.html
title: Convolutional Neural Networks (CNN)
description: Convolutional Neural Networks (CNNs) applied to text for natural language processing (NLP) tasks.
keywords: convolutional neural networks, CNN, computer vision, image classification, image recognition, batchnorm, batch normalization, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/foundations.png
repository: https://github.com/GokuMohandas/MadeWithML
notebook: https://colab.research.google.com/github/GokuMohandas/MadeWithML/blob/main/notebooks/11_Convolutional_Neural_Networks.ipynb
---

{% include "styles/lesson.md" %}

## Overview
At the core of CNNs are filters (aka weights, kernels, etc.) which convolve (slide) across our input to extract relevant features. The filters are initialized randomly but learn to act as feature extractors via parameter sharing.

<div class="ai-center-all">
    <img width="500" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/foundations/cnn/convolution.gif">
</div>

- **Objective**:
    - Extract meaningful spatial substructure from encoded data.
- **Advantages**:
    - Small number of weights (shared)
    - Parallelizable
    - Detects spatial substrcutures (feature extractors)
    - [Interpretability](#interpretability) via filters
    - Can be used for processing in images, text, time-series, etc.
- **Disadvantages**:
    - Many hyperparameters (kernel size, strides, etc.) to tune.
- **Miscellaneous**:
    - Lot's of deep CNN architectures constantly updated for SOTA performance.
    - Very popular feature extractor that acts as a foundation for many architectures.


## Set up
Let's set our seed and device.
```python linenums="1"
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
```
```python linenums="1"
SEED = 1234
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
# Set seeds for reproducibility
set_seeds(seed=SEED)
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

### Load data
We will download the [AG News dataset](http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html){:target="_blank"}, which consists of 120K text samples from 4 unique classes (`Business`, `Sci/Tech`, `Sports`, `World`)
```python linenums="1"
# Load data
url = "https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/datasets/news.csv"
df = pd.read_csv(url, header=0) # load
df = df.sample(frac=1).reset_index(drop=True) # shuffle
df.head()
```
<div class="output_subarea output_html rendered_html"><div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Sharon Accepts Plan to Reduce Gaza Army Operat...</td>
      <td>World</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Internet Key Battleground in Wildlife Crime Fight</td>
      <td>Sci/Tech</td>
    </tr>
    <tr>
      <th>2</th>
      <td>July Durable Good Orders Rise 1.7 Percent</td>
      <td>Business</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Growing Signs of a Slowing on Wall Street</td>
      <td>Business</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The New Faces of Reality TV</td>
      <td>World</td>
    </tr>
  </tbody>
</table>
</div></div>

### Preprocessing
We're going to clean up our input data first by doing operations such as lower text, removing stop (filler) words, filters using regular expressions, etc.
```python linenums="1"
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
```
```python linenums="1"
nltk.download("stopwords")
STOPWORDS = stopwords.words("english")
print (STOPWORDS[:5])
porter = PorterStemmer()
```
<pre class="output">
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data]   Package stopwords is already up-to-date!
['i', 'me', 'my', 'myself', 'we']
</pre>
```python linenums="1"
def preprocess(text, stopwords=STOPWORDS):
    """Conditional preprocessing on our text unique to our task."""
    # Lower
    text = text.lower()

    # Remove stopwords
    pattern = re.compile(r"\b(" + r"|".join(stopwords) + r")\b\s*")
    text = pattern.sub("", text)

    # Remove words in parenthesis
    text = re.sub(r"\([^)]*\)", "", text)

    # Spacing and filters
    text = re.sub(r"([-;;.,!?<=>])", r" \1 ", text)  # separate punctuation tied to words
    text = re.sub("[^A-Za-z0-9]+", " ", text)  # remove non alphanumeric chars
    text = re.sub(" +", " ", text)  # remove multiple spaces
    text = text.strip()

    return text
```
```python linenums="1"
# Sample
text = "Great week for the NYSE!"
preprocess(text=text)
```
<pre class="output">
great week nyse
</pre>
```python linenums="1"
# Apply to dataframe
preprocessed_df = df.copy()
preprocessed_df.title = preprocessed_df.title.apply(preprocess)
print (f"{df.title.values[0]}\n\n{preprocessed_df.title.values[0]}")
```
<pre class="output">
Sharon Accepts Plan to Reduce Gaza Army Operation, Haaretz Says

sharon accepts plan reduce gaza army operation haaretz says
</pre>

### Split data
```python linenums="1"
import collections
from sklearn.model_selection import train_test_split
```
```python linenums="1"
TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15
```
```python linenums="1"
def train_val_test_split(X, y, train_size):
    """Split dataset into data splits."""
    X_train, X_, y_train, y_ = train_test_split(X, y, train_size=TRAIN_SIZE, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_, y_, train_size=0.5, stratify=y_)
    return X_train, X_val, X_test, y_train, y_val, y_test
```
```python linenums="1"
# Data
X = preprocessed_df["title"].values
y = preprocessed_df["category"].values
```
```python linenums="1"
# Create data splits
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
    X=X, y=y, train_size=TRAIN_SIZE)
print (f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print (f"X_val: {X_val.shape}, y_val: {y_val.shape}")
print (f"X_test: {X_test.shape}, y_test: {y_test.shape}")
print (f"Sample point: {X_train[0]} → {y_train[0]}")
```
<pre class="output">
X_train: (84000,), y_train: (84000,)
X_val: (18000,), y_val: (18000,)
X_test: (18000,), y_test: (18000,)
Sample point: china battles north korea nuclear talks → World
</pre>

### Label encoding
Next we'll define a `LabelEncoder` to encode our text labels into unique indices
```python linenums="1"
import itertools
```
```python linenums="1"
class LabelEncoder(object):
    """Label encoder for tag labels."""
    def __init__(self, class_to_index={}):
        self.class_to_index = class_to_index
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        self.classes = list(self.class_to_index.keys())

    def __len__(self):
        return len(self.class_to_index)

    def __str__(self):
        return f"<LabelEncoder(num_classes={len(self)})>"

    def fit(self, y):
        classes = np.unique(y)
        for i, class_ in enumerate(classes):
            self.class_to_index[class_] = i
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        self.classes = list(self.class_to_index.keys())
        return self

    def encode(self, y):
        encoded = np.zeros((len(y)), dtype=int)
        for i, item in enumerate(y):
            encoded[i] = self.class_to_index[item]
        return encoded

    def decode(self, y):
        classes = []
        for i, item in enumerate(y):
            classes.append(self.index_to_class[item])
        return classes

    def save(self, fp):
        with open(fp, "w") as fp:
            contents = {'class_to_index': self.class_to_index}
            json.dump(contents, fp, indent=4, sort_keys=False)

    @classmethod
    def load(cls, fp):
        with open(fp, "r") as fp:
            kwargs = json.load(fp=fp)
        return cls(**kwargs)
```
```python linenums="1"
# Encode
label_encoder = LabelEncoder()
label_encoder.fit(y_train)
NUM_CLASSES = len(label_encoder)
label_encoder.class_to_index
```
<pre class="output">
{'Business': 0, 'Sci/Tech': 1, 'Sports': 2, 'World': 3}
</pre>
```python linenums="1"
# Convert labels to tokens
print (f"y_train[0]: {y_train[0]}")
y_train = label_encoder.encode(y_train)
y_val = label_encoder.encode(y_val)
y_test = label_encoder.encode(y_test)
print (f"y_train[0]: {y_train[0]}")
```
<pre class="output">
y_train[0]: World
y_train[0]: 3
</pre>
```python linenums="1"
# Class weights
counts = np.bincount(y_train)
class_weights = {i: 1.0/count for i, count in enumerate(counts)}
print (f"counts: {counts}\nweights: {class_weights}")
```
<pre class="output">
counts: [21000 21000 21000 21000]
weights: {0: 4.761904761904762e-05, 1: 4.761904761904762e-05, 2: 4.761904761904762e-05, 3: 4.761904761904762e-05}
</pre>


## Tokenizer
Our input data is text and we can't feed it directly to our models. So, we'll define a `Tokenizer` to convert our text input data into token indices. This means that every token (we can decide what a token is char, word, sub-word, etc.) is mapped to a unique index which allows us to represent our text as an array of indices.

```python linenums="1"
import json
from collections import Counter
from more_itertools import take
```
```python linenums="1"
class Tokenizer(object):
    def __init__(self, char_level, num_tokens=None,
                 pad_token="<PAD>", oov_token="<UNK>",
                 token_to_index=None):
        self.char_level = char_level
        self.separator = "" if self.char_level else " "
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
                text = text.split(" ")
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
We're going to restrict the number of tokens in our `Tokenizer` to the top 500 most frequent tokens (stop words already removed) because the full vocabulary size (~30K) is too large to run on Google Colab notebooks.

```python linenums="1"
# Tokenize
tokenizer = Tokenizer(char_level=False, num_tokens=500)
tokenizer.fit_on_texts(texts=X_train)
VOCAB_SIZE = len(tokenizer)
print (tokenizer)
```
<pre class="output">
Tokenizer(num_tokens=500)
</pre>
```python linenums="1"
# Sample of tokens
print (take(5, tokenizer.token_to_index.items()))
print (f"least freq token's freq: {tokenizer.min_token_freq}") # use this to adjust num_tokens
```
<pre class="output">
[('&lt;PAD&gt;', 0), ('&lt;UNK&gt;', 1), ('39', 2), ('b', 3), ('gt', 4)]
least freq token's freq: 166
</pre>
```python linenums="1"
# Convert texts to sequences of indices
X_train = tokenizer.texts_to_sequences(X_train)
X_val = tokenizer.texts_to_sequences(X_val)
X_test = tokenizer.texts_to_sequences(X_test)
preprocessed_text = tokenizer.sequences_to_texts([X_train[0]])[0]
print ("Text to indices:\n"
    f"  (preprocessed) → {preprocessed_text}\n"
    f"  (tokenized) → {X_train[0]}")
```
<pre class="output">
Text to indices:
  (preprocessed) → china &lt;UNK&gt; north korea nuclear talks
  (tokenized) → [ 16   1 285 142 114  24]
</pre>

!!! question "Did we need to split the data first?"
    How come we applied the preprocessing functions to the entire dataset but tokenization after splitting the dataset? Does it matter?

    ??? quote "Show answer"
        If you have preprocessing steps like standardization, etc. that are calculated, you need to separate the training and test set first before applying those operations. This is because we cannot apply any knowledge gained from the test set accidentally (data leak) during preprocessing/training. So for the tokenization process, it's important that we only fit using our train data split because during inference, our model will not always know every token so it's important to replicate that scenario with our validation and test splits as well. However for global preprocessing steps, like the preprocessing function where we aren't learning anything from the data itself, we can perform before splitting the data.

## One-hot encoding
One-hot encoding creates a binary column for each unique value for the feature we're trying to map.  All of the values in each token's array will be 0 except at the index that this specific token is represented by.

There are 5 words in the vocabulary:
```json linenums="1"
{
    "a": 0,
    "e": 1,
    "i": 2,
    "o": 3,
    "u": 4
}
```

Then the text `aou` would be represented by:
```python linenums="1"
[[1. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 1.]]
```

One-hot encoding allows us to represent our data in a way that our models can process the data and isn't biased by the actual value of the token (ex. if your labels were actual numbers).

!!! note
    We have already applied one-hot encoding in the previous lessons when we encoded our labels. Each label was represented by a unique index but when determining loss, we effectively use it's one hot representation and compared it to the predicted probability distribution. We never explicitly wrote this out since all of our previous tasks were multi-class which means every input had just one output class, so the 0s didn't affect the loss (though it did matter during back propagation).

```python linenums="1"
def to_categorical(seq, num_classes):
    """One-hot encode a sequence of tokens."""
    one_hot = np.zeros((len(seq), num_classes))
    for i, item in enumerate(seq):
        one_hot[i, item] = 1.
    return one_hot
```
```python linenums="1"
# One-hot encoding
print (X_train[0])
print (len(X_train[0]))
cat = to_categorical(seq=X_train[0], num_classes=len(tokenizer))
print (cat)
print (cat.shape)
```
<pre class="output">
[ 16   1 285 142 114  24]
6
[[0. 0. 0. ... 0. 0. 0.]
 [0. 1. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]]
(6, 500)
</pre>
```python linenums="1"
# Convert tokens to one-hot
vocab_size = len(tokenizer)
X_train = [to_categorical(seq, num_classes=vocab_size) for seq in X_train]
X_val = [to_categorical(seq, num_classes=vocab_size) for seq in X_val]
X_test = [to_categorical(seq, num_classes=vocab_size) for seq in X_test]
```

## Padding
Our inputs are all of varying length but we need each batch to be uniformly shaped. Therefore, we will use padding to make all the inputs in the batch the same length. Our padding index will be 0 (note that this is consistent with the `<PAD>` token defined in our `Tokenizer`).

!!! note
    One-hot encoding creates a batch of shape (`N`, `max_seq_len`, `vocab_size`) so we'll need to be able to pad 3D sequences.

```python linenums="1"
def pad_sequences(sequences, max_seq_len=0):
    """Pad sequences to max length in sequence."""
    max_seq_len = max(max_seq_len, max(len(sequence) for sequence in sequences))
    num_classes = sequences[0].shape[-1]
    padded_sequences = np.zeros((len(sequences), max_seq_len, num_classes))
    for i, sequence in enumerate(sequences):
        padded_sequences[i][:len(sequence)] = sequence
    return padded_sequences
```
```python linenums="1"
# 3D sequences
print (X_train[0].shape, X_train[1].shape, X_train[2].shape)
padded = pad_sequences(X_train[0:3])
print (padded.shape)
```
<pre class="output">
(6, 500) (5, 500) (6, 500)
(3, 6, 500)
</pre>

## Dataset
We're going to create Datasets and DataLoaders to be able to efficiently create batches with our data splits.

```python linenums="1"
FILTER_SIZE = 1 # unigram
```
```python linenums="1"
class Dataset(torch.utils.data.Dataset):
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

        # Pad sequences
        X = pad_sequences(X, max_seq_len=self.max_filter_size)

        # Cast
        X = torch.FloatTensor(X.astype(np.int32))
        y = torch.LongTensor(y.astype(np.int32))

        return X, y

    def create_dataloader(self, batch_size, shuffle=False, drop_last=False):
        return torch.utils.data.DataLoader(
            dataset=self, batch_size=batch_size, collate_fn=self.collate_fn,
            shuffle=shuffle, drop_last=drop_last, pin_memory=True)
```
```python linenums="1"
# Create datasets for embedding
train_dataset = Dataset(X=X_train, y=y_train, max_filter_size=FILTER_SIZE)
val_dataset = Dataset(X=X_val, y=y_val, max_filter_size=FILTER_SIZE)
test_dataset = Dataset(X=X_test, y=y_test, max_filter_size=FILTER_SIZE)
print ("Datasets:\n"
    f"  Train dataset:{train_dataset.__str__()}\n"
    f"  Val dataset: {val_dataset.__str__()}\n"
    f"  Test dataset: {test_dataset.__str__()}\n"
    "Sample point:\n"
    f"  X: {test_dataset[0][0]}\n"
    f"  y: {test_dataset[0][1]}")
```
<pre class="output">
Datasets:
  Train dataset: &lt;Dataset(N=84000)&gt;
  Val dataset: &lt;Dataset(N=18000)&gt;
  Test dataset: &lt;Dataset(N=18000)&gt;
Sample point:
  X: [[0. 0. 0. ... 0. 0. 0.]
 [0. 1. 0. ... 0. 0. 0.]
 [0. 1. 0. ... 0. 0. 0.]
 [0. 1. 0. ... 0. 0. 0.]]
  y: 1
</pre>
```python linenums="1"
# Create dataloaders
batch_size = 64
train_dataloader = train_dataset.create_dataloader(batch_size=batch_size)
val_dataloader = val_dataset.create_dataloader(batch_size=batch_size)
test_dataloader = test_dataset.create_dataloader(batch_size=batch_size)
batch_X, batch_y = next(iter(test_dataloader))
print ("Sample batch:\n"
    f"  X: {list(batch_X.size())}\n"
    f"  y: {list(batch_y.size())}\n"
    "Sample point:\n"
    f"  X: {batch_X[0]}\n"
    f"  y: {batch_y[0]}")
```
<pre class="output">
Sample batch:
  X: [64, 14, 500]
  y: [64]
Sample point:
  X: tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 1., 0.,  ..., 0., 0., 0.],
        [0., 1., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device="cpu")
  y: 1
</pre>

## CNN
We're going to learn about CNNs by applying them on 1D text data.

### Inputs
In the dummy example below, our inputs are composed of character tokens that are one-hot encoded. We have a batch of N samples, where each sample has 8 characters and each character is represented by an array of 10 values (`vocab size=10`). This gives our inputs the size `(N, 8, 10)`.

!!! note
    With PyTorch, when dealing with convolution, our inputs (X) need to have the channels as the second dimension, so our inputs will be `(N, 10, 8)`.

```python linenums="1"
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
```
```python linenums="1"
# Assume all our inputs are padded to have the same # of words
batch_size = 64
max_seq_len = 8 # words per input
vocab_size = 10 # one hot size
x = torch.randn(batch_size, max_seq_len, vocab_size)
print(f"X: {x.shape}")
x = x.transpose(1, 2)
print(f"X: {x.shape}")
```
<pre class="output">
X: torch.Size([64, 8, 10])
X: torch.Size([64, 10, 8])
</pre>

<div class="ai-center-all">
    <img width="500" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/foundations/cnn/inputs.png">
</div>
<div class="ai-center-all">
  <small>This diagram above is for char-level tokens but extends to any level of tokenization.</small>
</div>

### Filters
At the core of CNNs are filters (aka weights, kernels, etc.) which convolve (slide) across our input to extract relevant features. The filters are initialized randomly but learn to act as feature extractors via parameter sharing.

We can see convolution in the diagram below where we simplified the filters and inputs to be 2D for ease of visualization. Also note that the values are 0/1s but in reality they can be any floating point value.

<div class="ai-center-all">
    <img width="500" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/foundations/cnn/convolution.gif">
</div>

Now let's return to our actual inputs `x`, which is of shape (8, 10) [`max_seq_len`, `vocab_size`] and we want to convolve on this input using filters. We will use 50 filters that are of size (1, 3) and has the same depth as the number of channels (`num_channels` = `vocab_size` = `one_hot_size` = 10). This gives our filter a shape of (3, 10, 50) [`kernel_size`, `vocab_size`, `num_filters`]

<div class="ai-center-all">
    <img width="500" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/foundations/cnn/filters.png">
</div>

* `stride`: amount the filters move from one convolution operation to the next.
* `padding`: values (typically zero) padded to the input, typically to create a volume with whole number dimensions.

So far we've used a `stride` of 1 and `VALID` padding (no padding) but let's look at an example with a higher stride and difference between different padding approaches.

Padding types:

* `VALID`: no padding, the filters only use the "valid" values in the input. If the filter cannot reach all the input values (filters go left to right), the extra values on the right are dropped.
* `SAME`: adds padding evenly to the right (preferred) and left sides of the input so that all values in the input are processed.

<div class="ai-center-all">
    <img width="500" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/foundations/cnn/padding.png">
</div>

We're going to use the [Conv1d](https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html#torch.nn.Conv1d){:target="_blank"} layer to process our inputs.

```python linenums="1"
# Convolutional filters (VALID padding)
vocab_size = 10 # one hot size
num_filters = 50 # num filters
filter_size = 3 # filters are 3X3
stride = 1
padding = 0 # valid padding (no padding)
conv1 = nn.Conv1d(in_channels=vocab_size, out_channels=num_filters,
                  kernel_size=filter_size, stride=stride,
                  padding=padding, padding_mode="zeros")
print("conv: {}".format(conv1.weight.shape))
```
<pre class="output">
conv: torch.Size([50, 10, 3])
</pre>
```python linenums="1"
# Forward pass
z = conv1(x)
print (f"z: {z.shape}")
```
<pre class="output">
z: torch.Size([64, 50, 6])
</pre>
<div class="ai-center-all">
    <img width="500" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/foundations/cnn/conv.png">
</div>

When we apply these filter on our inputs, we receive an output of shape (N, 6, 50). We get 50 for the output channel dim because we used 50 filters and 6 for the conv outputs because:

$$ W_1 = \frac{W_2 - F + 2P}{S} + 1 = \frac{8 - 3 + 2(0)}{1} + 1 = 6 $$

$$ H_1 = \frac{H_2 - F + 2P}{S} + 1 = \frac{1 - 1 + 2(0)}{1} + 1 = 1 $$

$$ D_2 = D_1 $$

<center>

| Variable    | Description                          |
| :---------- | :----------------------------------- |
| $W$         | width of each input = 8              |
| $H$         | height of each input = 1             |
| $D$         | depth (# of channels)                |
| $F$         | filter size = 3                      |
| $P$         | padding = 0                          |
| $S$         | stride = 1                           |

</center>

Now we'll add padding so that the convolutional outputs are the same shape as our inputs. The amount of padding for the `SAME` padding can be determined using the same equation. We want out output to have the same width as our input, so we solve for P:

$$ \frac{W-F+2P}{S} + 1 = W $$

$$ P = \frac{S(W-1) - W + F}{2} $$

If $P$ is not a whole number, we round up (using `math.ceil`) and place the extra padding on the right side.

```python linenums="1"
# Convolutional filters (SAME padding)
vocab_size = 10 # one hot size
num_filters = 50 # num filters
filter_size = 3 # filters are 3X3
stride = 1
conv = nn.Conv1d(in_channels=vocab_size, out_channels=num_filters,
                 kernel_size=filter_size, stride=stride)
print("conv: {}".format(conv.weight.shape))
```
<pre class="output">
conv: torch.Size([50, 10, 3])
</pre>
```python linenums="1"
# `SAME` padding
padding_left = int((conv.stride[0]*(max_seq_len-1) - max_seq_len + filter_size)/2)
padding_right = int(math.ceil((conv.stride[0]*(max_seq_len-1) - max_seq_len + filter_size)/2))
print (f"padding: {(padding_left, padding_right)}")
```
<pre class="output">
padding: (1, 1)
</pre>
```python linenums="1"
# Forward pass
z = conv(F.pad(x, (padding_left, padding_right)))
print (f"z: {z.shape}")
```
<pre class="output">
z: torch.Size([64, 50, 8])
</pre>

!!! note
    We will explore larger dimensional convolution layers in subsequent lessons. For example, [Conv2D](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d){:target="_blank"} is used with 3D inputs (images, char-level text, etc.) and [Conv3D](https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html#torch.nn.Conv3d){:target="_blank"} is used for 4D inputs (videos, time-series, etc.).

### Pooling
The result of convolving filters on an input is a feature map. Due to the nature of convolution and overlaps, our feature map will have lots of redundant information. Pooling is a way to summarize a high-dimensional feature map into a lower dimensional one for simplified downstream computation. The pooling operation can be the max value, average, etc. in a certain receptive field. Below is an example of pooling where the outputs from a conv layer are `4X4` and we're going to apply max pool filters of size `2X2`.
<div class="ai-center-all">
    <img width="500" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/foundations/cnn/pooling.png">
</div>

$$ W_2 = \frac{W_1 - F}{S} + 1 = \frac{4 - 2}{2} + 1 = 2 $$

$$ H_2 = \frac{H_1 - F}{S} + 1 = \frac{4 - 2}{2} + 1 = 2 $$

$$ D_2 = D_1 $$

<center>

| Variable    | Description                          |
| :---------- | :----------------------------------- |
| $W$         | width of each input = 4              |
| $H$         | height of each input = 4             |
| $D$         | depth (# of channels)                |
| $F$         | filter size = 2                      |
| $S$         | stride = 2                           |

</center>

In our use case, we want to just take the one max value so we will use the [MaxPool1D](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html#torch.nn.MaxPool1d){:target="_blank"} layer, so our max-pool filter size will be max_seq_len.
```python linenums="1"
# Max pooling
pool_output = F.max_pool1d(z, z.size(2))
print("Size: {}".format(pool_output.shape))
```
<pre class="output">
Size: torch.Size([64, 50, 1])
</pre>

### Batch normalization
The last topic we'll cover before constructing our model is [batch normalization](https://arxiv.org/abs/1502.03167){:target="_blank"}. It's an operation that will standardize (mean=0, std=1) the activations from the previous layer. Recall that we used to standardize our inputs in previous notebooks so our model can optimize quickly with larger learning rates. It's the same concept here but we continue to maintain standardized values throughout the forward pass to further aid optimization.

```python linenums="1"
# Batch normalization
batch_norm = nn.BatchNorm1d(num_features=num_filters)
z = batch_norm(conv(x)) # applied to activations (after conv layer & before pooling)
print (f"z: {z.shape}")
```
<pre class="output">
z: torch.Size([64, 50, 6])
</pre>
```python linenums="1"
# Mean and std before batchnorm
print (f"mean: {torch.mean(conv1(x)):.2f}, std: {torch.std(conv(x)):.2f}")
```
<pre class="output">
mean: 0.01, std: 0.57
</pre>
```python linenums="1"
# Mean and std after batchnorm
print (f"mean: {torch.mean(z):.2f}, std: {torch.std(z):.2f}")
```
<pre class="output">
mean: 0.00, std: 1.00
</pre>

## Modeling

### Model

Let's visualize the model's forward pass.

1. We'll first tokenize our inputs (`batch_size`, `max_seq_len`).
2. Then we'll one-hot encode our tokenized inputs (`batch_size`, `max_seq_len`, `vocab_size`).
3. We'll apply convolution via filters (`filter_size`, `vocab_size`, `num_filters`) followed by batch normalization. Our filters act as character level n-gram detectors.
4. We'll apply 1D global max pooling which will extract the most relevant information from the feature maps for making the decision.
5. We feed the pool outputs to a fully-connected (FC) layer (with dropout).
6. We use one more FC layer with softmax to derive class probabilities.

<div class="ai-center-all">
    <img width="1000" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/foundations/cnn/model.png">
</div>

```python linenums="1"
NUM_FILTERS = 50
HIDDEN_DIM = 100
DROPOUT_P = 0.1
```
```python linenums="1"
class CNN(nn.Module):
    def __init__(self, vocab_size, num_filters, filter_size,
                 hidden_dim, dropout_p, num_classes):
        super(CNN, self).__init__()

        # Convolutional filters
        self.filter_size = filter_size
        self.conv = nn.Conv1d(
            in_channels=vocab_size, out_channels=num_filters,
            kernel_size=filter_size, stride=1, padding=0, padding_mode="zeros")
        self.batch_norm = nn.BatchNorm1d(num_features=num_filters)

        # FC layers
        self.fc1 = nn.Linear(num_filters, hidden_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, inputs, channel_first=False,):

        # Rearrange input so num_channels is in dim 1 (N, C, L)
        x_in, = inputs
        if not channel_first:
            x_in = x_in.transpose(1, 2)

        # Padding for `SAME` padding
        max_seq_len = x_in.shape[2]
        padding_left = int((self.conv.stride[0]*(max_seq_len-1) - max_seq_len + self.filter_size)/2)
        padding_right = int(math.ceil((self.conv.stride[0]*(max_seq_len-1) - max_seq_len + self.filter_size)/2))

        # Conv outputs
        z = self.conv(F.pad(x_in, (padding_left, padding_right)))
        z = F.max_pool1d(z, z.size(2)).squeeze(2)

        # FC layer
        z = self.fc1(z)
        z = self.dropout(z)
        z = self.fc2(z)
        return z
```
```python linenums="1"
# Initialize model
model = CNN(vocab_size=VOCAB_SIZE, num_filters=NUM_FILTERS, filter_size=FILTER_SIZE,
            hidden_dim=HIDDEN_DIM, dropout_p=DROPOUT_P, num_classes=NUM_CLASSES)
model = model.to(device) # set device
print (model.named_parameters)
```
<pre class="output">
&lt;bound method Module.named_parameters of CNN(
  (conv): Conv1d(500, 50, kernel_size=(1,), stride=(1,))
  (batch_norm): BatchNorm1d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc1): Linear(in_features=50, out_features=100, bias=True)
  (dropout): Dropout(p=0.1, inplace=False)
  (fc2): Linear(in_features=100, out_features=4, bias=True)
)&gt;
</pre>

!!! note
    We used `SAME` padding (w/ stride=1) which means that the conv outputs will have the same width (`max_seq_len`) as our inputs. The amount of padding differs for each batch based on the `max_seq_len` but you can calculate it by solving for P in the equation below.

$$ \frac{W_1 - F + 2P}{S} + 1 = W_2 $$

$$ \frac{\text{max_seq_len } - \text{ filter_size } + 2P}{\text{stride}} + 1 = \text{max_seq_len} $$

$$ P = \frac{\text{stride}(\text{max_seq_len}-1) - \text{max_seq_len} + \text{filter_size}}{2} $$

If $P$ is not a whole number, we round up (using `math.ceil`) and place the extra padding on the right side.

### Training
Let's create the `Trainer` class that we'll use to facilitate training for our experiments. Notice that we're now moving the `train` function inside this class.

```python linenums="1"
from torch.optim import Adam
```
```python linenums="1"
LEARNING_RATE = 1e-3
PATIENCE = 5
NUM_EPOCHS = 10
```
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
                y_prob = F.softmax(z).cpu().numpy()
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
                y_prob = F.softmax(z).cpu().numpy()
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
```python linenums="1"
# Define Loss
class_weights_tensor = torch.Tensor(list(class_weights.values())).to(device)
loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)
```
```python linenums="1"
# Define optimizer & scheduler
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.1, patience=3)
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
    NUM_EPOCHS, PATIENCE, train_dataloader, val_dataloader)
```
<pre class="output">
Epoch: 1 | train_loss: 0.87388, val_loss: 0.79013, lr: 1.00E-03, _patience: 3
Epoch: 2 | train_loss: 0.78354, val_loss: 0.78657, lr: 1.00E-03, _patience: 3
Epoch: 3 | train_loss: 0.77743, val_loss: 0.78433, lr: 1.00E-03, _patience: 3
Epoch: 4 | train_loss: 0.77242, val_loss: 0.78260, lr: 1.00E-03, _patience: 3
Epoch: 5 | train_loss: 0.76900, val_loss: 0.78169, lr: 1.00E-03, _patience: 3
Epoch: 6 | train_loss: 0.76613, val_loss: 0.78064, lr: 1.00E-03, _patience: 3
Epoch: 7 | train_loss: 0.76413, val_loss: 0.78019, lr: 1.00E-03, _patience: 3
Epoch: 8 | train_loss: 0.76215, val_loss: 0.78016, lr: 1.00E-03, _patience: 3
Epoch: 9 | train_loss: 0.76034, val_loss: 0.77974, lr: 1.00E-03, _patience: 3
Epoch: 10 | train_loss: 0.75859, val_loss: 0.77978, lr: 1.00E-03, _patience: 2
</pre>

### Evaluation
```python linenums="1"
import json
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support
```
```python linenums="1"
def get_metrics(y_true, y_pred, classes):
    """Per-class performance metrics."""
    # Performance
    performance = {"overall": {}, "class": {}}

    # Overall performance
    metrics = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    performance["overall"]["precision"] = metrics[0]
    performance["overall"]["recall"] = metrics[1]
    performance["overall"]["f1"] = metrics[2]
    performance["overall"]["num_samples"] = np.float64(len(y_true))

    # Per-class performance
    metrics = precision_recall_fscore_support(y_true, y_pred, average=None)
    for i in range(len(classes)):
        performance["class"][classes[i]] = {
            "precision": metrics[0][i],
            "recall": metrics[1][i],
            "f1": metrics[2][i],
            "num_samples": np.float64(metrics[3][i]),
        }

    return performance
```
```python linenums="1"
# Get predictions
test_loss, y_true, y_prob = trainer.eval_step(dataloader=test_dataloader)
y_pred = np.argmax(y_prob, axis=1)
```
```python linenums="1"
# Determine performance
performance = get_metrics(
    y_true=y_test, y_pred=y_pred, classes=label_encoder.classes)
print (json.dumps(performance["overall"], indent=2))
```
<pre class="output">
{
  "precision": 0.7120047175492572,
  "recall": 0.6935,
  "f1": 0.6931471439737603,
  "num_samples": 18000.0
}
</pre>
```python linenums="1"
# Save artifacts
dir = Path("cnn")
dir.mkdir(parents=True, exist_ok=True)
label_encoder.save(fp=Path(dir, "label_encoder.json"))
tokenizer.save(fp=Path(dir, 'tokenizer.json'))
torch.save(best_model.state_dict(), Path(dir, "model.pt"))
with open(Path(dir, 'performance.json'), "w") as fp:
    json.dump(performance, indent=2, sort_keys=False, fp=fp)
```

### Inference
```python linenums="1"
def get_probability_distribution(y_prob, classes):
    """Create a dict of class probabilities from an array."""
    results = {}
    for i, class_ in enumerate(classes):
        results[class_] = np.float64(y_prob[i])
    sorted_results = {k: v for k, v in sorted(
        results.items(), key=lambda item: item[1], reverse=True)}
    return sorted_results
```
```python linenums="1"
# Load artifacts
device = torch.device("cpu")
label_encoder = LabelEncoder.load(fp=Path(dir, "label_encoder.json"))
tokenizer = Tokenizer.load(fp=Path(dir, 'tokenizer.json'))
model = CNN(
    vocab_size=VOCAB_SIZE, num_filters=NUM_FILTERS, filter_size=FILTER_SIZE,
    hidden_dim=HIDDEN_DIM, dropout_p=DROPOUT_P, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(Path(dir, "model.pt"), map_location=device))
model.to(device)
```
<pre class="output">
CNN(
  (conv): Conv1d(500, 50, kernel_size=(1,), stride=(1,))
  (batch_norm): BatchNorm1d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc1): Linear(in_features=50, out_features=100, bias=True)
  (dropout): Dropout(p=0.1, inplace=False)
  (fc2): Linear(in_features=100, out_features=4, bias=True)
)
</pre>
```python linenums="1"
# Initialize trainer
trainer = Trainer(model=model, device=device)
```
```python linenums="1"
# Dataloader
text = "What a day for the new york stock market to go bust!"
sequences = tokenizer.texts_to_sequences([preprocess(text)])
print (tokenizer.sequences_to_texts(sequences))
X = [to_categorical(seq, num_classes=len(tokenizer)) for seq in sequences]
y_filler = label_encoder.encode([label_encoder.classes[0]]*len(X))
dataset = Dataset(X=X, y=y_filler, max_filter_size=FILTER_SIZE)
dataloader = dataset.create_dataloader(batch_size=batch_size)
```
<pre class="output">
['day new &lt;UNK&gt; stock market go &lt;UNK&gt;']
</pre>
```python linenums="1"
# Inference
y_prob = trainer.predict_step(dataloader)
y_pred = np.argmax(y_prob, axis=1)
label_encoder.decode(y_pred)
```
<pre class="output">
['Business']
</pre>
```python linenums="1"
# Class distributions
prob_dist = get_probability_distribution(y_prob=y_prob[0], classes=label_encoder.classes)
print (json.dumps(prob_dist, indent=2))
```
<pre class="output">
{
  "Business": 0.8670833110809326,
  "Sci/Tech": 0.10699427127838135,
  "World": 0.021050667390227318,
  "Sports": 0.004871787969022989
}
</pre>

## Interpretability
We went through all the trouble of padding our inputs before convolution to result is outputs of the same shape as our inputs so we can try to get some interpretability. Since every token is mapped to a convolutional output on which we apply max pooling, we can see which token's output was most influential towards the prediction. We first need to get the conv outputs from our model:

```python linenums="1"
import collections
import seaborn as sns
```
```python linenums="1"
class InterpretableCNN(nn.Module):
    def __init__(self, vocab_size, num_filters, filter_size,
                 hidden_dim, dropout_p, num_classes):
        super(InterpretableCNN, self).__init__()

        # Convolutional filters
        self.filter_size = filter_size
        self.conv = nn.Conv1d(
            in_channels=vocab_size, out_channels=num_filters,
            kernel_size=filter_size, stride=1, padding=0, padding_mode="zeros")
        self.batch_norm = nn.BatchNorm1d(num_features=num_filters)

        # FC layers
        self.fc1 = nn.Linear(num_filters, hidden_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, inputs, channel_first=False):

        # Rearrange input so num_channels is in dim 1 (N, C, L)
        x_in, = inputs
        if not channel_first:
            x_in = x_in.transpose(1, 2)

        # Padding for `SAME` padding
        max_seq_len = x_in.shape[2]
        padding_left = int((self.conv.stride[0]*(max_seq_len-1) - max_seq_len + self.filter_size)/2)
        padding_right = int(math.ceil((self.conv.stride[0]*(max_seq_len-1) - max_seq_len + self.filter_size)/2))

        # Conv outputs
        z = self.conv(F.pad(x_in, (padding_left, padding_right)))
        return z
```
```python linenums="1"
# Initialize
interpretable_model = InterpretableCNN(
    vocab_size=len(tokenizer), num_filters=NUM_FILTERS, filter_size=FILTER_SIZE,
    hidden_dim=HIDDEN_DIM, dropout_p=DROPOUT_P, num_classes=NUM_CLASSES)
```
```python linenums="1"
# Load weights (same architecture)
interpretable_model.load_state_dict(torch.load(Path(dir, "model.pt"), map_location=device))
interpretable_model.to(device)
```
<pre class="output">
InterpretableCNN(
  (conv): Conv1d(500, 50, kernel_size=(1,), stride=(1,))
  (batch_norm): BatchNorm1d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc1): Linear(in_features=50, out_features=100, bias=True)
  (dropout): Dropout(p=0.1, inplace=False)
  (fc2): Linear(in_features=100, out_features=4, bias=True)
)
</pre>
```python linenums="1"
# Initialize trainer
interpretable_trainer = Trainer(model=interpretable_model, device=device)
```
```python linenums="1"
# Get conv outputs
conv_outputs = interpretable_trainer.predict_step(dataloader)
print (conv_outputs.shape) # (num_filters, max_seq_len)
```
<pre class="output">
(50, 7)
</pre>
```python linenums="1"
# Visualize a bi-gram filter's outputs
tokens = tokenizer.sequences_to_texts(sequences)[0].split(" ")
sns.heatmap(conv_outputs, xticklabels=tokens)
```
<div class="ai-center-all">
    <img width="400" src="data:image/png;base64,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">
</div>

The filters have high values for the words `stock` and `market` which influenced the `Business` category classification.

!!! warning
    This is a crude technique loosely based off of more elaborate [interpretability](https://arxiv.org/abs/1312.6034){:target="_blank"} methods.


<!-- Citation -->
{% include "cite.md" %}