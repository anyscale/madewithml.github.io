---
template: lesson.html
title: Recurrent Neural Networks (RNN)
description: Explore and motivate the need for representation via embeddings.
keywords: recurrent neural networks, numpy, pytorch, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/foundations.png
repository: https://github.com/GokuMohandas/Made-With-ML
notebook: https://github.com/GokuMohandas/Made-With-ML/blob/main/notebooks/13_Recurrent_Neural_Networks.ipynb
---

{% include "styles/lesson.md" %}

## Overview
So far we've processed inputs as whole (ex. applying filters across the entire input to extract features) but we can also process our inputs sequentially. For example we can think of each token in our text as an event in time (timestep). We can process each timestep, one at a time, and predict the class after the last timestep (token) has been processed. This is very powerful because the model now has a meaningful way to account for the sequential order of tokens in our sequence and predict accordingly.

<div class="ai-center-all">
    <img width="500" src="/static/images/foundations/rnn/vanilla.png" alt="vanilla RNN">
</div>

$$ \text{RNN forward pass for a single time step } X_t $$:

$$ h_t = tanh(W_{hh}h_{t-1} + W_{xh}X_t+b_h) $$

<center>

| Variable    | Description                          |
| :---------- | :----------------------------------- |
| $N$         | batch size                                                         |
| $E$         | embeddings dimension                                               |
| $H$         | # of hidden units                                                  |
| $W_{hh}$    | RNN weights $\in \mathbb{R}^{HXH}$                                 |
| $h_{t-1}$   | previous timestep's hidden state $\in in \mathbb{R}^{NXH}$         |
| $W_{xh}$    | input weights $\in \mathbb{R}^{EXH}$                               |
| $X_t$       | input at time step $t \in \mathbb{R}^{NXE}$                        |
| $b_h$       | hidden units bias $\in \mathbb{R}^{HX1}$                           |
| $h_t$       | output from RNN for timestep $t$                                   |

</center>

* **Objective**:
    - Process sequential data by accounting for the current input and also what has been learned from previous inputs.
* **Advantages**:
    - Account for order and previous inputs in a meaningful way.
    - Conditioned generation for generating sequences.
* **Disadvantages**:
    - Each time step's prediction depends on the previous prediction so it's difficult to parallelize RNN operations.
    - Processing long sequences can yield memory and computation issues.
    - Interpretability is difficult but there are few [techniques](https://arxiv.org/abs/1506.02078){:target="_blank"} that use the activations from RNNs to see what parts of the inputs are processed.
* **Miscellaneous**:
    - Architectural tweaks to make RNNs faster and interpretable is an ongoing area of research.

## Set up
Let's set our seed and device for our main task.
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
url = "https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/news.csv"
df = pd.read_csv(url, header=0) # load
df = df.sample(frac=1).reset_index(drop=True) # shuffle
df.head()
```
<div class="output_subarea output_html rendered_html ai-center-all"><div>
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
      <td>Sharon Accepts Plan to Reduce Gaza Army Operation...</td>
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
    text = re.sub(r"([-;;.,!?<=>])", r" \1 ", text)
    text = re.sub("[^A-Za-z0-9]+", " ", text) # remove non alphanumeric chars
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

!!! warning
    If you have preprocessing steps like standardization, etc. that are calculated, you need to separate the training and test set first before applying those operations. This is because we cannot apply any knowledge gained from the test set accidentally (data leak) during preprocessing/training. However for global preprocessing steps like the function above where we aren't learning anything from the data itself, we can perform before splitting the data.

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
        self.class_to_index = class_to_index or {}  # mutable defaults ;)
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

### Tokenizer
We'll define a `Tokenizer` to convert our text input data into token indices.

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

!!! warning
    It's important that we only fit using our train data split because during inference, our model will not always know every token so it's important to replicate that scenario with our validation and test splits as well.

```python linenums="1"
# Tokenize
tokenizer = Tokenizer(char_level=False, num_tokens=5000)
tokenizer.fit_on_texts(texts=X_train)
VOCAB_SIZE = len(tokenizer)
print (tokenizer)
```
<pre class="output">
&lt;Tokenizer(num_tokens=5000)&gt;

</pre>
```python linenums="1"
# Sample of tokens
print (take(5, tokenizer.token_to_index.items()))
print (f"least freq token's freq: {tokenizer.min_token_freq}") # use this to adjust num_tokens
```
<pre class="output">
[('&lt;PAD&gt;', 0), ('&lt;UNK&gt;', 1), ('39', 2), ('b', 3), ('gt', 4)]
least freq token's freq: 14
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
  (preprocessed) → china battles north korea nuclear talks
  (tokenized) → [  16 1491  285  142  114   24]
</pre>

### Padding
We'll need to do 2D padding to our tokenized text.
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
# 2D sequences
padded = pad_sequences(X_train[0:3])
print (padded.shape)
print (padded)
```
<pre class="output">
(3, 6)
[[1.600e+01 1.491e+03 2.850e+02 1.420e+02 1.140e+02 2.400e+01]
 [1.445e+03 2.300e+01 6.560e+02 2.197e+03 1.000e+00 0.000e+00]
 [1.200e+02 1.400e+01 1.955e+03 1.005e+03 1.529e+03 4.014e+03]]
</pre>

### Datasets
We're going to create Datasets and DataLoaders to be able to efficiently create batches with our data splits.

```python linenums="1"
class Dataset(torch.utils.data.Dataset):
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
        batch = np.array(batch)
        X = batch[:, 0]
        seq_lens = batch[:, 1]
        y = batch[:, 2]

        # Pad inputs
        X = pad_sequences(sequences=X)

        # Cast
        X = torch.LongTensor(X.astype(np.int32))
        seq_lens = torch.LongTensor(seq_lens.astype(np.int32))
        y = torch.LongTensor(y.astype(np.int32))

        return X, seq_lens, y

    def create_dataloader(self, batch_size, shuffle=False, drop_last=False):
        return torch.utils.data.DataLoader(
            dataset=self, batch_size=batch_size, collate_fn=self.collate_fn,
            shuffle=shuffle, drop_last=drop_last, pin_memory=True)
```
```python linenums="1"
# Create datasets
train_dataset = Dataset(X=X_train, y=y_train)
val_dataset = Dataset(X=X_val, y=y_val)
test_dataset = Dataset(X=X_test, y=y_test)
print ("Datasets:\n"
    f"  Train dataset:{train_dataset.__str__()}\n"
    f"  Val dataset: {val_dataset.__str__()}\n"
    f"  Test dataset: {test_dataset.__str__()}\n"
    "Sample point:\n"
    f"  X: {train_dataset[0][0]}\n"
    f"  seq_len: {train_dataset[0][1]}\n"
    f"  y: {train_dataset[0][2]}")
```
<pre class="output">
Datasets:
  Train dataset: &lt;Dataset(N=84000)&gt;
  Val dataset: &lt;Dataset(N=18000)&gt;
  Test dataset: &lt;Dataset(N=18000)&gt;
Sample point:
  X: [  16 1491  285  142  114   24]
  seq_len: 6
  y: 3
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
batch_X, batch_seq_lens, batch_y = next(iter(train_dataloader))
print ("Sample batch:\n"
    f"  X: {list(batch_X.size())}\n"
    f"  seq_lens: {list(batch_seq_lens.size())}\n"
    f"  y: {list(batch_y.size())}\n"
    "Sample point:\n"
    f"  X: {batch_X[0]}\n"
    f" seq_len: {batch_seq_lens[0]}\n"
    f"  y: {batch_y[0]}")
```
<pre class="output">
Sample batch:
  X: [64, 14]
  seq_lens: [64]
  y: [64]
Sample point:
  X: tensor([  16, 1491,  285,  142,  114,   24,    0,    0,    0,    0,    0,    0,
           0,    0])
 seq_len: 6
  y: 3
</pre>

### Trainer
Let's create the `Trainer` class that we'll use to facilitate training for our experiments.

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


## Vanilla RNN

### RNN
Inputs to RNNs are sequential like text or time-series.

```python linenums="1"
BATCH_SIZE = 64
EMBEDDING_DIM = 100
```
```python linenums="1"
# Input
sequence_size = 8 # words per input
x = torch.rand((BATCH_SIZE, sequence_size, EMBEDDING_DIM))
seq_lens = torch.randint(high=sequence_size, size=(BATCH_SIZE, ))
print (x.shape)
print (seq_lens.shape)
```
<pre class="output">
torch.Size([64, 8, 100])
torch.Size([1, 64])
</pre>
<div class="ai-center-all">
    <img width="500" src="/static/images/foundations/rnn/vanilla.png" alt="vanilla RNN">
</div>

$$ \text{RNN forward pass for a single time step } X_t $$:

$$ h_t = tanh(W_{hh}h_{t-1} + W_{xh}X_t+b_h) $$

<center>

| Variable    | Description                          |
| :---------- | :----------------------------------- |
| $N$         | batch size                                                         |
| $E$         | embeddings dimension                                               |
| $H$         | # of hidden units                                                  |
| $W_{hh}$    | RNN weights $\in \mathbb{R}^{HXH}$                                 |
| $h_{t-1}$   | previous timestep's hidden state $\in in \mathbb{R}^{NXH}$         |
| $W_{xh}$    | input weights $\in \mathbb{R}^{EXH}$                               |
| $X_t$       | input at time step $t \in \mathbb{R}^{NXE}$                        |
| $b_h$       | hidden units bias $\in \mathbb{R}^{HX1}$                           |
| $h_t$       | output from RNN for timestep $t$                                   |

</center>

> At the first time step, the previous hidden state $h_{t-1}$ can either be a zero vector (unconditioned) or initialized (conditioned). If we are conditioning the RNN, the first hidden state $h_0$ can belong to a specific condition or we can concat the specific condition to the randomly initialized hidden vectors at each time step. More on this in the subsequent notebooks on RNNs.

```python linenums="1"
RNN_HIDDEN_DIM = 128
DROPOUT_P = 0.1
```
```python linenums="1"
# Initialize hidden state
hidden_t = torch.zeros((BATCH_SIZE, RNN_HIDDEN_DIM))
print (hidden_t.size())
```
<pre class="output">
torch.Size([64, 128])
</pre>

We'll show how to create an RNN cell using PyTorch's [`RNNCell`](https://pytorch.org/docs/stable/generated/torch.nn.RNNCell.html#torch.nn.RNNCell){:target="_blank"} and the more abstracted [`RNN`](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html#torch.nn.RNN){:target="_blank"}.

```python linenums="1"
# Initialize RNN cell
rnn_cell = nn.RNNCell(EMBEDDING_DIM, RNN_HIDDEN_DIM)
print (rnn_cell)
```
<pre class="output">
RNNCell(100, 128)
</pre>
```python linenums="1"
# Forward pass through RNN
x = x.permute(1, 0, 2) # RNN needs batch_size to be at dim 1

# Loop through the inputs time steps
hiddens = []
for t in range(sequence_size):
    hidden_t = rnn_cell(x[t], hidden_t)
    hiddens.append(hidden_t)
hiddens = torch.stack(hiddens)
hiddens = hiddens.permute(1, 0, 2) # bring batch_size back to dim 0
print (hiddens.size())
```
<pre class="output">
torch.Size([64, 8, 128])
</pre>
```python linenums="1"
# We also could've used a more abstracted layer
x = torch.rand((BATCH_SIZE, sequence_size, EMBEDDING_DIM))
rnn = nn.RNN(EMBEDDING_DIM, RNN_HIDDEN_DIM, batch_first=True)
out, h_n = rnn(x) # h_n is the last hidden state
print ("out: ", out.shape)
print ("h_n: ", h_n.shape)
```
<pre class="output">
out:  torch.Size([64, 8, 128])
h_n:  torch.Size([1, 64, 128])
</pre>
```python linenums="1"
# The same tensors
print (out[:,-1,:])
print (h_n.squeeze(0))
```
<pre class="output">
tensor([[-0.0359, -0.3819,  0.2162,  ..., -0.3397,  0.0468,  0.1937],
        [-0.4914, -0.3056, -0.0837,  ..., -0.3507, -0.4320,  0.3593],
        [-0.0989, -0.2852,  0.1170,  ..., -0.0805, -0.0786,  0.3922],
        ...,
        [-0.3115, -0.4169,  0.2611,  ..., -0.3214,  0.0620,  0.0338],
        [-0.2455, -0.3380,  0.2048,  ..., -0.4198, -0.0075,  0.0372],
        [-0.2092, -0.4594,  0.1654,  ..., -0.5397, -0.1709,  0.0023]],
       grad_fn=&lt;SliceBackward&gt;)
tensor([[-0.0359, -0.3819,  0.2162,  ..., -0.3397,  0.0468,  0.1937],
        [-0.4914, -0.3056, -0.0837,  ..., -0.3507, -0.4320,  0.3593],
        [-0.0989, -0.2852,  0.1170,  ..., -0.0805, -0.0786,  0.3922],
        ...,
        [-0.3115, -0.4169,  0.2611,  ..., -0.3214,  0.0620,  0.0338],
        [-0.2455, -0.3380,  0.2048,  ..., -0.4198, -0.0075,  0.0372],
        [-0.2092, -0.4594,  0.1654,  ..., -0.5397, -0.1709,  0.0023]],
       grad_fn=&lt;SqueezeBackward1&gt;)
</pre>

In our model, we want to use the RNN's output after the last relevant token in the sentence is processed. The last relevant token doesn't refer the `<PAD>` tokens but to the last actual word in the sentence and its index is different for each input in the batch. This is why we included a `seq_lens` tensor in our batches.

```python linenums="1"
def gather_last_relevant_hidden(hiddens, seq_lens):
    """Extract and collect the last relevant
    hidden state based on the sequence length."""
    seq_lens = seq_lens.long().detach().cpu().numpy() - 1
    out = []
    for batch_index, column_index in enumerate(seq_lens):
        out.append(hiddens[batch_index, column_index])
    return torch.stack(out)
```
```python linenums="1"
# Get the last relevant hidden state
gather_last_relevant_hidden(hiddens=out, seq_lens=seq_lens).squeeze(0).shape
```
<pre class="output">
torch.Size([64, 128])
</pre>

There are many different ways to use RNNs. So far we've processed our inputs one timestep at a time and we could either use the RNN's output at each time step or just use the final input timestep's RNN output. Let's look at a few other possibilities.
<div class="ai-center-all">
    <img width="1000" src="/static/images/foundations/rnn/architectures.png" alt="RNN architecture">
</div>

### Model
```python linenums="1"
import torch.nn.functional as F
```
```python linenums="1"
HIDDEN_DIM = 100
```
```python linenums="1"
class RNN(nn.Module):
    def __init__(self, embedding_dim, vocab_size, rnn_hidden_dim,
                 hidden_dim, dropout_p, num_classes, padding_idx=0):
        super(RNN, self).__init__()

        # Initialize embeddings
        self.embeddings = nn.Embedding(
            embedding_dim=embedding_dim, num_embeddings=vocab_size,
            padding_idx=padding_idx)

        # RNN
        self.rnn = nn.RNN(embedding_dim, rnn_hidden_dim, batch_first=True)

        # FC weights
        self.dropout = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(rnn_hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, inputs):
        # Embed
        x_in, seq_lens = inputs
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
```python linenums="1"
# Simple RNN cell
model = RNN(
    embedding_dim=EMBEDDING_DIM, vocab_size=VOCAB_SIZE,
    rnn_hidden_dim=RNN_HIDDEN_DIM, hidden_dim=HIDDEN_DIM,
    dropout_p=DROPOUT_P, num_classes=NUM_CLASSES)
model = model.to(device) # set device
print (model.named_parameters)
```
<pre class="output">
&lt;bound method Module.named_parameters of RNN(
  (embeddings): Embedding(5000, 100, padding_idx=0)
  (rnn): RNN(100, 128, batch_first=True)
  (dropout): Dropout(p=0.1, inplace=False)
  (fc1): Linear(in_features=128, out_features=100, bias=True)
  (fc2): Linear(in_features=100, out_features=4, bias=True)
)&gt;
</pre>

### Training
```python linenums="1"
from torch.optim import Adam
```
```python linenums="1"
NUM_LAYERS = 1
LEARNING_RATE = 1e-4
PATIENCE = 10
NUM_EPOCHS = 50
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
Epoch: 1 | train_loss: 1.25605, val_loss: 1.10880, lr: 1.00E-04, _patience: 10
Epoch: 2 | train_loss: 1.03074, val_loss: 0.96749, lr: 1.00E-04, _patience: 10
Epoch: 3 | train_loss: 0.90110, val_loss: 0.86424, lr: 1.00E-04, _patience: 10
...
Epoch: 31 | train_loss: 0.32206, val_loss: 0.53581, lr: 1.00E-06, _patience: 3
Epoch: 32 | train_loss: 0.32233, val_loss: 0.53587, lr: 1.00E-07, _patience: 2
Epoch: 33 | train_loss: 0.32215, val_loss: 0.53572, lr: 1.00E-07, _patience: 1
Stopping early!
</pre>

### Evaluation
```python linenums="1"
import json
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
  "precision": 0.8171357577653572,
  "recall": 0.8176111111111112,
  "f1": 0.8171696173843819,
  "num_samples": 18000.0
}
</pre>

## Gated RNN
While our simple RNNs so far are great for sequentially processing our inputs, they have quite a few disadvantages. They commonly suffer from exploding or vanishing gradients as a result using the same set of weights ($W_{xh}$ and $W_{hh}$) with each timestep's input. During backpropagation, this can cause gradients to explode (>1) or vanish (<1). If you multiply any number greater than 1 with itself over and over, it moves towards infinity (exploding gradients) and similarly,  If you multiply any number less than 1 with itself over and over, it moves towards zero (vanishing gradients). To mitigate this issue, gated RNNs were devised to selectively retain information. If you're interested in learning more of the specifics, this [post](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) is a must-read.

There are two popular types of gated RNNs: Long Short-term Memory (LSTMs) units and Gated Recurrent Units (GRUs).

> When deciding between LSTMs and GRUs, empirical performance is the best factor but in general GRUs offer similar performance with less complexity (less weights).

<div class="ai-center-all">
    <img width="550" src="/static/images/foundations/rnn/gated.png" alt="gated RNN">
</div>
<div class="ai-center-all">
  <small><a href="http://colah.github.io/posts/2015-08-Understanding-LSTMs/" target="_blank">Understanding LSTM Networks</a> - Chris Olah</small>
</div>

```python linenums="1"
# Input
sequence_size = 8 # words per input
x = torch.rand((BATCH_SIZE, sequence_size, EMBEDDING_DIM))
print (x.shape)
```
<pre class="output">
torch.Size([64, 8, 100])
</pre>
```python linenums="1"
# GRU
gru = nn.GRU(input_size=EMBEDDING_DIM, hidden_size=RNN_HIDDEN_DIM, batch_first=True)
```
```python linenums="1"
# Forward pass
out, h_n = gru(x)
print (f"out: {out.shape}")
print (f"h_n: {h_n.shape}")
```
<pre class="output">
out: torch.Size([64, 8, 128])
h_n: torch.Size([1, 64, 128])
</pre>

### Bidirectional RNN
We can also have RNNs that process inputs from both directions (first token to last token and vice versa) and combine their outputs. This architecture is known as a bidirectional RNN.
```python linenums="1"
# GRU
gru = nn.GRU(input_size=EMBEDDING_DIM, hidden_size=RNN_HIDDEN_DIM,
             batch_first=True, bidirectional=True)
```
```python linenums="1"
# Forward pass
out, h_n = gru(x)
print (f"out: {out.shape}")
print (f"h_n: {h_n.shape}")
```
<pre class="output">
out: torch.Size([64, 8, 256])
h_n: torch.Size([2, 64, 128])
</pre>
Notice that the output for each sample at each timestamp has size 256 (double the `RNN_HIDDEN_DIM`). This is because this includes both the forward and backward directions from the BiRNN.

<h4 id="model_gated">Model</h4>

```python linenums="1"
class GRU(nn.Module):
    def __init__(self, embedding_dim, vocab_size, rnn_hidden_dim,
                 hidden_dim, dropout_p, num_classes, padding_idx=0):
        super(GRU, self).__init__()

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

    def forward(self, inputs:
        # Embed
        x_in, seq_lens = inputs
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
```python linenums="1"
# Simple RNN cell
model = GRU(
    embedding_dim=EMBEDDING_DIM, vocab_size=VOCAB_SIZE,
    rnn_hidden_dim=RNN_HIDDEN_DIM, hidden_dim=HIDDEN_DIM,
    dropout_p=DROPOUT_P, num_classes=NUM_CLASSES)
model = model.to(device) # set device
print (model.named_parameters)
```
<pre class="output">
&lt;bound method Module.named_parameters of GRU(
  (embeddings): Embedding(5000, 100, padding_idx=0)
  (rnn): GRU(100, 128, batch_first=True, bidirectional=True)
  (dropout): Dropout(p=0.1, inplace=False)
  (fc1): Linear(in_features=256, out_features=100, bias=True)
  (fc2): Linear(in_features=100, out_features=4, bias=True)
)&gt;
</pre>

### Training
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
Epoch: 1 | train_loss: 1.18125, val_loss: 0.93827, lr: 1.00E-04, _patience: 10
Epoch: 2 | train_loss: 0.81291, val_loss: 0.72564, lr: 1.00E-04, _patience: 10
Epoch: 3 | train_loss: 0.65413, val_loss: 0.64487, lr: 1.00E-04, _patience: 10
...
Epoch: 23 | train_loss: 0.30351, val_loss: 0.53904, lr: 1.00E-06, _patience: 3
Epoch: 24 | train_loss: 0.30332, val_loss: 0.53912, lr: 1.00E-07, _patience: 2
Epoch: 25 | train_loss: 0.30300, val_loss: 0.53909, lr: 1.00E-07, _patience: 1
Stopping early!
</pre>

### Evaluation
```python linenums="1"
from pathlib import Path
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
  "precision": 0.8192635071011053,
  "recall": 0.8196111111111111,
  "f1": 0.8192710197821547,
  "num_samples": 18000.0
}
</pre>
```python linenums="1"
# Save artifacts
dir = Path("gru")
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
model = GRU(
    embedding_dim=EMBEDDING_DIM, vocab_size=VOCAB_SIZE,
    rnn_hidden_dim=RNN_HIDDEN_DIM, hidden_dim=HIDDEN_DIM,
    dropout_p=DROPOUT_P, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(Path(dir, "model.pt"), map_location=device))
model.to(device)
```
<pre class="output">
GRU(
  (embeddings): Embedding(5000, 100, padding_idx=0)
  (rnn): GRU(100, 128, batch_first=True, bidirectional=True)
  (dropout): Dropout(p=0.1, inplace=False)
  (fc1): Linear(in_features=256, out_features=100, bias=True)
  (fc2): Linear(in_features=100, out_features=4, bias=True)
)
</pre>
```python linenums="1"
# Initialize trainer
trainer = Trainer(model=model, device=device)
```
```python linenums="1"
# Dataloader
text = "The final tennis tournament starts next week."
X = tokenizer.texts_to_sequences([preprocess(text)])
print (tokenizer.sequences_to_texts(X))
y_filler = label_encoder.encode([label_encoder.classes[0]]*len(X))
dataset = Dataset(X=X, y=y_filler)
dataloader = dataset.create_dataloader(batch_size=batch_size)
```
<pre class="output">
['final tennis tournament starts next week']
</pre>
```python linenums="1"
# Inference
y_prob = trainer.predict_step(dataloader)
y_pred = np.argmax(y_prob, axis=1)
label_encoder.decode(y_pred)
```
<pre class="output">
['Sports']
</pre>
```python linenums="1"
# Class distributions
prob_dist = get_probability_distribution(y_prob=y_prob[0], classes=label_encoder.classes)
print (json.dumps(prob_dist, indent=2))
```
<pre class="output">
{
  "Sports": 0.49753469228744507,
  "World": 0.2925860285758972,
  "Business": 0.1932886838912964,
  "Sci/Tech": 0.01659061387181282
}
</pre>

> We will learn how to create more context-aware representations and a little bit of interpretability with RNNs in the next lesson on [attention](attention.md){:target="_blank"}.


<!-- Citation -->
{% include "templates/cite.md" %}