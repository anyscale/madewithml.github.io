---
template: lesson.html
title: Attention
description: Incorporating attention mechanisms to create context-aware representations.
keywords: attention, transformers, self-attention, positional encoding, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/foundations.png
repository: https://github.com/GokuMohandas/Made-With-ML
notebook: https://colab.research.google.com/github/GokuMohandas/Made-With-ML/blob/main/notebooks/14_Attention.ipynb
---

{% include "styles/lesson.md" %}

## Overview

In the <a target="_blank" href="https://madewithml.com/courses/foundations/recurrent-neural-networks/">RNN lesson</a>, we were constrained to using the representation at the very end but what if we could give contextual weight to each encoded input ($h_i$) when making our prediction? This is also preferred because it can help mitigate the vanishing gradient issue which stems from processing very long sequences. Below is attention applied to the outputs from an RNN. In theory, the outputs can come from anywhere where we want to learn how to weight amongst them but since we're working with the context of an RNN from the previous lesson , we'll continue with that.

<div class="ai-center-all">
    <img src="/static/images/foundations/attention/attention.png" width="500">
</div>

$$ \alpha = softmax(W_{attn}h) $$

$$ c_t = \sum_{i=1}^{n} \alpha_{t,i}h_i $$

<center>

| Variable        | Description                                                                           |
| :-------------- | :------------------------------------------------------------------------------------ |
| $N$             | batch size                                                                            |
| $M$             | max sequence length in the batch                                                      |
| $H$             | hidden dim, model dim, etc.                                                           |
| $h$             | RNN outputs (or any group of outputs you want to attend to) $\in \mathbb{R}^{NXMXH}$  |
| $\alpha_{t,i}$  | alignment function context vector $c_t$ (attention in our case) $                     |
| $W_{attn}$      | attention weights to learn $\in \mathbb{R}^{HX1}$                                     |
| $c_t$           | context vector that accounts for the different inputs with attention                  |

</center>

- **Objective**:
    - At it's core, attention is about learning how to weigh a group of encoded representations to produce a context-aware representation to use for downstream tasks. This is done by learning a set of attention weights and then using softmax to create attention values that sum to 1.
- **Advantages**:
    - Learn how to account for the appropriate encoded representations regardless of position.
- **Disadvantages**:
    - Another compute step that involves learning weights.
- **Miscellaneous**:
    - Several state-of-the-art approaches extend on basic attention to deliver highly context-aware representations (ex. self-attention).

## Set up

Let's set our seed and device for our main task.
```python linenums="1"
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
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
                y_prob = F.softmax(model(inputs), dim=1)

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

## Attention

Attention applied to the outputs from an RNN. In theory, the outputs can come from anywhere where we want to learn how to weight amongst them but since we're working with the context of an RNN from the previous lesson , we'll continue with that.

<div class="ai-center-all">
    <img src="/static/images/foundations/attention/attention.png" width="500">
</div>

$$ \alpha = softmax(W_{attn}h) $$

$$ c_t = \sum_{i=1}^{n} \alpha_{t,i}h_i $$

<center>

| Variable        | Description                                                                           |
| :-------------- | :------------------------------------------------------------------------------------ |
| $N$             | batch size                                                                            |
| $M$             | max sequence length in the batch                                                      |
| $H$             | hidden dim, model dim, etc.                                                           |
| $h$             | RNN outputs (or any group of outputs you want to attend to) $\in \mathbb{R}^{NXMXH}$  |
| $\alpha_{t,i}$  | alignment function context vector $c_t$ (attention in our case) $                     |
| $W_{attn}$      | attention weights to learn $\in \mathbb{R}^{HX1}$                                     |
| $c_t$           | context vector that accounts for the different inputs with attention                  |

</center>

```python linenums="1"
import torch.nn.functional as F
```

The RNN will create an encoded representation for each word in our input resulting in a stacked vector that has dimensions $NXMXH$, where N is the # of samples in the batch, M is the max sequence length in the batch, and H is the number of hidden units in the RNN.

```python linenums="1"
BATCH_SIZE = 64
SEQ_LEN = 8
EMBEDDING_DIM = 100
RNN_HIDDEN_DIM = 128
```

```python linenums="1"
# Embed
x = torch.rand((BATCH_SIZE, SEQ_LEN, EMBEDDING_DIM))
```

```python linenums="1"
# Encode
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
# Attend
attn = nn.Linear(RNN_HIDDEN_DIM, 1)
e = attn(out)
attn_vals = F.softmax(e.squeeze(2), dim=1)
c = torch.bmm(attn_vals.unsqueeze(1), out).squeeze(1)
print ("e: ", e.shape)
print ("attn_vals: ", attn_vals.shape)
print ("attn_vals[0]: ", attn_vals[0])
print ("sum(attn_vals[0]): ", sum(attn_vals[0]))
print ("c: ", c.shape)
```

<pre class="output">
e:  torch.Size([64, 8, 1])
attn_vals:  torch.Size([64, 8])
attn_vals[0]:  tensor([0.1131, 0.1161, 0.1438, 0.1181, 0.1244, 0.1234, 0.1351, 0.1261],
       grad_fn=<SelectBackward>)
sum(attn_vals[0]):  tensor(1.0000, grad_fn=<AddBackward0>)
c:  torch.Size([64, 128])
</pre>

```python linenums="1"
# Predict
fc1 = nn.Linear(RNN_HIDDEN_DIM, NUM_CLASSES)
output = F.softmax(fc1(c), dim=1)
print ("output: ", output.shape)
```

<pre class="output">
output:  torch.Size([64, 4])
</pre>

> In a many-to-many task such as machine translation, our attentional interface will also account for the encoded representation of token in the output as well (via concatenation) so we can know which encoded inputs to attend to based on the encoded output we're focusing on. For more on this, be sure to explore [Bahdanau's attention paper](https://arxiv.org/abs/1409.0473){:target="_blank"}.

### Model

Now let's create our RNN based model but with the addition of the attention layer on top of the RNN's outputs.

```python linenums="1"
RNN_HIDDEN_DIM = 128
DROPOUT_P = 0.1
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

        # Attention
        self.attn = nn.Linear(rnn_hidden_dim, 1)

        # FC weights
        self.dropout = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(rnn_hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, inputs):
        # Embed
        x_in, seq_lens = inputs
        x_in = self.embeddings(x_in)

        # Encode
        out, h_n = self.rnn(x_in)

        # Attend
        e = self.attn(out)
        attn_vals = F.softmax(e.squeeze(2), dim=1)
        c = torch.bmm(attn_vals.unsqueeze(1), out).squeeze(1)

        # Predict
        z = self.fc1(c)
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
  (attn): Linear(in_features=128, out_features=1, bias=True)
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
Epoch: 1 | train_loss: 1.21680, val_loss: 1.08622, lr: 1.00E-04, _patience: 10
Epoch: 2 | train_loss: 1.00379, val_loss: 0.93546, lr: 1.00E-04, _patience: 10
Epoch: 3 | train_loss: 0.87091, val_loss: 0.83399, lr: 1.00E-04, _patience: 10
...
Epoch: 48 | train_loss: 0.35045, val_loss: 0.54718, lr: 1.00E-08, _patience: 10
Epoch: 49 | train_loss: 0.35055, val_loss: 0.54718, lr: 1.00E-08, _patience: 10
Epoch: 50 | train_loss: 0.35086, val_loss: 0.54717, lr: 1.00E-08, _patience: 10
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
  "precision": 0.8133385428975775,
  "recall": 0.8137222222222222,
  "f1": 0.8133454847232977,
  "num_samples": 18000.0
}
</pre>

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
RNN(
  (embeddings): Embedding(5000, 100, padding_idx=0)
  (rnn): RNN(100, 128, batch_first=True)
  (attn): Linear(in_features=128, out_features=1, bias=True)
  (dropout): Dropout(p=0.1, inplace=False)
  (fc1): Linear(in_features=128, out_features=100, bias=True)
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
  "Sports": 0.9651875495910645,
  "World": 0.03468644618988037,
  "Sci/Tech": 8.490968320984393e-05,
  "Business": 4.112234091735445e-05
}
</pre>

## Interpretability

Let's use the attention values to see which encoded tokens were most useful in predicting the appropriate label.

```python linenums="1"
import collections
import seaborn as sns
```

```python linenums="1"
class InterpretAttn(nn.Module):
    def __init__(self, embedding_dim, vocab_size, rnn_hidden_dim,
                 hidden_dim, dropout_p, num_classes, padding_idx=0):
        super(InterpretAttn, self).__init__()

        # Initialize embeddings
        self.embeddings = nn.Embedding(
            embedding_dim=embedding_dim, num_embeddings=vocab_size,
            padding_idx=padding_idx)

        # RNN
        self.rnn = nn.RNN(embedding_dim, rnn_hidden_dim, batch_first=True)

        # Attention
        self.attn = nn.Linear(rnn_hidden_dim, 1)

        # FC weights
        self.dropout = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(rnn_hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, inputs):
        # Embed
        x_in, seq_lens = inputs
        x_in = self.embeddings(x_in)

        # Encode
        out, h_n = self.rnn(x_in)

        # Attend
        e = self.attn(out)  # could add optional activation function (ex. tanh)
        attn_vals = F.softmax(e.squeeze(2), dim=1)

        return attn_vals
```

```python linenums="1"
# Initialize model
interpretable_model = InterpretAttn(
    embedding_dim=EMBEDDING_DIM, vocab_size=VOCAB_SIZE,
    rnn_hidden_dim=RNN_HIDDEN_DIM, hidden_dim=HIDDEN_DIM,
    dropout_p=DROPOUT_P, num_classes=NUM_CLASSES)
interpretable_model.load_state_dict(torch.load(Path(dir, "model.pt"), map_location=device))
interpretable_model.to(device)
```

<pre class="output">
InterpretAttn(
  (embeddings): Embedding(5000, 100, padding_idx=0)
  (rnn): RNN(100, 128, batch_first=True)
  (attn): Linear(in_features=128, out_features=1, bias=True)
  (dropout): Dropout(p=0.1, inplace=False)
  (fc1): Linear(in_features=128, out_features=100, bias=True)
  (fc2): Linear(in_features=100, out_features=4, bias=True)
)
</pre>

```python linenums="1"
# Initialize trainer
interpretable_trainer = Trainer(model=interpretable_model, device=device)
```

```python linenums="1"
# Get attention values
attn_vals  = interpretable_trainer.predict_step(dataloader)
print (attn_vals.shape) # (N, max_seq_len)
```

```python linenums="1"
# Visualize a bi-gram filter's outputs
sns.set(rc={"figure.figsize":(10, 1)})
tokens = tokenizer.sequences_to_texts(X)[0].split(" ")
sns.heatmap(attn_vals, xticklabels=tokens)
```

<div class="ai-center-all">
    <img width="700" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAiAAAABYCAYAAADMZ0yZAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAASsUlEQVR4nO3de1BU5f8H8PfuCmoihn69oJKABaKA4BiXQcbLUCJCKziJqVnR2DSO4l0upoglX9HMMkTSMZVKxWrUxGy0UVMydEwTRsyUH8glWBSS9cIusPv8/vDrSURgNTgL2/s1szN7OM85z+c57Bne+5xdjkIIIUBEREQkI6W5CyAiIqJ/HwYQIiIikh0DCBEREcmOAYSIiIhkxwBCREREsmMAISIiItl1krUz6wFydtdh1fx5ytwldAjdB44xdwkdgrvdIHOX0GE4WtuZu4QO4cLdEnOX0GHk3zwva391N/9Pem71H2dZ+35SsgYQIiIiajtCd9fcJZiMAYSIiMhS1NaYuwKTMYAQERFZCKHnDAgRERHJjAGEiIiI5MdLMERERCQ3fgiViIiI5Ke7Z+4KTMYAQkREZClqdeauwGQMIERERJaCMyBEREQkOx0/hEpERERyYwAhIiIiuQl9rblLMBkDCBERkaWo4YdQiYiISGZCxwBCREREcuMlGCIiIpKbqNGbuwSTMYAQERFZCHGv48yAKM1dABEREbUOY02d9HgSBQUFiIyMxPjx4xEZGYnCwsJGbTZt2oSJEyciLCwMEREROHXqlLSupqYG8+fPx0svvYTg4GAcP368xT45A0JERGQhhM7wVNslJCRg2rRpUKvVOHDgAFasWIH09PQGbTw9PREVFYWuXbvi999/x4wZM5CVlYUuXbpg27ZtsLGxwdGjR1FYWIjp06fjyJEj6NatW5N9cgaEiIjIQhjvGaWHqSorK5GXl4fQ0FAAQGhoKPLy8lBVVdWgXWBgILp27QoAcHV1hRACt27dAgAcPnwYkZGRAABHR0e4u7vj5MmTzfbLGRAiIiILYXjoVjBarRZarbZRG1tbW9ja2krLZWVl6Nu3L1QqFQBApVKhT58+KCsrQ8+ePR/bz/79+/Hcc8+hX79+AIA///wTAwYMkNbb29ujvLy82VoZQIiIiCyEQa+Qnu/cuRMpKSmN2syZMwdz58596j7Onj2LTz75BJ9//vlT7wNgACEiIrIYdTUq6fkbb7yB8PDwRm0env0A7s9WaDQaGAwGqFQqGAwGVFRUwN7evtG2Fy5cwJIlS5CamgpnZ2fp5/3790dpaak0Y1JWVgZfX99mazUpgPz111/SVEq/fv1gZ2dnymZEREQkozrd3wHk0UstTenVqxfc3NyQmZkJtVqNzMxMuLm5Nbr8kpOTgwULFmDjxo0YNmxYg3XBwcHIyMiAh4cHCgsLkZubi/Xr1zfbr0IIIZpaWVRUhOXLlyMvLw99+vQBAFRUVGDo0KFITEyEo6NjiwN7WCfrAS03ItT8earlRoTuA8eYu4QOwd1ukLlL6DAcrfnmyhQX7paYu4QOI//meVn7+90lRHo+5I/vTd4uPz8fsbGx0Gq1sLW1RXJyMpydnTFr1ixER0fDw8MDkydPRmlpKfr27Sttt3btWri6uuLevXuIjY3F5cuXoVQqsWTJEgQFBTXbZ7MBZOrUqZg2bRpCQ0OhVN7/wozRaMTBgwexa9cuZGRkmDw4gAHEVAwgpmEAMQ0DiOkYQEzDAGI6uQNIrlOY9Nyj4KCsfT+pZr+Ge+vWLbzyyitS+AAApVIJtVqN6urqNi+OiIiITKevU0mP9q7ZAPLss88iMzMTD0+SCCHw3XffmXRdiYiIiOSjr+8kPdq7Zitcs2YNEhISsGrVKumaj0ajwZAhQ7BmzRpZCiQiIiLT6EXH+f+izQYQR0dH7Ny5E1VVVSgrKwNw/+s6Tf1jEiIiIjIfXQf6B+cmzdH07NmToYOIiKid0yksLIAQERFR+6dXKlpu1E4wgBAREVkInYIBhIiIiGRW03GuwDCAEBERWYqajjMBwgBCRERkKWoZQIiIiEhuNYom767S7jCAEBERWQgdjOYuwWQMIERERBZCD86AEBERkcx0MJi7BJMxgBAREVkIXoIhIiIi2ekEZ0CIiIhIZrUMIERERCQ3nag3dwkmYwAhIiKyEAwgREREJLuOdAmmA922hoiIiJqjN9ZJjydRUFCAyMhIjB8/HpGRkSgsLGzUJisrCxEREXB3d0dycnKDdZ9++in8/f2hVquhVquRmJjYYp+cASEiIrIQuicMHg8kJCRg2rRpUKvVOHDgAFasWIH09PQGbRwcHLB69Wr88MMPqK2tbbSPSZMmISYmxuQ+OQNCRERkIZ5mBqSyshJ5eXkIDQ0FAISGhiIvLw9VVVUN2g0aNAhubm7o1Kl15i44A0JERGQhah8KHlqtFlqttlEbW1tb2NraSstlZWXo27cvVCoVAEClUqFPnz4oKytDz549Te770KFDyMrKQu/evTF37lx4e3s3254BhIiIyELoDX8HkJ07dyIlJaVRmzlz5mDu3Lmt2u/UqVPx7rvvwsrKCj///DNmz56N77//HnZ2dk1uwwBCRERkIfT1fweQN954A+Hh4Y3aPDz7AQD29vbQaDQwGAxQqVQwGAyoqKiAvb29yf327t1beh4QEAB7e3tcvXoVPj4+TW7DAEJERGQhao1//x+QRy+1NKVXr15wc3NDZmYm1Go1MjMz4ebm9kSXXzQaDfr27QsAuHz5MkpLS+Hk5NTsNgohRMe5dy8RERG1uvz8fMTGxkKr1cLW1hbJyclwdnbGrFmzEB0dDQ8PD5w7dw4LFy7EnTt3IIRA9+7dsXr1agQGBiImJgaXLl2CUqmElZUVoqOjMXr06Gb7ZAAhIiIi2fFruERERCQ7BhAiIiKSHQMIERERyY4BhIiIiGTHAEJERESyYwAhIiIi2TGAEBERkewYQIiIiEh2FhlAfvzxR0yYMAGTJk2Cp6cndDrdU++rpKQEvr6+rVid+Xz66aeora1ts/0vW7YM586da7P9m6qtx9melJSUICMjw9xlNPBPazpz5gyysrJasSLLt2PHDlRWVpq7jHbvzJkziIiIMHcZ9D8WGUD27NmD6Oho7N+/Hzk5OejSpYu5S2oXUlJSUFdX13LDp7R69WqMHDmyzfZvqtYeZ319fcuNzKS0tLTdBZB/UlN9fT3Onj2Ln3/+uZWrsmzp6ekMINThWNzN6JKSkvDrr7+ioKAAu3btwtmzZ3H+/Hl069YN48aNg1qtxunTp3Hjxg1ERUVhxowZAIDk5GScPXsWdXV1sLOzQ1JSEgYMGGDm0bSexMREAPdvmaxUKrF582Zs2rQJV65cgV6vh6+vL+Li4qBSqfD666/D3d0dv/32GyoqKjBhwgQsXrwYAFpcFxUVhbFjxyIjIwM7duyAtbU1jEYjPv74YwwePFj2cW7btg0JCQkoKioCALz99tuYNGkSAMDV1VV6bTy67Orqijlz5uDEiRMIDAxEeXk5rK2tUVhYiPLycnh5eSE5ORkKhQIHDx5Eenq6FHpiYmLg7+8PABg3bhzCwsKQnZ0NjUaDRYsWobKyEpmZmaiurkZSUhJefPFFAMBPP/2EzZs3o7a2FlZWVoiLi4OXlxfOnDmDpKQkDB8+HBcuXIBCocCGDRswePBgrFq1CiUlJVCr1Rg0aBA2btzY5sf4YTU1NYiJicG1a9fQqVMnODk54dq1a41qaur8KikpweTJkxEREYHs7GxERERgz549MBqNOH36NCZOnIjJkydLxw0A/P39ER8fL+s4W5urqysWLFiAo0eP4tatW1i6dCnGjx8PALh48SI+/PBD3L17FwAQHR2NMWPGIDU1FXl5eUhJSUFNTQ2mTJmCxYsXIy8vDxUVFYiOjkbnzp2xfv16PP/88+Yc3j+2Z88eXLlyBQkJCcjJycGrr76Kr7/+Gp6enli5ciXc3NwwZMiQxx4noOlz6WFarRZz5szBuHHj8Oabb8o8QgIACAs0Y8YMcezYMSGEEC4uLuLOnTtCCCHGjh0r1qxZI4QQori4WHh5eUnrKisrpe337t0r5s+fL7Xz8fGRs/w28/CxiI+PF/v27RNCCGEwGMSCBQtERkaGEOL+8Zs3b54wGAxCq9UKHx8fUVBQYNK6B8d9xIgRQqPRCCGE0Ov14t69e2YZ57x588SGDRuEEEJoNBoREBAgrly50qjdo8suLi7is88+k9bFxMSIqVOnCp1OJ/R6vQgJCRFZWVlCCCGqqqqE0WgUQgiRn58vAgMDpe0efs1dvHhRDB8+XHz55ZdCCCEOHTokpk6dKoQQ4vr162LKlCni9u3bQggh/vjjDzF69GghhBDZ2dli6NCh4tKlS0IIIVJTU8XChQuldeHh4a1y3J7GkSNHRFRUlLR869atx9bU3Pnl4uIiDh06JK3fuHGjdMyEEGL79u1i+fLlDfro6FxcXMQXX3whhBDi3LlzYtSoUUIIIaqrq4VarZbOHY1GIwIDA0V1dbUwGAzirbfeEunp6SI2NlYkJydL+xs7dqz0urYEhYWFYvz48UIIIdLS0kRkZKR0Pr788ssiNze3yePU0rkUHh4uSkpKRHh4uDh8+LD8gyOJxc2AtCQkJAQAMHDgQNja2qK8vByDBw/GyZMnsWvXLty7d69dT7m3lmPHjiEnJwfbt28HAOh0OulWygAQHBwMpVKJ7t27Y/DgwSgqKoKjo2OL6x7w8/NDbGwsxo4dizFjxsDBwUGuoTXwyy+/IDY2FgDQp08fjB49GmfOnIGLi0uL24aHhzdYDgoKQufOnQEAQ4cORVFREQICAlBcXIxFixZBo9GgU6dOuHnzJm7cuIHevXsD+Ps1N2zYMNTU1GDChAkAAHd3d2lm5tSpUygqKsL06dOl/urr63Hz5k0AgJOTE4YOHQoA8PLywvHjx5/6mLSmIUOGID8/H4mJifDx8ZHegT6qufOrc+fO0jF5nOHDh2PHjh1ITk6Gj48PRo0a1ZpDMJsHrwsvLy9UVFRAr9fjwoULKCkpwaxZs6R2CoUC169fh4eHB9atWwe1Wo3+/ftj165d5iq9zQ0aNAh6vR7l5eX45ZdfsGDBAqSlpSEsLAx1dXWorKxs8jjl5OQ0ey7duHEDM2fORHJycru4ZPxv9q8LIA/+gACASqWCwWBAaWkp/vvf/+Kbb76Bg4MDzp8/L11WsFRCCKSmpjYZDB53nExZ90BKSgpyc3ORnZ2NmTNnYuXKlS3emlluKpUK4n83g9br9Y3WP/PMMw2Wmxr3woULERsbi6CgIBiNRgwfPrzB/h5sp1KpGiwrlcoGf4wDAwOxdu3aRnXk5+fD2tpaWn50O3NycHBAZmYmsrOzcfLkSWzYsAHvvfdegzYtnV9du3aFQqFosg9vb2/s27cPp0+fxoEDB7Blyxbs3r27zcYkl0dfF/X19RBCwNXVFV999dVjtykpKYFSqYRWq4VOp4ONjY1s9crNz88Px48fR2VlJXx9ffH+++/jxIkT8PX1bfY45eTkNHsu9ejRA/369cPJkycZQMzMIj+E+qTu3LkDKysr9O7dG0ajEXv27DF3SW2iW7duuHPnDoD7n03YsmWL9Ee0qqoKxcXFrdJPfX09iouL4enpiXfeeQcBAQG4fPlyq+zbFA+P09/fH3v37gVw/53PTz/9BD8/PwDAc889h9zcXADAwYMHn7q/27dvY+DAgQCAb7/99qm+gRMQEIBTp07h6tWr0s9ycnJa3M7GxkYaqzmUl5dDpVIhKCgIcXFxqKqqalTTk55fNjY2uH37trRcXFwMGxsbTJw4EXFxcbh06RKMRmObjcmcvL29cf36dWRnZ0s/y8nJgRAC1dXVWLx4MT766COEhIRg+fLlUptu3bo1OGaWwM/PD1u3boW3tzcAYMSIEdi6dSv8/f2bPU4tnUvW1tZITU3FtWvX8MEHH0hvQkh+/7oZkMdxdXVFcHAwQkJCYGdnh9GjR7eLr5O2tqioKMycORNdunRBWloa0tLSoFaroVAoYGVlhfj4+Fa5VGI0GhEbG4vbt29DoVDA3t4eixYtaoURmObhcW7btg0rVqxAWFgYAGDx4sV44YUXAABxcXFYsWIFunfvjuDg4KfuLy4uDrNnz0aPHj0QGBiIZ5999on34ejoiHXr1mHZsmXQ6XSoq6vDiBEj4Onp2ex2rq6ucHJyQmhoKJydnWX/EOqVK1ewfv16APd/7++88w48PT0b1fQk51dQUBD2798PtVqNiRMnolevXtixYweUSiWMRiMSExOhVFrme6cePXogNTUV69atQ1JSEurq6uDg4IC0tDTEx8dj8uTJGDlyJLy9vfHmm29i9+7deO211zBz5kzEx8ejS5cuFvEhVOB+AFm6dKn0gW4/Pz9kZGTAz8+v2eNkyrlkbW2NjRs3YsmSJVi+fDlWrVplsa+p9kwhGP+IiIhIZox8REREJDsGECIiIpIdAwgRERHJjgGEiIiIZMcAQkRERLJjACEiIiLZMYAQERGR7BhAiIiISHb/D1FkCgksR3kAAAAAAElFTkSuQmCC">
</div>

The word `tennis` was attended to the most to result in the `Sports` label.

## Types of attention

We'll briefly look at the different types of attention and when to use each them.

### Soft (global) attention

Soft attention the type of attention we've implemented so far, where we attend to all encoded inputs when creating our context vector.

- **advantages**: we always have the ability to attend to all inputs in case something we saw much earlier/ see later are crucial for determing the output.
- **disadvantages**: if our input sequence is very long, this can lead to expensive compute.

### Hard attention

Hard attention is focusing on a specific set of the encoded inputs at each time step.

- **advantages**: we can save a lot of compute on long sequences by only focusing on a local patch each time.
- **disadvantages**: non-differentiable and so we need to use more complex techniques (variance reduction, reinforcement learning, etc.) to train.

<div class="ai-center-all">
<img src="/static/images/foundations/attention/soft_attention.png" width="700">
</div>
<div class="ai-center-all">
<small><a href="https://arxiv.org/abs/1502.03044" target="_blank">Show, Attend and Tell: Neural Image Caption Generation with Visual Attention</a></small>
</div>

### Local attention

[Local attention](https://arxiv.org/abs/1508.04025){:target="_blank"} blends the advantages of soft and hard attention. It involves learning an aligned position vector and empirically determining a local window of encoded inputs to attend to.

- **advantages**: apply attention to a local patch of inputs yet remain differentiable.
- **disadvantages**: need to determine the alignment vector for each output but it's a worthwhile trade off to determine the right window of inputs to attend to in order to avoid attending to all of them.

<div class="ai-center-all">
<img src="/static/images/foundations/attention/local_attention.png" width="700">
</div>
<div class="ai-center-all">
<small><a href="https://arxiv.org/abs/1508.04025" target="_blank">Effective Approaches to Attention-based Neural Machine Translation
</a></small>
</div>

### Self-attention

We can also use attention within the encoded input sequence to create a weighted representation that based on the similarity between input pairs. This will allow us to create rich representations of the input sequence that are aware of the relationships between each other. For example, in the image below you can see that when composing the representation of the token "its", this specific attention head will be incorporating signal from the token "Law" (it's learned that "its" is referring to the "Law").

<div class="ai-center-all">
<img src="/static/images/foundations/attention/self_attention.png" width="300">
</div>
<div class="ai-center-all">
<small><a href="https://arxiv.org/abs/1706.03762" target="_blank">Attention Is All You Need</a></small>
</div>

In the [next lesson](transformers.md), we'll implement Transformers that leverage self-attention to create contextual representations of our inputs for downstream applications.