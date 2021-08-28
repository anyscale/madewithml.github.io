---
template: lesson.html
title: Attention
description: Incorporating attention mechanisms to create context-aware representations.
keywords: attention, transformers, self-attention, positional encoding, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/foundations.png
repository: https://github.com/GokuMohandas/MadeWithML
notebook: https://colab.research.google.com/github/GokuMohandas/MadeWithML/blob/main/notebooks/14_Attention.ipynb
---

{% include "styles/lesson.md" %}

## Overview

In the <a target="_blank" href="https://madewithml.com/courses/foundations/recurrent-neural-networks/">RNN lesson</a>, we were constrained to using the representation at the very end but what if we could give contextual weight to each encoded input ($h_i$) when making our prediction? This is also preferred because it can help mitigate the vanishing gradient issue which stems from processing very long sequences. Below is attention applied to the outputs from an RNN. In theory, the outputs can come from anywhere where we want to learn how to weight amongst them but since we're working with the context of an RNN from the previous lesson , we'll continue with that.

<div class="ai-center-all">
    <img src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/foundations/attention/attention.png" width="500">
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

- `Objective`:
    - At it's core, attention is about learning how to weigh a group of encoded representations to produce a context-aware representation to use for downstream tasks. This is done by learning a set of attention weights and then using softmax to create attention values that sum to 1.
- `Advantages`:
    - Learn how to account for the appropriate encoded representations regardless of position.
- `Disadvantages`:
    - Another compute step that involves learning weights.
- `Miscellaneous`:
    - Several state-of-the-art approaches extend on basic attention to deliver highly context-aware representations (ex. self-attention).

## Set up

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
nltk.download('stopwords')
STOPWORDS = stopwords.words('english')
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
    pattern = re.compile(r'\b(' + r'|'.join(stopwords) + r')\b\s*')
    text = pattern.sub('', text)

    # Remove words in paranthesis
    text = re.sub(r'\([^)]*\)', '', text)

    # Spacing and filters
    text = re.sub(r"([-;;.,!?<=>])", r" \1 ", text)
    text = re.sub('[^A-Za-z0-9]+', ' ', text) # remove non alphanumeric chars
    text = re.sub(' +', ' ', text)  # remove multiple spaces
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
        with open(fp, 'w') as fp:
            contents = {'class_to_index': self.class_to_index}
            json.dump(contents, fp, indent=4, sort_keys=False)

    @classmethod
    def load(cls, fp):
        with open(fp, 'r') as fp:
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
                 pad_token='<PAD>', oov_token='<UNK>',
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
<Tokenizer(num_tokens=5000)>

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
        batch = np.array(batch, dtype=object)
        X = batch[:, 0]
        seq_lens = batch[:, 1]
        y = np.stack(batch[:, 2], axis=0)

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
  Train dataset:<Dataset(N=84000)>
  Val dataset: <Dataset(N=18000)>
  Test dataset: <Dataset(N=18000)>
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
                y_prob = self.model(inputs, apply_softmax=True)

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
    <img src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/foundations/attention/attention.png" width="500">
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

!!! note
    In a many-to-many task such as machine translation, our attentional interface will also account for the encoded representation of token in the output as well (via concatenation) so we can know which encoded inputs to attend to based on the encoded output we're focusing on. For more on this, be sure to explore [Bahdanau's attention paper](https://arxiv.org/abs/1409.0473){:target="_blank"}.

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

    def forward(self, inputs, apply_softmax=False):
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
        y_pred = self.fc2(z)

        if apply_softmax:
            y_pred = F.softmax(y_pred, dim=1)
        return y_pred
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
bound method Module.named_parameters of RNN(
  (embeddings): Embedding(5000, 100, padding_idx=0)
  (rnn): RNN(100, 128, batch_first=True)
  (attn): Linear(in_features=128, out_features=1, bias=True)
  (dropout): Dropout(p=0.1, inplace=False)
  (fc1): Linear(in_features=128, out_features=100, bias=True)
  (fc2): Linear(in_features=100, out_features=4, bias=True)
)
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
    optimizer, mode='min', factor=0.1, patience=3)
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
Epoch: 1 | train_loss: 1.22600, val_loss: 1.08924, lr: 1.00E-04, _patience: 10
Epoch: 2 | train_loss: 1.01283, val_loss: 0.93986, lr: 1.00E-04, _patience: 10
Epoch: 3 | train_loss: 0.88110, val_loss: 0.83812, lr: 1.00E-04, _patience: 10
...
Epoch: 30 | train_loss: 0.34505, val_loss: 0.54407, lr: 1.00E-06, _patience: 3
Epoch: 31 | train_loss: 0.34477, val_loss: 0.54422, lr: 1.00E-07, _patience: 2
Epoch: 32 | train_loss: 0.34430, val_loss: 0.54394, lr: 1.00E-07, _patience: 1
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
print (json.dumps(performance['overall'], indent=2))
```
<pre class="output">
{
  "precision": 0.8187010197413059,
  "recall": 0.8189444444444445,
  "f1": 0.8187055904309233,
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
label_encoder = LabelEncoder.load(fp=Path(dir, 'label_encoder.json'))
tokenizer = Tokenizer.load(fp=Path(dir, 'tokenizer.json'))
model = GRU(
    embedding_dim=EMBEDDING_DIM, vocab_size=VOCAB_SIZE,
    rnn_hidden_dim=RNN_HIDDEN_DIM, hidden_dim=HIDDEN_DIM,
    dropout_p=DROPOUT_P, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(Path(dir, 'model.pt'), map_location=device))
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
  "Sports": 0.9116348028182983,
  "World": 0.08557619899511337,
  "Sci/Tech": 0.0019216578220948577,
  "Business": 0.0008673836709931493
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

    def forward(self, inputs, apply_softmax=False):
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
tokens = tokenizer.sequences_to_texts(X)[0].split(' ')
sns.heatmap(attn_vals, xticklabels=tokens)
```

<div class="ai-center-all">
    <img width="700" src="data:image/png;base64,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
">
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
<img src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/foundations/attention/soft_attention.png" width="700">
</div>
<div class="ai-center-all">
<small><a href="https://arxiv.org/abs/1502.03044" target="_blank">Show, Attend and Tell: Neural Image Caption Generation with Visual Attention</a></small>
</div>

### Local attention

[Local attention](https://arxiv.org/abs/1508.04025){:target="_blank"} blends the advantages of soft and hard attention. It involves learning an aligned position vector and empirically determining a local window of encoded inputs to attend to.

- **advantages**: apply attention to a local patch of inputs yet remain differentiable.
- **disadvantages**: need to determine the alignment vector for each output but it's a worthwhile trade off to determine the right window of inputs to attend to in order to avoid attending to all of them.

<div class="ai-center-all">
<img src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/foundations/attention/local_attention.png" width="700">
</div>
<div class="ai-center-all">
<small><a href="https://arxiv.org/abs/1508.04025" target="_blank">Effective Approaches to Attention-based Neural Machine Translation
</a></small>
</div>

### Self-attention

We can also use attention within the encoded input sequence to create a weighted representation that based on the similarity between input pairs. This will allow us to create rich representations of the input sequence that are aware of the relationships between each other. For example, in the image below you can see that when composing the representation of the token "its", this specific attention head will be incorporating signal from the token "Law" (it's learned that "its" is referring to the "Law").

<div class="ai-center-all">
<img src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/foundations/attention/self_attention.png" width="300">
</div>
<div class="ai-center-all">
<small><a href="https://arxiv.org/abs/1706.03762" target="_blank">Attention Is All You Need</a></small>
</div>

## Transformers

Transformers are a very popular architecture that leverage and extend the concept of self-attention to create very useful representations of our input data for a downstream task.

### Scaled dot-product attention

The most popular type of self-attention is scaled dot-product attention from the widely-cited [Attention is all you need](https://arxiv.org/abs/1706.03762){:target="_blank"} paper. This type of attention involves projecting our encoded input sequences onto three matrices, queries (Q), keys (K) and values (V), whose weights we learn.


$$ Q = XW_q \text{where} W_q \in \mathbb{R}^{HXd_q} $$

$$ K = XW_k \text{where} W_k \in \mathbb{R}^{HXd_k} $$

$$ V = XW_v \text{where} W_v \in \mathbb{R}^{HXd_v} $$

$$ attention (Q, K, V) = softmax( \frac{Q K^{T}}{\sqrt{d_k}} )V \in \mathbb{R}^{MXd_v} $$

<center>

| Variable  | Description                              |
| :-------- | :--------------------------------------- |
| $X$       | encoded inputs $\in \mathbb{R}^{NXMXH}$  |
| $N$       | batch size                               |
| $M$       | max sequence length in the batch         |
| $H$       | hidden dim, model dim, etc.              |
| $W_q$     | query weights $\in \mathbb{R}^{HXd_q}$   |
| $W_k$     | key weights $\in \mathbb{R}^{HXd_k}$     |
| $W_v$     | value weights $\in \mathbb{R}^{HXd_v}$   |

</center>

### Multi-head attention

Instead of applying self-attention only once across the entire encoded input, we can also separate the input and apply self-attention in parallel (heads) to each input section and concatenate them. This allows the different head to learn unique representations while maintaining the complexity since we split the input into smaller subspaces.

$$ MultiHead(Q, K, V) = concat({head}_1, ..., {head}_{h})W_O $$

$$ {head}_i = attention(Q_i, K_i, V_i) $$

<center>

| Variable  | Description                                             |
| :-------- | :------------------------------------------------------ |
| $h$       | number of attention heads                               |
| $W_O$     | multi-head attention weights $\in \mathbb{R}^{hd_vXH}$  |
| $H$       | hidden dim (or dimension of the model $d_{model}$)      |

</center>

### Positional encoding

With self-attention, we aren't able to account for the sequential position of our input tokens. To address this, we can use positional encoding to create a representation of the location of each token with respect to the entire sequence. This can either be learned (with weights) or we can use a fixed function that can better extend to create positional encoding for lengths during inference that were not observed during training.

$$ PE_{(pos,2i)} = sin({pos}/{10000^{2i/H}}) $$

$$ PE_{(pos,2i+1)} = cos({pos}/{10000^{2i/H}}) $$

<center>

| Variable  | Description                      |
| :-------- | :------------------------------- |
| $pos$     | position of the token $(1...M)$  |
| $i$       | hidden dim $(1..H)$              |

</center>

This effectively allows us to represent each token's relative position using a fixed function for very large sequences. And because we've constrained the positional encodings to have the same dimensions as our encoded inputs, we can simply concatenate them before feeding them into the multi-head attention heads.

### Architecture

And here's how it all fits together! It's an end-to-end architecture that creates these contextual representations and uses an encoder-decoder architecture to predict the outcomes (one-to-one, many-to-one, many-to-many, etc.) Due to the complexity of the architecture, they require massive amounts of data for training without overfitting, however, they can be leveraged as pretrained models to finetune with smaller datasets that are similar to the larger set it was initially trained on.

<div class="ai-center-all">
<img src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/foundations/attention/transformer.png" width="800">
</div>
<div class="ai-center-all">
<small><a href="https://arxiv.org/abs/1706.03762" target="_blank">Attention Is All You Need</a></small>
</div>

!!! note
    We're not going to the implement the Transformer [from scratch](https://nlp.seas.harvard.edu/2018/04/03/attention.html){:target="_blank"} but we will use the[ Hugging Face library](https://github.com/huggingface/transformers){:target="_blank"} to do so in the [baselines](https://madewithml.com/courses/mlops/baselines/#transformers-w-contextual-embeddings){:target="_blank"} lesson!