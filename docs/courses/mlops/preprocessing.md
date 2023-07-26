---
template: lesson.html
title: Data Preprocessing
description: Preprocessing our dataset, through preparations and transformations, before training our models.
keywords: preprocessing, preparation, cleaning, feature engineering, filtering, transformations, mlops, machine learning
image: https://madewithml.com/static/images/mlops.png
repository: https://github.com/GokuMohandas/Made-With-ML
notebook: https://github.com/GokuMohandas/Made-With-ML/blob/main/notebooks/madewithml.ipynb
---

{% include "styles/lesson.md" %}

## Intuition

Data preprocessing can be categorized into two types of processes: *preparation* and *transformation*. We'll explore common preprocessing techniques and then we'll preprocess our dataset.

!!! warning
    Certain preprocessing steps are `global` (don't depend on our dataset, ex. lower casing text, removing stop words, etc.) and others are `local` (constructs are learned only from the training split, ex. vocabulary, standardization, etc.). For the local, dataset-dependent preprocessing steps, we want to ensure that we [split](splitting.md){:target="_blank"} the data first before preprocessing to avoid data leaks.

## Preparing

Preparing the data involves organizing and cleaning the data.

### Joins
Performing SQL joins with existing data tables to organize all the relevant data you need into one view. This makes working with our dataset a whole lot easier.

```sql linenums="1"
SELECT * FROM A
INNER JOIN B on A.id == B.id
```

!!! warning
    We need to be careful to perform point-in-time valid joins to avoid data leaks. For example, if Table B may have features for objects in Table A that were not available at the time inference would have been needed.

### Missing values

First, we'll have to identify the rows with missing values and once we do, there are several approaches to dealing with them.

- omit samples with missing values (if only a small subset are missing it)
```python linenums="1"
# Drop a row (sample) by index
df.drop([4, 10, ...])
# Conditionally drop rows (samples)
df = df[df.value > 0]
# Drop samples with any missing feature
df = df[df.isnull().any(axis=1)]
```

- omit the entire feature (if too many samples are missing the value)
```python linenums="1"
# Drop a column (feature)
df.drop(["A"], axis=1)
```

- fill in missing values for features (using domain knowledge, heuristics, etc.)
```python linenums="1"
# Fill in missing values with mean
df.A = df.A.fillna(df.A.mean())
```

- may not always seem "missing" (ex. 0, null, NA, etc.)
```python linenums="1"
# Replace zeros to NaNs
import numpy as np
df.A = df.A.replace({"0": np.nan, 0: np.nan})
```

### Outliers (anomalies)

- craft assumptions about what is a "normal" expected value
```python linenums="1"
# Ex. Feature value must be within 2 standard deviations
df[np.abs(df.A - df.A.mean()) <= (2 * df.A.std())]
```
- be careful not to remove important outliers (ex. fraud)
- values may not be outliers when we apply a transformation (ex. power law)
- anomalies can be global (point), contextual (conditional) or collective (individual points are not anomalous and the collective group is an outlier)

### Feature engineering

Feature engineering involves combining features in unique ways to draw out signal.

```python linenums="1"
# Input
df.C = df.A + df.B
```

!!! tip
    Feature engineering can be done in collaboration with domain experts that can guide us on what features to engineer and use.

### Cleaning

Cleaning our data involves apply constraints to make it easier for our models to extract signal from the data.

- use domain expertise and EDA
- apply constraints via filters
- ensure data type consistency
- removing data points with certain or null column values
- images (crop, resize, clip, etc.)
```python linenums="1"
# Resize
import cv2
dims = (height, width)
resized_img = cv2.resize(src=img, dsize=dims, interpolation=cv2.INTER_LINEAR)
```
- text (lower, stem, lemmatize, regex, etc.)
```python linenums="1"
# Lower case the text
text = text.lower()
```

## Transformations
Transforming the data involves feature encoding and engineering.

### Scaling
- required for models where the scale of the input affects the processes
- learn constructs from train split and apply to other splits (local)
- don't blindly scale features (ex. categorical features)

- [standardization](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler){:target="_blank"}: rescale values to mean 0, std 1

    ```python linenums="1"
    # Standardization
    import numpy as np
    x = np.random.random(4) # values between 0 and 1
    print ("x:\n", x)
    print (f"mean: {np.mean(x):.2f}, std: {np.std(x):.2f}")
    x_standardized = (x - np.mean(x)) / np.std(x)
    print ("x_standardized:\n", x_standardized)
    print (f"mean: {np.mean(x_standardized):.2f}, std: {np.std(x_standardized):.2f}")
    ```
    <pre class="output">
    x: [0.36769939 0.82302265 0.9891467  0.56200803]
    mean: 0.69, std: 0.24
    x_standardized: [-1.33285946  0.57695671  1.27375049 -0.51784775]
    mean: 0.00, std: 1.00
    </pre>


- [min-max](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.minmax_scale.html#sklearn.preprocessing.minmax_scale){:target="_blank"}: rescale values between a min and max

    ```python linenums="1"
    # Min-max
    import numpy as np
    x = np.random.random(4) # values between 0 and 1
    print ("x:", x)
    print (f"min: {x.min():.2f}, max: {x.max():.2f}")
    x_scaled = (x - x.min()) / (x.max() - x.min())
    print ("x_scaled:", x_scaled)
    print (f"min: {x_scaled.min():.2f}, max: {x_scaled.max():.2f}")
    ```
    <pre class="output">
    x: [0.20195674 0.99108855 0.73005081 0.02540603]
    min: 0.03, max: 0.99
    x_scaled: [0.18282479 1.         0.72968575 0.        ]
    min: 0.00, max: 1.00
    </pre>

- [binning](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html){:target="_blank"}: convert a continuous feature into categorical using bins

    ```python linenums="1"
    # Binning
    import numpy as np
    x = np.random.random(4) # values between 0 and 1
    print ("x:", x)
    bins = np.linspace(0, 1, 5) # bins between 0 and 1
    print ("bins:", bins)
    binned = np.digitize(x, bins)
    print ("binned:", binned)
    ```
    <pre class="output">
    x: [0.54906364 0.1051404  0.2737904  0.2926313 ]
    bins: [0.   0.25 0.5  0.75 1.  ]
    binned: [3 1 2 2]
    </pre>

- and many [more](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing){:target="_blank"}!


### Encoding

- allows for representing data efficiently (maintains signal) and effectively (learns patterns, ex. one-hot vs embeddings)

- [label](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html#sklearn.preprocessing.LabelEncoder){:target="_blank"}: unique index for categorical value

    ```python linenums="1"
    # Label encoding
    label_encoder.class_to_index = {
    "attention": 0,
    "autoencoders": 1,
    "convolutional-neural-networks": 2,
    "data-augmentation": 3,
    ... }
    label_encoder.transform(["attention", "data-augmentation"])
    ```
    <pre class="output">
    array([2, 2, 1])
    </pre>

- [one-hot](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html#sklearn.preprocessing.OneHotEncoder){:target="_blank"}: representation as binary vector

    ```python linenums="1"
    # One-hot encoding
    one_hot_encoder.transform(["attention", "data-augmentation"])
    ```
    <pre class="output">
    array([1, 0, 0, 1, 0, ..., 0])
    </pre>


- [embeddings](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html){:target="_blank"}: dense representations capable of representing context

    ```python linenums="1"
    # Embeddings
    self.embeddings = nn.Embedding(
        embedding_dim=embedding_dim, num_embeddings=vocab_size)
    x_in = self.embeddings(x_in)
    print (x_in.shape)
    ```
    <pre class="output">
    (len(X), embedding_dim)
    </pre>

- and many [more](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing){:target="_blank"}!

### Extraction

- signal extraction from existing features
- combine existing features
- [transfer learning](https://en.wikipedia.org/wiki/Transfer_learning){:target="_blank"}: using a pretrained model as a feature extractor and finetuning on it's results
- [autoencoders](https://en.wikipedia.org/wiki/Autoencoder){:target="_blank"}: learn to encode inputs for compressed knowledge representation

- [principle component analysis (PCA)](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA){:target="_blank"}: linear dimensionality reduction to project data in a lower dimensional space.

    ```python linenums="1"
    # PCA
    import numpy as np
    from sklearn.decomposition import PCA
    X = np.array([[-1, -1, 3], [-2, -1, 2], [-3, -2, 1]])
    pca = PCA(n_components=2)
    pca.fit(X)
    print (pca.transform(X))
    print (pca.explained_variance_ratio_)
    print (pca.singular_values_)
    ```
    <pre class="output">
    [[-1.44245791 -0.1744313 ]
     [-0.1148688   0.31291575]
     [ 1.55732672 -0.13848446]]
    [0.96838847 0.03161153]
    [2.12582835 0.38408396]
    </pre>

- [counts (ngram)](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html){:target="_blank"}: sparse representation of text as matrix of token counts — useful if feature values have lot's of meaningful, separable signal.

    ```python linenums="1"
    # Counts (ngram)
    from sklearn.feature_extraction.text import CountVectorizer
    y = [
        "acetyl acetone",
        "acetyl chloride",
        "chloride hydroxide",
    ]
    vectorizer = CountVectorizer()
    y = vectorizer.fit_transform(y)
    print (vectorizer.get_feature_names())
    print (y.toarray())
    # 💡 Repeat above with char-level ngram vectorizer
    # vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 3)) # uni, bi and trigrams
    ```
    <pre class="output">
    ['acetone', 'acetyl', 'chloride', 'hydroxide']
    [[1 1 0 0]
     [0 1 1 0]
     [0 0 1 1]]
    </pre>

- [similarity](https://github.com/dirty-cat/dirty_cat){:target="_blank"}: similar to count vectorization but based on similarities in tokens
- and many [more](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction){:target="_blank"}!

> We'll often was to retrieve feature values for an entity (user, item, etc.) over time and reuse the same features across different projects. To ensure that we're retrieving the proper feature values and to avoid duplication of efforts, we can use a [feature store](feature-store.md){:target="_blank"}.

!!! question "Curse of dimensionality"
    What can we do if a feature has lots of unique values but enough data points for each unique value (ex. URL as a feature)?

    ??? quote "Show answer"
        We can encode our data with [hashing](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.FeatureHasher.html){:target="_blank"} or using it's attributes instead of the exact entity itself. For example, representing a user by their location and favorites as opposed to using their user ID or representing a webpage with it's domain as opposed to the exact url. This methods effectively decrease the total number of unique feature values and increase the number of data points for each.

## Implementation

For our application, we'll be implementing a few of these preprocessing steps that are relevant for our dataset.

```python linenums="1"
import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
```

### Feature engineering
We can combine existing input features to create new meaningful signal for helping the model learn. However, there's usually no simple way to know if certain feature combinations will help or not without empirically experimenting with the different combinations. Here, we could use a project's title and description separately as features but we'll combine them to create one input feature.

```python linenums="1"
# Input
df["text"] = df.title + " " + df.description
```

### Cleaning

Since we're dealing with text data, we can apply some common text preprocessing operations. Here, we'll be using Python's built-in regular expressions library [`re`](https://developers.google.com/edu/python/regular-expressions){:target="_blank"} and the Natural Language Toolkit [`nltk`](https://www.nltk.org/){:target="_blank"}.

```python linenums="1"
nltk.download("stopwords")
STOPWORDS = stopwords.words("english")
```

```python linenums="1"
def clean_text(text, stopwords=STOPWORDS):
    """Clean raw text string."""
    # Lower
    text = text.lower()

    # Remove stopwords
    pattern = re.compile(r'\b(' + r"|".join(stopwords) + r")\b\s*")
    text = pattern.sub('', text)

    # Spacing and filters
    text = re.sub(r"([!\"'#$%&()*\+,-./:;<=>?@\\\[\]^_`{|}~])", r" \1 ", text)  # add spacing
    text = re.sub("[^A-Za-z0-9]+", " ", text)  # remove non alphanumeric chars
    text = re.sub(" +", " ", text)  # remove multiple spaces
    text = text.strip()  # strip white space at the ends
    text = re.sub(r"http\S+", "", text)  #  remove links

    return text
```

!!! note
    We could definitely try and include emojis, punctuations, etc. because they do have a lot of signal for the task but it's best to simplify the initial feature set we use to just what we think are the most influential and then we can slowly introduce other features and assess utility.

Once we're defined our function, we can apply it to each row in our dataframe via [`pandas.DataFrame.apply`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.apply.html){:target="_blank"}.

```python linenums="1"
# Apply to dataframe
original_df = df.copy()
df.text = df.text.apply(clean_text)
print (f"{original_df.text.values[0]}\n{df.text.values[0]}")
```
<pre class="output">
Comparison between YOLO and RCNN on real world videos Bringing theory to experiment is cool. We can easily train models in colab and find the results in minutes.
comparison yolo rcnn real world videos bringing theory experiment cool easily train models colab find results minutes
</pre>

!!! warning
    We'll want to introduce less frequent features as they become more frequent or encode them in a clever way (ex. binning, extract general attributes, common n-grams, mean encoding using other feature values, etc.) so that we can mitigate the feature value dimensionality issue until we're able to collect more data.

We'll wrap up our cleaning operation by removing columns ([`pandas.DataFrame.drop`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html){:target="_blank"}) and rows with null tag values ([`pandas.DataFrame.dropna`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dropna.html){:target="_blank"}).

```python linenums="1"
# DataFrame cleanup
df = df.drop(columns=["id", "created_on", "title", "description"], errors="ignore")  # drop cols
df = df.dropna(subset=["tag"])  # drop nulls
df = df[["text", "tag"]]  # rearrange cols
df.head()
```

<div class="output_subarea output_html rendered_html output_result" dir="auto"><div>
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
      <th>0</th>
      <td>comparison yolo rcnn real world videos bringin...</td>
      <td>computer-vision</td>
    </tr>
    <tr>
      <th>1</th>
      <td>show infer tell contextual inference creative ...</td>
      <td>computer-vision</td>
    </tr>
    <tr>
      <th>2</th>
      <td>awesome graph classification collection import...</td>
      <td>other</td>
    </tr>
    <tr>
      <th>3</th>
      <td>awesome monte carlo tree search curated list m...</td>
      <td>other</td>
    </tr>
    <tr>
      <th>4</th>
      <td>attentionwalk pytorch implementation watch ste...</td>
      <td>other</td>
    </tr>
  </tbody>
</table>
</div></div>

### Encoding

We need to encode our data into numerical values so that our models can process them. We'll start by encoding our text labels into unique indices.

```python linenums="1"
# Label to index
tags = train_df.tag.unique().tolist()
num_classes = len(tags)
class_to_index = {tag: i for i, tag in enumerate(tags)}
class_to_index
```

<pre class="output">
{'mlops': 0,
 'natural-language-processing': 1,
 'computer-vision': 2,
 'other': 3}
</pre>

Next, we can use the [`pandas.Series.map`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.map.html){:target="_blank"} function to map our `class_to_index` dictionary on our tag column to encode our labels.

```python linenums="1"
# Encode labels
df["tag"] = df["tag"].map(class_to_index)
df.head()
```

<div class="output_subarea output_html rendered_html output_result" dir="auto"><div>
<table class="dataframe" border="1">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>tag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>comparison yolo rcnn real world videos bringin...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>show infer tell contextual inference creative ...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>awesome graph classification collection import...</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>awesome monte carlo tree search curated list m...</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>attentionwalk pytorch implementation watch ste...</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div></div>

We'll also want to be able to decode our predictions back into text labels. We can do this by creating an `index_to_class` dictionary and using that to convert encoded labels back into text labels.

```python linenums="1"
def decode(indices, index_to_class):
    return [index_to_class[index] for index in indices]
```

```python linenums="1"
index_to_class = {v:k for k, v in class_to_index.items()}
decode(df.head()["tag"].values, index_to_class=index_to_class)
```

<pre class="output">
['computer-vision', 'computer-vision', 'other', 'other', 'other']
</pre>

### Tokenizer

Next we'll encode our text as well. Instead of using a random dictionary, we'll use a [tokenizer](https://huggingface.co/allenai/scibert_scivocab_uncased){:target="_blank"} that was used for a pretrained LLM ([scibert](https://huggingface.co/allenai/scibert_scivocab_uncased){:target="_blank"}) to tokenize our text. We'll be fine-tuning this exact model later when we train our model.

> Here is a quick refresher on [attention](../foundations/attention.md){:target="_blank"} and [Transformers](../foundations/pandas.md){:target="_blank"}.

```python linenums="1"
import numpy as np
from transformers import BertTokenizer
```

The tokenizer will convert our input text into a list of token ids and a list of attention masks. The token ids are the indices of the tokens in the [vocabulary](https://huggingface.co/allenai/scibert_scivocab_uncased/blob/main/vocab.txt){:target="_blank"}. The attention mask is a binary mask indicating the position of the token indices so that the model can attend to them (and ignore the pad tokens).

```python linenums="1"
# Bert tokenizer
tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased", return_dict=False)
text = "Transfer learning with transformers for text classification."
encoded_inputs = tokenizer([text], return_tensors="np", padding="longest")  # pad to longest item in batch
print ("input_ids:", encoded_inputs["input_ids"])
print ("attention_mask:", encoded_inputs["attention_mask"])
print (tokenizer.decode(encoded_inputs["input_ids"][0]))
```

<pre class="output">
input_ids: [[  102  2268  1904   190 29155   168  3267  2998   205   103]]
attention_mask: [[1 1 1 1 1 1 1 1 1 1]]
[CLS] transfer learning with transformers for text classification. [SEP]
</pre>

> Note that we use `padding="longest"` in our tokenizer function to pad our inputs to the longest item in the batch. This becomes important when we use batches of inputs later and want to create a uniform input size, where shorted text sequences will be padded with zeros to meet the length of the longest input in the batch.

We'll wrap our tokenization into a `tokenize` function that we can use to tokenize batches of our data.

```python linenums="1"
def tokenize(batch):
    tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased", return_dict=False)
    encoded_inputs = tokenizer(batch["text"].tolist(), return_tensors="np", padding="longest")
    return dict(ids=encoded_inputs["input_ids"], masks=encoded_inputs["attention_mask"], targets=np.array(batch["tag"]))
```

```python linenums="1"
# Tokenization
tokenize(df.head(1))
```

<pre class="output">
{'ids': array([[  102,  2029,  1778,   609,  6446,  4857,  1332,  2399, 13572,
         19125,  1983,  1954,  6240,  3717,  7434,  1262,   537,   201,
          1040,   545,  4714,   103]]),
 'masks': array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]),
 'targets': array([2])}
</pre>

### Best practices

We'll wrap up by combining all of our preprocessing operations into function. This way we can easily apply it to different datasets (training, inference, etc.)

```python linenums="1"
def preprocess(df, class_to_index):
    """Preprocess the data."""
    df["text"] = df.title + " " + df.description  # feature engineering
    df["text"] = df.text.apply(clean_text)  # clean text
    df = df.drop(columns=["id", "created_on", "title", "description"], errors="ignore")  # clean dataframe
    df = df[["text", "tag"]]  # rearrange columns
    df["tag"] = df["tag"].map(class_to_index)  # label encoding
    outputs = tokenize(df)
    return outputs
```

```python linenums="1"
# Apply
preprocess(df=train_df, class_to_index=class_to_index)
```

<pre class="output">
{'ids': array([[  102,   856,   532, ...,     0,     0,     0],
        [  102,  2177, 29155, ...,     0,     0,     0],
        [  102,  2180,  3241, ...,     0,     0,     0],
        ...,
        [  102,   453,  2068, ...,  5730,   432,   103],
        [  102, 11268,  1782, ...,     0,     0,     0],
        [  102,  1596,   122, ...,     0,     0,     0]]),
 'masks': array([[1, 1, 1, ..., 0, 0, 0],
        [1, 1, 1, ..., 0, 0, 0],
        [1, 1, 1, ..., 0, 0, 0],
        ...,
        [1, 1, 1, ..., 1, 1, 1],
        [1, 1, 1, ..., 0, 0, 0],
        [1, 1, 1, ..., 0, 0, 0]]),
 'targets': array([0, 1, 1, ... 0, 2, 3])}
</pre>

<!-- Course signup -->
{% include "templates/signup.md" %}

<!-- Citation -->
{% include "templates/cite.md" %}
