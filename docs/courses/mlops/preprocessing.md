---
template: lesson.html
title: Data Preprocessing
description: Preprocessing our dataset, via through preparations and transformations, to use for training.
keywords: preprocessing, preparation, cleaning, feature engineering, filtering, transformations, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/mlops.png
repository: https://github.com/GokuMohandas/mlops-course
notebook: https://github.com/GokuMohandas/mlops-course/blob/main/notebooks/tagifai.ipynb
---

{% include "styles/lesson.md" %}

## Intuition

Data preprocessing can be categorized into two types of processes: *preparation* and *transformation*. We'll explore common preprocessing techniques and then walk through the relevant processes for our specific application.

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

Cleaning our data involves apply constraints to make it easier for our models to draw our signal from the data.

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

## Application

For our application, we'll be implementing a few of these preprocessing steps that are relevant for our dataset.

### Feature engineering
We can combine existing input features to create new meaningful signal (helping the model learn). However, there's usually no simple way to know if certain feature combinations will help or not without empirically experimenting with the different combinations. Here, we could use a project's title and description separately as features but we'll combine them to create one input feature.

```python linenums="1"
# Input
df["text"] = df.title + " " + df.description
```


### Cleaning

Since we're dealing with text data, we can apply some of the common text preprocessing steps:

```bash
!pip install nltk==3.7 -q
```

```python linenums="1"
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
```

```python linenums="1"
nltk.download("stopwords")
STOPWORDS = stopwords.words("english")
stemmer = PorterStemmer()
```

```python linenums="1"
def clean_text(text, lower=True, stem=False, stopwords=STOPWORDS):
    """Clean raw text."""
    # Lower
    if lower:
        text = text.lower()

    # Remove stopwords
    if len(stopwords):
        pattern = re.compile(r'\b(' + r"|".join(stopwords) + r")\b\s*")
        text = pattern.sub('', text)

    # Spacing and filters
    text = re.sub(
        r"([!\"'#$%&()*\+,-./:;<=>?@\\\[\]^_`{|}~])", r" \1 ", text
    )  # add spacing between objects to be filtered
    text = re.sub("[^A-Za-z0-9]+", " ", text)  # remove non alphanumeric chars
    text = re.sub(" +", " ", text)  # remove multiple spaces
    text = text.strip()  # strip white space at the ends

    # Remove links
    text = re.sub(r"http\S+", "", text)

    # Stemming
    if stem:
        text = " ".join([stemmer.stem(word, to_lowercase=lower) for word in text.split(" ")])

    return text
```

    !!! note
        We could definitely try and include emojis, punctuations, etc. because they do have a lot of signal for the task but it's best to simplify the initial feature set we use to just what we think are the most influential and then we can slowly introduce other features and assess utility.

```python linenums="1"
# Apply to dataframe
original_df = df.copy()
df.text = df.text.apply(clean_text, lower=True, stem=False)
print (f"{original_df.text.values[0]}\n{df.text.values[0]}")
```
<pre class="output">
Comparison between YOLO and RCNN on real world videos Bringing theory to experiment is cool. We can easily train models in colab and find the results in minutes.
comparison yolo rcnn real world videos bringing theory experiment cool easily train models colab find results minutes
</pre>

!!! warning
    We'll want to introduce less frequent features as they become more frequent or encode them in a clever way (ex. binning, extract general attributes, common n-grams, mean encoding using other feature values, etc.) so that we can mitigate the feature value dimensionality issue until we're able to collect more data.


### Replace labels

Based on our findings from [EDA](exploratory-data-analysis.md){:target="_blank"}, we're going to apply several constraints for labeling our data:

- if a data point has a tag that we currently don't support, we'll replace it with `other`
- if a certain tag doesn't have *enough* samples, we'll replace it with `other`

```python linenums="1"
import json
```

```python linenums="1"
# Accepted tags (external constraint)
ACCEPTED_TAGS = ["natural-language-processing", "computer-vision", "mlops", "graph-learning"]
```

```python linenums="1"
# Out of scope (OOS) tags
oos_tags = [item for item in df.tag.unique() if item not in ACCEPTED_TAGS]
oos_tags
```
<pre class="output">
['reinforcement-learning', 'time-series']
</pre>

```python linenums="1"
# Samples with OOS tags
oos_indices = df[df.tag.isin(oos_tags)].index
df[df.tag.isin(oos_tags)].head()
```
<pre class="output">
<div class="output_subarea output_html rendered_html output_result" dir="auto"><div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>created_on</th>
      <th>title</th>
      <th>description</th>
      <th>tag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>15</td>
      <td>2020-02-28 23:55:26</td>
      <td>Awesome Monte Carlo Tree Search</td>
      <td>A curated list of Monte Carlo tree search papers...</td>
      <td>reinforcement-learning</td>
    </tr>
    <tr>
      <th>37</th>
      <td>121</td>
      <td>2020-03-24 04:56:38</td>
      <td>Deep Reinforcement Learning in TensorFlow2</td>
      <td>deep-rl-tf2 is a repository that implements a ...</td>
      <td>reinforcement-learning</td>
    </tr>
    <tr>
      <th>67</th>
      <td>218</td>
      <td>2020-04-06 11:29:57</td>
      <td>Distributional RL using TensorFlow2</td>
      <td>🐳 Implementation of various Distributional Rei...</td>
      <td>reinforcement-learning</td>
    </tr>
    <tr>
      <th>74</th>
      <td>239</td>
      <td>2020-04-06 18:39:48</td>
      <td>Prophet: Forecasting At Scale</td>
      <td>Tool for producing high quality forecasts for ...</td>
      <td>time-series</td>
    </tr>
    <tr>
      <th>95</th>
      <td>277</td>
      <td>2020-04-07 00:30:33</td>
      <td>Curriculum for Reinforcement Learning</td>
      <td>Curriculum learning applied to reinforcement l...</td>
      <td>reinforcement-learning</td>
    </tr>
  </tbody>
</table>
</div></div>
</pre>

```python linenums="1"
# Replace this tag with "other"
df.tag = df.tag.apply(lambda x: "other" if x in oos_tags else x)
df.iloc[oos_indices].head()
```
<pre class="output">
<div class="output_subarea output_html rendered_html output_result" dir="auto"><div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>created_on</th>
      <th>title</th>
      <th>description</th>
      <th>tag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>15</td>
      <td>2020-02-28 23:55:26</td>
      <td>Awesome Monte Carlo Tree Search</td>
      <td>A curated list of Monte Carlo tree search papers...</td>
      <td>other</td>
    </tr>
    <tr>
      <th>37</th>
      <td>121</td>
      <td>2020-03-24 04:56:38</td>
      <td>Deep Reinforcement Learning in TensorFlow2</td>
      <td>deep-rl-tf2 is a repository that implements a ...</td>
      <td>other</td>
    </tr>
    <tr>
      <th>67</th>
      <td>218</td>
      <td>2020-04-06 11:29:57</td>
      <td>Distributional RL using TensorFlow2</td>
      <td>🐳 Implementation of various Distributional Rei...</td>
      <td>other</td>
    </tr>
    <tr>
      <th>74</th>
      <td>239</td>
      <td>2020-04-06 18:39:48</td>
      <td>Prophet: Forecasting At Scale</td>
      <td>Tool for producing high quality forecasts for ...</td>
      <td>other</td>
    </tr>
    <tr>
      <th>95</th>
      <td>277</td>
      <td>2020-04-07 00:30:33</td>
      <td>Curriculum for Reinforcement Learning</td>
      <td>Curriculum learning applied to reinforcement l...</td>
      <td>other</td>
    </tr>
  </tbody>
</table>
</div></div>
</pre>

We're also going to restrict the mapping to only tags that are above a certain frequency threshold. The tags that don't have enough projects will not have enough samples to model their relationships.

```python linenums="1"
# Minimum frequency required for a tag
min_freq = 75
tags = Counter(df.tag.values)
```

```python linenums="1"
# Tags that just made / missed the cut
@widgets.interact(min_freq=(0, tags.most_common()[0][1]))
def separate_tags_by_freq(min_freq=min_freq):
    tags_above_freq = Counter(tag for tag in tags.elements()
                                    if tags[tag] >= min_freq)
    tags_below_freq = Counter(tag for tag in tags.elements()
                                    if tags[tag] < min_freq)
    print ("Most popular tags:\n", tags_above_freq.most_common(3))
    print ("\nTags that just made the cut:\n", tags_above_freq.most_common()[-3:])
    print ("\nTags that just missed the cut:\n", tags_below_freq.most_common(3))
```
<pre class="output">
Most popular tags:
 [('natural-language-processing', 388), ('computer-vision', 356), ('other', 87)]

Tags that just made the cut:
 [('computer-vision', 356), ('other', 87), ('mlops', 79)]

Tags that just missed the cut:
 [('graph-learning', 45)]
</pre>

```python linenums="1"
def filter(tag, include=[]):
    """Determine if a given tag is to be included."""
    if tag not in include:
        tag = None
    return tag
```

```python linenums="1"
# Filter tags that have fewer than <min_freq> occurrences
tags_above_freq = Counter(tag for tag in tags.elements()
                          if (tags[tag] >= min_freq))
df.tag = df.tag.apply(filter, include=list(tags_above_freq.keys()))
```

```python linenums="1"
# Fill None with other
df.tag = df.tag.fillna("other")
```

### Encoding

We're going to encode our output labels where we'll assign each tag a unique index.

```python linenums="1"
import numpy as np
import random
```

```python linenums="1"
# Get data
X = df.text.to_numpy()
y = df.tag
```

We'll be writing our own LabelEncoder which is based on scikit-learn's [implementation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html){:target="_blank"}. It's an extremely valuable skill to be able to write clean classes for objects we want to create.

```python linenums="1"
class LabelEncoder(object):
    """Encode labels into unique indices"""
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
            contents = {"class_to_index": self.class_to_index}
            json.dump(contents, fp, indent=4, sort_keys=False)

    @classmethod
    def load(cls, fp):
        with open(fp, "r") as fp:
            kwargs = json.load(fp=fp)
        return cls(**kwargs)
```

> If you're not familiar with the `@classmethod` decorator, learn more about it from our [Python lesson](../foundations/python.md#methods){:target="_blank"}.

```python linenums="1"
# Encode
label_encoder = LabelEncoder()
label_encoder.fit(y)
num_classes = len(label_encoder)
```
```python linenums="1"
label_encoder.class_to_index
```
<pre class="output">
{'computer-vision': 0,
 'mlops': 1,
 'natural-language-processing': 2,
 'other': 3}
</pre>
```python linenums="1"
label_encoder.index_to_class
```
<pre class="output">
{0: 'computer-vision',
 1: 'mlops',
 2: 'natural-language-processing',
 3: 'other'}
</pre>

```python linenums="1"
# Encode
label_encoder.encode(["computer-vision", "mlops", "mlops"])
```
<pre class="output">
array([0, 1, 1])
</pre>
```python linenums="1"
# Decode
label_encoder.decode(np.array([0, 1, 1]))
```
<pre class="output">
['computer-vision', 'mlops', 'mlops']
</pre>

```python linenums="1"
# Encode all our labels
y = label_encoder.encode(y)
print (y.shape)
```

Many of the *transformations* we're going to do on our input text features are model specific. For example, for our simple baselines we may do `label encoding` → `tf-idf` while for the more involved architectures we may do `label encoding` → `one-hot encoding` → `embeddings`. So we'll cover these in the next suite of lessons as we implement our [baselines](baselines.md){:target="_blank"}.

> In the next section we'll be performing exploratory data analysis (EDA) on our preprocessed dataset. However, the order of the steps can be reversed depending on how well the problem is defined. If we're unsure about how to prepare the data, we can use EDA to figure it out and vice versa.


<!-- Citation -->
{% include "styles/cite.md" %}
