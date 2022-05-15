---
template: lesson.html
title: Data Preprocessing
description: Preprocessing our dataset, via through preparations and transformations, to use for training.
keywords: preprocessing, preparation, cleaning, feature engineering, filtering, transformations, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/mlops.png
repository: https://github.com/GokuMohandas/MLOps
notebook: https://colab.research.google.com/github/GokuMohandas/MLOps/blob/main/notebooks/tagifai.ipynb
---

{% include "styles/lesson.md" %}

## Intuition

Data preprocessing can be categorized into two types of processes: *preparation* and *transformation*. We'll explore common preprocessing techniques and then walkthrough the relevant processes for our specific application.

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
- combine features in unique ways to draw out signal
```python linenums="1"
# Input
df.C = df.A + df.B
```

!!! tip
    Feature engineering can be done in collaboration with domain experts that can guide us on what features to engineer and use.

> We can use techniques such as [SHAP](https://github.com/slundberg/shap){:target="_blank"} (SHapley Additive exPlanations) or [LIME](https://github.com/marcotcr/lime){:target="_blank"} (Local Interpretable Model-agnostic Explanations) to inspect feature importance. On a high level, these techniques learn which features have the most signal by assessing the performance in their absence. These inspections can be done on a model's single prediction or at a coarse-grained, overall level.

### Cleaning
- use domain expertise and EDA
- apply constraints via filters
- ensure data type consistency
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

> We can also encode our data with hashing or using it's attributes instead of the exact entity itself. For example, representing a user by their location and favorites as opposed to using their user ID. These methods are great when we want to use features that suffer from the curse of dimensionality (lots of feature values for a feature but not enough data samples for each one) or [online learning](infrastructure.md#online-learning){:target="_blank"} scenarios.

### Extraction

- signal extraction from existing features
- combine existing features
- transfer learning: using a pretrained model as a feature extractor and finetuning on it's results
- autoencoders: learn to encode inputs for compressed knowledge representation

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
    <pre class="outout">
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

> Often, teams will want to reuse the same features for different tasks so how can we avoid duplication of efforts? A solution is [feature stores](https://www.tecton.ai/blog/what-is-a-feature-store/){:target="_blank"} which will enable sharing of features and the workflows around feature pipelines. We'll cover feature stores during *Production*.


## Application

For our application, we'll be implementing a few of these preprocessing steps that are relevant for our dataset.

## Feature engineering
We can combine existing input features to create new meaningful signal (helping the model learn). However, there's usually no simple way to know if certain feature combinations will help or not without empirically experimenting with the different combinations. Here, we could use a project's title and description separately as features but we'll combine them to create one input feature.

```python linenums="1"
# Input
df["text"] = df.title + " " + df.description
```

And since we're dealing with text data, we can apply some of the common preparation processes:

1. lower (conditional)
```python linenums="1"
text = text.lower()
```
2. remove stopwords (from [NLTK](https://github.com/nltk/nltk){:target="_blank"} package)
```python linenums="1"
import re
# Remove stopwords
if len(stopwords):
    pattern = re.compile(r"\b(" + r"|".join(stopwords) + r")\b\s*")
    text = pattern.sub("", text)
```

3. Filters and spacing
```python linenums="1"
# Separate filters attached to tokens
filters = r"([-;;.,!?<=>])"
text = re.sub(filters, r" \1 ", text)

# Remove non alphanumeric chars
text = re.sub("[^A-Za-z0-9]+", " ", text)

# Remove multiple spaces
text = re.sub(" +", " ", text)

# Strip white space at the ends
text = text.strip()
```

    !!! note
        We could definitely try and include emojis, punctuations, etc. because they do have a lot of signal for the task but it's best to simplify the initial feature set we use to just what we think are the most influential and then we can slowly introduce other features and assess utility.

    !!! warning
        We'll want to introduce less frequent features as they become more frequent or encode them in a clever way (ex. binning, extract general attributes, common n-grams, mean encoding using other feature values, etc.) so that we can mitigate the feature value dimensionality issue until we're able to collect more data.

4. remove URLs using regex (discovered during EDA)
```python linenums="1"
text = re.sub(r"http\S+", "", text)
```
5. stemming (conditional)
```python linenums="1"
text = " ".join([porter.stem(word) for word in text.split(" ")])
```

We can apply our preprocessing steps to our text feature in the dataframe by wrapping all these processes under a function.

```python linenums="1"
# Define preprocessing function
def preprocess(text):
    ...
    return text
```

```python linenums="1"
# Apply to dataframe
original_df = df.copy()
df.text = df.text.apply(preprocess, lower=True, stem=False)
print (f"{original_df.text.values[0]}\n{df.text.values[0]}")
```
<pre class="output">
Comparison between YOLO and RCNN on real world videos Bringing theory to experiment is cool. We can easily train models in colab and find the results in minutes.
comparison yolo rcnn real world videos bringing theory experiment cool easily train models colab find results minutes
</pre>

## Transformations

Many of the *transformations* we're going to do are model specific. For example, for our simple baselines we may do `label encoding` → `tf-idf` while for the more involved architectures we may do `label encoding` → `one-hot encoding` → `embeddings`. So we'll cover these in the next suite of lessons as we implement each of the [baselines](baselines.md){:target="_blank"}.

> In the next section we'll be performing exploratory data analysis (EDA) on our preprocessed dataset. However, the order of the steps can be reversed depending on how well the problem is defined. If we're unsure about how to prepare the data, we can use EDA to figure it out. In fact in our [dashboard](dashboard.md){:target="_blank"} lesson, we can interactively apply data processing and EDA back and forth until we have finalized on constraints.


<!-- Citation -->
{% include "cite.md" %}
