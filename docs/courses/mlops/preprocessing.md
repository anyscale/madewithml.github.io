---
template: lesson.html
title: Preprocessing Data for Machine Learning
description: Preparing and transforming our data for modeling.
keywords: preprocessing, cleaning, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/applied_ml.png
repository: https://github.com/GokuMohandas/MLOps
notebook: https://colab.research.google.com/github/GokuMohandas/MLOps/blob/main/notebooks/tagifai.ipynb
---

{% include "styles/lesson.md" %}

## Intuition

Data preprocessing can be categorized into two types of processes: *preparation* and *transformation*.

!!! note
    Certain preprocessing steps are `global` (don't depend on our dataset, ex. removing stop words) and others are `local` (constructs are learned only from the training split, ex. vocabulary). For the local, dataset-dependent preprocessing steps, we want to ensure that we split the data first before preprocessing to avoid data leaks.

## Preparing

Preparing the data involves organizing and cleaning the data.

- `joins`:
    - performing SQL joins with existing data tables to *organize* all the relevant data you need in one view.
- `typing`:
    - ensure that all the values for a specific feature are of the same data type, otherwise you won't be able to compare them.
- `missing values`:
    - omit samples with missing values (if only a small subset are missing it)
    - omit the entire feature (if too many samples are missing the value)
    - fill in missing values for features (using domain knowledge, heuristics, etc.)
    - may not always seem "missing" (ex. 0, null, NA, etc.)
- `outliers (anomalies)`:
    - craft assumptions about what is a "normal" expected value
    - be careful about removing important outliers (ex. fraud)
    - anomalies can be global (point), contextual (conditional) or collective
- `clean`:
    - use domain expertise and EDA
    - images (crop, resize, clip, etc.)
    - text (lower, stem, lemmatize, regex, etc.)

!!! note
    You need to clean your data first before splitting, at least for the features that splitting depends on. So the process is more like: preprocessing (global, cleaning) → splitting → preprocessing (local, transformations). We covered splitting first since many preprocessing transformations depend on the training split.

## Transforming
Transforming the data involves feature encoding and engineering.

!!! warning
    Before transformation, be sure to detect (and potentially remove) outliers using distributions and/or domain expertise. You should constantly revisit these explicit decisions because they may change over time and you don’t want to be including or removing data you shouldn’t be.

### Scaling
- required for most models that are not decision tree based
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

!!! note
    When we move our code from notebooks to Python scripts, we'll be testing all our preprocessing functions (these workflows can also be captured in feature stores and applied as features are updated).


### Encoding

- allows for representing data efficiently (maintains signal) & effectively (learns pattern)

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

- [target](https://contrib.scikit-learn.org/category_encoders/targetencoder.html){:target="_blank"}: represent a categorical feature with the average of the target values that share that categorical value
- and many [more](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing){:target="_blank"}!

### Extraction
- signal extraction from existing features
- combine existing features
- [transfer learning](https://ruder.io/transfer-learning/){:target="_blank"}: using a pretrained model as a feature extractor and finetuning on it's results
- [autoencoders](https://www.jeremyjordan.me/autoencoders/){:target="_blank"}: learn to encode inputs for compressed knowledge representation

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
        'acetyl acetone',
        'acetyl chloride',
        'chloride hydroxide',
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

!!! note
    Often, teams will want to reuse the same features for different tasks so how can we avoid duplication of efforts? A solution is [feature stores](https://www.tecton.ai/blog/what-is-a-feature-store/){:target="_blank"} which will enable sharing of features and the workflows around feature pipelines. We'll cover feature stores during *Production*.


## Application

Our dataset is pretty straight forward so we'll only need to use a few of the preprocessing techniques from above. To *prepare* our data, we're going to clean up our input text (all global actions).
1. lower (conditional)
```python linenums="1"
text = text.lower()
```
2. remove stopwords ([NLTK](https://github.com/nltk/nltk){:target="_blank"} package)
```python linenums="1"
import re
pattern = re.compile(r'\b(' + r'|'.join(stopwords) + r')\b\s*')
text = pattern.sub('', text)
```
3. spacing and filters
```python linenums="1"
text = re.sub(r"([-;;.,!?<=>])", r" \1 ", text)
text = re.sub(filters, r"", text)
text = re.sub(' +', ' ', text)  # remove multiple spaces
text = text.strip()
```
4. remove URLs using regex (discovered during EDA)
```python linenums="1"
text = re.sub(r'http\S+', '', text)
```
5. stemming (conditional)
```python linenums="1"
text = " ".join([porter.stem(word) for word in text.split(' ')])
```

We can apply our preprocessing steps to our text feature in the dataframe.
```python linenums="1"
# Apply to dataframe
preprocessed_df = df.copy()
preprocessed_df.text = preprocessed_df.text.apply(preprocess, lower=True, stem=False)
print (f"{df.text.values[0]}\n\n{preprocessed_df.text.values[0]}")
```
<pre class="output">
Albumentations Fast image augmentation library and easy to use wrapper around other libraries.
albumentations fast image augmentation library easy use wrapper around libraries
</pre>

!!! info
    Our data splits were dependent only on the target labels (tags) which were already cleaned. However, if your splits depend on other features as well, you need to at least clean them first before splitting. So the process is more like: preprocessing (global, cleaning) → splitting → preprocessing (local, transformations).

Many of the *transformations* we're going to do are model specific. For example, for our simple baselines we may do `label encoding` → `tf-idf` while for the more involved architectures we may do `label encoding` → `one-hot encoding` → `embeddings`. So we'll cover these in the next suite of lessons as we implement each baseline.


<!-- Citation -->
{% include "cite.md" %}