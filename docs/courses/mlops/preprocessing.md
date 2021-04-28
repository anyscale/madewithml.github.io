---
template: lesson.html
title: Data Preprocessing
description: Preprocessing our dataset, via through preparations and transformations, to use for training.
keywords: preprocessing, preparation, cleaning, feature engineering, filtering, transformations, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/applied_ml.png
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
- be careful about removing important outliers (ex. fraud)
- anomalies can be global (point), contextual (conditional) or collective (individual points are not anomalous and the collective group is an outlier)

### Feature engineering
- combine features in unique ways to draw out signal
```python linenums="1"
# Input
df.C = df.A + df.B
```

!!! note
    This is also when we would save these features to a central [feature store](feature_stores.md){:target="_blank"} for the benefits of:

    - reduce duplication of effort when engineering features.
    - remove training and serving skew for creating and using features.
    - avoid data leaks with point-in-time validation (es. during SQL joins).
    - data validation and monitoring on features distributions.

    Learn more about feature stores and implementing them with [Feast](https://github.com/feast-dev/feast){:target="_blank"} in our [Feature Stores lesson](feature_stores.md){:target="_blank"}.

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

!!! warning
    Before transformation, be sure to detect (and potentially remove) outliers using distributions and/or domain expertise because they can severely skew transformations. We should regularly revisit these explicit decisions because they may change over time and you don’t want to be including or removing data we shouldn’t be.

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


## Data augmentation
We'll often want to increase the size and diversity of our training data split through data augmentation. It involves using the existing samples to generate synthetic, yet realistic, examples.

1. **Split the dataset**. We want to split our dataset first because many augmentation techniques will cause a form of data leak if we allow the generated samples to be placed across different data splits. For example, support augmentation involves generating synonyms for certain key tokens in a sentence. If we allow the generated sentences from the same origin sentence to go into different splits, we could be potentially leaking samples with nearly identical embedding representations across our different splits.

2. **Augment the training split only**. We want to apply data augmentation on only the training set because our validation and testing splits should be used to provide an accurate estimate on actual data points.

Depending on the feature types and tasks, there are many data augmentation libraries which allow us to extend our training data.

### Natural language processing (NLP)
- [NLPAug](https://github.com/makcedward/nlpaug){:target="_blank"}: data augmentation for NLP.
- [TextAttack](https://github.com/QData/TextAttack){:target="_blank"}: a framework for adversarial attacks, data augmentation, and model training in NLP.
- [TextAugment](https://github.com/dsfsi/textaugment){:target="_blank"}: text augmentation library.

### Computer vision (CV)
- [Imgaug](https://github.com/aleju/imgaug){:target="_blank"}: image augmentation for machine learning experiments.
- [Albumentations](https://github.com/albumentations-team/albumentations){:target="_blank"}: fast image augmentation library.
- [Augmentor](https://github.com/mdbloice/Augmentor){:target="_blank"}: image augmentation library in Python for machine learning.
- [Kornia.augmentation](https://github.com/kornia/kornia){:target="_blank"}: a module to perform data augmentation in the GPU.
- [SOLT](https://github.com/MIPT-Oulu/solt){:target="_blank"}: data augmentation library for Deep Learning, which supports images, segmentation masks, labels and key points.

### Other
- [Snorkel](https://github.com/snorkel-team/snorkel){:target="_blank"}: system for generating training data with weak supervision.
- [DeltaPy⁠⁠](https://github.com/firmai/deltapy){:target="_blank"}: tabular data augmentation and feature engineering.
- [Audiomentations](https://github.com/iver56/audiomentations){:target="_blank"}: a Python library for audio data augmentation.
- [Tsaug](https://github.com/arundo/tsaug){:target="_blank"}: a Python package for time series augmentation.

Regardless of what tool we use, it's important to validate that we're not just augmenting for the sake of augmentation. For example, in many NLP data augmentation scenarios, the adjectives are replaced with other adjectives. We need to ensure that this generalized change doesn't affect key aspects of our dataset. For more fine-grained data augmentation, we can use concepts like [transformation functions](https://www.snorkel.org/use-cases/02-spam-data-augmentation-tutorial){:target="_blank"} to apply specific types of augmentation to a subset of our dataset.


## Application

For our application, we'll be implementing a few of these preprocessing steps that are relevant for our dataset.

## Feature engineering
We can combine existing input features to create new meaningful signal (helping the model learn). However, there's usually no simple way to know if certain feature combinations will help or not without empirically experimenting with the different combinations. Here, we could use a project's title and description separately as features but we'll combine them to create one input feature.

```python linenums="1"
# Input
df['text'] = df.title + " " + df.description
```

## Filtering
In the same vain, we can also reduce the size of our data by placing constraints as to what data is worth annotation or labeling. Here we decide to filter tags above a certain frequency threshold because those with fewer samples won't be adequate for training.

```python linenums="1"
def filter(l, include=[], exclude=[]):
    """Filter a list using inclusion and exclusion lists of items."""
    filtered = [item for item in l if item in include and item not in exclude]
    return filtered
```

We're going to *include* only these tags because they're the tags we care about and we've allowed authors to add any tag they want (noise). We'll also be *excluding* some general tags because they are automatically added when their children tags are present.
```python linenums="1"
# Inclusion/exclusion criteria for tags
include = list(tags_dict.keys())
exclude = ['machine-learning', 'deep-learning',  'data-science',
           'neural-networks', 'python', 'r', 'visualization']
```
!!! note
    Since we're *constraining* the output space here, we'll want to monitor the prevalence of new tags over time so we can capture them.


```python linenums="1"
# Filter tags for each project
df.tags = df.tags.apply(filter, include=include, exclude=exclude)
tags = Counter(itertools.chain.from_iterable(df.tags.values))
```

We're also going to restrict the mapping to only tags that are above a certain frequency threshold. The tags that don't have enough projects will not have enough samples to model their relationships.
```python linenums="1"
@widgets.interact(min_tag_freq=(0, tags.most_common()[0][1]))
def separate_tags_by_freq(min_tag_freq=30):
    tags_above_freq = Counter(tag for tag in tags.elements()
                                    if tags[tag] >= min_tag_freq)
    tags_below_freq = Counter(tag for tag in tags.elements()
                                    if tags[tag] < min_tag_freq)
    print ("Most popular tags:\n", tags_above_freq.most_common(5))
    print ("\nTags that just made the cut:\n", tags_above_freq.most_common()[-5:])
    print ("\nTags that just missed the cut:\n", tags_below_freq.most_common(5))
```
<pre class="output">
Most popular tags:
 [('natural-language-processing', 429),
  ('computer-vision', 388),
  ('pytorch', 258),
  ('tensorflow', 213),
  ('transformers', 196)]

Tags that just made the cut:
 [('time-series', 34),
  ('flask', 34),
  ('node-classification', 33),
  ('question-answering', 32),
  ('pretraining', 30)]

Tags that just missed the cut:
 [('model-compression', 29),
  ('fastai', 29),
  ('graph-classification', 29),
  ('recurrent-neural-networks', 28),
  ('adversarial-learning', 28)]
</pre>
```python linenums="1"
# Filter tags that have fewer than <min_tag_freq> occurances
min_tag_freq = 30
tags_above_freq = Counter(tag for tag in tags.elements()
                          if tags[tag] >= min_tag_freq)
df.tags = df.tags.apply(filter, include=list(tags_above_freq.keys()))
```

## Cleaning

After applying our filters, it's important that we remove any samples that didn't make the cut. In our case, we'll want to remove inputs that have no remaining (not enough frequency) tags.

```python linenums="1"
# Remove projects with no more remaining relevant tags
df = df[df.tags.map(len) > 0]
print (f"{len(df)} projects")
```
<pre class="output">
1444 projects
</pre>

And since we're dealing with text data, we can apply some of the common preparation processes:

1. lower (conditional)
```python linenums="1"
text = text.lower()
```
2. remove stopwords (from [NLTK](https://github.com/nltk/nltk){:target="_blank"} package)
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

We can apply our preprocessing steps to our text feature in the dataframe by wrapping all these processes under a function.

```python linenums="1"
# Apply to dataframe
preprocessed_df.text = preprocessed_df.text.apply(preprocess, lower=True, stem=False)
print (f"{df.text.values[0]}\n\n{preprocessed_df.text.values[0]}")
```
<pre class="output">
Albumentations Fast image augmentation library and easy to use wrapper around other libraries.
albumentations fast image augmentation library easy use wrapper around libraries
</pre>

## Transformations
Many of the *transformations* we're going to do are model specific. For example, for our simple baselines we may do `label encoding` → `tf-idf` while for the more involved architectures we may do `label encoding` → `one-hot encoding` → `embeddings`. So we'll cover these in the next suite of lessons as we implement each [baseline](baselines.md){:target="_blank"}.


## Pipelines

In a production setting, each of these preprocessing steps can be individual workflows. For example, if we use [KubeFlow Pipelines](https://www.kubeflow.org/docs/components/pipelines/overview/pipelines-overview/){:target="_blank"} (KFP) to manage our workflows (I recommend using [k3d](https://k3d.io/){:target="_blank"} to play with KFP [locally](https://www.kubeflow.org/docs/components/pipelines/installation/standalone-deployment/){:target="_blank"}), each of them would execute in a [Docker](docker.md){:target="_blank"} container that can be individually scaled, inspected and modified.

<div class="ai-center-all">
    <img width="1000" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/preprocessing/kfp.png">
</div>

!!! note
    In the next section we'll be performing exploratory data analysis (EDA) on our preprocessed dataset. However, the order of the steps can be reversed depending on how well the problem is defined. If we're unsure about how to prepare the data, we can use EDA to figure it out. In fact in our [dashboard](dashboard.md){:target="_blank"} lesson, we can interactively apply data processing and EDA back and forth until we have finalized on constraints.


<!-- Citation -->
{% include "cite.md" %}
