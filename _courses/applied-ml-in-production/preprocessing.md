---
layout: page
title: Preprocessing · Applied ML in Production
description: Preparing and transforming our data for modeling.
image: /static/images/courses/applied-ml-in-production/preprocessing.png
tags: preprocessing feature-engineering

course-url: /courses/applied-ml-in-production/
next-lesson-url: /courses/applied-ml-in-production/baselines/
---

<!-- Header -->
<div class="row">
  <div class="col-md-8 col-6 mr-auto">
    <h1 class="page-title">{{ page.title | split: " · " | first }}</h1>
  </div>
  <div class="col-md-4 col-6">
    <div class="btn-group float-right mb-0" role="group">
      <a href="{{ page.course-url }}" class="btn btn-sm btn-outline-secondary"><i
          class="fas fa-sm fa-arrow-left mr-1"></i>Return to course</a>
    </div>
  </div>
</div>
<hr class="mt-0">

<!-- Video -->
<div class="ai-center-all mt-2">
    <iframe width="600" height="337.5" src="https://www.youtube.com/embed/i524r-nZi6Q?rel=0" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen></iframe>
</div>
<div class="ai-center-all mt-2">
  <small>Accompanying video for this lesson. <a href="https://www.youtube.com/madewithml?sub_confirmation=1" target="_blank">Subscribe</a> for updates!</small>
</div>

<div class="alert info mt-4" role="alert">
  <span style="text-align: left;">
    <i class="fas fa-info-circle mr-1"></i> Connect with the author, <i>Goku Mohandas</i>, on
    <a href="https://twitter.com/GokuMohandas" target="_blank">Twitter</a> and
    <a href="https://www.linkedin.com/in/goku" target="_blank">LinkedIn</a> for
    interactive conversations on upcoming lessons.
  </span>
</div>

<h3><u>Intuition</u></h3>

Data preprocessing can be categorized into two types of processes: *preparation* and *transformation*.

> Certain preprocessing steps are `global` (don't depend on our dataset, ex. removing stop words) and others are `local` (constructs are learned only from the training split, ex. vocabulary or min-max scaling). For the local, dataset-dependent preprocessing steps, we want to ensure that we split the data first before preprocessing to avoid data leaks.

🧹 *Preparing* the data involves organizing and cleaning the data.
- **joins**:
    - performing SQL joins with existing data tables to *organize* all the relevant data you need in one view.
- **typing**:
    - ensure that all the values for a specific feature are of the same data type, otherwise you won't be able to compare them.
- **missing values**:
    - omit samples with missing values (if only a small subset are missing it)
    - omit the entire feature (if too many samples are missing the value)
    - fill in missing values for features (using domain knowledge, heuristics, etc.)
    - may not always seem "missing" (ex. 0, null, NA, etc.)
- **outliers (anomalies)**:
    - craft assumptions about what is a "normal" expected value
    - be careful about removing important outliers (ex. fraud)
    - anomalies can be global (point), contextual (conditional) or collective
- **clean**:
    - use domain expertise and EDA
    - images (crop, resize, clip, etc.)
    - text (lower, stem, lemmatize, regex, etc.)

> You need to clean your data first before splitting, at least for the features that splitting depends on. So the process is more like: preprocessing (global, cleaning) → splitting → preprocessing (local, transformations). We covered splitting first since many preprocessing transformations depend on the training split.

🤖 *Transforming* the data involves feature encoding and engineering.
- **scaling**:
    - required for most models that are not decision tree based
    - learn constructs from train split and apply to other splits (local)
    - don't blindly scale features (ex. categorical features)
    - examples:
        - [standardization](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler){:target="_blank"}: rescales values to mean 0 and std 1
```python
import numpy as np
x_standardized = (x - np.mean(x)) / np.std(x)
```
        - [min-max](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.minmax_scale.html#sklearn.preprocessing.minmax_scale){:target="_blank"}: rescales values between 0 and 1 (wary of outliers)
```python
x_scaled = (x - x.min()) / (x.max() - x.min())
```
        - [binning](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html){:target="_blank"}: bin continuous values into categorical ones using your own range of values of binning or perform optimal binning used LightGBM, etc.
```python
import numpy as np
data = np.random.random(4) # values between 0 and 1
> array([0.94874339, 0.22385892, 0.51003595, 0.68040127])
bins = np.linspace(0, 1, 5) # bins between 0 and 1
> array([0.  , 0.25, 0.5 , 0.75, 1.  ])
x_binned = np.digitize(data, bins)
> array([4, 1, 3, 3])
```
        - and many [more](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing){:target="_blank"}!

- **encoding**:
    - allows for representing data efficiently (maintains signal) & effectively (learns pattern)
    - examples:
        - [label](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html#sklearn.preprocessing.LabelEncoder){:target="_blank"}: assign a unique index for every unique categorical value
```python
label_encoder.class_to_index = {
    "attention": 0,
    "autoencoders": 1,
    "convolutional-neural-networks": 2,
    "data-augmentation": 3,
    ... }
label_encoder.transform(["attention", "data-augmentation"])
> array([2, 2, 1])
```
        - [one-hot](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html#sklearn.preprocessing.OneHotEncoder){:target="_blank"}: representation as binary vectors
```python
one_hot_encoder.transform(["attention", "data-augmentation"])
> array([1, 0, 0, 1, 0, ..., 0])
```
        - [embeddings](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html){:target="_blank"}: efficient, dense representations that can capture meaningful relationships amongst the different values
```python
# Initialize embeddings
self.embeddings = nn.Embedding(
    embedding_dim=embedding_dim, num_embeddings=vocab_size)
x_in = self.embeddings(x_in)
x_in.shape
> (len(X), embedding_dim)
```
        - [target](https://contrib.scikit-learn.org/category_encoders/targetencoder.html){:target="_blank"}: represent a categorical feature with the average of the target values that share that categorical value
        - and many [more](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing){:target="_blank"}!

- **extraction**:
    - signal extraction from existing features
    - examples:
        - combine existing features
        - [transfer learning](https://ruder.io/transfer-learning/){:target="_blank"}: using a pretrained model as a feature extractor and finetuning on it's results
        - [autoencoders](https://www.jeremyjordan.me/autoencoders/){:target="_blank"}: learn to encode inputs for compressed knowledge representation
        - [principle component analysis (PCA)](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA){:target="_blank"}: linear dimensionality reduction to project data in a lower dimensional space.
```python
import numpy as np
from sklearn.decomposition import PCA
X = np.array([[-1, -1, 3], [-2, -1, 2], [-3, -2, 1]])
pca = PCA(n_components=2)
pca.fit(X)
pca.transform(X)
> array([
    [-1.44245791, -0.1744313 ],
    [-0.1148688 ,  0.31291575],
    [ 1.55732672, -0.13848446]])
pca.explained_variance_ratio_
> array([0.96838847, 0.03161153])
pca.singular_values_
> array([2.12582835, 0.38408396])
```
        - [counts (ngram)](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html){:target="_blank"}: sparse representation of text as matrix of token counts — useful if feature values have lot's of meaningful, separable signal.
```python
from sklearn.feature_extraction.text import CountVectorizer
y = [
    'acetyl acetone',
    'acetyl chloride',
    'chloride hydroxide',
]
vectorizer = CountVectorizer()
y = vectorizer.fit_transform(y)
vectorizer.get_feature_names()
> ['acetone', 'acetyl', 'chloride', 'hydroxide']
y.toarray()
> [[1, 1, 0, 0],
   [0, 1, 1, 0],
   [0, 0, 1, 1]]
# 💡 Repeat above with char-level ngram vectorizer
# vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 3)) # uni, bi and trigrams
```
        - [similarity](https://github.com/dirty-cat/dirty_cat){:target="_blank"}: similar to count vectorization but based on similarities in tokens
        - and many [more](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction){:target="_blank"}!

> Often, teams will want to reuse the same features for different tasks so how can we avoid duplication of efforts? A solution is `feature stores` which will enable sharing of features and the workflows around feature pipelines. We'll cover feature stores during *Production*.

> When we move our code from notebooks to Python scripts, we'll be testing all our preprocessing functions (these workflows can also be captured in feature stores and applied as features are updated).

<i class="fab fa-youtube ai-color-youtube mr-1"></i> Watch from [0:00](https://www.youtube.com/watch?v=i524r-nZi6Q&list=PLqy_sIcckLC2jrxQhyqWDhL_9Uwxz8UFq&index=8&t=0s){:target="_blank"} for a video walkthrough of this section.

<h3><u>Application</u></h3>

> The notebook for this section can be found [here](https://github.com/madewithml/applied-ml-in-production/blob/master/notebooks/tagifai.ipynb){:target="_blank"}.

Our dataset is pretty straight forward so we'll only need to use a few of the preprocessing techniques from above. To *prepare* our data, we're going to clean up our input text (all global actions).
1. lower (conditional)
```python
text = text.lower()
```
2. remove stopwords ([NLTK](https://github.com/nltk/nltk){:target="_blank"} package)
```python
pattern = re.compile(r'\b(' + r'|'.join(stopwords) + r')\b\s*')
text = pattern.sub('', text)
```
3. spacing and filters
```python
text = re.sub(r"([-;;.,!?<=>])", r" \1 ", text)
text = re.sub(filters, r"", text)
text = re.sub(' +', ' ', text)  # remove multiple spaces
text = text.strip()
```
4. remove URLs using regex (discovered during EDA)
```python
text = re.sub(r'http\S+', '', text)
```
5. stemming (conditional)
```python
text = " ".join([porter.stem(word) for word in text.split(' ')])
```

> Our data splits were dependent only on the target labels (tags) which were already cleaned. However, if your splits depend on other features as well, you need to at least clean them first before splitting. So the process is more like: preprocessing (global, cleaning) → splitting → preprocessing (local, transformations).

Many of the *transformations* we're going to do are model specific. For example, for our simple baselines we may do `label encoding` → `tf-idf` while for the more involved architectures we may do `label encoding` → `one-hot encoding` → `embeddings`. So we'll cover these in the next suite of lessons as we implement each baseline.

<i class="fab fa-youtube ai-color-youtube mr-1"></i> Watch from [21:10](https://www.youtube.com/watch?v=i524r-nZi6Q&list=PLqy_sIcckLC2jrxQhyqWDhL_9Uwxz8UFq&index=8&t=1270s){:target="_blank"} for a video walk-through of this section.


<!-- Footer -->
<hr>
<div class="row mb-4">
  <div class="col-6 mr-auto">
    <a href="{{ page.course-url }}" class="btn btn-sm btn-outline-secondary"><i class="fas fa-sm fa-arrow-left mr-1"></i>Return to course</a>
  </div>
  <div class="col-6">
    <div class="float-right">
      <a href="{{ page.next-lesson-url }}" class="btn btn-sm btn-outline-secondary"><i class="fas fa-sm fa-arrow-right mr-1"></i>Next lesson</a>
    </div>
  </div>
</div>