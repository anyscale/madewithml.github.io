---
template: lesson.html
title: Data Augmentation
description: Assessing data augmentation on our training data split to increase the number of quality training samples.
keywords: data augmentation, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/mlops.png
repository: https://github.com/GokuMohandas/mlops-course
notebook: https://github.com/GokuMohandas/mlops-course/blob/main/notebooks/tagifai.ipynb
---


{% include "styles/lesson.md" %}

## Intuition

We'll often want to increase the size and diversity of our training data split through data augmentation. It involves using the existing samples to generate synthetic, yet realistic, examples.

1. **Split the dataset**. We want to split our dataset first because many augmentation techniques will cause a form of data leak if we allow the generated samples to be placed across different data splits.
> For example, some augmentation involves generating synonyms for certain key tokens in a sentence. If we allow the generated sentences from the same origin sentence to go into different splits, we could be potentially leaking samples with nearly identical embedding representations across our different splits.

2. **Augment the training split**. We want to apply data augmentation on only the training set because our validation and testing splits should be used to provide an accurate estimate on actual data points.

3. **Inspect and validate**.  It's useless to augment just for the same of increasing our training sample size if the augmented data samples are not probable inputs that our model could encounter in production.

The exact method of data augmentation depends largely on the type of data and the application. Here are a few ways different modalities of data can be augmented:

<div class="ai-center-all mt-4">
    <img width="700" src="/static/images/mlops/augmentation/snorkel.png" alt="data augmentation types">
</div>
<div class="ai-center-all mb-4">
    <small><a href="https://www.snorkel.org/blog/tanda" target="_blank">Data Augmentation with Snorkel</a></small>
</div>

- **General**: normalization, smoothing, random noise, synthetic oversampling ([SMOTE](https://arxiv.org/abs/1106.1813){:target="_blank"}), etc.
- **Natural language processing (NLP)**: substitutions (synonyms, tfidf, embeddings, masked models), random noise, spelling errors, etc.
- **Computer vision (CV)**: crop, flip, rotate, pad, saturate, increase brightness, etc.

!!! warning
    While the transformations on some data modalities, such as images, are easy to inspect and validate, others may introduce silent errors. For example, shifting the order of tokens in text can significantly alter the meaning (“this is really cool” → “is this really cool”). Therefore, it’s important to measure the noise that our augmentation policies will introduce and do have granular control over the transformations that take place.

## Libraries

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


## Application

Let's use the [nlpaug](https://github.com/makcedward/nlpaug){:target="_blank"} library to augment our dataset and assess the quality of the generated samples.

```bash
pip install nlpaug==1.1.0 transformers==3.0.2 -q
pip install snorkel==0.9.8 -q
```

```python linenums="1"
import nlpaug.augmenter.word as naw
```
``` python linenums="1"
# Load tokenizers and transformers
substitution = naw.ContextualWordEmbsAug(model_path="distilbert-base-uncased", action="substitute")
insertion = naw.ContextualWordEmbsAug(model_path="distilbert-base-uncased", action="insert")
text = "Conditional image generation using Variational Autoencoders and GANs."
```

```python linenums="1"
# Substitutions
substitution.augment(text)
```
<pre class="output">
hierarchical risk mapping using variational signals and gans.
</pre>

Substitution doesn't seem like a great idea for us because there are certain keywords that provide strong signal for our tags so we don't want to alter those. Also, note that these augmentations are NOT deterministic and will vary every time we run them. Let's try insertion...

```python linenums="1"
# Insertions
insertion.augment(text)
```
<pre class="output">
automated conditional inverse image generation algorithms using multiple variational autoencoders and gans.
</pre>

A little better but still quite fragile and now it can potentially insert key words that can influence false positive tags to appear. Maybe instead of substituting or inserting new tokens, let's try simply swapping machine learning related keywords with their aliases from our [auxiliary data](https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/tags.json){:target="_blank"}. We'll use Snorkel's [transformation functions](https://www.snorkel.org/use-cases/02-spam-data-augmentation-tutorial){:target="_blank"} to easily achieve this.

```python linenums="1"
# Replace dashes from tags & aliases
def replace_dash(x):
    return x.replace("-", " ")
```
```python linenums="1"
# Create dict of aliases
flat_tags_dict = {}
for tag, info in tags_dict.items():
    tag = replace_dash(x=tag)
    aliases = list(map(replace_dash, info["aliases"]))
    if len(aliases):
        flat_tags_dict[tag] = aliases
    for alias in aliases:
        _aliases = aliases + [tag]
        _aliases.remove(alias)
        flat_tags_dict[alias] = _aliases
```
```python linenums="1"
print (flat_tags_dict["natural language processing"])
print (flat_tags_dict["nlp"])
```
<pre class="output">
['nlp', 'nlproc']
['nlproc', 'natural language processing']
</pre>

> For now we'll use tags and aliases as they are in `tags.json` but we could account for plurality of tags using the [inflect](https://github.com/jaraco/inflect){:target="_blank"} package or apply stemming before replacing aliases, etc.

```python linenums="1"
# We want to match with the whole word only
print ("gan" in "This is a gan.")
print ("gan" in "This is gandalf.")
```

```python linenums="1"
# \b matches spaces
def find_word(word, text):
    word = word.replace("+", "\+")
    pattern = re.compile(fr"\b({word})\b", flags=re.IGNORECASE)
    return pattern.search(text)
```

```python linenums="1"
# Correct behavior (single instance)
print (find_word("gan", "This is a gan."))
print (find_word("gan", "This is gandalf."))
```

<pre class="output">
&lt;re.Match object; span=(10, 13), match='gan'&gt;
None
</pre>

Now let's use snorkel's [`transformation_function`](https://snorkel.readthedocs.io/en/latest/packages/_autosummary/augmentation/snorkel.augmentation.transformation_function.html){:target="_blank"} to systematically apply this transformation to our data.

```python linenums="1"
from snorkel.augmentation import transformation_function
```

```python linenums="1"
@transformation_function()
def swap_aliases(x):
    """Swap ML keywords with their aliases."""
    # Find all matches
    matches = []
    for i, tag in enumerate(flat_tags_dict):
        match = find_word(tag, x.text)
        if match:
            matches.append(match)
    # Swap a random match with a random alias
    if len(matches):
        match = random.choice(matches)
        tag = x.text[match.start():match.end()]
        x.text = f"{x.text[:match.start()]}{random.choice(flat_tags_dict[tag])}{x.text[match.end():]}"
    return x
```

```python linenums="1"
# Swap
for i in range(3):
    sample_df = pd.DataFrame([{"text": "a survey of reinforcement learning for nlp tasks."}])
    sample_df.text = sample_df.text.apply(preprocess, lower=True, stem=False)
    print (swap_aliases(sample_df.iloc[0]).text)
```

```python linenums="1"
# Undesired behavior (needs contextual insight)
for i in range(3):
    sample_df = pd.DataFrame([{"text": "Autogenerate your CV to apply for jobs using NLP."}])
    sample_df.text = sample_df.text.apply(preprocess, lower=True, stem=False)
    print (swap_aliases(sample_df.iloc[0]).text)
```

<pre class="output">
autogenerate vision apply jobs using nlp
autogenerate cv apply jobs using natural language processing
autogenerate cv apply jobs using nlproc
</pre>

Now we'll define a [augmentation policy](https://snorkel.readthedocs.io/en/latest/packages/augmentation.html){:target="_blank"} to apply our transformation functions with certain rules (how many samples to generate, whether to keep the original data point, etc.)


```python linenums="1"
from snorkel.augmentation import ApplyOnePolicy, PandasTFApplier
```

```python linenums="1"
# Transformation function (TF) policy
policy = ApplyOnePolicy(n_per_original=5, keep_original=True)
tf_applier = PandasTFApplier([swap_aliases], policy)
train_df_augmented = tf_applier.apply(train_df)
train_df_augmented.drop_duplicates(subset=["text"], inplace=True)
train_df_augmented.head()
```

<div class="output_subarea output_html rendered_html ai-center-all"><div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>laplacian pyramid reconstruction refinement se...</td>
      <td>computer-vision</td>
    </tr>
    <tr>
      <th>1</th>
      <td>extract stock sentiment news headlines project...</td>
      <td>natural-language-processing</td>
    </tr>
    <tr>
      <th>2</th>
      <td>big bad nlp database collection 400 nlp datasets...</td>
      <td>natural-language-processing</td>
    </tr>
    <tr>
      <th>2</th>
      <td>big bad natural language processing database c...</td>
      <td>natural-language-processing</td>
    </tr>
    <tr>
      <th>2</th>
      <td>big bad nlproc database collection 400 nlp dat...</td>
      <td>natural-language-processing</td>
    </tr>
  </tbody>
</table>
</div></div>

```python linenums="1"
len(train_df), len(train_df_augmented)
```
<pre class="output">
(668, 913)
</pre>

For now, we'll skip the data augmentation because it's quite fickle and empirically it doesn't improvement performance much. But we can see how this can be very effective once we can control what type of vocabulary to augment on and what exactly to augment with.

!!! warning
    Regardless of what method we use, it's important to validate that we're not just augmenting for the sake of augmentation. We can do this by executing any existing [data validation tests](https://madewithml.com/courses/mlops/testing#data){:target="_blank"} and even creating specific tests to apply on augmented data.

## Resources

- [Learning to Compose Domain-Specific Transformations for Data Augmentation](https://arxiv.org/abs/1709.01643){:target="_blank"}