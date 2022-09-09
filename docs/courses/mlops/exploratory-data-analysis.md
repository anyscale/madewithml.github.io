---
template: lesson.html
title: Exploratory Data Analysis (EDA)
description: Exploring our dataset for insights, with intention.
keywords: EDA, datasets, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/mlops.png
repository: https://github.com/GokuMohandas/mlops-course
notebook: https://github.com/GokuMohandas/mlops-course/blob/main/notebooks/tagifai.ipynb
---

{% include "styles/lesson.md" %}

## Intuition

Exploratory data analysis (EDA) to understand the signals and nuances of our dataset. It's a cyclical process that can be done at various points of our development process (before/after labeling, preprocessing, etc. depending on how well the problem is defined. For example, if we're unsure how to label or preprocess our data, we can use EDA to figure it out.

We're going to start our project with EDA, a vital (and fun) process that's often misconstrued. Here's how to think about EDA:

- not just to visualize a prescribed set of plots (correlation matrix, etc.).
- goal is to *convince* yourself that the data you have is sufficient for the task.
- use EDA to answer important questions and to make it easier to extract insight
- not a one time process; as your data grows, you want to revisit EDA to catch distribution shifts, anomalies, etc.

Let's answer a few key questions using EDA.

```python linenums="1"
from collections import Counter
import ipywidgets as widgets
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from wordcloud import WordCloud, STOPWORDS
sns.set_theme()
warnings.filterwarnings("ignore")
```

## Tag distribution
How many data points do we have per tag?

```python linenums="1"
# Distribution of tags
tags, tag_counts = zip(*Counter(df.tag.values).most_common())
plt.figure(figsize=(10, 3))
ax = sns.barplot(list(tags), list(tag_counts))
plt.title("Tag distribution", fontsize=20)
plt.xlabel("Tag", fontsize=16)
ax.set_xticklabels(tags, rotation=90, fontsize=14)
plt.ylabel("Number of projects", fontsize=16)
plt.show()
```
<div class="ai-center-all">
  <img src="/static/images/mlops/eda/tag_distribution.png" width="600" alt="data points per tag">
</div>

```python linenums="1"
# Most common tags
tags = Counter(df.tag.values)
tags.most_common()
```
<pre class="output">
[('natural-language-processing', 388),
 ('computer-vision', 356),
 ('mlops', 79),
 ('reinforcement-learning', 56),
 ('graph-learning', 45),
 ('time-series', 31)]
</pre>

> We'll address the [data imbalance](baselines.md#data-imbalance){:target="_blank"} after splitting into our train split and prior to training our model.

## Wordcloud

Is there enough signal in the title and description that's unique to each tag? This is important because we want to verify our initial hypothesis that the project's title and description are highly influential features.

```python linenums="1"
# Most frequent tokens for each tag
@widgets.interact(tag=list(tags))
def display_word_cloud(tag="natural-language-processing"):
    # Plot word clouds top top tags
    plt.figure(figsize=(15, 5))
    subset = df[df.tag==tag]
    text = subset.title.values
    cloud = WordCloud(
        stopwords=STOPWORDS, background_color="black", collocations=False,
        width=500, height=300).generate(" ".join(text))
    plt.axis("off")
    plt.imshow(cloud)
```
<div class="ai-center-all">
  <img src="/static/images/mlops/eda/word_cloud.png" width="500" alt="word cloud">
</div>

Looks like the `title` text feature has some good signal for the respective classes and matches our intuition. We can repeat this for the `description` text feature as well. This information will become useful when we decide how to use our features for modeling.

> All of the work we've done so far are inside IPython notebooks but in our [dashboard lesson](dashboard.md){:target="_blank"}, we'll transfer all of this into an interactive dashboard using [Streamlit](https://streamlit.io/){:target="_blank"}.

<!-- Citation -->
{% include "styles/cite.md" %}