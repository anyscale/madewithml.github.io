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

## Application

- [projects.json](https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/projects.json){:target="_blank"}: projects with title, description and tag.
- [tags.json](https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/tags.json){:target="_blank"}: auxiliary information on the tags we are about of our platform.

> Recall that our [objective](purpose.md#objective){:target="_blank"} was to classify incoming content so that the community can discover them.

### Projects
We'll first load our dataset from the JSON file.

```python linenums="1"
from collections import Counter
import ipywidgets as widgets
import itertools
import json
import pandas as pd
from urllib.request import urlopen
```

> Traditionally, our data assets will be stored, versioned and updated in a database, warehouse, etc. We'll learn more about these different [data systems](data-stack.md){:target="_blank"} later, but for now, we'll load our data as a JSON file from our repository.

```python linenums="1"
# Load projects
url = "https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/projects.json"
projects = json.loads(urlopen(url).read())
print (f"{len(projects)} projects")
print (json.dumps(projects[0], indent=2))
```
<pre class="output">
955 projects
{
  "id": 6,
  "created_on": "2020-02-20 06:43:18",
  "title": "Comparison between YOLO and RCNN on real world videos",
  "description": "Bringing theory to experiment is cool. We can easily train models in colab and find the results in minutes.",
  "tag": "computer-vision"
}
</pre>
Now we can load our data into a Pandas DataFrame.
```python linenums="1"
# Create dataframe
df = pd.DataFrame(projects)
print (f"{len(df)} projects")
df.head(5)
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
      <th>0</th>
      <td>6</td>
      <td>2020-02-20 06:43:18</td>
      <td>Comparison between YOLO and RCNN on real world...</td>
      <td>Bringing theory to experiment is cool. We can ...</td>
      <td>computer-vision</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>2020-02-20 06:47:21</td>
      <td>Show, Infer &amp; Tell: Contextual Inference for C...</td>
      <td>The beauty of the work lies in the way it arch...</td>
      <td>computer-vision</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>2020-02-24 16:24:45</td>
      <td>Awesome Graph Classification</td>
      <td>A collection of important graph embedding, cla...</td>
      <td>graph-learning</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15</td>
      <td>2020-02-28 23:55:26</td>
      <td>Awesome Monte Carlo Tree Search</td>
      <td>A curated list of Monte Carlo tree search papers...</td>
      <td>reinforcement-learning</td>
    </tr>
    <tr>
      <th>4</th>
      <td>19</td>
      <td>2020-03-03 13:54:31</td>
      <td>Diffusion to Vector</td>
      <td>Reference implementation of Diffusion2Vec (Com...</td>
      <td>graph-learning</td>
    </tr>
  </tbody>
</table>
</div></div>
</pre>

```python linenums="1"
df = df[df.tag.notnull()]  # remove projects with no tag
```

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

### Tags

We're also going to be using an [auxiliary dataset](https://github.com/GokuMohandas/Made-With-ML/blob/main/datasets/tags.json){:target="_blank"} which contains a collection of all the tags that are currently relevant to us.

```python linenums="1"
# Load tags
url = "https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/tags.json"
tags_dict = {}
for item in json.loads(urlopen(url).read()):
    key = item.pop("tag")
    tags_dict[key] = item
print (f"{len(tags_dict)} tags")
```
<pre class="output">
4 tags
</pre>
```python linenums="1"
@widgets.interact(tag=list(tags_dict.keys()))
def display_tag_details(tag="computer-vision"):
    print (json.dumps(tags_dict[tag], indent=2))
```
<pre class="output">
"computer-vision": {
  "aliases": [
    "cv",
    "vision"
  ]
}
</pre>

> It's important that this auxillary information about our tags resides in a separate location. This way, everyone can use the same source of truth as it's [versioned](versioning.md){:target="_blank"} and kept up-to-date.

Let's answer a few key questions using EDA.

```python linenums="1"
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

We can see that `reinforcement-learning` and `time-series` is not our list of tags (`tags_dict`). These out of scope classes (and `graph-learning`) also don't have a lot of data points. We'll keep these details in mind when labeling and preprocessing our data.

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

## Resources
- [Fundamentals of Data Visualization](https://clauswilke.com/dataviz/){:target="_blank"}
- [Data Viz](https://armsp.github.io/covidviz/){:target="_blank"}

<!-- Citation -->
{% include "cite.md" %}