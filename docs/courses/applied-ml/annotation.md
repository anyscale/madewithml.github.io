---
description: Preparing and labeling our data for exploration.
image: https://madewithml.com/static/images/applied_ml.png
---

:octicons-mark-github-16: [Repository](https://github.com/GokuMohandas/applied-ml){:target="_blank"} · :octicons-book-24: [Notebook](https://colab.research.google.com/github/GokuMohandas/applied-ml/blob/main/notebooks/tagifai.ipynb){:target="_blank"}

Preparing and labeling our data for exploration.

## Intuition
Annotation is the process of identifying the inputs and outputs that are **worth** modeling (*not* just what could be modeled).

- use objective as a guide to determine the necessary signals
- explore creating new signals (via combining data, collecting new data, etc.)
-  iteratively add more features to control complexity and effort

!!! warning
    Be careful not to include features in the dataset that will not be available during inference time, causing *data leakage*.

It's also the phase where we can use our deep understanding of the problem, processes, constraints and domain expertise to:

- enhance using auxiliary data
- simplify using constraints

And it isn't just about identifying and labeling our initial dataset but also involves thinking about how to make the annotation process more efficient as our dataset grows.

- who will annotate new (streaming) data
- what tools will be used to accelerate the annotation process
- what workflows will be established to track the annotation process

!!! note
    You should have overlaps where different annotators are working on the same samples. A meaningful *inter-labeler discrepancy* (>2%) indicates that the annotation task is subjective and requires more explicit labeling criteria.

## Application

### Datasets
- [projects.json](https://raw.githubusercontent.com/GokuMohandas/applied-ml/main/datasets/projects.json){:target="_blank"}: projects with title, description and tags (cleaned by mods).
- [projects_detailed.json](https://raw.githubusercontent.com/GokuMohandas/applied-ml/main/datasets/projects_detailed.json){:target="_blank"}: projects with full-text details and additional URLs.
- [tags.json](https://raw.githubusercontent.com/GokuMohandas/applied-ml/main/datasets/tags.json){:target="_blank"}: tags used in dropdown to aid autocompletion.

!!! note
    We'll have a small GitHub Action that runs on a schedule (cron) to constantly update these datasets over time. We'll learn about how these work when we get to the CI/CD lesson.

Recall that our objective was to augment authors to add the appropriate tags for their project so the community can discover them. So we want to use the metadata provided in each project to determine what the relevant tags are. We'll want to start with the highly influential features and iteratively experiment with additional features:

- title + description
- ^ + details
- ^ + relevant html text from URLs

### Load data
We'll first load our dataset from the JSON file.

```python linenums="1"
from collections import Counter, OrderedDict
import ipywidgets as widgets
import itertools
import json
import pandas as pd
from urllib.request import urlopen
```
```python linenums="1"
# Load projects
url = "https://raw.githubusercontent.com/madewithml/datasets/main/projects.json"
projects = json.loads(urlopen(url).read())
print (json.dumps(projects[-305], indent=2))
```
<pre class="output">
{
  "id": 324,
  "title": "AdverTorch",
  "description": "A Toolbox for Adversarial Robustness Research",
  "tags": [
    "code",
    "library",
    "security",
    "adversarial-learning",
    "adversarial-attacks",
    "adversarial-perturbations"
  ]
}
</pre>
Now we can load our data into a Pandas DataFrame.
```python linenums="1"
# Create dataframe
df = pd.DataFrame(projects)
print (f"{len(df)} projects")
df.head(5)
```
<div class="output_subarea output_html rendered_html"><div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>title</th>
      <th>description</th>
      <th>tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2438</td>
      <td>How to Deal with Files in Google Colab: What Y...</td>
      <td>How to supercharge your Google Colab experienc...</td>
      <td>[article, google-colab, colab, file-system]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2437</td>
      <td>Rasoee</td>
      <td>A powerful web and mobile application that ide...</td>
      <td>[api, article, code, dataset, paper, research,...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2436</td>
      <td>Machine Learning Methods Explained (+ Examples)</td>
      <td>Most common techniques used in data science pr...</td>
      <td>[article, deep-learning, machine-learning, dim...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2435</td>
      <td>Top “Applied Data Science” Papers from ECML-PK...</td>
      <td>Explore the innovative world of Machine Learni...</td>
      <td>[article, deep-learning, machine-learning, adv...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2434</td>
      <td>OpenMMLab Computer Vision</td>
      <td>MMCV is a python library for CV research and s...</td>
      <td>[article, code, pytorch, library, 3d, computer...</td>
    </tr>
  </tbody>
</table>
</div></div>


The reason we want to iteratively add more features is because it introduces more complexity and effort. For example, extracting the relevant HTML from the URLs is not trivial but recall that we want to *close the loop* with a simple solution first. We're going to use just the title and description because we hypothesize that the project's core concepts will be there whereas the details may have many other keywords.

### Auxiliary data

We're also going to be using an [auxiliary dataset](https://raw.githubusercontent.com/madewithml/datasets/main/tags.json) which contains a collection of all the tags with their aliases and parent/child relationships.
```python linenums="1"
# Load tags
url = "https://raw.githubusercontent.com/madewithml/datasets/main/tags.json"
tags_dict = OrderedDict(json.loads(urlopen(url).read()))
print (f"{len(tags_dict)} tags")
```
<pre class="output">
400
</pre>
```python linenums="1"
@widgets.interact(tag=list(tags_dict.keys()))
def display_tag_details(tag='question-answering'):
    print (json.dumps(tags_dict[tag], indent=2))
```
<pre class="output">
"question-answering": {
    "aliases": [
      "qa"
    ],
    "parents": [
      "natural-language-processing"
    ]
  }
</pre>

### Features
We could use a project's title and description separately as features but we'll combine them to create one input feature.
```python linenums="1"
# Input
df['text'] = df.title + " " + df.description
```

### Constraints
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
```python linenums="1"
# Remove projects with no more remaining relevant tags
df = df[df.tags.map(len) > 0]
print (f"{len(df)} projects")
```
<pre class="output">
1444 projects
</pre>

Over time, our dataset will grow and we'll need to label new data. So far, we had a team of moderators clean the existing data but we'll need to establish proper workflow to make this process easier and reliable. Typically, we'll use collaborative UIs where annotators can fix errors, etc. and then use a tool like [Airflow](https://airflow.apache.org/){:target="_blank"} for workflow management to know when new data is ready to be annotated and also when it's ready to be used for modeling.

!!! note
    In the next section we'll be performing exploratory data analysis (EDA) on our labeled dataset. However, the order of the `annotation` and `EDA` steps can be reversed depending on how well the problem is defined. If you're unsure about what inputs and outputs are worth mapping, use can use EDA to figure it out.

## Resources
- [Human in the Loop: Deep Learning without Wasteful Labelling](https://oatml.cs.ox.ac.uk/blog/2019/06/24/batchbald.html){:target="_blank"}
- [Harnessing Organizational Knowledge for Machine Learning](https://ai.googleblog.com/2019/03/harnessing-organizational-knowledge-for.html){:target="_blank"}
