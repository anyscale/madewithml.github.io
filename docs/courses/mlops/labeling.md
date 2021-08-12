---
template: lesson.html
title: Data Labeling
description: Labeling our data with intention before using it to construct our ML systems.
keywords: labeling, annotation, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/mlops.png
repository: https://github.com/GokuMohandas/MLOps
notebook: https://colab.research.google.com/github/GokuMohandas/MLOps/blob/main/notebooks/tagifai.ipynb
---

{% include "styles/lesson.md" %}

## Intuition

Labeling (or annotation) is the process of identifying the inputs and outputs that are **worth** modeling (*not* just what could be modeled).

- use objective as a guide to determine the necessary signals
- explore creating new signals (via combining data, collecting new data, etc.)
- iteratively add more features to control complexity and effort

!!! warning
    Be careful not to include features in the dataset that will not be available during inference time, causing *data leakage*.

It's also the phase where we can use our deep understanding of the problem, processes, constraints and domain expertise to:

- augment the training data split
- enhance using auxiliary data
- simplify using constraints
- remove noisy samples

And it isn't just about identifying and labeling our initial dataset but also involves thinking about how to make the labeling process more efficient as our dataset grows.

- who will labeling new (streaming) data
- what tools will be used to accelerate the labeling process (ie. labeling functions)
- what workflows will be established to track the labeling process

!!! note
    We should always have  multiple labelers working on an overlap amongst the samples so we can easy discover labeling inconsistencies. A meaningful *inter-labeler discrepancy* (>2%) indicates that the labeling task is subjective and requires more explicit labeling criteria / instructions.

## Datasets
- [projects.json](https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/datasets/projects.json){:target="_blank"}: projects with title, description and tags (cleaned by mods).
- [tags.json](https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/datasets/tags.json){:target="_blank"}: tags used in dropdown to aid autocompletion.

!!! note
    We'll have a small GitHub Action that runs on a schedule (cron) to constantly update these datasets over time. We'll learn about how these work when we get to the CI/CD lesson.

Recall that our objective was to augment authors to add the appropriate tags for their project so the community can discover them. So we want to use the metadata provided in each project to determine what the relevant tags are. We'll want to start with the highly influential features and iteratively experiment with additional features.

## Load data
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
url = "https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/datasets/projects.json"
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
      <th>tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2020-02-17 06:30:41</td>
      <td>Machine Learning Basics</td>
      <td>A practical set of notebooks on machine learni...</td>
      <td>[code, tutorial, keras, pytorch, tensorflow, d...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2020-02-17 06:41:45</td>
      <td>Deep Learning with Electronic Health Record (E...</td>
      <td>A comprehensive look at recent machine learnin...</td>
      <td>[article, tutorial, deep-learning, health, ehr]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2020-02-20 06:07:59</td>
      <td>Automatic Parking Management using computer vi...</td>
      <td>Detecting empty and parked spaces in car parki...</td>
      <td>[code, tutorial, video, python, machine-learni...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2020-02-20 06:21:57</td>
      <td>Easy street parking using region proposal netw...</td>
      <td>Get a text on your phone whenever a nearby par...</td>
      <td>[code, tutorial, python, pytorch, machine-lear...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2020-02-20 06:29:18</td>
      <td>Deep Learning based parking management system ...</td>
      <td>Fastai provides easy to use wrappers to quickl...</td>
      <td>[code, tutorial, fastai, deep-learning, parkin...</td>
    </tr>
  </tbody>
</table>
</div></div>
</pre>

The reason we want to iteratively add more features is because it introduces more complexity and effort. For example, extracting the relevant HTML from the URLs is not trivial but recall that we want to *close the loop* with a simple solution first. We're going to use just the title and description because we hypothesize that the project's core concepts will be there whereas the details may have many other keywords.

!!! note
    Over time, our dataset will grow and we'll need to label new data. So far, we had a team of moderators clean the existing data but we'll need to establish proper workflow to make this process easier and reliable. Typically, we'll use collaborative UIs where annotators can fix errors, etc. and then use a tool like [Airflow](https://airflow.apache.org/){:target="_blank"} or [KubeFlow Pipelines](https://www.kubeflow.org/docs/components/pipelines/overview/pipelines-overview/){:target="_blank"} for workflow / pipeline orchestration to know when new data is ready to be labeled and also when it's ready to be used for modeling.

## Auxiliary data

We're also going to be using an [auxiliary dataset](https://github.com/GokuMohandas/MadeWithML/blob/main/datasets/tags.json){:target="_blank"} which contains a collection of all the tags with their aliases and parent/child relationships. This auxiliary dataset was used by our application to automatically add the relevant parent tags when the child tags were present.

```python linenums="1"
# Load tags
url = "https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/datasets/tags.json"
tags = json.loads(urlopen(url).read())
tags_dict = {}
for item in tags:
    key = item.pop("tag")
    tags_dict[key] = item
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


## Data imbalance

With our datasets, we may often notice a data imbalance problem where a range of continuous values (regression) or certain classes (classification) may have insufficient amounts of data to learn from. This becomes a major issue when training because the model will learn to generalize to the data available and perform poorly on regions where the data is sparse. There are several techniques to mitigate data imbalance, including [resampling](https://github.com/scikit-learn-contrib/imbalanced-learn){:target="_blank"} (oversampling from minority classes / undersampling from majority classes), account for the [data distributions via the loss function](baselines.md#data-imbalance){:target="_blank"} (since that drives the learning process), etc.

!!! note
    While these data imbalance mitigation techniques will allow our model to perform, the best long term approach is to directly address the imbalance issue. Identify which areas of the data need more samples and go collect them! This becomes a more more robust approach compared to focusing the model to learn from repeated samples or ignoring samples.



## Libraries

We could have used the user provided tags as our labels but what if the user added a wrong tag or forgot to add a relevant one. To remove this dependency on the user to provide the gold standard labels, we can leverage labeling tools and platforms. These tools allow for quick and organized labeling of the dataset to ensure its quality. And instead of starting from scratch and asking our labeler to provide all the relevant tags for a given project, we can provide the author's original tags and ask the labeler to add / remove as necessary. The specific labeling tool may be something that needs to be custom built or leverages something from the ecosystem.

### General
- [Scale AI](https://scale.com/){:target="_blank"}: the data platform for high quality training and validation data for AI applications.
- [Label Studio](https://github.com/heartexlabs/label-studio){:target="_blank"}: a multi-type data labeling and annotation tool with standardized output format.
- [Universal Data Tool](https://github.com/UniversalDataTool/universal-data-tool){:target="_blank"}: collaborate and label any type of data, images, text, or documents in an easy web interface or desktop app.
- [Prodigy](https://github.com/explosion/prodigy-recipes){:target="_blank"}: recipes for the Prodigy, our fully scriptable annotation tool.
- [Superintendent](https://github.com/janfreyberg/superintendent){:target="_blank"}: an ipywidget-based interactive labelling tool for your data to enable active learning.
### Natural language processing
- [Doccano](https://github.com/doccano/doccano){:target="_blank"}: an open source text annotation tool for text classification, sequence labeling and sequence to sequence tasks.
- [BRAT](https://github.com/nlplab/brat){:target="_blank"}: a rapid annotation tool for all your textual annotation needs.
### Computer vision
- [LabelImg](https://github.com/tzutalin/labelImg){:target="_blank"}: a graphical image annotation tool and label object bounding boxes in images.
- [CVAT](https://github.com/openvinotoolkit/cvat){:target="_blank"}: a free, online, interactive video and image annotation tool for computer vision.
- [VoTT](https://github.com/Microsoft/VoTT){:target="_blank"}: an electron app for building end-to-end object detection models from images and videos.
- [makesense.ai](https://github.com/SkalskiP/make-sense){:target="_blank"}: a free to use online tool for labelling photos.
- [remo](https://github.com/rediscovery-io/remo-python){:target="_blank"}: an app for annotations and images management in computer vision.
- [Labelai](https://github.com/aralroca/labelai){:target="_blank"}: an online tool designed to label images, useful for training AI models.
### Audio
- [Audino](https://github.com/midas-research/audino){:target="_blank"}: an open source audio annotation tool for voice activity detection (VAD), diarization, speaker identification, automated speech recognition, emotion recognition tasks, etc.
- [audio-annotator](https://github.com/CrowdCurio/audio-annotator){:target="_blank"}: a JavaScript interface for annotating and labeling audio files.
- [EchoML](https://github.com/ritazh/EchoML){:target="_blank"}: a web app to play, visualize, and annotate your audio files for machine learning.
### Miscellaneous
- [MedCAT](https://github.com/CogStack/MedCAT){:target="_blank"}: a medical concept annotation tool that can extract information from Electronic Health Records (EHRs) and link it to biomedical ontologies like SNOMED-CT and UMLS.


## Active learning

Even with a powerful labeling tool and established workflows, it's easy to see how involved and expensive labeling can be. Therefore, many teams employ active learning to iteratively label the dataset and evaluate the model.

In active learning, you first provide a small number of labelled examples. The model is trained on this "seed" dataset. Then, the model "asks questions" by selecting the unlabeled data points it is unsure about, so the human can "answer" the questions by providing labels for those points. The model updates again and the process is repeated until the performance is good enough. By having the human iteratively teach the model, it's possible to make a better model, in less time, with much less labelled data.

1. Label a small, initial dataset to train a model.
2. Ask the trained model to predict on some unlabeled data.
3. Determine which new data points to label from the unlabeled data based on:
    - entropy over the predicted class probabilities
    - samples with lowest predicted, [calibrated](https://arxiv.org/abs/1706.04599){:target="_blank"}, confidence (uncertainty sampling)
    - discrepancy in predictions from an ensemble of trained models
4. Repeat until the desired performance is achieved.

<div class="ai-center-all">
    <img src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/labeling/active_learning.png" width="700" alt="active learning">
</div>
<div class="ai-center-all mb-3">
  <small><a href="http://burrsettles.com/pub/settles.activelearning.pdf" target="_blank">Active Learning Literature Survey</a></small>
</div>

### Libraries
- [modAL](https://github.com/modAL-python/modAL){:target="_blank"}: a modular active learning framework for Python.
- [libact](https://github.com/ntucllab/libact){:target="_blank"}: pool-based active learning in Python.
- [ALiPy](https://github.com/NUAA-AL/ALiPy){:target="_blank"}: active learning python toolbox, which allows users to conveniently evaluate, compare and analyze the performance of active learning methods.


## Weak supervision

If we had samples that needed labeling or if we simply wanted to validate existing labels, we can use weak supervision to generate labels as opposed to hand labeling all of them. We could utilize weak supervision via [labeling functions](https://www.snorkel.org/use-cases/01-spam-tutorial){:target="_blank"} to label our existing and new data. We can create constructs based on keywords, pattern expressions, knowledge bases and generalized models to create these labeling functions to label our data. And we can add to the labeling functions over time and even mitigate conflicts amongst the different labeling functions.

```python linenums="1"
from snorkel.labeling import labeling_function

@labeling_function()
def contains_tensorflow(text):
    condition = any(tag in text.lower() for tag in ("tensorflow", "tf"))
    return "tensorflow" if condition else None
```

!!! note
    An easy way to validate our labels (before modeling) is to use our auxillary datasets to create labeling functions for the different classes. Then we can look for false positives and negatives to identify potentially mislabeled samples. We'll actually implement a similar kind of inspection approach, but using a trained model as a heuristic, in our [dashboards lesson](dashboard.md#inspection){:target="_blank"}.

## Iteration

Labeling isn't just a one time event or something we repeat identically. As new data is available, we'll want to strategically label the appropriate samples and improve [slices](testing.md#evaluation){:target="_blank"} of our data that are lacking in [quality](../foundations/data-quality.md){:target="_blank"}. In fact, there's an entire workflow related to labeling that is initiated when we want to iterate. We'll learn more about this iterative labeling process in our [continual learning lesson](continual-learning.md){:target="_blank"}.

## Resources
- [Human in the Loop: Deep Learning without Wasteful Labelling](https://oatml.cs.ox.ac.uk/blog/2019/06/24/batchbald.html){:target="_blank"}
- [Harnessing Organizational Knowledge for Machine Learning](https://ai.googleblog.com/2019/03/harnessing-organizational-knowledge-for.html){:target="_blank"}

<!-- Citation -->
{% include "cite.md" %}