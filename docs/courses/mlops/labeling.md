---
template: lesson.html
title: Data Labeling
description: Labeling our data with intention before using it to construct our ML systems.
keywords: labeling, annotation, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/mlops.png
repository: https://github.com/GokuMohandas/mlops-course
notebook: https://github.com/GokuMohandas/mlops-course/blob/main/notebooks/tagifai.ipynb
---

{% include "styles/lesson.md" %}

## Intuition

Labeling (or annotation) is the process of identifying the inputs and outputs that are **worth** modeling (*not* just what could be modeled).

- use objective as a guide to determine the necessary signals.
- explore creating new signals (via combining features, collecting new data, etc.).
- iteratively add more features to justify complexity and effort.

!!! warning
    Be careful not to include features in the dataset that will not be available during prediction, causing [data leaks](feature-store.md#intuition){:target="_blank"}.

!!! question "What else can we learn?"
    It's not just about identifying and labeling our initial dataset. What else can we learn from it?

    ??? quote "Show answer"

        It's also the phase where we can use our deep understanding of the problem, processes, constraints and domain expertise to:

            - augment the training data split
            - enhance using auxiliary data
            - simplify using constraints
            - remove noisy samples
            - improve the labeling process

## Process

Regardless of whether we have a custom labeling platform or we choose a generalized platform, the process of labeling and all it's related workflows (QA, data import/export, etc.) follow a similar approach.

### Preliminary steps
- `#!js [WHAT]` Decide what needs to be labeled:
    - identify natural labels you may already have (ex. time-series)
    - consult with domain experts to ensure you're labeling the appropriate signals
    - decide on the appropriate labels (and [hierarchy](https://aws.amazon.com/blogs/machine-learning/creating-hierarchical-label-taxonomies-using-amazon-sagemaker-ground-truth/){:target="_blank"}) for your task
- `#!js [WHERE]` Design the labeling interface:
    - intuitive, data modality dependent and quick (keybindings are a must!)
    - avoid option paralysis by allowing the labeler to dig deeper or suggesting likely labels
    - measure and resolve inter-labeler discrepancy
- `#!js [HOW]` Compose labeling instructions:
    - examples of each labeling scenario
    - course of action for discrepancies

<div class="ai-center-all">
    <img src="/static/images/mlops/labeling/ui.png" width="400" alt="labeling view">
</div>
<div class="ai-center-all mt-2">
  <small>Multi-label text classification for our task using  <a href="https://prodi.gy/" target="_blank">Prodigy</a> (labeling + QA)</small>
</div>

### Workflow setup
- Establish data pipelines:
    - `#!js [IMPORT]` *new data* for annotation
    - `#!js [EXPORT]` *annotated data* for QA, [testing](testing.md#data){:target="_blank"}, modeling, etc.
- Create a quality assurance (QA) workflow:
    - separate from labeling workflow (no bias)
    - communicates with labeling workflow to escalate errors

<div class="ai-center-all">
    <img src="/static/images/mlops/labeling/workflow.png" width="800" alt="labeling workflow">
</div>

### Iterative setup
- Implement strategies to reduce labeling efforts
    - identify subsets of the data to label next using [active learning](#active-learning)
    - auto-label entire or parts of a dataset using [weak supervision](#weak-supervision)
    - focus labeling efforts on long tail of edge cases over time

## Labeled data

For the purpose of this course, our data is already labeled, so we'll perform a basic version of ELT (extract, load, transform) to construct the labeled dataset.

> In our [data-stack](data-stack.md){:target="_blank"} and [orchestration](orchestration.md){:target="_blank"} lessons, we'll construct a modern data stack and programmatically deliver high quality data via DataOps workflows.

- [projects.csv](https://github.com/GokuMohandas/Made-With-ML/tree/main/datasets/projects.csv): projects with id, created time, title and description.
- [tags.csv](https://github.com/GokuMohandas/Made-With-ML/tree/main/datasets/tags.csv): labels (tag category) for the projects by id.

Recall that our [objective](https://madewithml.com/courses/mlops/design#objectives) was to classify incoming content so that the community can discover them easily. These data assets will act as the training data for our first model.

### Extract

We'll start by extracting data from our sources (external CSV files). Traditionally, our data assets will be stored, versioned and updated in a database, warehouse, etc. We'll learn more about these different [data systems](data-stack.md){:target="_blank"} later, but for now, we'll load our data as a stand-alone CSV file.

```python linenums="1"
import pandas as pd
```

```python linenums="1"
# Extract projects
PROJECTS_URL = "https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/projects.csv"
projects = pd.read_csv(PROJECTS_URL)
projects.head(5)
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>2020-02-20 06:43:18</td>
      <td>Comparison between YOLO and RCNN on real world...</td>
      <td>Bringing theory to experiment is cool. We can ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>2020-02-20 06:47:21</td>
      <td>Show, Infer &amp; Tell: Contextual Inference for C...</td>
      <td>The beauty of the work lies in the way it arch...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>2020-02-24 16:24:45</td>
      <td>Awesome Graph Classification</td>
      <td>A collection of important graph embedding, cla...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15</td>
      <td>2020-02-28 23:55:26</td>
      <td>Awesome Monte Carlo Tree Search</td>
      <td>A curated list of Monte Carlo tree search papers...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>19</td>
      <td>2020-03-03 13:54:31</td>
      <td>Diffusion to Vector</td>
      <td>Reference implementation of Diffusion2Vec (Com...</td>
    </tr>
  </tbody>
</table>
</div></div>
</pre>

We'll also load the labels (tag category) for our projects.

```python linenums="1"
# Extract tags
TAGS_URL = "https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/tags.csv"
tags = pd.read_csv(TAGS_URL)
tags.head(5)
```
<pre class="output">
<div class="output_subarea output_html rendered_html output_result" dir="auto"><div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>tag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>computer-vision</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>computer-vision</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>graph-learning</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15</td>
      <td>reinforcement-learning</td>
    </tr>
    <tr>
      <th>4</th>
      <td>19</td>
      <td>graph-learning</td>
    </tr>
  </tbody>
</table>
</div></div>
</pre>

### Transform

Apply basic transformations to create our labeled dataset.

```python linenums="1"
# Join projects and tags
df = pd.merge(projects, tags, on="id")
df.head()
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

### Load

Finally, we'll load our transformed data locally so that we can use it for our machine learning application.

```python linenums="1"
# Save locally
df.to_csv("labeled_projects.csv", index=False)
```

## Libraries

We could have used the user provided tags as our labels but what if the user added a wrong tag or forgot to add a relevant one. To remove this dependency on the user to provide the gold standard labels, we can leverage labeling tools and platforms. These tools allow for quick and organized labeling of the dataset to ensure its quality. And instead of starting from scratch and asking our labeler to provide all the relevant tags for a given project, we can provide the author's original tags and ask the labeler to add / remove as necessary. The specific labeling tool may be something that needs to be custom built or leverages something from the ecosystem.

> As our platform grows, so too will our dataset and labeling needs so it's imperative to use the proper tooling that supports the workflows we'll depend on.

### General
- [Labelbox](https://labelbox.com/){:target="_blank"}: the data platform for high quality training and validation data for AI applications.
- [Scale AI](https://scale.com/){:target="_blank"}: data platform for AI that provides high quality training data.
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

!!! question "Generalized labeling solutions"

    What criteria should we use to evaluate what labeling platform to use?

    ??? quote "Show answer"

        It's important to pick a generalized platform that has all the major labeling features for your data modality with the capability to easily customize the experience.

        - how easy is it to connect to our data sources (DB, QA, etc.)?
        - how easy was it to make changes (new features, labeling paradigms)?
        - how securely is our data treated (on-prem, trust, etc.)

        However, as an industry trend, this balance between generalization and specificity is difficult to strike. So many teams put in the upfront effort to create bespoke labeling platforms or used industry specific, niche,  labeling tools.


## Active learning

Even with a powerful labeling tool and established workflows, it's easy to see how involved and expensive labeling can be. Therefore, many teams employ active learning to iteratively label the dataset and evaluate the model.

1. Label a small, initial dataset to train a model.
2. Ask the trained model to predict on some unlabeled data.
3. Determine which new data points to label from the unlabeled data based on:
    - entropy over the predicted class probabilities
    - samples with lowest predicted, [calibrated](https://arxiv.org/abs/1706.04599){:target="_blank"}, confidence (uncertainty sampling)
    - discrepancy in predictions from an ensemble of trained models
4. Repeat until the desired performance is achieved.

> This can be significantly more cost-effective and faster than labeling the entire dataset.

<div class="ai-center-all">
    <img src="/static/images/mlops/labeling/active_learning.png" width="700" alt="active learning">
</div>
<div class="ai-center-all mt-1 mb-3">
  <small><a href="http://burrsettles.com/pub/settles.activelearning.pdf" target="_blank">Active Learning Literature Survey</a></small>
</div>

### Libraries
- [modAL](https://github.com/modAL-python/modAL){:target="_blank"}: a modular active learning framework for Python.
- [libact](https://github.com/ntucllab/libact){:target="_blank"}: pool-based active learning in Python.
- [ALiPy](https://github.com/NUAA-AL/ALiPy){:target="_blank"}: active learning python toolbox, which allows users to conveniently evaluate, compare and analyze the performance of active learning methods.


## Weak supervision

If we had samples that needed labeling or if we simply wanted to validate existing labels, we can use weak supervision to generate labels as opposed to hand labeling all of them. We could utilize weak supervision via [labeling functions](https://www.snorkel.org/use-cases/01-spam-tutorial){:target="_blank"} to label our existing and new data, where we can create constructs based on keywords, pattern expressions, knowledge bases, etc. And we can add to the labeling functions over time and even mitigate conflicts amongst the different labeling functions. We'll use these labeling functions to create and evaluate slices of our data in the [evaluation lesson](evaluation.md#slices){:target="_blank"}.

```python linenums="1"
from snorkel.labeling import labeling_function

@labeling_function()
def contains_tensorflow(text):
    condition = any(tag in text.lower() for tag in ("tensorflow", "tf"))
    return "tensorflow" if condition else None
```

> An easy way to validate our labels (before modeling) is to use the aliases in our auxillary datasets to create labeling functions for the different classes. Then we can look for false positives and negatives to identify potentially mislabeled samples. We'll actually implement a similar kind of inspection approach, but using a trained model as a heuristic, in our [dashboards lesson](dashboard.md#inspection){:target="_blank"}.


## Iteration

Labeling isn't just a one time event or something we repeat identically. As new data is available, we'll want to strategically label the appropriate samples and improve [slices](testing.md#evaluation){:target="_blank"} of our data that are lacking in [quality](../foundations/data-quality.md){:target="_blank"}. Once new data is labeled, we can have workflows that are triggered to start the (re)training process to deploy a new version of our system.

<!-- Citation -->
{% include "styles/cite.md" %}