---
layout: page
title: Style · Made With ML
description: Different styling options for content.
display-title: false
permalink: /style/
---

<!-- Header -->
<div class="row">
  <div class="col-12 mr-auto">
    <h1 class="page-title">{{ page.title | split: " · " | first }}</h1>
  </div>
</div>
<hr class="mt-0">

<!-- Text -->
Different styling options for content.

<!-- Text styles -->
### Text styling
- **bolded text**
- *Italicized text*
- <u>underlined text</u>
- <small>small text</small>
- This is a `tag`
- This is a [link](https://madewithml.com/){:target="_blank"}

<!-- Block quote -->
> This is a block quote

<!-- Alert -->
<div class="alert info mt-4" role="alert">
  <span style="text-align: left;">
    <i class="fas fa-info-circle mr-1"></i> Connect with the author, <i>Goku Mohandas</i>, on
    <a href="https://twitter.com/GokuMohandas" target="_blank">Twitter</a> and
    <a href="https://www.linkedin.com/in/goku" target="_blank">LinkedIn</a> for
    interactive conversations on upcoming lessons.
  </span>
</div>

### Code

**What are they?** Baselines are simple benchmarks which pave the way for iterative deve...

```python
from collections import Counter, OrderedDict
import ipywidgets as widgets
import itertools
import json
import pandas as pd
from urllib.request import urlopen
```
```python
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
```python
@widgets.interact(tag=list(tags_dict.keys()))
def display_tag_details(tag='question-answering'):
    print (json.dumps(tags_dict[tag], indent=2))
```

### MathJax

So far we've treated the words in our input text as isolated tokens and we haven't really captured any meaning between tokens. Let's use term frequency–inverse document frequency (**TF-IDF**) to capture the significance of a token to a particular input with respect to all the inputs.

$$ w_{i, j} = {tf}_{i, j} * log(\frac{N}{df}_i) $$

<div class="ai-center-all">
<table class="mathjax-table">
  <tbody>
    <tr>
      <th>$$ w_{i, j} $$</th>
      <th>$$ \text{tf-idf score for term i in document j} $$</th>
    </tr>
    <tr>
      <td>$$ {tf}_{i, j} $$</td>
      <td>$$ \text{num of times term i appear in document j} $$</td>
    </tr>
    <tr>
      <td>$$ N $$</td>
      <td>$$ \text{total num of documents} $$</td>
    </tr>
    <tr>
      <td>$$ {df}_i $$</td>
      <td>$$ \text{num of documents with token i} $$</td>
    </tr>
  </tbody>
</table>
</div>

### Plots
<div id="tags_per_project" class="ai-center-all">
  <img src="data:image/png;base64,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">
</div>

### Dataframes

<div class="output_subarea output_html rendered_html">
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>(15,)</th>
      <th>(19,)</th>
      <th>(33,)</th>
      <th>(21,)</th>
      <th>(32,)</th>
      <th>(14,)</th>
      <th>(2,)</th>
      <th>(20,)</th>
      <th>(24,)</th>
      <th>(5,)</th>
      <th>(16,)</th>
      <th>(9,)</th>
      <th>(22,)</th>
      <th>(12,)</th>
      <th>(23,)</th>
      <th>(0,)</th>
      <th>(25,)</th>
      <th>(34,)</th>
      <th>(28,)</th>
      <th>(10,)</th>
      <th>(26,)</th>
      <th>(27,)</th>
      <th>(13,)</th>
      <th>(17,)</th>
      <th>(3,)</th>
      <th>(1,)</th>
      <th>(7,)</th>
      <th>(11,)</th>
      <th>(18,)</th>
      <th>(4,)</th>
      <th>(6,)</th>
      <th>(29,)</th>
      <th>(8,)</th>
      <th>(31,)</th>
      <th>(30,)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>train</th>
      <td>314</td>
      <td>37</td>
      <td>26</td>
      <td>26</td>
      <td>145</td>
      <td>33</td>
      <td>274</td>
      <td>191</td>
      <td>41</td>
      <td>55</td>
      <td>26</td>
      <td>56</td>
      <td>34</td>
      <td>41</td>
      <td>40</td>
      <td>90</td>
      <td>44</td>
      <td>28</td>
      <td>136</td>
      <td>44</td>
      <td>30</td>
      <td>31</td>
      <td>63</td>
      <td>45</td>
      <td>64</td>
      <td>27</td>
      <td>50</td>
      <td>34</td>
      <td>21</td>
      <td>33</td>
      <td>24</td>
      <td>24</td>
      <td>32</td>
      <td>33</td>
      <td>23</td>
    </tr>
    <tr>
      <th>val</th>
      <td>58</td>
      <td>8</td>
      <td>4</td>
      <td>2</td>
      <td>29</td>
      <td>8</td>
      <td>53</td>
      <td>33</td>
      <td>7</td>
      <td>9</td>
      <td>4</td>
      <td>11</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
      <td>20</td>
      <td>7</td>
      <td>2</td>
      <td>42</td>
      <td>13</td>
      <td>12</td>
      <td>4</td>
      <td>14</td>
      <td>14</td>
      <td>17</td>
      <td>3</td>
      <td>7</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>5</td>
      <td>7</td>
      <td>10</td>
      <td>7</td>
      <td>4</td>
    </tr>
    <tr>
      <th>test</th>
      <td>57</td>
      <td>6</td>
      <td>9</td>
      <td>4</td>
      <td>22</td>
      <td>10</td>
      <td>61</td>
      <td>34</td>
      <td>9</td>
      <td>11</td>
      <td>3</td>
      <td>11</td>
      <td>6</td>
      <td>4</td>
      <td>8</td>
      <td>10</td>
      <td>9</td>
      <td>9</td>
      <td>35</td>
      <td>7</td>
      <td>6</td>
      <td>5</td>
      <td>16</td>
      <td>10</td>
      <td>25</td>
      <td>11</td>
      <td>16</td>
      <td>11</td>
      <td>6</td>
      <td>5</td>
      <td>5</td>
      <td>9</td>
      <td>9</td>
      <td>6</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div></div>

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