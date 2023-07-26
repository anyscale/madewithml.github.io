---
template: lesson.html
title: Distributed training
description: Training models on our prepared data to optimize on our objective.
keywords: training, distributed training, llms, modeling, pytorch, transformers, huggingface, mlops, machine learning
image: https://madewithml.com/static/images/mlops.png
repository: https://github.com/GokuMohandas/Made-With-ML
notebook: https://github.com/GokuMohandas/Made-With-ML/blob/main/notebooks/madewithml.ipynb
---

{% include "styles/lesson.md" %}

## Intuition

Now that we have our data prepared, we can start training our models to optimize on our objective. Ideally, we would start with the simplest possible baseline and slowly add complexity to our models:

1. Start with a random (chance) model.
> Since we have four classes, we may expect a random model to be correct around 25% of the time but recall that [not](exploratory-data-analysis.md#tag-distribution){:target="_blank"} all of our classes have equal counts.
2. Develop a rule-based approach using if-else statements, regular expressions, etc.
> We could build a list of common words for each class and if a word in the input matches a word in the list, we can predict that class.
3. Slowly add complexity by *addressing* limitations and *motivating* representations and model architectures.
> We could start with a simple term frequency ([TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html){:target="_blank"}) mode and then move onto embeddings with [CNNs](../foundations/convolutional-neural-networks.md){:target="_blank"}, [RNNs](../foundations/recurrent-neural-networks.md){:target="_blank"}, [Transformers](../foundations/transformers.md){:target="_blank"}, etc.
4. Weigh *tradeoffs* (performance, latency, size, etc.) between performant baselines.
5. Revisit and iterate on baselines as your dataset grows and new model architectures are developed.

We're going to skip straight to step 3 of developing a complex model because this task involves unstructured data and rule-based systems are not well suited for this. And with the increase adoption of **large language models (LLMs)** as a proven model architecture for NLP tasks, we'll fine-tune a pretrained LLM on our dataset.

!!! tip "Iterate on the data"
    Instead of using a fixed dataset and iterating on the models, we could keep the model constant and iterate on the dataset. This is useful to improve the quality of our datasets.

    - remove or fix data samples (false positives & negatives)
    - prepare and transform features
    - expand or consolidate classes
    - incorporate auxiliary datasets
    - identify unique slices to boost

## Distributed training

With the rapid increase in data (unstructured) and model sizes (ex. LLMs), it's becoming increasingly difficult to train models on a single machine. We need to be able to distribute our training across multiple machines in order to train our models in a reasonable amount of time. And we want to be able to do this **without** having to:

- set up a cluster by individually (and painstakingly) provisioning compute resources (CPU, GPU, etc.)
- writing complex code to distribute our training across multiple machines
- worry about communication and resource utilization between our different distributed compute resources
- worry about fault tolerance and recovery from our large training workloads

To address all of these concerns, we'll be using [Ray Train](https://docs.ray.io/en/master/train/train.html){:target="_blank"} here in order to create a training workflow that can scale across multiple machines. While there are many options to choose from for distributed training, such as Pytorch [Distributed Data Parallel (DDP)](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html){:target="_blank"}, [Horovod](https://horovod.readthedocs.io/en/stable/ray_include.html){:target="_blank"}, etc., none of them allow us to scale across *different* machines with ease and do so with [*minimal*](https://docs.ray.io/en/latest/ray-air/examples/convert_existing_pytorch_code_to_ray_air.html){:target="_blank"} changes to our single-machine training code as Ray does.

!!! note "Primer on distributed training"
    With distributed training, there will be a head node that's responsible for orchestrating the training process. While the worker nodes that will be responsible for training the model and communicating results back to the head node. From a user's perspective, Ray abstracts away all of this complexity and we can simply define our training functionality with **minimal** changes to our code (as if we were training on a single machine).

## Generative AI

In this lesson, we're going to be fine-tuning a pretrained large language model (LLM) using our labeled dataset. The specific class of LLMs we'll be using is called [BERT](https://en.wikipedia.org/wiki/BERT_(language_model)){:target="_blank"}. Bert models are encoder-only models and are the gold-standard for supervised NLP tasks. However, you may be wondering how do all the (much larger) LLM, created for generative applications, fare ([GPT 4](https://openai.com/research/gpt-4){:target="_blank"}, [Falcon 40B](https://huggingface.co/tiiuae/falcon-40b){:target="_blank"}, [Llama 2](https://ai.meta.com/llama/){:target="_blank"}, etc.)?

> We chose the smaller BERT model for our course because it's easier to train and fine-tune. However, the workflow for fine-tuning the larger LLMs are quite similar as well. They do require much more compute but Ray abstracts away the scaling complexities involved with that.

!!! note
    All the code for this section can be found in our separate [benchmarks.ipynb](https://github.com/GokuMohandas/Made-With-ML/blob/main/notebooks/benchmarks.ipynb){:target="_blank"} notebook.

### Set up

You'll need to first sign up for an [OpenAI account](https://platform.openai.com/signup){:target="_blank"} and then grab your API key from [here](https://platform.openai.com/account/api-keys){:target="_blank"}.

```python linenums="1"
import openai
openai.api_key = "YOUR_API_KEY"
```

### Load data

We'll first load the our training and inference data into dataframes.

```python linenums="1"
import pandas as pd
```

```python linenums="1"
# Load training data
DATASET_LOC = "https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/dataset.csv"
train_df = pd.read_csv(DATASET_LOC)
train_df.head()
```

<div class="output_subarea output_html rendered_html output_result" dir="auto"><div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align:right">
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
      <td>other</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15</td>
      <td>2020-02-28 23:55:26</td>
      <td>Awesome Monte Carlo Tree Search</td>
      <td>A curated list of Monte Carlo tree search pape...</td>
      <td>other</td>
    </tr>
    <tr>
      <th>4</th>
      <td>25</td>
      <td>2020-03-07 23:04:31</td>
      <td>AttentionWalk</td>
      <td>A PyTorch Implementation of "Watch Your Step: ...</td>
      <td>other</td>
    </tr>
  </tbody>
</table>
</div></div>

```python linenums="1"
# Unique labels
tags = train_df.tag.unique().tolist()
tags
```

<pre class="output">
['computer-vision', 'other', 'natural-language-processing', 'mlops']
</pre>

```python linenums="1"
# Load inference dataset
HOLDOUT_LOC = "https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/holdout.csv"
test_df = pd.read_csv(HOLDOUT_LOC)
```

### Utilities

We'll define a few utility functions to make the OpenAI request and to store our predictions. While we could perform batch prediction by loading samples until the context length is reached, we'll just perform one at a time since it's not too many data points and we can have fully deterministic behavior (if you insert new data, etc.). We'll also added some reliability in case we overload the endpoints with too many request at once.

```python linenums="1"
import json
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
from sklearn.metrics import precision_recall_fscore_support
import time
from tqdm import tqdm
```

We'll first define what a sample call to the OpenAI endpoint looks like. We'll pass in:
- `system_content` that has information about how the LLM should behave.
- `assistant_content` for any additional context it should have for answering our questions.
- `user_content` that has our message or query to the LLM.
- `model` should specify which specific model we want to send our request to.

We can pass all of this information in through the [`openai.ChatCompletion.create`](https://platform.openai.com/docs/guides/gpt/chat-completions-api){:target="_blank"} function to receive our response.

```python linenums="1"
# Query OpenAI endpoint
system_content = "you only answer in rhymes"  # system content (behavior)
assistant_content = ""  # assistant content (context)
user_content = "how are you"  # user content (message)
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    messages=[
        {"role": "system", "content": system_content},
        {"role": "assistant", "content": assistant_content},
        {"role": "user", "content": user_content},
    ],
)
print (response.to_dict()["choices"][0].to_dict()["message"]["content"])
```

<pre class="output">
I'm doing just fine, so glad you ask,
Rhyming away, up to the task.
How about you, my dear friend?
Tell me how your day did ascend.
</pre>

Now, let's create a function that can predict tags for a given sample.

```python linenums="1"
def get_tag(model, system_content="", assistant_content="", user_content=""):
    try:
        # Get response from OpenAI
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "assistant", "content": assistant_content},
                {"role": "user", "content": user_content},
            ],
        )
        predicted_tag = response.to_dict()["choices"][0].to_dict()["message"]["content"]
        return predicted_tag

    except (openai.error.ServiceUnavailableError, openai.error.APIError) as e:
        return None
```

```python linenums="1"
# Get tag
model = "gpt-3.5-turbo-0613"
system_context = f"""
    You are a NLP prediction service that predicts the label given an input's title and description.
    You must choose between one of the following labels for each input: {tags}.
    Only respond with the label name and nothing else.
    """
assistant_content = ""
user_context = "Transfer learning with transformers: Using transformers for transfer learning on text classification tasks."
tag = get_tag(model=model, system_content=system_context, assistant_content=assistant_content, user_content=user_context)
print (tag)
```

<pre class="output">
natural-language-processing
</pre>

Next, let's create a function that can predict tags for a list of inputs.

```python linenums="1"
# List of dicts w/ {title, description} (just the first 3 samples for now)
samples = test_df[["title", "description"]].to_dict(orient="records")[:3]
samples
```
<pre class="output">
[{'title': 'Diffusion to Vector',
  'description': 'Reference implementation of Diffusion2Vec (Complenet 2018) built on Gensim and NetworkX. '},
 {'title': 'Graph Wavelet Neural Network',
  'description': 'A PyTorch implementation of "Graph Wavelet Neural Network" (ICLR 2019) '},
 {'title': 'Capsule Graph Neural Network',
  'description': 'A PyTorch implementation of "Capsule Graph Neural Network" (ICLR 2019).'}]
</pre>

```python linenums="1"
def get_predictions(inputs, model, system_content, assistant_content=""):
    y_pred = []
    for item in tqdm(inputs):
        # Convert item dict to string
        user_content = str(item)

        # Get prediction
        predicted_tag = get_tag(
            model=model, system_content=system_content,
            assistant_content=assistant_content, user_content=user_content)

        # If error, try again after pause (repeatedly until success)
        while predicted_tag is None:
            time.sleep(30)  # could also do exponential backoff
            predicted_tag = get_tag(
                model=model, system_content=system_content,
                assistant_content=assistant_content, user_content=user_content)

        # Add to list of predictions
        y_pred.append(predicted_tag)

    return y_pred
```

```python linenums="1"
# Get predictions for a list of inputs
get_predictions(inputs=samples, model=model, system_content=system_context)
```

<pre class="output">
100%|██████████| 3/3 [00:01<00:00,  2.96its]
['computer-vision', 'computer-vision', 'computer-vision']
</pre>

Next we'll define a function that can clean our predictions in the event that it's not the proper format or has hallucinated a tag outside of our expected tags.

```python linenums="1"
def clean_predictions(y_pred, tags, default="other"):
    for i, item in enumerate(y_pred):
        if item not in tags:  # hallucinations
            y_pred[i] = default
        if item.startswith("'") and item.endswith("'"):  # GPT 4 likes to places quotes
            y_pred[i] = item[1:-1]
    return y_pred
```

!!! tip
    Open AI has now released [function calling](https://openai.com/blog/function-calling-and-other-api-updates) and [custom instructions](https://openai.com/blog/custom-instructions-for-chatgpt) which is worth exploring to avoid this manual cleaning.

Next, we'll define a function that will plot our ground truth labels and predictions.

```python linenums="1"
def plot_tag_dist(y_true, y_pred):
    # Distribution of tags
    true_tag_freq = dict(Counter(y_true))
    pred_tag_freq = dict(Counter(y_pred))
    df_true = pd.DataFrame({"tag": list(true_tag_freq.keys()), "freq": list(true_tag_freq.values()), "source": "true"})
    df_pred = pd.DataFrame({"tag": list(pred_tag_freq.keys()), "freq": list(pred_tag_freq.values()), "source": "pred"})
    df = pd.concat([df_true, df_pred], ignore_index=True)

    # Plot
    plt.figure(figsize=(10, 3))
    plt.title("Tag distribution", fontsize=14)
    ax = sns.barplot(x="tag", y="freq", hue="source", data=df)
    ax.set_xticklabels(list(true_tag_freq.keys()), rotation=0, fontsize=8)
    plt.legend()
    plt.show()
```

And finally, we'll define a function that will combine all the utilities above to predict, clean and plot our results.

```python linenums="1"
def evaluate(test_df, model, system_content, assistant_content="", tags):
    # Predictions
    y_test = test_df.tag.to_list()
    test_samples = test_df[["title", "description"]].to_dict(orient="records")
    y_pred = get_predictions(
        inputs=test_samples, model=model,
        system_content=system_content, assistant_content=assistant_content)
    y_pred = clean_predictions(y_pred=y_pred, tags=tags)

    # Performance
    metrics = precision_recall_fscore_support(y_test, y_pred, average="weighted")
    performance = {"precision": metrics[0], "recall": metrics[1], "f1": metrics[2]}
    print(json.dumps(performance, indent=2))
    plot_tag_dist(y_true=y_test, y_pred=y_pred)
    return y_pred, performance
```

### Zero-shot learning

Now we're ready to start benchmarking our different LLMs with different context.

```python linenums="1"
y_pred = {"zero_shot": {}, "few_shot": {}}
performance = {"zero_shot": {}, "few_shot": {}}
```

We'll start with zero-shot learning which involves providing the model with the `system_content` that tells it how to behave but no examples of the behavior (no `assistant_content`).

```python linenums="1"
system_content = f"""
    You are a NLP prediction service that predicts the label given an input's title and description.
    You must choose between one of the following labels for each input: {tags}.
    Only respond with the label name and nothing else.
    """
```

```python linenums="1"
# Zero-shot with GPT 3.5
method = "zero_shot"
model = "gpt-3.5-turbo-0613"
y_pred[method][model], performance[method][model] = evaluate(
    test_df=test_df, model=model, system_content=system_content, tags=tags)
```

<pre class="output">
100%|██████████| 191/191 [11:01<00:00,  3.46s/it]
{
  "precision": 0.7919133278407181,
  "recall": 0.806282722513089,
  "f1": 0.7807530967691199
}
</pre>

<div class="ai-center-all">
    <img src="/static/images/mlops/training/zero_shot_35.png" width="600" alt="zero-shot GPT 3.5">
</div>

```python linenums="1"
# Zero-shot with GPT 4
method = "zero_shot"
model = "gpt-4-0613"
y_pred[method][model], performance[method][model] = evaluate(
    test_df=test_df, model=model, system_content=system_content, tags=tags)
```

<pre class="output">
100%|██████████| 191/191 [02:28<00:00,  1.29it/s]
{
  "precision": 0.9314722577069027,
  "recall": 0.9267015706806283,
  "f1": 0.9271956481845013
}
</pre>

<div class="ai-center-all">
    <img src="/static/images/mlops/training/zero_shot_4.png" width="600" alt="zero-shot GPT 4">
</div>

### Few-shot learning

Now, we'll be adding a `assistant_context` with a few samples from our training data for each class. The intuition here is that we're giving the model a few examples (few-shot learning) of what each class looks like so that it can learn to generalize better.

```python linenums="1"
# Create additional context with few samples from each class
num_samples = 2
additional_context = []
cols_to_keep = ["title", "description", "tag"]
for tag in tags:
    samples = train_df[cols_to_keep][train_df.tag == tag][:num_samples].to_dict(orient="records")
    additional_context.extend(samples)
additional_context
```

<pre class="output">
[{'title': 'Comparison between YOLO and RCNN on real world videos',
  'description': 'Bringing theory to experiment is cool. We can easily train models in colab and find the results in minutes.',
  'tag': 'computer-vision'},
 {'title': 'Show, Infer & Tell: Contextual Inference for Creative Captioning',
  'description': 'The beauty of the work lies in the way it architects the fundamental idea that humans look at the overall image and then individual pieces of it.\r\n',
  'tag': 'computer-vision'},
 {'title': 'Awesome Graph Classification',
  'description': 'A collection of important graph embedding, classification and representation learning papers with implementations.',
  'tag': 'other'},
 {'title': 'Awesome Monte Carlo Tree Search',
  'description': 'A curated list of Monte Carlo tree search papers with implementations. ',
  'tag': 'other'},
 {'title': 'Rethinking Batch Normalization in Transformers',
  'description': 'We found that NLP batch statistics exhibit large variance throughout training, which leads to poor BN performance.',
  'tag': 'natural-language-processing'},
 {'title': 'ELECTRA: Pre-training Text Encoders as Discriminators',
  'description': 'PyTorch implementation of the electra model from the paper: ELECTRA - Pre-training Text Encoders as Discriminators Rather Than Generators',
  'tag': 'natural-language-processing'},
 {'title': 'Pytest Board',
  'description': 'Continuous pytest runner with awesome visualization.',
  'tag': 'mlops'},
 {'title': 'Debugging Neural Networks with PyTorch and W&B',
  'description': 'A closer look at debugging common issues when training neural networks.',
  'tag': 'mlops'}]
</pre>

```python linenums="1"
# Add assistant context
assistant_content = f"""Here are some examples with the correct labels: {additional_context}"""
print (assistant_content)
```

<pre class="output">
Here are some examples with the correct labels: [{'title': 'Comparison between YOLO and RCNN on real world videos', ... 'description': 'A closer look at debugging common issues when training neural networks.', 'tag': 'mlops'}]
</pre>

!!! tip
    We could increase the number of samples by increasing the context length. We could also retrieve better few-shot samples by extracting examples from the training data that are similar to the current sample (ex. similar unique vocabulary).

```python linenums="1"
# Few-shot with GPT 3.5
method = "few_shot"
model = "gpt-3.5-turbo-0613"
y_pred[method][model], performance[method][model] = evaluate(
    test_df=test_df, model=model, system_content=system_content,
    assistant_content=assistant_content, tags=tags)
```

<pre class="output">
100%|██████████| 191/191 [04:18<00:00,  1.35s/it]
{
  "precision": 0.8435247936255214,
  "recall": 0.8586387434554974,
  "f1": 0.8447984162323493
}
</pre>

<div class="ai-center-all">
    <img src="/static/images/mlops/training/few_shot_35.png" width="600" alt="few-shot GPT 3.5">
</div>

```python linenums="1"
# Few-shot with GPT 4
method = "few_shot"
model = "gpt-4-0613"
y_pred[method][model], performance[method][model] = evaluate(
    test_df=test_df, model=model, system_content=few_shot_context,
    assistant_content=assistant_content, tags=tags)
```

<pre class="output">
100%|██████████| 191/191 [02:11<00:00,  1.46it/s]
{
  "precision": 0.9407759040163695,
  "recall": 0.9267015706806283,
  "f1": 0.9302632275594479
}
</pre>

<div class="ai-center-all">
    <img src="/static/images/mlops/training/few_shot_4.png" width="600" alt="few-shot GPT 4">
</div>

As we can see, few shot learning performs better than it's respective zero shot counter part. GPT 4 has had considerable improvements in reducing hallucinations but for our supervised task this comes at an expense of high precision but lower recall and f1 scores. When GPT 4 is not confident, it would rather predict `other`.

### OSS LLMs

So far, we've only been using closed-source models from OpenAI. While these are *currently* the gold-standard, there are many open-source models that are rapidly catching up ([Falcon 40B](https://huggingface.co/tiiuae/falcon-40b), [Llama 2](https://ai.meta.com/llama/), etc.). Before we see how these models perform on our task, let's first consider a few reasons why we should care about open-source models.

- **data ownership**: you can serve your models and pass data to your models, without having to share it with a third-party API endpoint.
- **fine-tune**: with access to our model's weights, we can actually fine-tune them, as opposed to experimenting with fickle prompting strategies.
- **optimization**: we have full freedom to optimize our deployed models for inference (ex. quantization, pruning, etc.) to reduce costs.

```python linenums="1"
# Coming soon in August!
```

### Results

Now let's compare all the results from our generative AI LLM benchmarks:

```python linenums="1"
print(json.dumps(performance, indent=2))
```

```json
{
  "zero_shot": {
    "gpt-3.5-turbo-0613": {
      "precision": 0.7919133278407181,
      "recall": 0.806282722513089,
      "f1": 0.7807530967691199
    },
    "gpt-4-0613": {
      "precision": 0.9314722577069027,
      "recall": 0.9267015706806283,
      "f1": 0.9271956481845013
    }
  },
  "few_shot": {
    "gpt-3.5-turbo-0613": {
      "precision": 0.8435247936255214,
      "recall": 0.8586387434554974,
      "f1": 0.8447984162323493
    },
    "gpt-4-0613": {
      "precision": 0.9407759040163695,
      "recall": 0.9267015706806283,
      "f1": 0.9302632275594479
    }
  }
}
```

And we can plot these on a bar plot to compare them visually.

```python linenums="1"
# Transform data into a new dictionary with four keys
by_model_and_context = {}
for context_type, models_data in performance.items():
    for model, metrics in models_data.items():
        key = f"{model}_{context_type}"
        by_model_and_context[key] = metrics
```

```python linenums="1"
# Extracting the model names and the metric values
models = list(by_model_and_context.keys())
metrics = list(by_model_and_context[models[0]].keys())

# Plotting the bar chart with metric scores on top of each bar
fig, ax = plt.subplots(figsize=(10, 4))
width = 0.2
x = range(len(models))

for i, metric in enumerate(metrics):
    metric_values = [by_model_and_context[model][metric] for model in models]
    ax.bar([pos + width * i for pos in x], metric_values, width, label=metric)
    # Displaying the metric scores on top of each bar
    for pos, val in zip(x, metric_values):
        ax.text(pos + width * i, val, f'{val:.3f}', ha='center', va='bottom', fontsize=9)

ax.set_xticks([pos + width for pos in x])
ax.set_xticklabels(models, rotation=0, ha='center', fontsize=8)
ax.set_ylabel('Performance')
ax.set_title('GPT Benchmarks')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.show()
```

<div class="ai-center-all">
  <img src="/static/images/mlops/training/benchmarks.png" width="800" alt="benchmarks">
</div>

Our best model is GPT 4 with few shot learning at an f1 score of ~93%. We will see, in the rest of the course, how fine-tuning an LLM with a proper training dataset to change the actual weights of the last N layers (as opposed to the hard prompt tuning here) will yield similar/slightly better results to GPT 4 (at a fraction of the model size and inference costs).

However, the best system might actually be a combination of using these few-shot hard prompt LLMs alongside fine-tuned LLMs. For example, our fine-tuned LLMs in the course will perform well when the test data is similar to the training data (similar distributions of vocabulary, etc.) but may not perform well on out of distribution. Whereas, these hard prompted LLMs, by themselves or augmented with additional context (ex. arXiv plugins in our case), could be used when our primary fine-tuned model is not so confident.

## Setup

We'll start by defining some setup utilities and configuring our model.

```python linenums="1"
import os
import random
import torch
from ray.data.preprocessor import Preprocessor
```

We'll define a `set_seeds` function that will set the seeds for reproducibility across our libraries ([`np.random.seed`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.seed.html){:target="_blank"}, [`random.seed`](https://docs.python.org/3/library/random.html#random.seed){:target="_blank"}, [`torch.manual_seed`](https://pytorch.org/docs/stable/generated/torch.manual_seed.html){:target="_blank"} and [`torch.cuda.manual_seed`](https://pytorch.org/docs/stable/generated/torch.cuda.manual_seed.html){:target="_blank"}). We'll also set the behavior for some [torch backends](https://pytorch.org/docs/stable/backends.html){:target="_blank"} to ensure deterministic results when we run our workloads on GPUs.

```python linenums="1"
def set_seeds(seed=42):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    eval("setattr(torch.backends.cudnn, 'deterministic', True)")
    eval("setattr(torch.backends.cudnn, 'benchmark', False)")
    os.environ["PYTHONHASHSEED"] = str(seed)
```

Next, we'll define a simple `load_data` function to ingest our data from source (CSV files) and load it as a Ray Dataset.

```python linenums="1"
def load_data(num_samples=None):
    ds = ray.data.read_csv(DATASET_LOC)
    ds = ds.random_shuffle(seed=1234)
    ds = ray.data.from_items(ds.take(num_samples)) if num_samples else ds
    return ds
```

!!! tip
    When working with very large datasets, it's a good idea to limit the number of samples in our dataset so that we can execute our code quickly and iterate on bugs, etc. This is why we have a `num_samples` input argument in our `load_data` function (`None` = no limit, all samples).

We'll also define a custom preprocessor class that we'll to conveniently preprocess our dataset but also to save/load for later. When defining a preprocessor, we'll need to define a `_fit` method to learn how to fit to our dataset and a `_transform_{pandas|numpy}` method to preprocess the dataset using any components from the `_fit` method. We can either define a `_transform_pandas` method to apply our preprocessing to a Pandas DataFrame or a `_transform_numpy` method to apply our preprocessing to a NumPy array. We'll define the `_transform_pandas` method since our [preprocessing function](preprocessing.md#best-practices){:target="_blank"} expects a batch of data as a Pandas DataFrame.

```python linenums="1"
class CustomPreprocessor(Preprocessor):
    """Custom preprocessor class."""
    def _fit(self, ds):
        tags = ds.unique(column="tag")
        self.class_to_index = {tag: i for i, tag in enumerate(tags)}
        self.index_to_class = {v:k for k, v in self.class_to_index.items()}
    def _transform_pandas(self, batch):  # could also do _transform_numpy
        return preprocess(batch, class_to_index=self.class_to_index)
```

### Model

Now we're ready to start defining our model architecture. We'll start by loading a pretrained LLM and then defining the components needed for fine-tuning it on our dataset. Our pretrained LLM here is a transformer-based model that has been pretrained on a large corpus of scientific text called [scibert](https://huggingface.co/allenai/scibert_scivocab_uncased){:target="_blank"}.

<div class="ai-center-all">
  <a href="https://media.arxiv-vanity.com/render-output/7086622/bert_pretraining.png" target="_blank"><img src="/static/images/mlops/training/bert.png" width="500" alt="bert architecture"></a>
</div>

> If you're not familiar with transformer-based models like LLMs, be sure to check out the [attention](../foundations/attention.md){:target="_blank"} and [Transformers](../foundations/transformers.md){:target="_blank"} lessons.

```python linenums="1"
import torch.nn as nn
from transformers import BertModel
```

We can load our pretrained model by using the [from_pretrained`](https://huggingface.co/docs/transformers/main/main_classes/model#transformers.PreTrainedModel.from_pretrained){:target="_blank"} method.

```python linenums="1"
# Pretrained LLM
llm = BertModel.from_pretrained("allenai/scibert_scivocab_uncased", return_dict=False)
embedding_dim = llm.config.hidden_size
```

Once our model is loaded, we can tokenize an input text, convert it to torch tensors and pass it through our model to get a sequence and pooled representation of the text.

```python linenums="1"
# Sample
text = "Transfer learning with transformers for text classification."
batch = tokenizer([text], return_tensors="np", padding="longest")
batch = {k:torch.tensor(v) for k,v in batch.items()}  # convert to torch tensors
seq, pool = llm(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
np.shape(seq), np.shape(pool)
```

<pre class="output">
(torch.Size([1, 10, 768]), torch.Size([1, 768]))
</pre>

We're going to use this pretrained model to represent our input text features and add additional layers (linear classifier) on top of it for our specific classification task. In short, the pretrained LLM will process the tokenized text and return a sequence (one representation after each token) and pooled (combined) representation of the text. We'll use the pooled representation as input to our final fully-connection layer (`fc1`) to result in a vector of size `num_classes` (number of classes) that we can use to make predictions.

```python linenums="1"
class FinetunedLLM(nn.Module):
    def __init__(self, llm, dropout_p, embedding_dim, num_classes):
        super(FinetunedLLM, self).__init__()
        self.llm = llm
        self.dropout = torch.nn.Dropout(dropout_p)
        self.fc1 = torch.nn.Linear(embedding_dim, num_classes)

    def forward(self, batch):
        ids, masks = batch["ids"], batch["masks"]
        seq, pool = self.llm(input_ids=ids, attention_mask=masks)
        z = self.dropout(pool)
        z = self.fc1(z)
        return z

    @torch.inference_mode()
    def predict(self, batch):
        self.eval()
        z = self(inputs)
        y_pred = torch.argmax(z, dim=1).cpu().numpy()
        return y_pred

    @torch.inference_mode()
    def predict_proba(self, batch):
        self.eval()
        z = self(batch)
        y_probs = F.softmax(z).cpu().numpy()
        return y_probs
```

Let's initialize our model and inspect its layers:

```python linenums="1"
# Initialize model
model = FinetunedLLM(llm=llm, dropout_p=0.5, embedding_dim=embedding_dim, num_classes=num_classes)
print (model.named_parameters)
```

<pre class="output">
(llm): BertModel(
(embeddings): BertEmbeddings(
    (word_embeddings): Embedding(31090, 768, padding_idx=0)
    (position_embeddings): Embedding(512, 768)
    (token_type_embeddings): Embedding(2, 768)
    (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
    (dropout): Dropout(p=0.1, inplace=False)
)
(encoder): BertEncoder(
    (layer): ModuleList(
    (0-11): 12 x BertLayer(
        (attention): BertAttention(
        (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
        )
        (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
        )
        )
        (intermediate): BertIntermediate(
        (dense): Linear(in_features=768, out_features=3072, bias=True)
        (intermediate_act_fn): GELUActivation()
        )
        (output): BertOutput(
        (dense): Linear(in_features=3072, out_features=768, bias=True)
        (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
        )
    )
    )
)
(pooler): BertPooler(
    (dense): Linear(in_features=768, out_features=768, bias=True)
    (activation): Tanh()
)
)
(dropout): Dropout(p=0.5, inplace=False)
(fc1): Linear(in_features=768, out_features=4, bias=True)
</pre>

### Batching

We can iterate through our dataset in batches however we may have batches of different sizes. Recall that our tokenizer padded the inputs to the longest item in the batch (`padding="longest"`). However, our batches for training will be smaller than our large data processing batches and so our batches here may have inputs with different lengths. To address this, we're going to define a custom `collate_fn` to repad the items in our training batches.

```python linenums="1"
from ray.train.torch import get_device
```

Our `pad_array` function will take an array of arrays and pad the inner arrays to the longest length.

```python linenums="1"
def pad_array(arr, dtype=np.int32):
    max_len = max(len(row) for row in arr)
    padded_arr = np.zeros((arr.shape[0], max_len), dtype=dtype)
    for i, row in enumerate(arr):
        padded_arr[i][:len(row)] = row
    return padded_arr
```

And our `collate_fn` will take a batch of data to pad them and convert them to the appropriate PyTorch tensor types.

```python linenums="1"
def collate_fn(batch):
    batch["ids"] = pad_array(batch["ids"])
    batch["masks"] = pad_array(batch["masks"])
    dtypes = {"ids": torch.int32, "masks": torch.int32, "targets": torch.int64}
    tensor_batch = {}
    for key, array in batch.items():
        tensor_batch[key] = torch.as_tensor(array, dtype=dtypes[key], device=get_device())
    return tensor_batch
```

Let's test our `collate_fn` on a sample batch from our dataset.

```python linenums="1"
# Sample batch
sample_batch = sample_ds.take_batch(batch_size=128)
collate_fn(batch=sample_batch)
```

<pre class="output">
{'ids': tensor([[  102,  5800, 14982,  ...,     0,     0,     0],
         [  102,  7746,  2824,  ...,     0,     0,     0],
         [  102,   502,  1371,  ...,     0,     0,     0],
         ...,
         [  102, 10431,   160,  ...,     0,     0,     0],
         [  102,   124,   132,  ...,     0,     0,     0],
         [  102, 12459, 28196,  ...,     0,     0,     0]], dtype=torch.int32),
 'masks': tensor([[1, 1, 1,  ..., 0, 0, 0],
         [1, 1, 1,  ..., 0, 0, 0],
         [1, 1, 1,  ..., 0, 0, 0],
         ...,
         [1, 1, 1,  ..., 0, 0, 0],
         [1, 1, 1,  ..., 0, 0, 0],
         [1, 1, 1,  ..., 0, 0, 0]], dtype=torch.int32),
 'targets': tensor([2, 0, 3, 2, 0, 3, 2, 0, 2, 0, 2, 2, 0, 3, 2, 0, 2, 3, 0, 2, 2, 0, 2, 2,
         0, 1, 1, 0, 2, 0, 3, 2, 0, 3, 2, 0, 2, 0, 2, 2, 0, 2, 0, 3, 2, 0, 3, 2,
         0, 2, 0, 2, 2, 0, 3, 2, 0, 2, 3, 0, 2, 2, 0, 2, 2, 0, 1, 1, 0, 3, 0, 0,
         0, 3, 0, 1, 1, 0, 3, 2, 0, 2, 3, 0, 2, 2, 0, 2, 2, 0, 1, 1, 0, 3, 2, 0,
         2, 3, 0, 2, 2, 0, 2, 2, 0, 1, 1, 0, 2, 0, 2, 2, 0, 2, 2, 0, 2, 0, 1, 1,
         0, 0, 0, 1, 0, 0, 1, 0])}
</pre>

### Utilities

Next, we'll implement set the necessary utility functions for distributed training.

<div class="ai-center-all">
  <img src="/static/images/mlops/ray/train.svg" width="700" alt="ray train">
</div>

```python linenums="1"
from ray.air import Checkpoint, session
from ray.air.config import CheckpointConfig, DatasetConfig, RunConfig, ScalingConfig
import ray.train as train
from ray.train.torch import TorchCheckpoint, TorchTrainer
import torch.nn.functional as F
```

We'll start by defining what one step (or iteration) of training looks like. This will be a function that takes in a batch of data, a model, a loss function, and an optimizer. It will then perform a forward pass, compute the loss, and perform a backward pass to update the model's weights. And finally, it will return the loss.

```python linenums="1"
def train_step(ds, batch_size, model, num_classes, loss_fn, optimizer):
    """Train step."""
    model.train()
    loss = 0.0
    ds_generator = ds.iter_torch_batches(batch_size=batch_size, collate_fn=collate_fn)
    for i, batch in enumerate(ds_generator):
        optimizer.zero_grad()  # reset gradients
        z = model(batch)  # forward pass
        targets = F.one_hot(batch["targets"], num_classes=num_classes).float()  # one-hot (for loss_fn)
        J = loss_fn(z, targets)  # define loss
        J.backward()  # backward pass
        optimizer.step()  # update weights
        loss += (J.detach().item() - loss) / (i + 1)  # cumulative loss
    return loss
```

> **Note:** We're using the [`ray.data.iter_torch_batches`](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.iter_torch_batches.html){:target="_blank"} method instead of [`torch.utils.data.DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader){:target="_blank"} to create a generator that will yield batches of data. In fact, this is the **only line** that's different from a typical PyTorch training loop and the actual training workflow remains untouched. Ray supports [many other ways](https://docs.ray.io/en/latest/data/api/dataset.html#consuming-data){:target="_blank"} to load/consume data for different frameworks as well.

The validation step is quite similar to the training step but we don't need to perform a backward pass or update the model's weights.

```python linenums="1"
def eval_step(ds, batch_size, model, num_classes, loss_fn):
    """Eval step."""
    model.eval()
    loss = 0.0
    y_trues, y_preds = [], []
    ds_generator = ds.iter_torch_batches(batch_size=batch_size, collate_fn=collate_fn)
    with torch.inference_mode():
        for i, batch in enumerate(ds_generator):
            z = model(batch)
            targets = F.one_hot(batch["targets"], num_classes=num_classes).float()  # one-hot (for loss_fn)
            J = loss_fn(z, targets).item()
            loss += (J - loss) / (i + 1)
            y_trues.extend(batch["targets"].cpu().numpy())
            y_preds.extend(torch.argmax(z, dim=1).cpu().numpy())
    return loss, np.vstack(y_trues), np.vstack(y_preds)
```

Next, we'll define the `train_loop_per_worker` which defines the overall training loop for each worker. It's important that we include operations like loading the datasets, models, etc. so that each worker will have its own copy of these objects. Ray takes care of combining all the workers' results at the end of each iteration, so from the user's perspective, it's the exact same as training on a single machine!

The only additional lines of code we need to add compared to a typical PyTorch training loop are the following:

- `#!python session.get_dataset_shard("train")` and `#!python session.get_dataset_shard("val")` to load the data splits ([`session.get_dataset_shard`](https://docs.ray.io/en/latest/ray-air/api/doc/ray.air.session.get_dataset_shard.html){:target="_blank"}).
- `#!python model = train.torch.prepare_model(model)` to prepare the torch model for distributed execution ([`train.torch.prepare_model`](https://docs.ray.io/en/latest/train/api/doc/ray.train.torch.prepare_model.html){:target="_blank"}).
- `#!python batch_size_per_worker = batch_size // session.get_world_size()` to adjust the batch size for each worker ([`session.get_world_size`](https://docs.ray.io/en/latest/ray-air/api/doc/ray.air.session.get_world_size.html){:target="_blank"}).
- `#!python session.report(metrics, checkpoint=checkpoint)` to report metrics and save our model checkpoint ([`session.report`](https://docs.ray.io/en/latest/ray-air/api/doc/ray.air.session.report.html){:target="_blank"}).

All the other lines of code are the same as a typical PyTorch training loop!

```python linenums="1" hl_lines="14 15 20 28 38"
# Training loop
def train_loop_per_worker(config):
    # Hyperparameters
    dropout_p = config["dropout_p"]
    lr = config["lr"]
    lr_factor = config["lr_factor"]
    lr_patience = config["lr_patience"]
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    num_classes = config["num_classes"]

    # Get datasets
    set_seeds()
    train_ds = session.get_dataset_shard("train")
    val_ds = session.get_dataset_shard("val")

    # Model
    llm = BertModel.from_pretrained("allenai/scibert_scivocab_uncased", return_dict=False)
    model = FinetunedLLM(llm=llm, dropout_p=dropout_p, embedding_dim=llm.config.hidden_size, num_classes=num_classes)
    model = train.torch.prepare_model(model)

    # Training components
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=lr_factor, patience=lr_patience)

    # Training
    batch_size_per_worker = batch_size // session.get_world_size()
    for epoch in range(num_epochs):
        # Step
        train_loss = train_step(train_ds, batch_size_per_worker, model, num_classes, loss_fn, optimizer)
        val_loss, _, _ = eval_step(val_ds, batch_size_per_worker, model, num_classes, loss_fn)
        scheduler.step(val_loss)

        # Checkpoint
        metrics = dict(epoch=epoch, lr=optimizer.param_groups[0]["lr"], train_loss=train_loss, val_loss=val_loss)
        checkpoint = TorchCheckpoint.from_model(model=model)
        session.report(metrics, checkpoint=checkpoint)
```

<span id="class-imbalance"></span>
!!! tip "Class imbalance"
    Our dataset doesn't suffer from horrible class imbalance, but if it did, we could easily account for it through our loss function. There are also other strategies such as [over-sampling](https://imbalanced-learn.org/stable/over_sampling.html){:target="_blank"} less frequent classes and [under-sampling](https://imbalanced-learn.org/stable/under_sampling.html){:target="_blank"} popular classes.

    ```python linenums="1"
    # Class weights
    batch_counts = []
    for batch in train_ds.iter_torch_batches(batch_size=256, collate_fn=collate_fn):
        batch_counts.append(np.bincount(batch["targets"].cpu().numpy()))
    counts = [sum(count) for count in zip(*batch_counts)]
    class_weights = np.array([1.0/count for i, count in enumerate(counts)])
    class_weights_tensor = torch.Tensor(class_weights).to(get_device())

    # Training components
    loss_fn = nn.BCEWithLogitsLoss(weight=class_weights_tensor)
    ...
    ```

### Configurations

Next, we'll define some configurations that will be used to train our model.

```python linenums="1"
# Train loop config
train_loop_config = {
    "dropout_p": 0.5,
    "lr": 1e-4,
    "lr_factor": 0.8,
    "lr_patience": 3,
    "num_epochs": 10,
    "batch_size": 256,
    "num_classes": num_classes,
}
```

Next we'll define our scaling configuration ([ScalingConfig](https://docs.ray.io/en/latest/ray-air/api/doc/ray.air.ScalingConfig.html){:target="_blank"}) that will specify how we want to scale our training workload. We specify the number of workers (`num_workers`), whether to use GPU or not (`use_gpu`), the resources per worker (`resources_per_worker`) and how much CPU each worker is allowed to use (`_max_cpu_fraction_per_node`).

```python linenums="1"
# Scaling config
scaling_config = ScalingConfig(
    num_workers=num_workers,
    use_gpu=bool(resources_per_worker["GPU"]),
    resources_per_worker=resources_per_worker,
    _max_cpu_fraction_per_node=0.8,
)
```

> `#!python _max_cpu_fraction_per_node=0.8` indicates that 20% of CPU is reserved for **non-training workloads** that our workers will do such as data preprocessing (which we do prior to training anyway).

Next, we'll define our [`CheckpointConfig`](https://docs.ray.io/en/latest/ray-air/api/doc/ray.air.CheckpointConfig.html){:target="_blank"} which will specify how we want to checkpoint our model. Here we will just save one checkpoint (`num_to_keep`) based on the checkpoint with the `min` `val_loss`. We'll also configure a [`RunConfig`](https://docs.ray.io/en/latest/ray-air/api/doc/ray.air.RunConfig.html){:target="_blank"} which will specify the `name` of our run and where we want to save our checkpoints.

```python linenums="1"
# Run config
checkpoint_config = CheckpointConfig(num_to_keep=1, checkpoint_score_attribute="val_loss", checkpoint_score_order="min")
run_config = RunConfig(name="llm", checkpoint_config=checkpoint_config, local_dir="~/ray_results")
```

We'll be naming our experiment `llm` and saving our results to `~/ray_results`, so a sample directory structure for our trained models would look like this:

```bash
/home/ray/ray_results/llm
├── TorchTrainer_fd40a_00000_0_2023-07-20_18-14-50/
├── basic-variant-state-2023-07-20_18-14-50.json
├── experiment_state-2023-07-20_18-14-50.json
├── trainer.pkl
└── tuner.pkl
```

The `TorchTrainer_` objects are the individuals runs in this experiment and each one will have the following contents:

```bash
/home/ray/ray_results/TorchTrainer_fd40a_00000_0_2023-07-20_18-14-50/
├── checkpoint_000009/  # we only save one checkpoint (the best)
├── events.out.tfevents.1689902160.ip-10-0-49-200
├── params.json
├── params.pkl
├── progress.csv
└── result.json
```

> There are several other configs that we could set with Ray (ex. [failure handling](https://docs.ray.io/en/latest/ray-air/api/doc/ray.air.FailureConfig.html#ray.air.FailureConfig){:target="_blank"}) so be sure to check them out [here](https://docs.ray.io/en/latest/ray-air/api/configs.html){:target="_blank"}.

!!! note "Stopping criteria"
    While we'll just let our experiments run for a certain number of epochs and stop automatically, our [`RunConfig`](https://docs.ray.io/en/latest/ray-air/api/doc/ray.air.RunConfig.html){:target="_blank"} accepts an optional stopping criteria (`stop`) which determines the conditions our training should stop for. It's entirely customizable and common examples include a certain [metric value](https://docs.ray.io/en/latest/tune/tutorials/tune-stopping.html#stop-using-metric-based-criteria){:target="_blank"}, [elapsed time](https://docs.ray.io/en/latest/tune/tutorials/tune-stopping.html#stop-using-metric-based-criteria){:target="_blank"} or even a [custom class](https://docs.ray.io/en/latest/tune/tutorials/tune-stopping.html#stop-using-metric-based-criteria){:target="_blank"}.

## Training

Now we're finally ready to train our model using all the components we've setup above.

```python linenums="1"
# Load and split data
ds = load_data()
train_ds, val_ds = stratify_split(ds, stratify="tag", test_size=test_size)
```

```python linenums="1"
# Preprocess
preprocessor = CustomPreprocessor()
train_ds =  preprocessor.fit_transform(train_ds)
val_ds = preprocessor.transform(val_ds)
train_ds = train_ds.materialize()
val_ds = val_ds.materialize()
```

> Calling materialize here is important because it will cache the preprocessed data in memory. This will allow us to train our model without having to reprocess the data each time.

Because we've preprocessed the data prior to training, we can use the `#!python fit=False` and `#!python transform=False` flags in our dataset config. This will allow us to skip the preprocessing step during training.

```python linenums="1"
# Dataset config
dataset_config = {
    "train": DatasetConfig(fit=False, transform=False, randomize_block_order=False),
    "val": DatasetConfig(fit=False, transform=False, randomize_block_order=False),
}
```

We'll pass all of our functions and configs to the [`TorchTrainer`](https://docs.ray.io/en/latest/train/api/doc/ray.train.torch.TorchTrainer.html){:target="_blank"} class to start training. Ray supports a wide variety of [framework Trainers](https://docs.ray.io/en/master/train/train.html#training-framework-catalog){:target="_blank"} so if you're using other frameworks, you can use the corresponding Trainer class instead.

<div class="ai-center-all">
  <img src="/static/images/mlops/training/trainers.png" width="700" alt="framework trainers">
</div>

```python linenums="1"
# Trainer
trainer = TorchTrainer(
    train_loop_per_worker=train_loop_per_worker,
    train_loop_config=train_loop_config,
    scaling_config=scaling_config,
    run_config=run_config,
    datasets={"train": train_ds, "val": val_ds},
    dataset_config=dataset_config,
    preprocessor=preprocessor,
)
```

Now let's fit our model to the data.

```python linenums="1"
# Train
results = trainer.fit()
```

<div class="output_subarea output_html rendered_html output_result" dir="auto"><div>
<table border="1" class="dataframe">
<thead>
<tr><th>Trial name              </th><th>status    </th><th>loc             </th><th style="text-align: right;">  iter</th><th style="text-align: right;">  total time (s)</th><th style="text-align: right;">  epoch</th><th style="text-align: right;">    lr</th><th style="text-align: right;">  train_loss</th></tr>
</thead>
<tbody>
<tr><td>TorchTrainer_8c960_00000</td><td>TERMINATED</td><td>10.0.18.44:68577</td><td style="text-align: right;">    10</td><td style="text-align: right;">         76.3089</td><td style="text-align: right;">      9</td><td style="text-align: right;">0.0001</td><td style="text-align: right;"> 0.000549661</td></tr>
</tbody>
</table></div></div>

```python linenums="1"
results.metrics_dataframe
```

<div class="output_subarea output_html rendered_html output_result" dir="auto"><div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>epoch</th>
      <th>lr</th>
      <th>train_loss</th>
      <th>val_loss</th>
      <th>timestamp</th>
      <th>time_this_iter_s</th>
      <th>should_checkpoint</th>
      <th>done</th>
      <th>training_iteration</th>
      <th>trial_id</th>
      <th>date</th>
      <th>time_total_s</th>
      <th>pid</th>
      <th>hostname</th>
      <th>node_ip</th>
      <th>time_since_restore</th>
      <th>iterations_since_restore</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.0001</td>
      <td>0.005196</td>
      <td>0.004071</td>
      <td>1689030896</td>
      <td>14.162520</td>
      <td>True</td>
      <td>False</td>
      <td>1</td>
      <td>8c960_00000</td>
      <td>2023-07-10_16-14-59</td>
      <td>14.162520</td>
      <td>68577</td>
      <td>ip-10-0-18-44</td>
      <td>10.0.18.44</td>
      <td>14.162520</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.0001</td>
      <td>0.004033</td>
      <td>0.003898</td>
      <td>1689030905</td>
      <td>8.704429</td>
      <td>True</td>
      <td>False</td>
      <td>2</td>
      <td>8c960_00000</td>
      <td>2023-07-10_16-15-08</td>
      <td>22.866948</td>
      <td>68577</td>
      <td>ip-10-0-18-44</td>
      <td>10.0.18.44</td>
      <td>22.866948</td>
      <td>2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>0.0001</td>
      <td>0.000550</td>
      <td>0.001182</td>
      <td>1689030958</td>
      <td>6.604867</td>
      <td>True</td>
      <td>False</td>
      <td>10</td>
      <td>8c960_00000</td>
      <td>2023-07-10_16-16-01</td>
      <td>76.308887</td>
      <td>68577</td>
      <td>ip-10-0-18-44</td>
      <td>10.0.18.44</td>
      <td>76.308887</td>
      <td>10</td>
    </tr>
  </tbody>
</table></div></div>

```python linenums="1"
results.best_checkpoints
```

<pre class="output">
[(TorchCheckpoint(local_path=/home/ray/ray_results/llm/TorchTrainer_8c960_00000_0_2023-07-10_16-14-41/checkpoint_000009),
  {'epoch': 9,
   'lr': 0.0001,
   'train_loss': 0.0005496611799268673,
   'val_loss': 0.0011818759376183152,
   'timestamp': 1689030958,
   'time_this_iter_s': 6.604866981506348,
   'should_checkpoint': True,
   'done': True,
   'training_iteration': 10,
   'trial_id': '8c960_00000',
   'date': '2023-07-10_16-16-01',
   'time_total_s': 76.30888652801514,
   'pid': 68577,
   'hostname': 'ip-10-0-18-44',
   'node_ip': '10.0.18.44',
   'config': {'train_loop_config': {'dropout_p': 0.5,
     'lr': 0.0001,
     'lr_factor': 0.8,
     'lr_patience': 3,
     'num_epochs': 10,
     'batch_size': 256,
     'num_classes': 4}},
   'time_since_restore': 76.30888652801514,
   'iterations_since_restore': 10,
   'experiment_tag': '0'})]
</pre>

## Observability

While our model is training, we can inspect our Ray dashboard to observe how our compute resources are being utilized.

!!! quote "💻 Local"
    We can inspect our Ray dashboard by opening [http://127.0.0.1:8265](http://127.0.0.1:8265){:target="_blank"} on a browser window. Click on **Cluster** on the top menu bar and then we will be able to see a list of our nodes (head and worker) and their utilizations.

!!! example "🚀 Anyscale"
    On Anyscale Workspaces, we can head over to the top right menu and click on **🛠️ Tools** → **Ray Dashboard** and this will open our dashboard on a new tab. Click on **Cluster** on the top menu bar and then we will be able to see a list of our nodes (head and worker) and their utilizations.

<div class="ai-center-all">
    <img src="/static/images/mlops/training/dashboard.png" width="800" alt="Ray dashboard">
</div>

> Learn about all the other observability features on the Ray Dashboard through this [video](https://www.youtube.com/playlist?list=PLzTswPQNepXlh3SWAgwZZxqLxYjXzcVbn){:target="_blank"}.

## Evaluation

Now that we've trained our model, we can evaluate it on a separate holdout test set. We'll cover the topic of evaluation much more extensively in our [evaluation lesson](evaluation.md){:target="_blank"} but for now we'll calculate some quick overall metrics.

```python linenums="1"
from ray.train.torch import TorchPredictor
from sklearn.metrics import precision_recall_fscore_support
```

We'll define a function that can take in a dataset and a predictor and return the performance metrics.

1. Load the predictor and preprocessor from the best checkpoint:
```python linenums="1"
# Predictor
best_checkpoint = results.best_checkpoints[0][0]
predictor = TorchPredictor.from_checkpoint(best_checkpoint)
preprocessor = predictor.get_preprocessor()
```
2. Load and preprocess the test dataset that we want to evaluate on:
```python linenums="1"
# Test (holdout) dataset
HOLDOUT_LOC = "https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/holdout.csv"
test_ds = ray.data.read_csv(HOLDOUT_LOC)
preprocessed_ds = preprocessor.transform(test_ds)
preprocessed_ds.take(1)
```
<pre class="output">
[{'ids': array([  102,  4905,  2069,  2470,  2848,  4905, 30132, 22081,   691,
          4324,  7491,  5896,   341,  6136,   934, 30137,   103,     0,
             0,     0,     0]),
  'masks': array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]),
  'targets': 3}]
</pre>
2. Retrieve the true label indices from the `targets` column by using [ray.data.Dataset.select_column](){:target="_blank"}:
```python linenums="1"
# y_true
values = preprocessed_ds.select_columns(cols=["targets"]).take_all()
y_true = np.stack([item["targets"] for item in values])
print (y_true)
```
<pre class="output">
[3 3 3 0 2 0 0 0 0 2 0 0 2 3 0 0 2 2 3 2 3 0 3 2 0 2 2 1 1 2 2 2 2 2 2 0 0
 0 0 0 1 1 2 0 0 3 1 2 0 2 2 3 3 0 2 3 2 3 3 3 3 0 0 0 0 2 2 0 2 1 0 2 3 0
 0 2 2 2 2 2 0 0 2 0 1 0 0 0 0 3 0 0 2 0 2 2 3 2 0 2 0 2 0 3 0 0 0 0 0 2 0
 0 2 2 2 2 3 0 2 0 2 0 2 3 3 3 2 0 2 2 2 2 0 2 2 2 0 1 2 2 2 2 2 1 2 0 3 0
 2 2 1 1 2 0 0 0 0 0 0 2 2 2 0 2 1 1 2 0 0 1 2 3 2 2 2 0 0 2 0 2 0 3 0 2 2
 0 1 2 1 2 2]
</pre>
3. Get our predicted label indices by using the `predictor`. Note that the `predictor` will automatically take care of the preprocessing for us.
```python linenums="1"
# y_pred
z = predictor.predict(data=test_ds.to_pandas())["predictions"]
y_pred = np.stack(z).argmax(1)
print (y_pred)
```
<pre class="output">
[3 3 3 0 2 0 0 0 0 2 0 0 2 3 0 0 0 2 3 2 3 0 3 2 0 0 2 1 1 2 2 2 2 2 2 0 0
 0 0 0 1 2 2 0 2 3 1 2 0 2 2 3 3 0 2 1 2 3 3 3 3 2 0 0 0 2 2 0 2 1 0 2 3 1
 0 2 2 2 2 2 0 0 2 1 1 0 0 0 0 3 0 0 2 0 2 2 3 2 0 2 0 2 2 0 2 0 0 3 0 2 0
 0 1 2 2 2 3 0 2 0 2 0 2 3 3 3 2 0 2 2 2 2 0 2 2 2 0 1 2 2 2 2 2 1 2 0 3 0
 2 2 2 1 2 0 2 0 0 0 0 2 2 2 0 2 1 2 2 0 0 1 2 3 2 2 2 0 0 2 0 2 1 3 0 2 2
 0 1 2 1 2 2]
</pre>
4. Compute our metrics using the true and predicted labels indices.
```python linenums="1"
# Evaluate
metrics = precision_recall_fscore_support(y_true, y_pred, average="weighted")
{"precision": metrics[0], "recall": metrics[1], "f1": metrics[2]}
```
<pre class="output">
{'precision': 0.9147673308349523,
 'recall': 0.9109947643979057,
 'f1': 0.9115810676649443}
</pre>

We're going to encapsulate all of these steps into one function so that we can call on it as we train more models soon.

```python linenums="1"
def evaluate(ds, predictor):
    # y_true
    preprocessor = predictor.get_preprocessor()
    preprocessed_ds = preprocessor.transform(ds)
    values = preprocessed_ds.select_columns(cols=["targets"]).take_all()
    y_true = np.stack([item["targets"] for item in values])

    # y_pred
    z = predictor.predict(data=ds.to_pandas())["predictions"]
    y_pred = np.stack(z).argmax(1)

    # Evaluate
    metrics = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    performance = {"precision": metrics[0], "recall": metrics[1], "f1": metrics[2]}
    return performance
```

## Inference

Now let's load our trained model for inference on new data. We'll create a few utility functions to format the probabilities into a dictionary for each class and to return predictions for each item in a dataframe.

```python linenums="1"
import pandas as pd
```

```python linenums="1"
def format_prob(prob, index_to_class):
    d = {}
    for i, item in enumerate(prob):
        d[index_to_class[i]] = item
    return d
```

```python linenums="1"
def predict_with_proba(df, predictor):
    preprocessor = predictor.get_preprocessor()
    z = predictor.predict(data=df)["predictions"]
    y_prob = torch.tensor(np.stack(z)).softmax(dim=1).numpy()
    results = []
    for i, prob in enumerate(y_prob):
        tag = decode([z[i].argmax()], preprocessor.index_to_class)[0]
        results.append({"prediction": tag, "probabilities": format_prob(prob, preprocessor.index_to_class)})
    return results
```

We'll load our `predictor` from the best checkpoint and load it's `preprocessor`.

```python linenums="1"
# Preprocessor
predictor = TorchPredictor.from_checkpoint(best_checkpoint)
preprocessor = predictor.get_preprocessor()
```

And now we're ready to apply our model to new data. We'll create a sample dataframe with a title and description and then use our `predict_with_proba` function to get the predictions. Note that we use a placeholder value for `tag` since our input dataframe will automatically be [preprocessed](preprocessing.md#best-practices){:target="_blank"} (and it expects a value in the `tag` column).

```python linenums="1"
# Predict on sample
title = "Transfer learning with transformers"
description = "Using transformers for transfer learning on text classification tasks."
sample_df = pd.DataFrame([{"title": title, "description": description, "tag": "other"}])
predict_with_proba(df=sample_df, predictor=predictor)
```

<pre class="output">
[{'prediction': 'natural-language-processing',
  'probabilities': {'computer-vision': 0.0007296873,
   'mlops': 0.0008382588,
   'natural-language-processing': 0.997829,
   'other': 0.00060295867}}]
</pre>

## Optimization

Distributed training strategies are great for when our data or models are too large for training but there are additional strategies to make the models itself smaller for serving. The following model compression techniques are commonly used to reduce the size of the model:

- [**Pruning**](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html){:target="_blank"}: remove weights (unstructured) or entire channels (structured) to reduce the size of the network. The objective is to preserve the model’s performance while increasing its sparsity.
- [**Quantization**](https://pytorch.org/docs/stable/torch.quantization.html){:target="_blank"}: reduce the memory footprint of the weights by reducing their precision (ex. 32 bit to 8 bit). We may loose some precision but it shouldn’t affect performance too much.
- [**Distillation**](https://arxiv.org/abs/2011.14691){:target="_blank"}: training smaller networks to “mimic” larger networks by having it reproduce the larger network’s layers’ outputs.

<div class="ai-center-all">
    <a href="https://nni.readthedocs.io/en/latest/TrialExample/KDExample.html" target="_blank"><img width="750" src="/static/images/mlops/baselines/kd.png" alt="knowledge distillation"></a>
</div>

<!-- Course signup -->
{% include "templates/signup.md" %}

<!-- Citation -->
{% include "templates/cite.md" %}