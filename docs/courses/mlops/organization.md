---
template: lesson.html
title: Organizing a Code Repository for ML Applications
description: Organizing our code when moving from notebooks to Python scripts.
keywords: git, github, organization, repository, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/mlops.png
repository: https://github.com/GokuMohandas/MLOps
---

{% include "styles/lesson.md" %}

## Intuition

To have organized code is to have readable, reproducible, scalable and efficient code. We'll cover all of these concepts throughout the scripting lessons.

## Organizing
There are several ways to organize our code when we're going from the notebooks to scripts but they're all based on utility. For example, we're organizing our code based on pipeline components (data processing, training, evaluation, prediction, etc.):

```bash linenums="1"
tagifai/
├── data.py       - data processing utilities
├── eval.py       - evaluation components
├── main.py       - training/optimization pipelines
├── models.py     - model architectures
├── predict.py    - inference utilities
├── train.py      - training utilities
└── utils.py      - supplementary utilities
```

Organizing our code base this way also makes it easier for us to understand (or modify) the code base. We could've also assumed a more granular stance for organization, such as breaking down `data.py` into `split.py`, `preprocess.py`, etc. This might make more sense if we have multiple ways of splitting, preprocessing, etc. but for our task, it's sufficient to be at a higher level.

### Functions and classes

Once we've decided on the directory architecture, we can start moving the functions and classes from the notebook under the appropriate scripts. It should be clear which function/class goes into which script based on how we've decided to organize our project (notebook headers can also be indicative).

!!! question "Streamlined process"
    How can we improve this process of moving code from notebooks to scripts?

    ??? quote "Show answer"

        As you work on more projects, you may find it useful for you and your team members to contribute your generalizable functions and classes to a central repository. Provided that all the code is [tested](testing.md){:target="_blank"} and [documented](documentation.md){:target="_blank"}, this can reduce boilerplate code and allow for reliable and faster development. To use your repository, you can [package](https://packaging.python.org/tutorials/packaging-projects/){:target="_blank"} it and install directly from your public/private repo or load it from a private PyPI mirror, etc.
        ```bash linenums="1"
        pip install git+https://github.com/GokuMohandas/MLOps#egg=tagifai
        ```

### Operations

Now that we've organized our functions/classes, it's time to create some new functions to encapsulate the ad-hoc processes in our notebooks. Recall that we repeatedly performed actions such as setting the device, reading from a JSON file, etc. We should organize these general [utilities](https://github.com/GokuMohandas/MLOps/blob/main/tagifai/utils.py){:target="_blank"} as separate functions that we can reuse later on. For example:

```python linenums="1"
# Set device
cuda = True
device = torch.device("cuda" if (
    torch.cuda.is_available() and cuda) else "cpu")
torch.set_default_tensor_type("torch.FloatTensor")
if device.type == "cuda":
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
print (device)
```

can be organize as a clean, reuseable function with the appropriate parameters:

```python linenums="1"
def set_device(cuda: bool):
    device = torch.device("cuda" if (torch.cuda.is_available() and cuda) else "cpu")
    torch.set_default_tensor_type("torch.FloatTensor")
    if device.type == "cuda":
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    return device
```

For the more main operations (computing features, training, etc.), we can organize them into functions under the [main.py](https://github.com/GokuMohandas/MLOps/blob/main/tagifai/main.py){:target="_blank"} script which we can call using various interfaces ([CLI](cli.md){:target="_blank"} or [API](cli.md){:target="_blank"}).


## Reading
So what's the best way to read a code base like this? We could look at the [documentation](https://gokumohandas.github.io/MLOps/){:target="_blank"} but that's usually useful if you're looking for specific functions or classes within a script. What if you want to understand the overall functionality and how it's all organized? Well, we can start with the operations defined in [`main.py`](https://github.com/GokuMohandas/MLOps/blob/main/tagifai/main.py){:target="_blank"} and dive deeper into the specific workflows (training, optimization, etc.).

For example, if we inspect the `train()` function that's responsible for training, we inspect the various steps involved.

```python linenums="1"
def train_model(params, trial):
    """Operations for training."""
    # Set up
    # Load data
    # Prepare data
    # Preprocess data
    # Encode labels
    # Class weights
    # Split data
    # Tokenize inputs
    # Create dataloaders
    # Initialize model
    # Train model
    # Evaluate model

    return artifacts
```

We can dive as deep as we'd like which really depends on your task (general understanding, modifying or extend the code base, etc.). Similarly, we can also zoom out and see which modules use this `train()` function, such as [CLI](cli.md){:target="_blank"} or [API](cli.md){:target="_blank"} endpoints.

<!-- Citation -->
{% include "cite.md" %}
