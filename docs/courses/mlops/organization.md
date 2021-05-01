---
template: lesson.html
title: Organizing a Code Repository for ML Applications
description: Organizing our code when moving from notebooks to Python scripts.
keywords: git, github, organization, repository, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/applied_ml.png
repository: https://github.com/GokuMohandas/MLOps
---

{% include "styles/lesson.md" %}

## Intuition

To have organized code is to have readable, reproducible, scalable and efficient code. We'll cover all of these concepts throughout the scripting lessons.

## Organizing
There are several ways to organize our code from the notebooks but they're all based on utility. For example, we're organizing our code based on pipeline components (data processing, training, evaluation, prediction, etc.):

```bash linenums="1"
app/
├── api.py        - FastAPI app
└── cli.py        - CLI app
├── schemas.py    - API model schemas
tagifai/
├── config.py     - configuration setup
├── data.py       - data processing utilities
├── eval.py       - evaluation components
├── main.py       - training/optimization pipelines
├── models.py     - model architectures
├── predict.py    - inference utilities
├── train.py      - training utilities
└── utils.py      - supplementary utilities
```

!!! note
    Don't worry about what all these different scripts do just yet! We'll be creating and going through them in the subsequent lessons.

Organizing our code base this way also makes it easier for us to understand (or modify) the code base. We could've also assumed a more granular stance for organization, such as breaking down `data.py` into `split.py`, `preprocess.py`, etc. This might make more sense if we have multiple ways of splitting, preprocessing, etc. but for our task, it's sufficient to be at a higher level.

!!! note
    Another way to supplement organized code is through [documentation](documentation.md){:target="_blank"}.

## Reading
So what's the best way to read a code base like this? We could look at the [documentation](https://gokumohandas.github.io/MLOps/){:target="_blank"} but that's usually useful if you're looking for specific functions or classes within a script. What if you want to understand the overall functionality and how it's all organized? Well, we can start with the operations defined in [`tagifai/main.py`](https://github.com/GokuMohandas/MLOps/blob/main/tagifai/main.py){:target="_blank"} and dive deeper into the specific workflows (training, optimization, etc.).

For example, if we inspect the `run()` function that's responsible for training, we inspect the various steps involved. We can dive as deep as we'd like which really depends on your task (general understanding, modifying or extend the code base, etc.). Similarly, we can also zoom out and see which modules use this `run()` function, such as CLI/API endpoints, etc.

```python linenums="1"
def run(params: Namespace, trial: optuna.trial._trial.Trial = None) -> Dict:
    """Operations for training.

    Args:
        params (Namespace): Input parameters for operations.
        trial (optuna.trial._trial.Trail, optional): Optuna optimization trial. Defaults to None.

    Returns:
        Artifacts to save and load for later.
    """
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

!!! note
    When looking a code base for the first, it's a good item to create a mental model of the entire application and writing it down for yourself so you easily navigate it in the future.

<!-- Citation -->
{% include "cite.md" %}
