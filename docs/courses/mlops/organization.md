---
template: lesson.html
title: Organizing a Code Repository
description: Organizing our code when moving from notebooks to Python scripts.
keywords: organization, repository, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/applied_ml.png
repository: https://github.com/GokuMohandas/MLOps
---

{% include "styles/lesson.md" %}

## Intuition

To have organized code is to have readable, reproducible, scalable and efficient code. We'll cover all of these concepts throughout the scripting lessons.

## Organizing
There are several ways to organize our code from the notebooks but they're all based on utility. For example, we're organizing our code based on the part of the pipeline (data, training, prediction, etc.):

```bash linenums="1"
app/
├── api.py        - FastAPI app
└── cli.py        - CLI app
tagifai/
├── config.py     - configuration setup
├── data.py       - data processing utilities
├── models.py     - model architectures
├── predict.py    - inference utilities
├── train.py      - training utilities
└── utils.py      - supplementary utilities
```

Organizing our code base this way also makes it easier for readers to understand (or modify) the code base. We could've also assumed a more granular stance for organization, such as breaking down `data.py` into `split.py`, `preprocess.py`, etc. This might make more sense if we have multiple ways of splitting, preprocessing, etc. but for our task, it's sufficient to be at a higher level.

!!! note
    Another way to supplement organized code is through [documentation](documentation.md){:target="_blank"}.

## Reading
So what's the best way to read a code base like this? We could look at the [documentation](https://gokumohandas.github.io/MLOps/){:target="_blank"} but that's usually useful if you're looking for specific functions or classes within a script. What if you want to understand the overall functionality and how it's all organized? Well, we can start with the options in `app/cli.py` and dive deeper into the specific utilities. Let's say we wanted to see how a single model is trained, then we'd go to the `train_model` function and inspect each line and build a mental model of the process. For example, when you reach the line:
```python linenums="1"
# Train
artifacts = train.run(params=params)
```
you'll want to go to `train.py` → `run` to see it's operations:
```bash linenums="1"
Operations for training.
1. Set seed
2. Set device
3. Load data
4. Clean data
5. Preprocess data
6. Encode labels
7. Split data
8. Tokenize inputs
9. Create dataloaders
10. Initialize model
11. Train model
12. Evaluate model
```
You can dive as deep as you'd like which really depends on your task (general understanding, modifying or extend the code base, etc.)

!!! note
    When looking a code base for the first, it's a good item to create a mental model of the entire application and writing it down for yourself so you easily navigate it in the future.

<!-- Citation -->
{% include "cite.md" %}
