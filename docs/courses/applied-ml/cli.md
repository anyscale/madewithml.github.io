---
template: lesson.html
title: Command Line Interface (CLI) Applications
description: Using a command line interface (CLI) application to organize our application's processes.
keywords: cli, application, applied ml, mlops, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/applied_ml.png
---

:octicons-mark-github-16: [Repository](https://github.com/GokuMohandas/applied-ml){:target="_blank"}

## Intuition

We want to enable others to be able to interact with our application without having to dig into the code and execute functions one at a time. One method is to build a CLI application that allows for interaction via any shell. It should designed such that we can see all possible operations as well the appropriate assistance needed for configuring options and other arguments for each of those operations. Let's see what a CLI looks like for our application which has many different commands (training, prediction, etc.)

## Application

The `app` that we defined inside our [`cli.py`](https://github.com/GokuMohandas/applied-ml/tree/main/app/cli.py){:target="_blank"} script is created using [Typer](https://typer.tiangolo.com/){:target="_blank"}, an open-source tool for building command line interface (CLI) applications. It starts by initializing the app and then adding the appropriate decorator to each function we wish to use as a CLI command.

```python
# Typer CLI app
app = typer.Typer()

@app.command()
def predict_tags(
    text: str = "Transfer learning with BERT for self-supervised learning",
    run_id: str = "",
) -> Dict:
...
```

!!! note
    We're combining console scripts (from `setup.py`) and our Typer app to create a CLI application but there are many [different ways](https://typer.tiangolo.com/typer-cli/){:target="_blank"} to use Typer as well. We're going to have other programs use our application so this approach works best.
    ```python
    # setup.py
    setup(
        name="tagifai",
        version="0.1",
        ...
        entry_points={
            "console_scripts": [
                "tagifai = app.cli:app",
            ],
        },
    )
    ```

## Commands

In [`cli.py`](https://github.com/GokuMohandas/applied-ml/tree/main/app/cli.py){:target="_blank"} script we have define the following commands:

- `download-data`: download data from online to local drive.
- `optimize`: optimize a subset of hyperparameters towards an objective.
- `train-model`: train a model using the specified parameters.
- `predict-tags`: predict tags for a give input text using a trained model.

We can list all the CLI commands for our application like so:

<div class="animated-code">

    ```console
    # View all Typer commands
    $ tagifai --help
    Usage: tagifai [OPTIONS] COMMAND [ARGS]
    👉  Commands:
        download-data  Download data from online to local drive.
        optimize       Optimize a subset of hyperparameters towards ...
        train-model    Predict tags for a give input text using a ...
        predict-tags   Train a model using the specified parameters.
    ```

</div>
<script src="../../../static/js/termynal.js"></script>

## Arguments

With Typer, a function's input arguments automatically get rendered as command line options. For example, our `predict_tags` function consumes `text` and an optional `run_id` as inputs which automatically become arguments for the `predict-tags` CLI command.

```python
@app.command()
def predict_tags(
    text: str = "Transfer learning with BERT for self-supervised learning",
    run_id: str = "",
) -> Dict:
    """Predict tags for a give input text using a trained model.

    Warning:
        Make sure that you have a trained model first!

    Args:
        text (str, optional): Input text to predict tags for.
                              Defaults to "Transfer learning with BERT for self-supervised learning".
        run_id (str, optional): ID of the run to load model artifacts from.
                                Defaults to model with lowest `best_val_loss` from the `best` experiment.

    Returns:
        Predicted tags for input text.
    """
    ...
```

<div class="animated-code">

    ```console
    # Help for a specific command
    $ tagifai predict-tags --help
    Usage: tagifai predict-tags [OPTIONS]

    Predict tags for a give input text using a trained model. Make sure that you have a trained model first!

    Args:
        text (str, optional):
            Input text to predict tags for.
            Defaults to "Transfer learning with BERT.".
        run_id(str, optional):
            ID of the run to load model artifacts from.
            Defaults to model with lowest `best_val_loss`
            from the `best` experiment.

    Returns:
        Predicted tags for input text.

    Options:
        --text TEXT    [default: Transfer learning with BERT.]
        --run-id TEXT  [default: ]
        --help         Show this message and exit.
    ```
</div>

## Executing

And we can easily use our CLI app to execute these commands with the appropriate options:
<div class="animated-code">

    ```console
    # Prediction
    $ tagifai predict-tags --text "Transfer learning with BERT."
    {
        "input_text": "Transfer learning with BERT.",
        "preprocessed_text": "transfer learning bert",
        "predicted_tags": [
            "attention",
            "natural-language-processing",
            "transfer-learning",
            "transformers"
        ]
    }
    ```

</div>

!!! note
    You'll most likely be using the CLI application to optimize and train you models. We'll cover how to train using compute instances on the cloud from Amazon Web Services (AWS) or Google Cloud Platforms (GCP) in a later lesson. But in the meantime, if you don't have access to GPUs, check out the [optimize.ipynb](https://colab.research.google.com/github/GokuMohandas/applied-ml/blob/main/notebooks/optimize.ipynb){:target="_blank"} notebook for how to train on Google Colab and transfer to local. We essentially run optimization, then train the best model to download and transfer it's arguments and artifacts. Once we have them in our local machine, we can run `tagifai set-artifact-metadata` to match all metadata as if it were run from your machine.

<!-- Citation -->
{% include "cite.md" %}