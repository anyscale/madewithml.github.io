---
template: lesson.html
title: Command Line Interface (CLI) Applications
description: Using a command line interface (CLI) application to organize our application's processes.
keywords: cli, application, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/mlops.png
repository: https://github.com/GokuMohandas/MLOps
---

{% include "styles/lesson.md" %}

## Intuition

We want to enable others to be able to interact with our application without having to dig into the code and execute functions one at a time. One method is to build a CLI application that allows for interaction via any shell. It should designed such that we can see all possible operations as well the appropriate assistance needed for configuring options and other arguments for each of those operations. Let's see what a CLI looks like for our application which has many different commands (training, prediction, etc.)

```bash linenums="1"
app/
└── cli.py        - CLI app
tagifai/
├── data.py       - data processing utilities
├── ...
└── utils.py      - supplementary utilities
```

## Application

The `app` that we defined inside our [`cli.py`](https://github.com/GokuMohandas/MLOps/tree/main/app/cli.py){:target="_blank"} script is created using [Typer](https://typer.tiangolo.com/){:target="_blank"}, an open-source tool for building command line interface (CLI) applications. It starts by initializing the app and then adding the appropriate decorator to each function we wish to use as a CLI command.

```python linenums="1"
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
    ```python linenums="1"
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

In [`cli.py`](https://github.com/GokuMohandas/MLOps/tree/main/app/cli.py){:target="_blank"} script we have define the following commands. You may see other operations as well but we can skip those for now because we'll be adding those later. But at this point in the course, we're concerned with the operations to load the data, create features, optimize, train and predict.

- `download-data`: download data from online to local drive.
- `compute-features`: compute and save features for training.
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
        download-data     Download data from online to local drive.
        compute-features  Compute and save features for training.
        optimize          Optimize a subset of hyperparameters towards ...
        train-model       Predict tags for a give input text using a ...
        predict-tags      Train a model using the specified parameters.
        ...
    ```

</div>
<script src="../../../static/js/termynal.js"></script>

## Arguments

With Typer, a function's input arguments automatically get rendered as command line options. For example, our `predict_tags` function consumes `text` and an optional `model_dir` as inputs which automatically become arguments for the `predict-tags` CLI command.

```python linenums="1"
@app.command()
def predict_tags(
    text: Optional[str] = "Transfer learning with BERT for self-supervised learning",
    run_id: Optional[str] = open(Path(config.MODEL_DIR, "run_id.txt")).read(),
) -> Dict:
    """Predict tags for a give input text using a trained model.

    Warning:
        Make sure that you have a trained model first!

    Args:
        text (str, optional): Input text to predict tags for.
                              Defaults to "Transfer learning with BERT for self-supervised learning".
        run_id (str, optional): ID of the model run to load artifacts. Defaults to run ID in config.MODEL_DIR.

    Raises:
        ValueError: Run id doesn't exist in experiment.

    Returns:
        Predicted tags for input text.
    """
    # Predict
    artifacts = main.load_artifacts(run_id=run_id)
    prediction = predict.predict(texts=[text], artifacts=artifacts)
    logger.info(json.dumps(prediction, indent=2))

    return prediction
```

<div class="animated-code">

    ```console
    # Help for a specific command
    $ tagifai predict-tags --help

    Usage: tagifai predict-tags [OPTIONS]
    ...
    Options:
        --text TEXT    [default: Transfer learning with BERT for self-supervised learning]
        --run-id TEXT  [default: dce5cc211fbb474e9b86af40939be0ca]
        --help         Show this message and exit.
    ```
</div>

## Usage

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
    You'll most likely be using the CLI application to optimize and train your models. If you don't have access to GPUs (personal machine, AWS, GCP, etc.), check out the [optimize.ipynb](https://colab.research.google.com/github/GokuMohandas/MLOps/blob/main/notebooks/optimize.ipynb){:target="_blank"} notebook for how to train on Google Colab and transfer the entire MLFlow experiment to your local machine. We essentially run optimization, then train the best model to download and transfer it's artifacts.

<!-- Citation -->
{% include "cite.md" %}