---
template: lesson.html
title: Command-Line Interface (CLI)
description: Using a command-line interface (CLI) to organize our application's workloads.
keywords: cli, application, mlops, machine learning
image: https://madewithml.com/static/images/mlops.png
repository: https://github.com/GokuMohandas/Made-With-ML
---

{% include "styles/lesson.md" %}

## Intuition

In the [previous lesson](scripting.md){:target="_blank"}, we organized our code from our notebook into individual Python scripts. We moved our [functions and classes](scripting.md#functions-and-classes){:target="_blank"} into their respective scripts and also created new [workload functions](scripting.md#workloads) to execute the main ML workloads (ex. `train_model` function from `madewithml/train.py` script). We now want to enable users to execute these workloads from the terminal without having to know anything about our code itself.

## Methods

One way to execute these workloads is to import the functions in the Python script and execute them one at a time:

```python linenums="1"
from madewithml import train
train.train_model(experiment_name="llm", ...)
```

> **Caution**: Don't forget to run `export PYTHONPATH=$PYTHONPATH:$PWD` in your terminal to ensure that Python can find the modules in our project.

While this may seem simple, it still requires us to import packages, identify the input arguments, etc. Therefore, another alternative is to place the main function call under a `#!python if __name__ == "__main__"` conditional so that it's only executed when we run the script directly. Here we can pass in the input arguments directly into the function in the code.

```python linenums="1"
# madewithml/train.py
if __name__ == "__main__":
    train_model(experiment_name="llm", ...)
```
Which we can call from the terminal like so:
```bash
python madewithml/train.py
```

However, the limitation here is that we can't choose which function from a particular script to execute. We have to set the one we want to execute under the `#!python if __name__ == "__main__"` conditional. It's also very rigid since we have to set the input argument values in the code, unless we use a library like [argparse](https://docs.python.org/3/library/argparse.html){:target="_blank"}.

```python linenums="1"
# madewithml/serve.py
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", help="run ID to use for serving.")
    parser.add_argument("--threshold", type=float, default=0.9, help="threshold for `other` class.")
    args = parser.parse_args()
    ray.init()
    serve.run(ModelDeployment.bind(run_id=args.run_id, threshold=args.threshold))
```
Which we can call from the terminal like so (note that `--threshold` is optional since it has a default value):
```bash
python madewithml/serve.py --run_id $RUN_ID
```

> We use [argparse](https://docs.python.org/3/library/argparse.html){:target="_blank"} in our `madewithml/serve.py` script because it's the only workload in the script and it's a one-line function call (`serve.run()`).

Compared to using functions or the `__main__` conditional, a much better user experience would be to execute these workloads from the terminal. In this lesson, we'll learn how to build a command-line interface (CLI) so that execute our main ML workloads.

## Typer

We're going to create our CLI using [Typer](https://typer.tiangolo.com/){:target="_blank"}. It's as simple as initializing the app and then adding the appropriate decorator to each function operation we wish to use as a CLI command in our script:

```python linenums="1" hl_lines="5"
import typer
from typing_extensions import Annotated
app = typer.Typer()

@app.command()
def train_model(
    experiment_name: Annotated[str, typer.Option(help="name of the experiment.")] = None,
    ...):
    pass


if __name__ == "__main__":
    app()
```

### Inputs

You may notice that our function inputs have a lot of information besides just the input name. We'll cover typing (`str`, `List`, etc.) in our [documentation lesson](documentation.md#typing){:target="_blank"} but for now, just know that `Annotated` allows us to specify metadata about the input argument's type and details about the (required) option ([`typer.Option`](https://typer.tiangolo.com/tutorial/options/required/){:target="_blank"}).

> We make all of our input arguments optional so that we can explicitly define them in our CLI commands (ex. `--experiment-name`).

We can also add some helpful information about the input parameter (with `typer.Option(help="...")`) and a default value (ex. `None`).

### Usage

With our CLI commands defined and our input arguments enriched, we can execute our workloads. Let's start by executing our `train_model` function by assuming that we don't know what the required input parameters are. Instead of having to look in the code, we can just do the following:

```bash
python madewithml/train.py --help
```

<pre class="output">
Usage: train.py [OPTIONS]
Main train function to train our model as a distributed workload.
</pre>

<div class="ai-center-all">
    <img src="/static/images/mlops/cli/options.png" width="800" alt="options">
</div>

We can follow this helpful message to execute our workload with the appropriate inputs.

```bash
export EXPERIMENT_NAME="llm"
export DATASET_LOC="https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/dataset.csv"
export TRAIN_LOOP_CONFIG='{"dropout_p": 0.5, "lr": 1e-4, "lr_factor": 0.8, "lr_patience": 3}'
python madewithml/train.py \
    --experiment-name "$EXPERIMENT_NAME" \
    --dataset-loc "$DATASET_LOC" \
    --train-loop-config "$TRAIN_LOOP_CONFIG" \
    --num-workers 1 \
    --cpu-per-worker 10 \
    --gpu-per-worker 1 \
    --num-epochs 10 \
    --batch-size 256 \
    --results-fp results/training_results.json
```

> Be sure to check out our [`README.md`](https://github.com/GokuMohandas/Made-With-ML/blob/main/README.md){:target="_blank"} file as it has examples of all the CLI commands for our ML workloads (train, tune, evaluate, inference and serve).

<!-- Course signup -->
{% include "templates/signup.md" %}

<!-- Citation -->
{% include "templates/cite.md" %}