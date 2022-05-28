---
template: lesson.html
title: Command Line Interface (CLI) Applications
description: Using a command line interface (CLI) application to organize our application's processes.
keywords: cli, application, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/mlops.png
repository: https://github.com/GokuMohandas/follow/tree/cli
---

{% include "styles/lesson.md" %}

## Intuition

When it comes to serving our models, we need to think about exposing the application's functionality to ourselves, team members and ultimately our end users. And the interfaces to achieve this will be different. Recall from our [Organization lesson](organization.md){:target="_blank"} where we executed our [main operations](organization.md#operations){:target="_blank"} via the terminal and Python interpreter.

```python
from tagifai import main
main.load_data()
```

or the alternative was to call the operation explicity inside our `main.py` file:

```python
# tagifai/main.py
if __name__ == "__main__":
    load_data()
```

This becomes extremely tedious having to dig into the code and execute functions one at a time. A solution is to build a command line interface (CLI) application that allows for interaction at the operational level. It should designed such that we can view all possible operations (and their required arguments) and execute them from the shell.

## Application

We're going to create our CLI using [Typer](https://typer.tiangolo.com/){:target="_blank"}:

```bash
pip install typer==0.4.1
```

```bash
# requirements.txt
typer==0.4.1
```

It's as simple as initializing the app and then adding the appropriate decorator to each function operation we wish to use as a CLI command in our `main.py`:

```python linenums="1"
# tagifai/main.py
import typer
app = typer.Typer()
```

```python linenums="1" hl_lines="1"
@app.command()
def load_data():
    ...
```

We'll repeat the same for all the other functions we want to access via the CLI: `load_data()`, `label_data()`, `train_model()`, `optimize()`, `predict_tag()`.

## Commands

To use our CLI app, we can first view the available commands thanks to the decorators we added to certain functions we wanted to expose to the CLI:

```bash
python tagifai/main.py --help
```

> Typer also comes with a utility tool called [typer-cli](https://typer.tiangolo.com/typer-cli/){:target="_blank"} but there are some dependency conflicts with our other libraries so we won't be using it.

<pre class="output">
Usage: main.py [OPTIONS] COMMAND [ARGS]...

Options:
  --help       Show this message and exit.

Commands:
  label-data   Label data with constraints.
  load-data    Load data from URLs and save to local drive.
  optimize     Optimize hyperparameters.
  predict-tag  Predict tag for text.
  train-model  Train a model given arguments.
</pre>

## Arguments

With Typer, a function's input arguments automatically get rendered as command line options. For example, our `predict_tags` function consumes `text` and an optional `run_id` as inputs which automatically become arguments for the `predict-tags` CLI command.

```python linenums="1"
@app.command()
def predict_tag(text: str, run_id: str = None) -> None:
    """Predict tag for text.

    Args:
        text (str): input text to predict label for.
        run_id (str, optional): run id to load artifacts for prediction. Defaults to None.
    """
    if not run_id:
        run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
    artifacts = load_artifacts(run_id=run_id)
    prediction = predict.predict(texts=[text], artifacts=artifacts)
    logger.info(json.dumps(prediction, indent=2))
```

But we can also ask for help with this specific command without having to go into the code:

```bash
python tagifai/main.py predict-tag --help
```

<pre class="output">
Usage: main.py predict-tag [OPTIONS] TEXT

  Predict tag for text.

  Args:
    text (str): input text to predict label for.
    run_id (str, optional): run id to load artifacts for prediction. Defaults to None.

Arguments:
  TEXT  [required]

Options:
  --run-id TEXT
  --help   Show this message and exit.
</pre>

## Usage

Finally, we can execute the specific command with all the arguments:

```bash
python tagifai/main.py predict-tag "Transfer learning with transformers for text classification."
```

<pre class="output">
[
    {
        "input_text": "Transfer learning with transformers for text classification.",
        "predicted_tag": "natural-language-processing"
    }
]
</pre>

<!-- Citation -->
{% include "cite.md" %}