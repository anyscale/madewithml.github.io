---
description: Organizing our code in Python scripts using OOPs principles.
image: https://madewithml.com/static/images/applied_ml.png
---

:octicons-mark-github-16: [Repository](https://github.com/GokuMohandas/applied-ml){:target="_blank"}

Organizing our code when moving from notebooks to Python scripts.

## Intuition

To have organized code is to have readable, reproducible, scalable and efficient code.

- **readable**: easy for your users, fellow developers and your future self to navigate and extend.
- **reproducible**: replicate the necessary environment and receive consistent results when executing the code.
- **scalable**: easily extend and combine with other applications
- **efficient**: minimizing repetitions minimizes errors and increases efficiency.

## Application

Let's look at what organizing a code base looks like for our [application](https://github.com/GokuMohandas/applied-ml){:target="_blank"}.

### Organizing
There are several ways to organize our code from the notebooks but they're all based on utility. For example, we're organizing our code based on the part of the pipeline (data, training, prediction, etc.):

```bash
tagifai/
├── config.py     - configuration setup
├── data.py       - data processing utilities
├── main.py       - main operations (CLI)
├── models.py     - model architectures
├── predict.py    - inference utilities
├── train.py      - training utilities
└── utils.py      - supplementary utilities
```

Organizing our code base this way also makes it easier for readers to understand (or modify) the code base. We could've also assumed a more granular stance for organization, such as breaking down `data.py` into `split.py`, `preprocess.py`, etc. This might make more sense if we have multiple ways of splitting, preprocessing, etc. but for our task, it's sufficient to be at a higher level.

!!! note
    Another way to supplement organized code is through documentation, which we'll cover in the next lesson.

### Reading
So what's the best way to read a code base like this? We could look at documentation (which we will cover in the next lesson) but that's usually useful if you're looking for specific functions or classes within a script. What if you want to understand the overall functionality and how it's all organized? Well, we can start with the options in `main.py` and dive deeper into the specific utilities. Let's say we wanted to see how a single model is trained, then we'd go to the `train_model` function and inspect each line and build a mental model of the process. For example, when you reach the line:
```python
# Train
artifacts = train.run(args=args)
```
you'll want to go to `train.py` → `run` to see it's operations:
```bash
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

### Virtual environment

Before we step into the code, we need to replicate the environment we were developing in. When we used our [notebook](https://colab.research.google.com/github/GokuMohandas/applied-ml/blob/main/notebooks/tagifai.ipynb){:target="_blank"}, we had a preloaded set of packages (run `!pip list` inside the notebook to see all of them). But now we want to define our environment so we can reproduce it for our Python scripts. There are [many recommended options](https://packaging.python.org/guides/tool-recommendations/){:target="_blank"} when it comes to packaging in Python and we'll be using the traditional and recommended [Pip](https://pip.pypa.io/en/stable/){:target="_blank"}.

!!! note
    I'm a huge fan (and user) of [Poetry](https://python-poetry.org/){:target="_blank"} which is a dependency management and packaging tool but there are still many things in flux. I'm sticking with Pip because it works for our application and don't want to deal with issues like [long resolve periods](https://github.com/python-poetry/poetry/issues/2094){:target="_blank"}.

First thing we'll do is set up a [virtual environment](https://docs.python.org/3/library/venv.html){:target="_blank"} so we can isolate our packages (and versions) necessary for application from our other projects which may have different dependencies. Once we create our virtual environment, we'll activate it and install our required packages.

```bash linenums="1"
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -e .
```

Let's unpack what's happening here:

1. Creating a vitual environment named `venv`
2. Activating our virtual environment. Type `deactivate` to exit out of the virtual environment.
3. Upgrading required packages so we download the latest package wheels.
4. Install from `setup.py` (`-e`, `--editable` installs a project in develop mode)

### Packaging

Let's dive into our [`setup.py`](https://github.com/GokuMohandas/applied-ml/blob/main/setup.py){:target="_blank"} to see how what we're installing inside our virtual environment. First, we're retrieving our required packages from our `requirements.txt` file. While we could place these requirements directly inside setup.py, many applications still look for a `requirements.txt` file so we'll keep it separate.
```python linenums="10"
# Load packages from requirements.txt
with open(Path(BASE_DIR, "requirements.txt"), "r") as file:
    required_packages = [ln.strip() for ln in file.readlines()]
```

The next several lines in our `setup.py` file include some packages required for testing (`test_packages`) and development (`dev_packages`). These will be situationally required when we're testing or developing. For example, a general user of our application won't need to to test or develop so they'll only need the required packages, however, a fellow developer will want both the test and dev packages to extend our code base.

!!! note
    You may notice some packages that we haven't used yet. Don't worry, they'll be used in future lessons and we'll also be adding more as we develop.

The heart of the `setup.py` file is the `setup` object which describes how to set up our package and it's dependencies. The first several lines cover metadata (name, description, etc.) and then we define the requirements. Here we're stating that we require a Python version equal to or above 3.6 and then passing in our required packages to `install_requires`. Finally, we define extra requirements that different types of users may require.

```python linenums="53"
setup(
    ...
    python_requires=">=3.6",
    install_requires=[required_packages],
    extras_require={
        "test": test_packages,
        "dev": test_packages + dev_packages,
    },
    ...
)
```

The final lines of the file define various entry points we can use to interact with the application. Here we define some console scripts (commands) we can type on our terminal to execute certain actions. For example, after we install our package, we can type the command `tagifai` to run the `app` variable inside `tagifai/main.py`.

```python linenums="59"
setup(
    ...
    entry_points={
        "console_scripts": [
            "tagifai = tagifai.main:app",
        ],
    },
)
```

We can install our package for different situations like so:
```bash
python -m pip install -e .            # installs required packages only
python -m pip install -e ".[dev]"     # installs required + dev packages
python -m pip install -e ".[test]"    # installs required + test packages
```

### Makefile

We have just started and there are already so many different commands to keep track of. To help with this, we're going to use a [`Makefile`](https://opensource.com/article/18/8/what-how-makefile){:target="_blank"} which is a automation tool that organizes our compilation commands. Inside our Makefile, we can see a list of functions (help, install, etc.). These functions (also known as `targets`) can sometimes have `prerequisites` that need to be met (can be other targets) and on the next line a ++tab++ followed by a `recipe`.

```bash
target: prerequisites
<TAB> recipe
```

We can execute any of our targets by typing `make <target>`:

<div class="animated-code">

    ```console
    # View all targets
    $ make help
    Usage: tagifai [OPTIONS] COMMAND [ARGS]
    👉  Commands:
        install         : installs required packages.
        install-dev     : installs development requirements.
        install-test    : installs test requirements.

    # Make a target
    $ make install-dev
    python -m pip install -e ".[dev]"
    ...
    ```

</div>
<script src="../../../static/js/termynal.js"></script>

We'll be adding more targets to our Makefile in subsequent lessons (testing, styling, etc.) but there's one more concept to illustrate. A Makefile is called as such because traditionally the `targets` are supposed to be files we can make. However, Makefiles are also commonly used as command shortcuts which can lead to confusion when a file with a certain name exists and a command with the same name exists! For example if you a directory called `docs` and a `target` in your Makefile called `docs`, when you run `make docs` you'll get this message:

<div class="animated-code">

    ```console
    $ make docs
    make: `docs` is up to date.
    ```

</div>

We can fix this by defining a `PHONY` target in our makefile by adding this line:
```bash
# Inside your Makefile
...
docs:
	mkdocs serve

.PHONY: docs
```

Putting this all together, we can now install our package for different situations like so:
```bash
make install         # installs required packages only
make install-dev     # installs required + dev packages
make install-test    # installs required + test packages
```

### Command line interface (CLI)

The `app` that we defined inside our `tagifai/main.py` is created using [Typer](https://typer.tiangolo.com/){:target="_blank"}, an open-source tool for building command line interface (CLI) applications. It starts by initializing the app and then adding the appropriate decorator to each function we wish to use as a CLI command.

!!! note
    We're combining console scripts (from `setup.py`) and our Typer app to create a CLI application but there are many [different ways](https://typer.tiangolo.com/typer-cli/){:target="_blank"} to use Typer as well. We're going to have other programs use our application so our approach works best for that.


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

We can list all the CLI commands for our application like so:

In `main.py` we have the following functions:

- `download-data`: download data from online to local drive.
- `optimize`: optimize a subset of hyperparameters towards an objective.
- `train-model`: train a model using the specified parameters.
- `predict-tags`: predict tags for a give input text using a trained model.


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

    # Help for a specific command
    $ tagifai train-model --help
    Usage: tagifai train-model [OPTIONS]
    Options:
        --args-fp  PATH [default: config/args.json]
        --help     Show this message and exit.

    # Train a model
    $ tagifai train-model --args-fp $PATH
    🚀 Training...
    ```

</div>


With Typer, a function's input arguments automatically get rendered as command line options. For example, our `predict_tags` function consumes `text` and an optional `run_id` as inputs which automatically become arguments for the `predict-tags` CLI command.

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
    We'll cover how to train using compute instances on the cloud from Amazon Web Services (AWS) or Google Cloud Platforms (GCP) in a later lesson. But in the meantime, if you don't have access to GPUs, check out the [optimize.ipynb](https://colab.research.google.com/github/GokuMohandas/applied-ml/blob/main/notebooks/optimize.ipynb){:target="_blank"} notebook for how to train on Colab and transfer to local. We essentially run optimization, then train the best model to download and transfer it's arguments and artifacts. Once we have them in our local machine, we can run `tagifai set-artifact-metadata` to match all metadata as if it were run from your machine.

<!--
```python

```
<pre class="output">

</pre>

<div class="ai-center-all">
    <img src="https://raw.githubusercontent.com/GokuMohandas/madewithml/main/images/applied-ml/solution/suggested_tags.png" width="550" alt="pivot">
</div>
<div class="ai-center-all">
  <small>UX of our hypothetical solution</small>
</div>

{:target="_blank"}
 -->
