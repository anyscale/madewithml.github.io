---
template: lesson.html
title: Organizing Machine Learning Code
description: Organizing our code when moving from notebooks to Python scripts.
keywords: git, github, organization, repository, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/mlops.png
repository: https://github.com/GokuMohandas/follow/tree/organization
---

{% include "styles/lesson.md" %}

## Intuition

To have organized code is to have readable, reproducible, scalable and efficient code. We'll cover all of these concepts throughout the scripting lessons.

## Editor
Before we can start moving our code from notebooks to proper Python scripts, we need a space to do so. There are several options for code editors, such as [Atom](https://atom.io/){:target="_blank"}, [Sublime](https://www.sublimetext.com/){:target="_blank"}, [PyCharm](https://www.jetbrains.com/pycharm/){:target="_blank"}, [Vim](https://www.vim.org/){:target="_blank"}, etc. and they all offer unique features while providing the basic operations for code editing and execution. We will be using [Visual Studio Code (VSCode)](https://code.visualstudio.com/){:target="_blank"} to edit and execute our code for it's simplicity, language support, add-ons and growing industry adoption.

> You are welcome to use any editor you want but we will be using some add-ons that may be specific to VSCode.

1. Install VSCode from source for your system: [https://code.visualstudio.com/](https://code.visualstudio.com/){:target="_blank"}
2. Open the Command Palette (`F1` or ++command++ + ++shift++ + `P` on mac) &rarr; type in "Preferences: Open Settings (UI)" &rarr; hit ++return++
3. Adjust any relevant settings you want to (spacing, font-size, etc.)
4. Install [VSCode extensions](https://marketplace.visualstudio.com/){:target="_blank"} (use the lego blocks icon on the editor's left panel)

??? quote "Recommended extensions"
    I recommend installing these extensions, which you can by copy/pasting this command:
    ```bash
    code --install-extension 74th.monokai-charcoal-high-contrast
    code --install-extension alefragnani.project-manager
    code --install-extension bierner.markdown-preview-github-styles
    code --install-extension bradgashler.htmltagwrap
    code --install-extension christian-kohler.path-intellisense
    code --install-extension euskadi31.json-pretty-printer
    code --install-extension formulahendry.auto-close-tag
    code --install-extension formulahendry.auto-rename-tag
    code --install-extension kamikillerto.vscode-colorize
    code --install-extension mechatroner.rainbow-csv
    code --install-extension mikestead.dotenv
    code --install-extension mohsen1.prettify-json
    code --install-extension ms-azuretools.vscode-docker
    code --install-extension ms-python.python
    code --install-extension ms-python.vscode-pylance
    code --install-extension ms-vscode.sublime-keybindings
    code --install-extension njpwerner.autodocstring
    code --install-extension PKief.material-icon-theme
    code --install-extension redhat.vscode-yaml
    code --install-extension ritwickdey.live-sass
    code --install-extension ritwickdey.LiveServer
    code --install-extension shardulm94.trailing-spaces
    code --install-extension streetsidesoftware.code-spell-checker
    code --install-extension zhuangtongfa.material-theme
    ```

    If you add your own extensions and want to share it with others, just run this command to generate the list of commands:
    ```bash
    code --list-extensions | xargs -L 1 echo code --install-extension
    ```

Once we're all set up with VSCode, we can start by creating our project directory, which we'll use to organize all our scripts. There are many ways to start a project, but here's my recommended path:

1. Use a terminal to create a directory (`#!bash mkdir <PROJECT_NAME>`).
2. Change into the project directory you just made (`#!bash cd <PROJECT_NAME>`).
3. Start VSCode from this directory by typing `#!bash code .`
> To open VSCode directly from the terminal with a `#!bash code $PATH` command, open the Command Palette (`F1` or ++command++ + ++shift++ + `P` on mac) &rarr; type "Shell Command: Install 'code' command in PATH" &rarr; hit ++return++ &rarr; restart the terminal.
4. Open a terminal within VSCode (`View` > `Terminal`) to continue creating scripts (`#!bash touch <FILE_NAME>`) or additional subdirectories (`#!bash mkdir <SUBDIR>`) as needed.

<div class="ai-center-all">
    <img src="/static/images/mlops/organization/vscode.png" width="650" alt="vscode">
</div>

## Organizing
There are several ways to organize our code when we're going from the notebooks to scripts but they're all based on utility. For example, we're organizing our code based on pipeline components (data processing, training, evaluation, prediction, etc.):

```bash
tagifai/
├── data.py       - data processing utilities
├── eval.py       - evaluation components
├── main.py       - training/optimization operations
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
        ```bash
        pip install git+https://github.com/GokuMohandas/MLOps#egg=tagifai
        ```

### Utilities

Now that we've organized our functions and classes, it's time to create some new functions to encapsulate the ad-hoc processes in our notebooks. Recall that we repeatedly performed actions such as setting the device, reading from a JSON file, etc. We should organize these general [utilities](https://github.com/GokuMohandas/MLOps/blob/main/tagifai/utils.py){:target="_blank"} as separate functions that we can reuse later on.

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

> As we move code into our scripts, we can format them to look via:  open the Command Palette (`F1` or ++command++ + ++shift++ + `P` on mac) &rarr; type "Format Document" &rarr; hit ++return++. Follow the same instructions by type "Prettify JSON" to format JSON documents.

### Operations

With all of our code modularized and organized, we're ready to start crafting functions that are responsible for our system's main operations. We can organize these in a [main.py](https://github.com/GokuMohandas/MLOps/blob/main/tagifai/main.py){:target="_blank"} script. These are some of the major operations in our script and we'll be adding more later as we go through the future lessons.

- `#!python load_data()`
- `#!python compute_features(params_fp)`
- `#!python optimize(params_fp, study_name, num_trials=100)`
- `#!python train_model(params_fp, experiment_name, run_name)`
- `#!python load_artifacts(run_id, device)`
- `#!python delete_experiment(experiment_name)`

We can test all of these via the terminal or with a separate Python script etc. Since this code isn't a perfect transplant from the notebooks, we may need to run it several times and resolve import issues, data passes, etc.

```python linenums="1"
from pathlib import Path
from config import config
from tagifai import main, utils

# Load data
main.load_data()

# Compute features
main.compute_features()

# Train model (test)
params_fp = Path(config.CONFIG_DIR, "test_params.json")
experiment_name = "test"
main.train_model(params_fp, experiment_name=experiment_name, run_name="model")

# Delete test experiment
utils.delete_experiment(experiment_name=experiment_name)
```

<pre class="output">
[01/01/20 16:36:49] INFO     ✅ Loaded data!
[01/01/20 16:36:49] INFO     ✅ Computed features!
[01/01/20 16:36:49] INFO     Run ID: b39c3a8d2c3c494984a3fa2d9d670402
[01/01/20 16:36:49] INFO     Epoch: 1 | train_loss: 0.00744, val_loss: 0.00648, lr: 1.02E-04, _patience: 10
[01/01/20 16:36:49] INFO     {
                               "precision": 0.5625,
                               "recall": 0.03125,
                               "f1": 0.05921052631578947,
                               "num_samples": 32.0
                             }
[01/01/20 16:36:49] INFO     ✅ Deleted experiment test!
</pre>

> Some developers may prefer to interact with the main operations through a command-line interface. We'll modify out main.py script in our ([CLI lesson](cli.md){:target="_blank"} to enable this.

## Config

To make our main operations work, we've also defined a [config directory](https://github.com/GokuMohandas/MLOps/blob/main/config){:target="_blank"} that has parameters for training, running a small test for training, etc. It also include the [config.py](https://github.com/GokuMohandas/MLOps/blob/main/config/config.py){:target="_blank"} script which specifies locations of our model stores (as well other stores that we will cover later), [logging](logging.md){:target="_blank"}, stopwords, etc.

```bash
config/
├── config.py       - configuration setup
├── params.json     - training parameters
└── test_params.py  - training test parameters
```

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
