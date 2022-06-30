---
template: lesson.html
title: Packaging a Python Codebase
description: Using configurations and virtual environments to create a setting for reproducing results.
keywords: packaging, pip, setup.py, virtual environment, reproducibility, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/mlops.png
repository: https://github.com/GokuMohandas/mlops-course
---

{% include "styles/lesson.md" %}

## Intuition

So far, we've been working inside notebooks, which has allowed us to train a model very quickly. However, notebooks are not easy to put into production and we don't always have control over the environment (ex. Google Colab updates its packages periodically). When we used our [notebook](https://colab.research.google.com/github/GokuMohandas/mlops-course/blob/main/notebooks/tagifai.ipynb){:target="_blank"}, we had a preloaded set of packages (run `!pip list` inside the notebook to see all of them). But now we want to explicitly define our environment so we can reproduce it locally (for us and team members) and when we deploy to production. There are [many recommended tools](https://packaging.python.org/guides/tool-recommendations/){:target="_blank"} for when it comes to packaging in Python and we'll be using the tried and tested [Pip](https://pip.pypa.io/en/stable/){:target="_blank"}.

> There are many alternative dependency management and packaging tools, such as [Poetry](https://python-poetry.org/){:target="_blank"}, but there are still many things in flux with these newer options. We're going to stick with Pip because it works for our application and don't want to deal with issues like [long resolve periods](https://github.com/python-poetry/poetry/issues/2094){:target="_blank"}.

## Terminal

Before we can start packaging, we need a way to create files and run commands. We can do this via the terminal, which will allow us to run languages such as bash, zsh, etc. to execute commands. All the commands we run should be the same regardless of your operating system or command-line interface (CLI) programming language.

!!! tip
    We highly recommend you use [iTerm2](https://iterm2.com/){:target="_blank"} (Mac) or [ConEmu](https://conemu.github.io/){:target="_blank"} (Windows) in lieu of the default terminal for its rich features.

## Project

While we'll organize our code from our notebook to scripts in the [next lesson](organization.md){:target="_blank"}, we'll create the main project directory now so that we can save our packaging components there. We'll call our main project directory `mlops` but feel free to name it anything you'd like.

```bash
# Create and change into the directory
mkdir mlops
cd mlops
```

## Python

First thing we'll do is set up the correct version of Python. We'll be using version `3.7.10` specifically but any version of Python 3 should work. Though you could download different Python versions online, we highly recommend using a version manager such as [pyenv](https://github.com/pyenv/pyenv){:target="_blank"}.

> Pyenv works for Mac & Linux, but if you're on windows, we recommend using [pyenv-win](https://github.com/pyenv-win/pyenv-win){:target="_blank"}.

<div class="animated-code">
    ```console
    # Install pyenv
    $ brew install pyenv

    # Check version of python
    $ python --version
    Python 3.6.9

    # Check available versions
    $ pyenv versions
    system
    *  3.6.9

    # Install new version
    $ pyenv install 3.7.10
    ---> 100%

    # Set new version
    $ pyenv local 3.7.10
    system
    3.6.9
    * 3.7.10

    # Validate
    $ python --version
    Python 3.7.10
    ```
</div>
<script src="../../../static/js/termynal.js"></script>

> We highly recommend using Python `3.7.10` because, while using another version of Python will work, we may face some conflicts with certain package versions that may need to be resolved.

## Virtual environment

Next, we'll set up a [virtual environment](https://docs.python.org/3/library/venv.html){:target="_blank"} so we can isolate the required packages for our application. This will also keep components separated from other projects which may have different dependencies. Once we create our virtual environment, we'll activate it and install our required packages.

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install pip setuptools wheel
```

Let's unpack what's happening here:

1. Creating a Python virtual environment named `venv`.
2. Activate our virtual environment. Type `deactivate` to exit the virtual environment.
3. Upgrading required packages so we download the latest package wheels.

Our virtual environment directory `venv` should be visible when we list the directories in our project:

```bash
ls
```

<pre class="output bash-output">
mlops/
├── <b>venv/</b>
├── requirements.txt
└── setup.py
</pre>

We'll know our virtual environment is active because we will it's name on the terminal. We can further validate by making sure `pip freeze` returns nothing.
```bash
(venv) ➜  mlops: pip freeze
```

### Requirements

We'll create a separate file called `requirements.txt` where we'll specify the packages (with their versions) that we want to install. While we could place these requirements directly inside `setup.py`, many applications still look for a separate `requirements.txt`.

```bash
touch requirements.txt
```

We should be adding packages with their versions to our `requirements.txt` as we require them for our project. It's inadvisable to install all packages and then do `pip freeze > requirements.txt` because it dumps the dependencies of all our packages into the file (even the ones we didn't explicitly install). To mitigate this, there are tools such as [pipreqs](https://github.com/bndr/pipreqs){:target="_blank"}, [pip-tools](https://github.com/jazzband/pip-tools){:target="_blank"}, [pipchill](https://github.com/rbanffy/pip-chill){:target="_blank"}, etc. that will only list the packages that are not dependencies. However, they're dependency resolving is not always accurate and don't work when you want to separate packages for different tasks (developing, testing, etc.).

!!! tip
    If we experience conflicts between package versions, we can relax constraints by specifying that the package needs to be above a certain version, as opposed to the exact version. We could also specify no version for all packages and allow pip to resolve all conflicts. And then we can see which version were actually installed and add that information to our `requirements.txt` file.
    ```bash
    # requirements.txt
    <PACKAGE>==<VERSION>  # exact version
    <PACKAGE>==<VERSION>  # above version
    <PACKAGE>             # no version
    ```

### Setup

Let's create a file called `setup.py` to provide instructions on how to set up our virtual environment.

```bash
touch setup.py
```

```python linenums="1"
# setup.py
from pathlib import Path
from setuptools import find_namespace_packages, setup
```

We'll start by extracting the require packaged from `requirements.txt`:

```python linenums="1"
# Load packages from requirements.txt
BASE_DIR = Path(__file__).parent
with open(Path(BASE_DIR, "requirements.txt"), "r") as file:
    required_packages = [ln.strip() for ln in file.readlines()]
```

The heart of the `setup.py` file is the `setup` object which describes how to set up our package and it's dependencies. Our package will be called `tagifai` and it will encompass all the requirements needed to run it. The first several lines cover [metadata](https://setuptools.pypa.io/en/latest/userguide/declarative_config.html#metadata){:target="_blank"} (name, description, etc.) and then we define the requirements. Here we're stating that we require a Python version equal to or above 3.7 and then passing in our required packages to `install_requires`.

```python linenums="1"
# setup.py
setup(
    name="tagifai",
    version=0.1,
    description="Classify machine learning projects.",
    author="Goku Mohandas",
    author_email="goku@madewithml.com",
    url="https://madewithml.com/",
    python_requires=">=3.7",
    install_requires=[required_packages],
)
```

??? example "View setup.py"

    ```python linenums="1"
    from pathlib import Path
    from setuptools import find_namespace_packages, setup

    # Load packages from requirements.txt
    BASE_DIR = Path(__file__).parent
    with open(Path(BASE_DIR, "requirements.txt"), "r") as file:
        required_packages = [ln.strip() for ln in file.readlines()]

    # Define our package
    setup(
        name="tagifai",
        version=0.1,
        description="Classify machine learning projects.",
        author="Goku Mohandas",
        author_email="goku@madewithml.com",
        url="https://madewithml.com/",
        python_requires=">=3.7",
        packages=find_namespace_packages(),
        install_requires=[required_packages],
    )
    ```

## Usage

We don't have any packages defined in our `requirements.txt` file but if we did, we can use the `setup.py` file, we can now install our packages like so:

```bash
python3 -m pip install -e .            # installs required packages only
```

<pre class="output bash-output">
Obtaining file:///Users/goku/Documents/madewithml/mlops
  Preparing metadata (setup.py) ... done
Installing collected packages: tagifai
  Running setup.py develop for tagifai
Successfully installed tagifai-0.1
</pre>

> The `-e` or `--editable` flag installs a project in develop mode so we can make changes without having to reinstall packages.

Now if we do `pip freeze` we should see that `tagifai` is installed.

```bash
pip freeze
```

<pre class="output bash-output">
# Editable install with no version control (tagifai==0.1)
-e /Users/goku/Documents/madewithml/mlops
</pre>

and we should also see a `tagifai.egg-info` directory in our project directory:

<pre class="output bash-output">
mlops/
├── <b>tagifai.egg-info/</b>
├── venv/
├── requirements.txt
└── setup.py
</pre>

There are many alternatives to a setup.py file such as the [`setup.cfg`](https://setuptools.pypa.io/en/latest/userguide/declarative_config.html){:target="_blank"} and the more recent [pyproject.toml](https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html){:target="_blank"}.

<!-- Citation -->
{% include "cite.md" %}