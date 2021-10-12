---
template: lesson.html
title: Packaging a Python Codebase
description: Using configurations and virtual environments to create a setting for reproducing results.
keywords: packaging, pip, setup.py, virtual environment, reproducibility, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/mlops.png
repository: https://github.com/GokuMohandas/follow/tree/packaging
---

{% include "styles/lesson.md" %}

## Set up

All of the work that we're doing with Python script is available in the [main repository](https://github.com/GokuMohandas/MLOps){:target="_blank"}, however, it's difficult to follow along this repository because all of the content we cover in this course is available there in one snapshot. So we highly recommend using the [branches](https://github.com/GokuMohandas/follow/branches){:target="_blank"} in the repository to follow along. Each branch's name will match with the lesson's name (ex. this lesson's branch is called [packaging](https://github.com/GokuMohandas/follow/tree/packaging){:target="_blank"}) and we can pull that branch to follow along the respective lesson.

```bash
git clone -b <BRANCH> <REMOTE_REPO_URL> <PATH_TO_PROJECT_DIR>
```

- `<REMOTE_REPO_URL>` is the location of the remote repo (ex. [https://github.com/GokuMohandas/follow](https://github.com/GokuMohandas/follow){:target="_blank"}).
- `<PATH_TO_PROJECT_DIR>` is the name of the local directory you want to clone the project into (ex. mlops).

## Intuition

It's integral to be able to consistently create an environment to develop in so that we can reliably reproduce the same results. To do this, we'll need to explicitly detail all the requirements (python version, packages, etc.) as well as create the environment that will load all the requirements. By doing this, we'll not only be able to consistently reproduce results but also enable others to arrive at the same results.

> We can set up our files below (setup.py & requirements.txt) with just the terminal and any text editor but you may use a [code editor](organization.md#editor){:target="_blank"} as well.

## Virtual environment

When we used our [notebook](https://colab.research.google.com/github/GokuMohandas/MLOps/blob/main/notebooks/tagifai.ipynb){:target="_blank"}, we had a preloaded set of packages (run `!pip list` inside the notebook to see all of them). But now we want to define our environment so we can reproduce it for our Python scripts. There are [many recommended options](https://packaging.python.org/guides/tool-recommendations/){:target="_blank"} when it comes to packaging in Python and we'll be using the traditional and recommended [Pip](https://pip.pypa.io/en/stable/){:target="_blank"}.

> There are many alternative dependency management and packaging tools, such as [Poetry](https://python-poetry.org/){:target="_blank"}, but there are still many things in flux with these newer options. We're going to stick with Pip because it works for our application and don't want to deal with issues like [long resolve periods](https://github.com/python-poetry/poetry/issues/2094){:target="_blank"}.

First thing we'll do is set up a [virtual environment](https://docs.python.org/3/library/venv.html){:target="_blank"} so we can isolate our packages (and versions) necessary for application from our other projects which may have different dependencies. Once we create our virtual environment, we'll activate it and install our required packages.

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Let's unpack what's happening here:

1. Creating a Python virtual environment named `venv`. Use Python 3.7.10 for our project.
2. Activate our virtual environment. Type `deactivate` to exit the virtual environment.
3. Upgrading required packages so we download the latest package wheels.

> We can use [pyenv](https://github.com/pyenv/pyenv){:target="_blank"} to manage different Python versions.

<div class="animated-code">

    ```console
    # Using pyenv to switch between Python versions
    $ python --version
    Python 3.6.9
    $ pyenv versions
    system
    *  3.6.9
    $ pyenv install 3.7.10
    $ pyenv local 3.7.10
    system
    3.6.9
    * 3.7.10 (set by /Users/goku/Documents/madewithml/mlops/.python-version)
    $ python --version
    Python 3.7.10
    ```

</div>
<script src="../../../static/js/termynal.js"></script>

## setup.py

Let's dive into our [`setup.py`](https://github.com/GokuMohandas/MLOps/blob/main/setup.py){:target="_blank"} to see how what we're installing inside our virtual environment.

### Requirements

First, we're retrieving our required packages from our `requirements.txt` file. While we could place these requirements directly inside `setup.py`, many applications still look for a `requirements.txt` file so we'll keep it separate.

```bash
touch requirements.txt setup.py
```

And we'll call these requirements in our setup.py script like so:

```python linenums="10"
# Load packages from requirements.txt
with open(Path(BASE_DIR, "requirements.txt"), "r") as file:
    required_packages = [ln.strip() for ln in file.readlines()]
```

> We've should add packages (with versions) to our `requirements.txt` as we've installed them but if we haven't, you can't just do `pip freeze > requirements.txt` because it dumps the dependencies of all our packages into the file (even the ones we didn't explicitly install). When a certain package updates, the stale dependency will still be there. To mitigate this, there are tools such as [pipreqs](https://github.com/bndr/pipreqs){:target="_blank"}, [pip-tools](https://github.com/jazzband/pip-tools){:target="_blank"}, [pipchill](https://github.com/rbanffy/pip-chill){:target="_blank"}, etc. that will only list the packages that are not dependencies. However, if you're separating packages for different environments, then these solutions are limited as well.

The next several lines in our `setup.py` file include some packages required for testing (`test_packages`) and development (`dev_packages`). These will be situationally required when we're testing or developing. For example, a general user of our application won't need to to test or develop so they'll only need the required packages, however, a technical developer will want both the test and dev packages to extend our code base.

> We have test and dev packages separated because in our [CI/CD lesson](cicd.md){:target="_blank"}, we'll be using [GitHub actions](https://github.com/features/actions){:target="_blank"} that will only be testing our code so we wanted to specify a way to load only the required packages for testing.

### Setup

The heart of the `setup.py` file is the `setup` object which describes how to set up our package and it's dependencies. The first several lines cover metadata (name, description, etc.) and then we define the requirements. Here we're stating that we require a Python version equal to or above 3.6 and then passing in our required packages to `install_requires`. Finally, we define extra requirements that different types of users may require.

```python linenums="53"
setup(
    ...
    python_requires=">=3.7",
    install_requires=[required_packages],
    extras_require={
        "test": test_packages,
        "dev": test_packages + dev_packages,
    },
    ...
)
```

### Entry points

The final lines of the file define various entry points we can use to interact with the application. Here we define some console scripts (commands) we can type on our terminal to execute certain actions. For example, after we install our package, we can type the command `tagifai` to run the `app` variable inside [`tagifai/main.py`](https://github.com/GokuMohandas/MLOps/blob/main/tagifai/main.py){:target="_blank"}.

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

## Usage

We can install our package for different situations like so:
```bash
python -m pip install -e .            # installs required packages only
python -m pip install -e ".[dev]"     # installs required + dev packages
python -m pip install -e ".[test]"    # installs required + test packages
```

> The `-e` or `--editable` flag installs a project in develop mode so we can make changes without having to reinstall packages.

There are many alternatives to a setup.py file such as the [`setup.cfg`](https://docs.python.org/3/distutils/configfile.html){:target="_blank"} and the more recent (and increasingly adopted) [pyproject.toml](https://github.com/GokuMohandas/MLOps/blob/main/pyproject.toml){:target="_blank"}.

<!-- Citation -->
{% include "cite.md" %}