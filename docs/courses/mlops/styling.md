---
template: lesson.html
title: Styling and Formatting Code
description: Style and formatting conventions to keep your code looking consistent.
keywords: styling, formatting, pep8, black, isort, flake8, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/applied_ml.png
repository: https://github.com/GokuMohandas/MLOps
---

{% include "styles/lesson.md" %}

## Intuition

> Code is read more often than it is written. -- [Guido Van Rossum](https://gvanrossum.github.io/){:target="_blank"} (author of Python)

When we write a piece of code, it's almost never the last time we see it or the last time it's edited. So we need to both explain what's going on ([documentation](documentaiton.md) but also make it easy to read. One of the easiest ways to make code more readable is to follow consistent style and formatting conventions.

There are many options when it comes to Python style conventions to adhere to, but most are based on [PEP8](https://www.python.org/dev/peps/pep-0008/) (link walks through the different components (blank lines, imports, etc.) that conventions were written for). You'll notice that different teams will default to different conventions and that's ok. The most important aspects are that everybody is consistently following the same convection and that there are pipelines in place to ensure that consistency. Let's see what this looks like in our application.

## Application

## Tools

We will be using a very popular blend of style and formatting conventions that makes some very opinionated decisions on our behalf (with configurable options).

- [`Black`](https://black.readthedocs.io/en/stable/){:target="_blank"}: an in-place reformatter that [adheres](https://black.readthedocs.io/en/stable/the_black_code_style.html){:target="_blank"} to PEP8. We can explore examples of the formatting adjustments that Black makes in this [demo](https://black.now.sh/){:target="_blank"}.
- [`isort`](https://pycqa.github.io/isort/){:target="_blank"}: sorts and formats import statements inside Python scripts.
- [`flake8`](https://flake8.pycqa.org/en/latest/index.html){:target="_blank"}: a code linter that with stylistic conventions that adhere to PEP8.

We installed all of these as they were defined in out `setup.py` file under `dev_packages`.
```bash linenums="1"
"black==20.8b1",
"flake8==3.8.3",
"isort==5.5.3",
```

## Configuration

Before we can properly use these tools, we'll have to configure them because they may have some discrepancies amongst them since they follow slightly different conventions that extend from PEP8. To configure Black, we could just pass in options using the [CLI method](https://black.readthedocs.io/en/stable/installation_and_usage.html#command-line-options){:target="_blank"}, but it's much more efficient (especially so others can easily find all our configurations) to do this through a file. So we'll need to create a [pyproject.toml](https://github.com/GokuMohandas/MLOps/blob/main/pyproject.toml){:target="_blank"} file and place the following configurations:

```toml linenums="1"
# Black formatting
[tool.black]
line-length = 79
include = '\.pyi?$'
exclude = '''
/(
      \.eggs         # exclude a few common directories in the
    | \.git          # root of the project
    | \.hg
    | \.mypy_cache
    | \.tox
    | \.venv
    | _build
    | buck-out
    | build
    | dist
  )/
'''
```

!!! note
    The [pyproject.toml](https://www.python.org/dev/peps/pep-0518/#file-format){:target="_blank"} was created to establish a more human-readable configuration file that is meant to replace a `setup.py` or `setup.cfg` file and is increasingly widely adopted by many open-source libraries.

Here we're telling Black that our maximum line length should be 79 characters and to include and exclude certain file extensions. We're going to follow the same configuration steps in our pyproject.toml file for configuring isort as well. Place the following configurations right below Black's configurations:

```bash linenums="20"
# iSort
[tool.isort]
profile = "black"
line_length = 79
multi_line_output = 3
include_trailing_comma = true
skip_gitignore = true
virtual_env = "venv"
```

Though there is a [complete list](https://pycqa.github.io/isort/docs/configuration/options/){:target="_blank"} of configuration options for isort, we've decided to set these explicitly so it works well with Black.

Lastly, we'll set up flake8 but this time we need to create a separate `.flake8` file and place the following configurations:

```toml linenums="1"
[flake8]
exclude = venv
ignore = E501, W503, E226
max-line-length = 79

# E501: Line too long
# W503: Line break occurred before binary operator
# E226: Missing white space around arithmetic operator
```

Here we setting up some configurations like before but we're including an `ignore` option to ignore certain [flake8 rules](https://www.flake8rules.com/){:target="_blank"} so everything works with our Black and isort configurations.

Besides defining configuration options here, which are applied globally, we can also choose specifically ignore certain conventions on a line-by-line basis. Here are a few example in our code of where we utilized this method:

```python linenums="1"
# tagifai/config.py
import pretty_errors  # NOQA: F401 (imported but unused)
...
# app/cli.py
with mlflow.start_run(
    run_name="cnn"
) as run:  # NOQA: F841 (assigned to but never used)
```

By placing the `# NOQA: <error-code>` on a line, we're telling flake8 to do NO Quality Assurance for that particular error on this line.

## Usage

To use these tools that we've configured, we could run these commands individually (the `.` signifies that the configuration file for that package is in the current directory) but we can also use the `style` target command from our `Makefile`:
```bash linenums="1"
black .
flake8
isort .
```
<pre class="output">
black .
All done! ✨ 🍰 ✨
9 files left unchanged.
flake8
isort .
Fixing ...
</pre>

!!! note
    We may sometimes forget to run these style checks after we finish development. We'll cover how to automate this process using [pre-commit](https://pre-commit.com/){:target="_blank"} so that these checks are automatically executed whenever we want to commit our code.


<!-- Citation -->
{% include "cite.md" %}