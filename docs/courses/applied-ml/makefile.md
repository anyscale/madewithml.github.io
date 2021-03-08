---
template: lesson.html
title: Makefiles
description: An automation tool that organizes commands for our application's processes.
keywords: makefile, applied ml, mlops, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/applied_ml.png
repository: https://github.com/GokuMohandas/applied-ml
---

## Intuition

We have just started and there are already so many different commands to keep track of. To help with this, we're going to use a [`Makefile`](https://opensource.com/article/18/8/what-how-makefile){:target="_blank"} which is a automation tool that organizes our commands. This makes it very easy for us to organize relevant commands as well as organize it for others who may be new to our application.

## Application

## Components

Inside our [Makefile](https://github.com/GokuMohandas/applied-ml/tree/main/Makefile){:target="_blank"}, we can see a list of rules (help, install, etc.). These rules have a `target` which can sometimes have `prerequisites` that need to be met (can be other targets) and on the next line a ++tab++ followed by a `recipe` which specifies how to create the target.

```bash linenums="1"
# Makefile
target: prerequisites
<TAB> recipe
```

## Targets
We can execute any of the rules by typing `make <target>`:

<div class="animated-code">

    ```console
    # View all rules
    $ make help
    Usage: tagifai [OPTIONS] COMMAND [ARGS]
    👉  Commands:
        install         : installs required packages.
        install-dev     : installs development requirements.
        install-test    : installs test requirements.
        ...

    # Make a target
    $ make install-dev
    python -m pip install -e ".[dev]"
    ...
    ```

</div>
<script src="../../../static/js/termynal.js"></script>

!!! note
    Each line in a recipe for a rule will execute in a separate sub-shell. However for certain recipes such as activating a virtual environment and loading packages, we want to do it all in one shell. To do this, we can add the [`.ONESHELL`](https://www.gnu.org/software/make/manual/make.html#One-Shell){:target="blank"} special target above any target like so:
    ```bash linenums="1" hl_lines="1"
    .ONESHELL:
    venv:
        python3 -m venv ${name}
        source ${name}/bin/activate
        python -m pip install --upgrade pip setuptools wheel
        make install-dev
    ```
    However this is only available in Make version 3.82 and above and most Macs currently user version 3.81. You can either update to the current version or chain your commands with `&&` like so:
    ```bash linenums="1"
    venv:
        python3 -m venv ${name}
        source ${name}/bin/activate && \
        python -m pip install --upgrade pip setuptools wheel && \
        make install-dev
    ```

## PHONY
A Makefile is called as such because traditionally the `targets` are supposed to be files we can make. However, Makefiles are also commonly used as command shortcuts which can lead to confusion when a file with a certain name exists and a Makefile rule with the same name exists! For example if you a directory called `docs` and a `target` in your Makefile called `docs`, when you run `make docs` you'll get this message:

<div class="animated-code">

    ```console
    $ make docs
    make: `docs` is up to date.
    ```

</div>

We can fix this by defining a [`PHONY`](https://www.gnu.org/software/make/manual/make.html#Phony-Targets){:target="_blank"} target in our makefile by adding this line above the target:
```bash linenums="1"
.PHONY: <target_name>
```

Most of the rules in our Makefile will require the `PHONY` target because we want them to execute even if there is a file sharing the target's name. An exception to this is the `venv` target because we don't want to create a `venv` directory if it already exists.

## Variables
We can also set and use [variables](https://www.gnu.org/software/make/manual/make.html#Using-Variables){:target="_blank"} inside our Makefile to organize all of our rules.

- We can set the variables directly inside the Makefile. If the variable isn't defined in the Makefile, then it would default to any environment variable with the same name.
```bash linenums="1"
# Set variable
MESSAGE := "hello world"

# Use variable
greeting:
    @echo ${MESSAGE}
```

- We can also use variables passed in when executing the rule like so (ensure that the variable is not overriden inside the Makefile):
```bash linenums="1"
make greeting MESSAGE="hi"
```

There's a whole lot [more](https://www.gnu.org/software/make/manual/make.html){:target="_blank"} to Makefiles but this is plenty for most applied ML projects.

<!-- Citation -->
{% include "cite.md" %}