---
template: lesson.html
title: Makefiles
description: An automation tool that organizes commands for our application's processes.
keywords: makefile, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/mlops.png
repository: https://github.com/GokuMohandas/mlops-course
---

{% include "styles/lesson.md" %}

## Intuition

Even though we're only about halfway through the course, there are already so many different commands to keep track of. To help organize everything, we're going to use a [`Makefile`](https://opensource.com/article/18/8/what-how-makefile){:target="_blank"} which is a automation tool that organizes our commands. We'll start by create this file in our project's root directory.

```bash
touch Makefile
```

At the top of our `Makefile` we need to specify the shell environment we want all of our commands to execute in:

```bash
# Makefile
SHELL = /bin/bash
```

## Components

Inside our [Makefile](https://github.com/GokuMohandas/mlops-course/tree/main/Makefile){:target="_blank"}, we'll be creating a list of rules. These rules have a `target` which can sometimes have `prerequisites` that need to be met (can be other targets) and on the next line a ++tab++ followed by a `recipe` which specifies how to create the target.

```bash
# Makefile
target: prerequisites
<TAB> recipe
```

For example, if we wanted to create a rule for styling our files, we would add the following to our `Makefile`:

```bash
# Styling
style:
	black .
	flake8
	python3 -m isort .
```

!!! warning "Tabs vs. spaces"
    Makefiles require that indention be done with a <TAB>, instead of spaces where we'll receive an error:
    <pre>
    Makefile:<line_no>: *** missing separator.  Stop.
    </pre>
    Luckily, editors like VSCode automatically change indentation to tabs even if other files use spaces.

## Targets
We can execute any of the rules by typing `make <target>` in the terminal:

```bash
# Make a target
$ make style
```

<pre class="output">
black .
All done! ✨ 🍰 ✨
8 files left unchanged.
flake8
python3 -m isort .
Skipped 1 files
</pre>

Similarly, we can set up our `Makefile` for creating a virtual environment:

```bash
# Environment
venv:
	python3 -m venv venv
	source venv/bin/activate && \
	python3 -m pip install pip setuptools wheel && \
	python3 -m pip install -e .
```

> `&&` signifies that we want these commands to execute in one shell (more on this [below](#shells)).

## PHONY
A Makefile is called as such because traditionally the `targets` are supposed to be files we can *make*. However, Makefiles are also commonly used as command shortcuts, which can lead to confusion when a Makefile target and a file share the same name! For example if we have a file called `venv` (which we do) and a `target` in your Makefile called `venv`, when you run `make venv` we'll get this message:

```bash
$ make venv
```

<pre class="output">
make: `venv' is up to date.
</pre>

In this situation, this is the intended behavior because if a virtual environment already exists, then we don't want ot *make* that target again. However, sometimes, we'll name our targets and want them to execute whether it exists as an actual file or not. In these scenarios, we want to define a [`PHONY`](https://www.gnu.org/software/make/manual/make.html#Phony-Targets){:target="_blank"} target in our makefile by adding this line above the target:
```bash
.PHONY: <target_name>
```

Most of the rules in our Makefile will require the `PHONY` target because we want them to execute even if there is a file sharing the target's name.

```bash hl_lines="2"
# Styling
.PHONY: style
style:
	black .
	flake8
	isort .
```

## Prerequisites

Before we make a target, we can attach prerequisites to them. These can either be file targets that must exist or PHONY target commands that need to be executed prior to *making* this target. For example, we'll set the *style* target as a prerequisite for the *clean* target so that all files are formatted appropriately prior to cleaning them.

```bash hl_lines="3"
# Cleaning
.PHONY: clean
clean: style
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
    find . | grep -E ".trash" | xargs rm -rf
	rm -f .coverage
```

## Variables
We can also set and use [variables](https://www.gnu.org/software/make/manual/make.html#Using-Variables){:target="_blank"} inside our Makefile to organize all of our rules.

- We can set the variables directly inside the Makefile. If the variable isn't defined in the Makefile, then it would default to any environment variable with the same name.
```bash
# Set variable
MESSAGE := "hello world"

# Use variable
greeting:
    @echo ${MESSAGE}
```

- We can also use variables passed in when executing the rule like so (ensure that the variable is not overridden inside the Makefile):
```bash
make greeting MESSAGE="hi"
```

## Shells

Each line in a recipe for a rule will execute in a separate sub-shell. However for certain recipes such as activating a virtual environment and loading packages, we want to execute all steps in one shell. To do this, we can add the [`.ONESHELL`](https://www.gnu.org/software/make/manual/make.html#One-Shell){:target="blank"} special target above any target.

```bash hl_lines="2"
# Environment
.ONESHELL:
venv:
	python3 -m venv venv
	source venv/bin/activate
	python3 -m pip install pip setuptools wheel
	python3 -m pip install -e .
```

However this is only available in Make version 3.82 and above and most Macs currently use version 3.81. You can either update to the current version or chain your commands with `&&` as we did previously:

```bash
# Environment
venv:
	python3 -m venv venv
	source venv/bin/activate && \
	python3 -m pip install pip setuptools wheel && \
	python3 -m pip install -e .
```

## Help

The last thing we'll add to our `Makefile` (for now at least) is a `help` target to the very top. This rule will provide an informative message for this Makefile's capabilities:

```bash
.PHONY: help
help:
	@echo "Commands:"
	@echo "venv    : creates a virtual environment."
	@echo "style   : executes style formatting."
	@echo "clean   : cleans all unnecessary files."
```

```bash
make help
```

<pre class="output">
Commands:
venv    : creates a virtual environment.
style   : executes style formatting.
clean   : cleans all unnecessary files.
</pre>

> There's a whole lot [more](https://www.gnu.org/software/make/manual/make.html){:target="_blank"} to Makefiles but this is plenty for most applied ML projects.

<!-- Citation -->
{% include "cite.md" %}