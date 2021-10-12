---
template: lesson.html
title: Documenting Code
description: Documenting code for your users and your future self.
keywords: documentation, mkdocs, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/mlops.png
repository: https://github.com/GokuMohandas/MLOps
documentation: https://github.com/GokuMohandas/follow/tree/documentation
---

{% include "styles/lesson.md" %}

## Intuition

> Code tells you *how*, comments tell you *why*. -- [Jeff Atwood](https://blog.codinghorror.com/code-tells-you-how-comments-tell-you-why/){:target="_blank"}

Another way to [organize](organization.md){:target="_blank"} our code is to document it. We want to do this so we can make it easier for others (and our future selves) to easily navigate the code base and build on it. We know our code base best the moment we finish writing it but fortunately documenting it will allow us to quickly get back to that stage time and time again. Documentation involves many different things to developers so let's define the most common (and required) components:

- `#!js comments`: short descriptions of why a piece of code exists.
- `#!js typing`: specification of a function's inputs and outputs data types, providing insight into what a function consumes and produces at a quick glance.
- `#!js docstrings`: meaningful descriptions for functions and classes that describe overall utility as well as arguments, returns, etc.
- `#!js documentation`: rendered webpage that summarizes all the functions, classes, API calls, workflows, examples, etc. so we can view and traverse through the code base without actually having to look at the code just yet.

!!! question "Code collaboration"
    How do you currently share your code with others on your team? What can be improved?

> Be sure to check out the auto-generated [documentation page](https://gokumohandas.github.io/MLOps/){:target="_blank"} for our [application](https://github.com/GokuMohandas/MLOps){:target="_blank"}.

## Typing
It's important to be as explicit as possible with our code. We're already discussed choosing explicit names for variables, functions, etc. but another way we can be explicit is by defining the types for our function's inputs and outputs. We want to do this so we can quickly know what data types a function expects and how we can utilize it's outputs for downstream processes.

So far, our functions have looked like this:
```python linenums="1"
def pad_sequences(sequences, max_seq_len):
    ...
    return padded_sequences
```

But we can incorporate so much more information using typing:
```python linenums="1"
def pad_sequences(sequences: np.ndarray, max_seq_len: int = 0) -> np.ndarray:
    ...
    return padded_sequences
```

Here we're defining that our input argument `sequences` is a NumPy array, `max_seq_len` is an integer with a default value of 0 and our output is also a NumPy array. There are many data types that we can work with, including but not limited to `List`, `Set`, `Dict`, `Tuple`, `Sequence` and [more](https://docs.python.org/3/library/typing.html){:target="_blank"}, as well as included types such as `int`, `float`, etc. You can also use any of your own defined classes as types (ex. `nn.Module`, `LabelEncoder`, etc.).

> Starting from Python 3.9+, common types are [built in](https://docs.python.org/3/whatsnew/3.9.html#type-hinting-generics-in-standard-collections){:target="_blank"} so we don't need to import them with ```from typing import List, Set, Dict, Tuple, Sequence``` anymore.


## Docstrings
We can make our code even more explicit by adding docstrings to functions and classes to describe overall utility, arguments, returns, exceptions and more. Let's take a look at an example:

```python linenums="1"
def pad_sequences(sequences: np.ndarray, max_seq_len: int = 0) -> np.ndarray:
    """Zero pad sequences to a specified `max_seq_len`
    or to the length of the largest sequence in `sequences`.

    Usage:

    ```python
    # Pad inputs
    seq = np.array([[1, 2, 3], [1, 2]], dtype=object)
    padded_seq = pad_sequences(sequences=seq, max_seq_len=5)
    print (padded_seq)
    ```
    <pre>
    [[1. 2. 3. 0. 0.]
     [1. 2. 0. 0. 0.]]
    </pre>

    Note:
        Input `sequences` must be 2D.

    Args:
        sequences (np.ndarray): 2D array of data to be padded.
        max_seq_len (int, optional): Length to pad sequences to. Defaults to 0.

    Raises:
        ValueError: Input sequences are not two-dimensional.

    Returns:
        An array with the zero padded sequences.

    """
    # Check shape
    if not sequences.ndim == 2:
        raise ValueError("Input sequences are not two-dimensional.")

    # Get max sequence length
    max_seq_len = max(
        max_seq_len, max(len(sequence) for sequence in sequences)
    )

    # Pad
    padded_sequences = np.zeros((len(sequences), max_seq_len))
    for i, sequence in enumerate(sequences):
        padded_sequences[i][: len(sequence)] = sequence
    return padded_sequences
```

Let's unpack the different parts of this function's docstring:

- `#!js [Lines 2-3]`: Summary of the overall utility of the function.
- `#!js [Lines 5-16]`: Example of how to use our function.
- `#!js [Lines 18-19]`: Insertion of a `Note` or other types of [admonitions](https://squidfunk.github.io/mkdocs-material/reference/admonitions/){:target="_blank"}.
- `#!js [Lines 21-23]`: Description of the function's input arguments.
- `#!js [Lines 25-26]`: Any exceptions that may be raised in the function.
- `#!js [Lines 28-29]`: Description of the function's output(s).

!!! note
    If you're using [Visual Studio Code](https://code.visualstudio.com/){:target="_blank"}, you should get the free [Python Docstrings Generator](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring){:target="_blank"} extension so you can type `"""` under a function and then hit the ++shift++ key to generate a template docstring. It will autofill parts of the docstring using the typing information and even exception in your code!

    ![vscode docstring generation](https://github.com/NilsJPWerner/autoDocstring/blob/13875f7e5d3a2ad2a2a7e42bad6a10d09fed7472/images/demo.gif?raw=true)

## Mkdocs

So we're going through all this effort to including typing and docstrings to our functions but it's all tucked away inside our scripts. But what if we can collect all this effort and **automatically** surface it as documentation? Well that's exactly what we'll do with the following open-source packages → final result [here](https://gokumohandas.github.io/MLOps/){:target="_blank"}.

- [mkdocs](https://github.com/mkdocs/mkdocs){:target="_blank"}                                  (generates project documentation)
- [mkdocs-material](https://github.com/squidfunk/mkdocs-material){:target="_blank"}             (styling to beautiful render documentation)
- [mkdocstrings](https://github.com/pawamoy/mkdocstrings){:target="_blank"}                     (fetch documentation automatically from docstrings)

Here are the steps we'll follow to automatically generate our documentation and serve it. You can find all the files we're talking about in our [repository](https://github.com/GokuMohandas/MLOps){:target="_blank"}.

1. Install the required packages. We already did this when we initially set up our dev environment.
```python linenums="1"
# setup.py
...
docs_packages = [
    "mkdocs==1.1.2",
    "mkdocs-material==7.2.3",
    "mkdocstrings==0.15.2",
]
...
setup(
    ...
    extras_require={
        "test": test_packages,
        "dev": test_packages + dev_packages + docs_packages,
        "docs": docs_packages,
    },
    ...
)
```
```bash
python -m pip install -e ".[dev]"
```
> Mkdocs related packages are part of the docs packages and not part of the required packages specified in [requirements.txt](https://github.com/GokuMohandas/MLOps/blob/main/requirements.txt){:target="_blank"} since not all users will be maintaining the documentation.
2. Initialize mkdocs
```bash
mkdocs new .
```
This will create the following files:
```bash
.
├─ docs/
│  └─ index.md
└─ mkdocs.yml
```
3. Create a documentation file for each item in our navigation tree and add ` ::: tagifai.<SCRIPT_NAME>` to each Markdown file to populate it with the information from function and class docstrings from `tagifai/<SCRIPT_NAME>.py`. We can add our own text directly to the Markdown file as well, like we do in [`docs/tagifai/config.md`](https://github.com/GokuMohandas/follow/blob/main/docs/config/config.md){:target="_blank"}.
```bash
# Documentation directory structure
docs/
├── app/
| ├── api.md
| └── schemas.md
├── config/
| ├── config.md
├── tagifai/
| ├── data.md
| ├── ...
| └── utils.md
├── getting_started.md
└── index.md
```
```bash
# docs/tagifai/data.md
::: tagifai.data
```
4. Fill in project metadata inside [mkdocs.yaml](https://github.com/GokuMohandas/follow/blob/main/mkdocs.yml){:target="_blank"}.
```yaml linenums="1"
# Project information
site_name: TagifAI
site_url: https://madewithml.com/#mlops
site_description: Tag suggestions for projects on Made With ML.
site_author: Goku Mohandas

# Repository
repo_url: https://github.com/GokuMohandas/MLOps
repo_name: GokuMohandas/MLOps
edit_uri: "" #disables edit button
...
```
5. Add theme, extensions and plugins (see more [options](https://squidfunk.github.io/mkdocs-material/setup/changing-the-colors/){:target="_blank"}).
```yaml linenums="1"
# Configuration
theme:
  name: material

# Extensions
markdown_extensions:
  - admonition # alerts
  - codehilite
  - pymdownx.highlight

# Plugins
plugins:
  - search
  - mkdocstrings
```
6. Create our documentation's navigation tree.
```yaml linenums="1"
# Page tree
nav:
  - Home:
      - TagIfAI: index.md
  - Getting started: getting_started.md
  - Application:
    - API: app/api.md
    - Schemas: app/schemas.md
  - Operations: tagifai/main.md
  - Configurations: config/config.md
  - Reference:
    - Data: tagifai/data.md
    - Eval: tagifai/eval.md
    - Models: tagifai/models.md
    - Training: tagifai/train.md
    - Inference: tagifai/predict.md
    - Utilities: tagifai/utils.md
```
7. Run `python -m mkdocs serve -a localhost:8000` to serve your docs.
<div class="animated-code">

    ```console
    # Serve documentation
    $ python -m mkdocs serve -a localhost:8000
    INFO    -  Building documentation...
    INFO    -  Cleaning site directory
    INFO    -  Serving on http://localhost:8000
    ```

</div>
<script src="../../../static/js/termynal.js"></script>

:octicons-info-24: View our rendered documentation via GitHub pages → [here](https://gokumohandas.github.io/MLOps/){:target="_blank"}.

> We can easily serve our documentation for free using [GitHub pages](https://squidfunk.github.io/mkdocs-material/publishing-your-site/){:target="_blank"} for public repositories as wells as [private documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/changing-the-visibility-of-your-github-pages-site){:target="_blank"} for private repositories. And we can even host it on a [custom domain](https://docs.github.com/en/github/working-with-github-pages/configuring-a-custom-domain-for-your-github-pages-site){:target="_blank"} (ex. company's subdomain). All we had to do was add the file [`.github/workflows/documentation.yml`](https://github.com/GokuMohandas/MLOps/blob/main/.github/workflows/documentation.yml){:target="_blank"} which [GitHub Actions](https://github.com/features/actions){:target="_blank"}  will use to build and deploy our documentation every time we push to the `main` branch (we'll learn about GitHub Actions in our CI/CD lesson soon).

<!-- Citation -->
{% include "cite.md" %}




