---
template: lesson.html
title: Documenting Code
description: Documenting our code to automatically generate documentation.
keywords: documentation, mkdocs, mlops, machine learning
image: https://madewithml.com/static/images/mlops.png
repository: https://github.com/GokuMohandas/Made-With-ML
documentation: https://gokumohandas.github.io/Made-With-ML
---

{% include "styles/lesson.md" %}

## Intuition

> Code tells you *how*, comments tell you *why*. -- [Jeff Atwood](https://en.wikipedia.org/wiki/Jeff_Atwood){:target="_blank"}

We can really improve the quality of our codebase by documenting it to make it easier for others (and our future selves) to easily navigate and extend it. We know our code base best the moment we finish writing it but fortunately documenting it will allow us to quickly get back to that familiar state of mind. Documentation can mean many different things to developers, so let's define the most common components:

- `#!js comments`: short descriptions as to why a piece of code exists.
- `#!js typing`: specification of a function's inputs and outputs' data types, providing information pertaining to what a function consumes and produces.
- `#!js docstrings`: meaningful descriptions for functions and classes that describe overall utility, arguments, returns, etc.
- `#!js docs`: rendered webpage that summarizes all the functions, classes, workflows, examples, etc.

## Typing
It's important to be as explicit as possible with our code. We've already discussed choosing explicit names for variables, functions but another way we can be explicit is by defining the types for our function's inputs and outputs by using the [typing library](https://docs.python.org/3/library/typing.html){:target="_blank"}.

So far, our functions have looked like this:
```python linenums="1"
def some_function(a, b):
    return c
```

But we can incorporate so much more information using typing:
```python linenums="1"
from typing import List
def some_function(a: List, b: int = 0) -> np.ndarray:
    return c
```

Here we've defined:

- input parameter `a` is a list
- input parameter `b` is an integer with default value 0
- output parameter `c` is a NumPy array

There are many other data types that we can work with, including `List`, `Set`, `Dict`, `Tuple`, `Sequence` and [more](https://docs.python.org/3/library/typing.html){:target="_blank"}, as well as included types such as `int`, `float`, etc. You can also use types from packages we install (ex. `np.ndarray`) and even from our own defined classes (ex. `LabelEncoder`).

> Starting from Python 3.9+, common types are [built in](https://docs.python.org/3/whatsnew/3.9.html#type-hinting-generics-in-standard-collections){:target="_blank"} so we don't need to import them with ```from typing import List, Set, Dict, Tuple, Sequence``` anymore.

## Docstrings
We can make our code even more explicit by adding docstrings to describe overall utility, arguments, returns, exceptions and more. Let's take a look at an example:

```python linenums="1"
from typing import List
def some_function(a: List, b: int = 0) -> np.ndarray:
    """Function description.

    ```python
    c = some_function(a=[], b=0)
    print (c)
    ```
    <pre>
    [[1 2]
     [3 4]]
    </pre>

    Args:
        a (List): description of `a`.
        b (int, optional): description of `b`. Defaults to 0.

    Raises:
        ValueError: Input list is not one-dimensional.

    Returns:
        np.ndarray: Description of `c`.

    """
    return c
```

Let's unpack the different parts of this function's docstring:

- `#!js [Line 3]`: Summary of the overall utility of the function.
- `#!js [Lines 5-12]`: Example of how to use our function.
- `#!js [Lines 14-16]`: Description of the function's input arguments.
- `#!js [Lines 18-19]`: Any exceptions that may be raised in the function.
- `#!js [Lines 21-22]`: Description of the function's output(s).

We'll render these docstrings in the [docs](#docs) section below to produce this:

<div class="ai-center-all">
    <img src="/static/images/mlops/documentation/docstrings.png" width="500" alt="docstrings">
</div>

Take a look at the docstrings of different functions and classes in our repository.

```python linenums="1" hl_lines="5"
# madewithml/data.py
from typing import List

def clean_text(text: str, stopwords: List = STOPWORDS) -> str:
    """Clean raw text string.

    Args:
        text (str): Raw text to clean.
        stopwords (List, optional): list of words to filter out. Defaults to STOPWORDS.

    Returns:
        str: cleaned text.
    """
    pass
```

!!! tip
    If using [Visual Studio Code](https://code.visualstudio.com/){:target="_blank"}, be sure to use the [Python Docstrings Generator](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring){:target="_blank"} extension so you can type `"""` under a function and then hit the ++shift++ key to generate a template docstring. It will autofill parts of the docstring using the typing information and even exception in your code!

    ![vscode docstring generation](https://github.com/NilsJPWerner/autoDocstring/blob/13875f7e5d3a2ad2a2a7e42bad6a10d09fed7472/images/demo.gif?raw=true)

## Docs

So we're going through all this effort of including typing and docstrings to our functions but it's all tucked away inside our scripts. What if we can collect all this effort and **automatically** surface it as documentation? Well that's exactly what we'll do with the following open-source packages → final result [here](https://gokumohandas.github.io/Made-With-ML){:target="_blank"}.

1. Initialize mkdocs
```bash
python3 -m mkdocs new .
```
This will create the following files:
```bash
.
├─ docs/
│  └─ index.md
└─ mkdocs.yml
```

2. We'll start by overwriting the default `index.md` file in our `docs` directory with information specific to our project:
```md linenums="1" title="index.md"
## Documentation
- [madewithml](madewithml/config.md): documentation for functions and classes.

## Course
Learn how to combine machine learning with software engineering to design, develop, deploy and iterate on production ML applications.

- Lessons: [https://madewithml.com/](https://madewithml.com/#course)
- Code: [GokuMohandas/Made-With-ML](https://github.com/GokuMohandas/Made-With-ML)
```

3. Next we'll create documentation files for each script in our `madewithml` directory:
```
mkdir docs/madewithml
cd docs/madewithml
touch config.md data.md evaluate.md models.md predict.md serve.md train.md tune.md util.md
cd ../../
```

!!! tip
    It's helpful to have the `docs` directory structure mimic our project's structure as much as possible.

4. Next we'll add `::: madewithml.<SCRIPT_NAME>` to each file under `docs/madewithml`. This will populate the file with information about the functions and classes (using their docstrings) from `madewithml/<SCRIPT_NAME>.py` thanks to the `mkdocstrings` plugin.
> Be sure to check out the complete list of [mkdocs plugins](https://github.com/mkdocs/mkdocs/wiki/MkDocs-Plugins){:target="_blank"}.
```bash
# docs/madewithml/data.md
::: madewithml.data
```

5. Finally, we'll add some configurations to our `mkdocs.yml` file that mkdocs automatically created:
```yaml
site_name: Made With ML
site_url: https://madewithml.com/
repo_url: https://github.com/GokuMohandas/Made-With-ML/
nav:
  - Home: index.md
  - madewithml:
    - data: madewithml/data.md
    - models: madewithml/models.md
    - train: madewithml/train.md
    - tune: madewithml/tune.md
    - evaluate: madewithml/evaluate.md
    - predict: madewithml/predict.md
    - serve: madewithml/serve.md
    - utils: madewithml/utils.md
theme: readthedocs
plugins:
  - mkdocstrings
watch:
  - .  # reload docs for any file changes
```

7. Serve our documentation locally:
```bash
python3 -m mkdocs serve
```

This will serve our docs at [http://localhost:8000/](http://localhost:8000/):

<div class="ai-center-all">
    <a href="https://gokumohandas.github.io/Made-With-ML" target="_blank">
        <img src="/static/images/mlops/documentation/mkdocs.png" width="700" alt="mkdocs">
    </a>
</div>

## Publishing

We can easily serve our documentation for free using [GitHub pages](https://www.mkdocs.org/user-guide/deploying-your-docs/){:target="_blank"} for public repositories as wells as [private documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/changing-the-visibility-of-your-github-pages-site){:target="_blank"} for private repositories. And we can even host it on a [custom domain](https://docs.github.com/en/github/working-with-github-pages/configuring-a-custom-domain-for-your-github-pages-site){:target="_blank"} (ex. company's subdomain).

> Be sure to check out the auto-generated [documentation page](https://gokumohandas.github.io/Made-With-ML){:target="_blank"} for our [repository](https://github.com/GokuMohandas/Made-With-ML){:target="_blank"}. We'll learn how to automatically generate and update this docs page every time we make changes to our codebase later in our [CI/CD](cicd.md#documentation){:target="_blank"} lesson.

In the next lesson, we'll learn how to [style and format](styling.md){:target="_blank"} our codebase in a consistent manner.

<!-- Course signup -->
{% include "templates/signup.md" %}

<!-- Citation -->
{% include "templates/cite.md" %}