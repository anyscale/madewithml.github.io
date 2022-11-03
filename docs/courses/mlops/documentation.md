---
template: lesson.html
title: Documenting Code
description: Documenting code for your team and your future self.
keywords: documentation, mkdocs, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/mlops.png
repository: https://github.com/GokuMohandas/mlops-course
documentation: https://gokumohandas.github.io/mlops-course
---

{% include "styles/lesson.md" %}

## Intuition

> Code tells you *how*, comments tell you *why*. -- [Jeff Atwood](https://en.wikipedia.org/wiki/Jeff_Atwood){:target="_blank"}

We can further [organize](organization.md){:target="_blank"} our code by documenting it to make it easier for others (and our future selves) to easily navigate and extend it. We know our code base best the moment we finish writing it but fortunately documenting it will allow us to quickly get back to that familiar state of mind. Documentation can mean many different things to developers, so let's define the most common components:

- `#!js comments`: short descriptions as to why a piece of code exists.
- `#!js typing`: specification of a function's inputs and outputs' data types, providing information pertaining to what a function consumes and produces.
- `#!js docstrings`: meaningful descriptions for functions and classes that describe overall utility, arguments, returns, etc.
- `#!js docs`: rendered webpage that summarizes all the functions, classes, workflows, examples, etc.

> For now, we'll produce our documentation locally but be sure to check out the auto-generated [documentation page](https://gokumohandas.github.io/mlops-course){:target="_blank"} for our [application](https://github.com/GokuMohandas/mlops-course){:target="_blank"}. We'll learn how to automatically create and keep our docs up-to-date in our [CI/CD](cicd.md){:target="_blank"} lesson every time we make changes to our code base.

!!! question "Code collaboration"
    How do you currently share your code with others on your team? What can be improved?

## Typing
It's important to be as explicit as possible with our code. We've already discussed choosing explicit names for variables, functions, etc. but another way we can be explicit is by defining the types for our function's inputs and outputs.

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

Take this time to update all the functions and classes in our project with docstrings and be sure to refer to the [repository](https://github.com/GokuMohandas/mlops-course){:target="_blank"} as a guide. Note that you my have to explicitly import some libraries to certain scripts because the `type` requires it. For example, we don't explicitly use the Pandas library in our `data.py` script, however, we do use pandas dataframes as input arguments.
```python linenums="1" hl_lines="5"
# tagifai/data.py
import pandas as pd
from typing import List

def replace_oos_labels(df: pd.DataFrame, labels: List, label_col: str, oos_label: str = "other"):
    ...
```

> Ideally we would add docstrings to our functions and classes as we develop them, as opposed to doing it all at once at the end.

!!! tip
    If using [Visual Studio Code](https://code.visualstudio.com/){:target="_blank"}, be sure to use the [Python Docstrings Generator](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring){:target="_blank"} extension so you can type `"""` under a function and then hit the ++shift++ key to generate a template docstring. It will autofill parts of the docstring using the typing information and even exception in your code!

    ![vscode docstring generation](https://github.com/NilsJPWerner/autoDocstring/blob/13875f7e5d3a2ad2a2a7e42bad6a10d09fed7472/images/demo.gif?raw=true)

## Docs

So we're going through all this effort of including typing and docstrings to our functions but it's all tucked away inside our scripts. What if we can collect all this effort and **automatically** surface it as documentation? Well that's exactly what we'll do with the following open-source packages → final result [here](https://gokumohandas.github.io/mlops-course){:target="_blank"}.

1. Install required packages:
```bash
pip install mkdocs==1.3.0 mkdocstrings==0.18.1
```
Instead of directly adding these requirements to our `requirements.txt` file, we're going to isolate it from our core required libraries. We want to do this because not everyone will need to create documentation as it's not a core machine learning operation (training, inference, etc.). We'll tweak our `setup.py` script to make this possible.
We'll define these packages under a `docs_packages` object:
```python
# setup.py
docs_packages = [
    "mkdocs==1.3.0",
    "mkdocstrings==0.18.1"
]
```
and then we'll add this to `setup()` object in the script:
```python linenums="1" hl_lines="5-8"
# Define our package
setup(
    ...
    install_requires=[required_packages],
    extras_require={
        "dev": docs_packages,
        "docs": docs_packages,
    },
)
```
Now we can install this package with:
```bash
python3 -m pip install -e ".[docs]"
```
We're also defining a `dev` option which we'll update over the course so that developers can install all required and extra packages in one call, instead of calling each extra required packages one at a time.
```bash
python3 -m pip install -e ".[dev]"
```
We created an explicit `doc` option because a user will want to only download the documentation packages to generate documentation (none of the other packages will be required). We'll see this in action when we use [CI/CD workflows](cicd.md){:target="_blank"} to autogenerate documentation via GitHub Actions.


2. Initialize mkdocs
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

3. We'll start by overwriting the default `index.md` file in our `docs` directory with information specific to our project:
```md linenums="1" title="index.md"
## Documentation

- [Workflows](tagifai/main.md): main workflows.
- [tagifai](tagifai/data.md): documentation of functionality.

## MLOps Lessons

Learn how to combine machine learning with software engineering to develop, deploy & maintain production ML applications.

- Lessons: [https://madewithml.com/](https://madewithml.com/#mlops)
- Code: [GokuMohandas/mlops-course](https://github.com/GokuMohandas/mlops-course)
```

4. Next we'll create documentation files for each script in our `tagifai` directory:
```
mkdir docs/tagifai
cd docs/tagifai
touch main.md utils.md data.md train.md evaluate.md predict.md
cd ../../
```
> It's helpful to have the `docs` directory structure mimic our project's structure as much as possible. This becomes even more important as we document more directories in future lessons.

5. Next we'll add `::: tagifai.<SCRIPT_NAME>` to each file under `docs/tagifai`. This will populate the file with information about the functions and classes (using their docstrings) from `tagifai/<SCRIPT_NAME>.py` thanks to the `mkdocstrings` plugin.
> Be sure to check out the complete list of [mkdocs plugins](https://github.com/mkdocs/mkdocs/wiki/MkDocs-Plugins){:target="_blank"}.
```bash
# docs/tagifai/data.md
::: tagifai.data
```

6. Finally, we'll add some configurations to our `mkdocs.yml` file that mkdocs automatically created:
```yml
# mkdocs.yml
site_name: Made With ML
site_url: https://madewithml.com/
repo_url: https://github.com/GokuMohandas/mlops-course/
nav:
  - Home: index.md
  - workflows:
    - main: tagifai/main.md
  - tagifai:
    - data: tagifai/data.md
    - evaluate: tagifai/evaluate.md
    - predict: tagifai/predict.md
    - train: tagifai/train.md
    - utils: tagifai/utils.md
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

## Publishing

We can easily serve our documentation for free using [GitHub pages](https://www.mkdocs.org/user-guide/deploying-your-docs/){:target="_blank"} for public repositories as wells as [private documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/changing-the-visibility-of-your-github-pages-site){:target="_blank"} for private repositories. And we can even host it on a [custom domain](https://docs.github.com/en/github/working-with-github-pages/configuring-a-custom-domain-for-your-github-pages-site){:target="_blank"} (ex. company's subdomain).

> Be sure to check out the auto-generated [documentation page](https://gokumohandas.github.io/mlops-course){:target="_blank"} for our [application](https://github.com/GokuMohandas/mlops-course){:target="_blank"}. We'll learn how to automatically create and keep our docs up-to-date in our [CI/CD](cicd.md){:target="_blank"} lesson every time we make changes to our code base.

<!-- Course signup -->
{% include "templates/course-signup.md" %}

<!-- Citation -->
{% include "templates/cite.md" %}