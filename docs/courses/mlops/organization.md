---
template: lesson.html
title: Organizing Machine Learning Code
description: Organizing our code when moving from notebooks to Python scripts.
keywords: git, github, organization, repository, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/mlops.png
repository: https://github.com/GokuMohandas/mlops-course
---

{% include "styles/lesson.md" %}

## Intuition

To have organized code is to have readable, reproducible, robust code. Your team, manager and most importantly, your future self, will thank you for putting in the initial effort towards organizing your work. In this lesson, we'll discuss how to migrate and organize code from our [notebook](https://github.com/GokuMohandas/mlops-course/blob/main/notebooks/tagifai.ipynb){:target="_blank"} to Python scripts.

## Editor
Before we can start coding, we need a space to do it. There are several options for code editors, such as [VSCode](https://code.visualstudio.com/){:target="_blank"}, [Atom](https://atom.io/){:target="_blank"}, [Sublime](https://www.sublimetext.com/){:target="_blank"}, [PyCharm](https://www.jetbrains.com/pycharm/){:target="_blank"}, [Vim](https://www.vim.org/){:target="_blank"}, etc. and they all offer unique features while providing the basic operations for code editing and execution. We will be using VSCode to edit and execute our code thanks to its simplicity, multi-language support, add-ons and growing industry adoption.

> You are welcome to use any editor but we will be using some add-ons that may be specific to VSCode.

1. Install VSCode from source for your system: [https://code.visualstudio.com/](https://code.visualstudio.com/){:target="_blank"}
2. Open the Command Palette (`F1` or ++command++ + ++shift++ + `P` on mac) &rarr; type in "Preferences: Open Settings (UI)" &rarr; hit ++return++
3. Adjust any relevant settings you want to (spacing, font-size, etc.)
4. Install [VSCode extensions](https://marketplace.visualstudio.com/){:target="_blank"} (use the lego blocks icon on the editor's left panel)

??? quote "Recommended VSCode extensions"
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

Once we're all set up with VSCode, we can start by creating our project directory, which we'll use to organize all our scripts. There are many ways to start a project, but here's our recommended path:

1. Use the terminal to create a directory (`#!bash mkdir <PROJECT_NAME>`).
2. Change into the project directory you just made (`#!bash cd <PROJECT_NAME>`).
3. Start VSCode from this directory by typing `#!bash code .`
> To open VSCode directly from the terminal with a `#!bash code $PATH` command, open the Command Palette (`F1` or ++command++ + ++shift++ + `P` on mac) &rarr; type "Shell Command: Install 'code' command in PATH" &rarr; hit ++return++ &rarr; restart the terminal.
4. Open a terminal within VSCode (`View` > `Terminal`) to continue creating scripts (`#!bash touch <FILE_NAME>`) or additional subdirectories (`#!bash mkdir <SUBDIR>`) as needed.

<div class="ai-center-all">
    <img src="/static/images/mlops/organization/vscode.png" width="650" alt="vscode">
</div>

## Setup

### README

We'll start our organization with a `README.md` file, which will provide information on the files in our directory, instructions to execute operations, etc. We'll constantly keep this file updated so that we can catalogue information for the future.

```bash
touch README.md
```

Let's start by adding the instructions we used for creating a [virtual environment](packaging.md#virtual-environment){:target="_blank"}:

```bash linenums="1"
# Inside README.md
python3 -m venv venv
source venv/bin/activate
python3 -m pip install pip setuptools wheel
python3 -m pip install -e .
```

If you press the Preview button located on the top right of the editor (button enclosed in red circle in the image below), you can see what the `README.md` will look like when we push to remote host for [git](git.md){:target="_blank"}.

<div class="ai-center-all">
    <img src="/static/images/mlops/organization/readme.png" width="650" alt="readme file">
</div>

### Configurations

Next we'll create a configuration directory called `config` where we can store components that will be required for our application. Inside this directory, we'll create a `config.py`  and a `args.json`.

```bash
mkdir config
touch config/main.py config/args.json
```

```bash
config/
├── args.json       - arguments
└── config.py       - configuration setup
```

Inside `config.py`, we'll add the code to define key directory locations (we'll add more configurations in later lessons as they're needed):
```python linenums="1"
# config.py
from pathlib import Path

# Directories
BASE_DIR = Path(__file__).parent.parent.absolute()
CONFIG_DIR = Path(BASE_DIR, "config")
```

and inside `args.json`, we'll add the parameters that are relevant to data processing and model training.
```json linenums="1"
{
    "shuffle": true,
    "subset": null,
    "min_freq": 75,
    "lower": true,
    "stem": false,
    "analyzer": "char",
    "ngram_max_range": 7,
    "alpha": 1e-4,
    "learning_rate": 1e-1,
    "power_t": 0.1
}
```

### Operations

We'll start by creating our package directory (`tagifai`) inside our project directory (`mlops`). Inside this package directory, we will create a `main.py` file that will define the core operations we want to be able to execute.

```bash
mkdir tagifai
touch tagifai/main.py
```

```bash
tagifai/
└── main.py       - training/optimization pipelines
```

We'll define these core operations inside `main.py` as we move code from notebooks to the appropriate scripts [below](#project):

- `#!js elt_data`: extract, load and transform data.
- `#!js optimize`: tune hyperparameters to optimize for objective.
- `#!js train_model`: train a model using best parameters from optimization study.
- `#!js load_artifacts`: load trained artifacts from a given run.
- `#!js predict_tag`: predict a tag for a given input.

### Utilities

Before we start moving code from our notebook, we should be intentional about *how* we move functionality over to scripts. It's common to have ad-hoc processes inside notebooks because it maintains state as long as the notebook is running. For example, we may set seeds in our notebooks like so:

```python linenums="1"
# Set seeds
np.random.seed(seed)
random.seed(seed)
```

But in our scripts, we should wrap this functionality as a clean, reuseable function with the appropriate parameters:

```python linenums="1"
def set_seeds(seed=42):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
```

We can store all of these inside a `utils.py` file inside our `tagifai` package directory.

```bash
touch tagifai/utils.py
```

```bash
tagifai/
├── main.py       - training/optimization pipelines
└── utils.py      - supplementary utilities
```

??? example "View utils.py"
    ```python linenums="1"
    import json
    import numpy as np
    import random

    def load_dict(filepath):
        """Load a dictionary from a JSON's filepath."""
        with open(filepath, "r") as fp:
            d = json.load(fp)
        return d

    def save_dict(d, filepath, cls=None, sortkeys=False):
        """Save a dictionary to a specific location."""
        with open(filepath, "w") as fp:
            json.dump(d, indent=2, fp=fp, cls=cls, sort_keys=sortkeys)

    def set_seeds(seed=42):
        """Set seed for reproducibility."""
        # Set seeds
        np.random.seed(seed)
        random.seed(seed)
    ```

> Don't worry about formatting our scripts just yet. We'll be automating all of it in our [styling](styling.md){:target="_blank"} lesson.

## Project

When it comes to migrating our code from notebooks to scripts, it's best to organize based on utility. For example, we can create scripts for the various stages of ML development such as data processing, training, evaluation, prediction, etc.:

We'll create the different python files to wrap our data and ML functionality:
```bash
cd tagifai
touch data.py train.py evaluate.py predict.py
```

```bash
tagifai/
├── data.py       - data processing utilities
├── evaluate.py   - evaluation components
├── main.py       - training/optimization pipelines
├── predict.py    - inference utilities
├── train.py      - training utilities
└── utils.py      - supplementary utilities
```

> We may have additional scripts in other projects, as they are necessary. For example, we'd typically have a `models.py` script we define explicit model architectures in Pytorch, Tensorflow, etc.

Organizing our code base this way also makes it easier for us to understand (or modify) the code base. We could've placed all the code into one `main.py` script but as our project grows, it will be hard to navigate one monolithic file. On the other hand, we could've  assumed a more granular stance by breaking down `data.py` into `split.py`, `preprocess.py`, etc. This might make more sense if we have multiple ways of splitting, preprocessing, etc. (ex. a library for ML operations) but for our task, it's sufficient to be at this higher level of organization.

## Principles

Through the migration process below, we'll be using several core software engineering principles repeatedly.

#### Wrapping functionality into functions

How do we decide when specific lines of code should be wrapped as a separate function? Functions should be atomic in that they each have a [single responsibility](https://en.wikipedia.org/wiki/Single-responsibility_principle){:target="_blank"} so that we can easily [test](testing.md){:target="_blank"} them. If not, we'll need to split them into more granular units. For example, we could replace tags in our projects with these lines:

```python linenums="1"
oos_tags = [item for item in df.tag.unique() if item not in tags_dict.keys()]
df.tag = df.tag.apply(lambda x: "other" if x in oos_tags else x)
```

<div class="ai-center-all">
    ──── &nbsp; compared to &nbsp; ────
</div>

```python linenums="1"
def replace_oos_tags(df, tags_dict):
    """Replace out of scope (oos) tags."""
    oos_tags = [item for item in df.tag.unique() if item not in tags_dict.keys()]
    df.tag = df.tag.apply(lambda x: "other" if x in oos_tags else x)
    return df
```

It's better to wrap them as a separate function because we may want to:

- repeat this functionality in other parts of the project or in other projects.
- test that these tags are actually being replaced properly.

#### Composing generalized functions

```python linenums="1" title="Specific"
def replace_oos_tags(df, tags_dict):
    """Replace out of scope (oos) tags."""
    oos_tags = [item for item in df.tag.unique() if item not in tags_dict.keys()]
    df.tag = df.tag.apply(lambda x: "other" if x in oos_tags else x)
    return df
```

<div class="ai-center-all">
    ──── &nbsp; compared to &nbsp; ────
</div>

```python linenums="1" title="Generalized"
def replace_oos_labels(df, labels, label_col, oos_label="other"):
    """Replace out of scope (oos) labels."""
    oos_tags = [item for item in df[label_col].unique() if item not in labels]
    df[label_col] = df[label_col].apply(lambda x: oos_label if x in oos_tags else x)
    return df
```

This way when the names of columns change or we want to replace with different labels, it's very easy to adjust our code. This also includes using generalized names in the functions such as `label` instead of the name of the specific label column (ex. `tag`). It also allows others to reuse this functionality for their use cases.

> However, it's important not to force generalization if it involves a lot of effort. We can spend time later if we see the similar functionality reoccurring.

## 🔢&nbsp; Data

### Load

??? quote "Load and save data"

    First, we'll name and create the directory to save our data assets to (raw data, labeled data, etc.):

    ```python linenums="1" hl_lines="7 10"
    # config/config.py
    from pathlib import Path
    import pretty_errors

    # Directories
    BASE_DIR = Path(__file__).parent.parent.absolute()
    CONFIG_DIR = Path(BASE_DIR, "config")
    DATA_DIR = Path(BASE_DIR, "data")

    # Create dirs
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ```

    Next, we'll add the location of our raw data assets to our `config.py`. It's important that we store this information in our central configuration file so we can easily discover and update it if needed, as opposed to being deeply buried inside the code somewhere.

    ```python linenums="1"
    # config/config.py
    ...
    # Assets
    PROJECTS_URL = "https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/projects.csv"
    TAGS_URL = "https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/tags.csv"
    ```

    Since this is a main operation, we'll define it in `main.py`:

    ```python linenums="1"
    # tagifai/main.py
    import pandas as pd
    from pathlib import Path
    import warnings

    from config import config
    from tagifai import utils

    warnings.filterwarnings("ignore")

    def elt_data():
        """Extract, load and transform our data assets."""
        # Extract + Load
        projects = pd.read_csv(config.PROJECTS_URL)
        tags = pd.read_csv(config.TAGS_URL)
        projects.to_csv(Path(config.DATA_DIR, "projects.csv"), index=False)
        tags.to_csv(Path(config.DATA_DIR, "tags.csv"), index=False)

        # Transform
        df = pd.merge(projects, tags, on="id")
        df = df[df.tag.notnull()]  # drop rows w/ no tag
        df.to_csv(Path(config.DATA_DIR, "labeled_projects.csv"), index=False)

        logger.info("✅ Saved data!")
    ```

    Before we can use this operation, we need to make sure we have the necessary packages loaded into our environment. Libraries such as `pathlib`, `json`, etc. are preloaded with native Python, but packages like `NumPy` are not. Let's load the required packages and add them to our `requirements.txt` file.

    ```bash
    pip install numpy==1.19.5 pandas==1.3.5 pretty-errors==1.2.19
    ```

    ```bash
    # Add to requirements.txt
    numpy==1.19.5
    pandas==1.3.5
    pretty-errors==1.2.19
    ```

    > We can fetch the exact version of the packages we used in our notebook by running `#!bash pip freeze` in a code cell.

    Though we're not using the NumPy package for this `elt_data()` operation, our Python interpreter will still require it because we invoke the `utils.py` script with the line `#!python from tagifai import utils`, which does use NumPy in its header. So if we don't install the package in our virtual environment, we'll receive an error.

    We'll run the operation using the Python interpreter via the terminal (type `python` in the terminal and types the commands below).

    ```python linenums="1"
    from tagifai import main
    main.elt_data()
    ```

    We could also call this operation directly through the `main.py` script but we'll have to change it every time we want to run a new operation.

    ```python linenums="1"
    # tagifai/main.py
    if __name__ == "__main__":
        elt_data()
    ```
    ```bash
    python tagifai/main.py
    ```

    We'll learn about a much easier way to execute these operations in our [CLI lesson](cli.md){:target="_blank"}. But for now, either of the methods above will produce the same result.

    <pre class="output">
    ✅ Saved data!
    </pre>

    We should also see the data assets saved to our `data` directory:

    ```bash
    data/
    ├── projects.csv
    └── tags.csv
    ```

    !!! question "Why save the raw data?"
            Why do we need to save our raw data? Why not just load it from the URL and save the downstream assets (labels, features, etc.)?

            ??? quote "Show answer"
                We'll be using the raw data to generate labeled data and other downstream assets (ex. features). If the source of our raw data changes, then we'll no longer be able to produce our downstream assets. By saving it locally, we can always reproduce our results without any external dependencies. We'll also be executing [data validation](testing.md#data){:target="_blank"} checks on the raw data before applying transformations on it.

                However, as our dataset grows, it may not scale to save the raw data or even labels or features. We'll talk about more scalable alternatives in our [versioning](versioning.md#operations){:target="_blank"} lesson where we aren't saving the physical data but the instructions to retrieve them from a specific point in time.

### Preprocess

??? quote "Preprocess features"

    Next, we're going to define the functions for preprocessing our input features. We'll be using these functions when we are preparing the data prior to training our model. We won't be saving the preprocessed data to a file because different experiment may preprocess them differently.

    ```python linenums="1"
    # tagifai/data.py
    def preprocess(df, lower, stem, min_freq):
        """Preprocess the data."""
        df["text"] = df.title + " " + df.description  # feature engineering
        df.text = df.text.apply(clean_text, lower=lower, stem=stem)  # clean text
        df = replace_oos_labels(
            df=df, labels=config.ACCEPTED_TAGS, label_col="tag", oos_label="other"
        )  # replace OOS labels
        df = replace_minority_labels(
            df=df, label_col="tag", min_freq=min_freq, new_label="other"
        )  # replace labels below min freq

        return df
    ```

    This function uses the `clean_text()` function which we can define right above it:

    ```python linenums="1"
    # tagifai/data.py
    from nltk.stem import PorterStemmer
    import re

    from config import config

    def clean_text(text, lower=True, stem=False, stopwords=config.STOPWORDS):
        """Clean raw text."""
        # Lower
        if lower:
            text = text.lower()

        # Remove stopwords
        if len(stopwords):
            pattern = re.compile(r'\b(' + r"|".join(stopwords) + r")\b\s*")
            text = pattern.sub('', text)

        # Spacing and filters
        text = re.sub(
            r"([!\"'#$%&()*\+,-./:;<=>?@\\\[\]^_`{|}~])", r" \1 ", text
        )  # add spacing between objects to be filtered
        text = re.sub("[^A-Za-z0-9]+", " ", text)  # remove non alphanumeric chars
        text = re.sub(" +", " ", text)  # remove multiple spaces
        text = text.strip()  # strip white space at the ends

        # Remove links
        text = re.sub(r"http\S+", "", text)

        # Stemming
        if stem:
            text = " ".join([stemmer.stem(word, to_lowercase=lower) for word in text.split(" ")])

        return text
    ```

    Install required packages and add to `requirements.txt`:

    ```bash
    pip install nltk==3.7
    ```

    ```bash
    # Add to requirements.txt
    nltk==3.7
    ```

    Notice that we're using an explicit set of stopwords instead of NLTK's default list:
    ```python linenums="1"
    # NLTK's default stopwords
    nltk.download("stopwords")
    STOPWORDS = stopwords.words("english")
    ```

    This is because we want to have full visibility into exactly what words we're filtering. The general list may have some valuable terms we may wish to keep and vice versa.

    ```bash
    # config/config.py
    STOPWORDS = [
        "i",
        "me",
        "my",
        ...
        "won't",
        "wouldn",
        "wouldn't",
    ]
    ```

    Next, we need to define the two functions we're calling from `data.py`:

    ```python linenums="1"
    # tagifai/data.py
    from collections import Counter

    def replace_oos_labels(df, labels, label_col, oos_label="other"):
        """Replace out of scope (oos) labels."""
        oos_tags = [item for item in df[label_col].unique() if item not in labels]
        df[label_col] = df[label_col].apply(lambda x: oos_label if x in oos_tags else x)
        return df

    def replace_minority_labels(df, label_col, min_freq, new_label="other"):
        """Replace minority labels with another label."""
        labels = Counter(df[label_col].values)
        labels_above_freq = Counter(label for label in labels.elements() if (labels[label] >= min_freq))
        df[label_col] = df[label_col].apply(lambda label: label if label in labels_above_freq else None)
        df[label_col] = df[label_col].fillna(new_label)
        return df
    ```

### Encode

??? quote "Encode labels"

    Now let's define the encoder for our labels, which we'll use prior to splitting our dataset:

    ```python linenums="1"
    # tagifai/data.py
    import json
    import numpy as np

    class LabelEncoder(object):
        """Encode labels into unique indices."""
        def __init__(self, class_to_index={}):
            self.class_to_index = class_to_index or {}  # mutable defaults ;)
            self.index_to_class = {v: k for k, v in self.class_to_index.items()}
            self.classes = list(self.class_to_index.keys())

        def __len__(self):
            return len(self.class_to_index)

        def __str__(self):
            return f"<LabelEncoder(num_classes={len(self)})>"

        def fit(self, y):
            classes = np.unique(y)
            for i, class_ in enumerate(classes):
                self.class_to_index[class_] = i
            self.index_to_class = {v: k for k, v in self.class_to_index.items()}
            self.classes = list(self.class_to_index.keys())
            return self

        def encode(self, y):
            encoded = np.zeros((len(y)), dtype=int)
            for i, item in enumerate(y):
                encoded[i] = self.class_to_index[item]
            return encoded

        def decode(self, y):
            classes = []
            for i, item in enumerate(y):
                classes.append(self.index_to_class[item])
            return classes

        def save(self, fp):
            with open(fp, "w") as fp:
                contents = {"class_to_index": self.class_to_index}
                json.dump(contents, fp, indent=4, sort_keys=False)

        @classmethod
        def load(cls, fp):
            with open(fp, "r") as fp:
                kwargs = json.load(fp=fp)
            return cls(**kwargs)
    ```

### Split

??? quote "Split dataset"

    And finally, we'll conclude our data operations with our split function:

    ```python linenums="1"
    from sklearn.model_selection import train_test_split

    def get_data_splits(X, y, train_size=0.7):
        """Generate balanced data splits."""
        X_train, X_, y_train, y_ = train_test_split(
            X, y, train_size=train_size, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(
            X_, y_, train_size=0.5, stratify=y_)
        return X_train, X_val, X_test, y_train, y_val, y_test
    ```

    Install required packages and add to `requirements.txt`:

    ```bash
    pip install scikit-learn==0.24.2
    ```

    ```bash
    # Add to requirements.txt
    scikit-learn==0.24.2
    ```

## 📈&nbsp; Modeling

### Train

??? quote "Train w/ default args"

    Now we're ready to kick off the training process. We'll start by defining the operation in our `main.py`:

    ```python linenums="1"
    # tagifai/main.py
    import json
    from tagifai import data, train, utils

    def train_model(args_fp):
        """Train a model given arguments."""
        # Load labeled data
        df = pd.read_csv(Path(config.DATA_DIR, "labeled_projects.csv"))

        # Train
        args = Namespace(**utils.load_dict(filepath=args_fp))
        artifacts = train.train(df=df, args=args)
        performance = artifacts["performance"]
        print(json.dumps(performance, indent=2))
    ```

    We'll be adding more to our `train_model()` operation when we factor in experiment tracking but, for now, it's quite simple. This function calls for a `train()` function inside our `train.py` script:

    ```python linenums="1" hl_lines="55-56"
    # tagifai/train.py
    from imblearn.over_sampling import RandomOverSampler
    import json
    import numpy as np
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import log_loss

    from tagifai import data, predict, utils


    def train(args, df, trial=None):
    """Train model on data."""

        # Setup
        utils.set_seeds()
        if args.shuffle: df = df.sample(frac=1).reset_index(drop=True)
        df = df[: args.subset]  # None = all samples
        df = data.preprocess(df, lower=args.lower, stem=args.stem)
        label_encoder = data.LabelEncoder().fit(df.tag)
        X_train, X_val, X_test, y_train, y_val, y_test = \
            data.get_data_splits(X=df.text.to_numpy(), y=label_encoder.encode(df.tag))
        test_df = pd.DataFrame({"text": X_test, "tag": label_encoder.decode(y_test)})

        # Tf-idf
        vectorizer = TfidfVectorizer(analyzer=args.analyzer, ngram_range=(2,args.ngram_max_range))  # char n-grams
        X_train = vectorizer.fit_transform(X_train)
        X_val = vectorizer.transform(X_val)
        X_test = vectorizer.transform(X_test)

        # Oversample
        oversample = RandomOverSampler(sampling_strategy="all")
        X_over, y_over = oversample.fit_resample(X_train, y_train)

        # Model
        model = SGDClassifier(
            loss="log", penalty="l2", alpha=args.alpha, max_iter=1,
            learning_rate="constant", eta0=args.learning_rate, power_t=args.power_t,
            warm_start=True)

        # Training
        for epoch in range(args.num_epochs):
            model.fit(X_over, y_over)
            train_loss = log_loss(y_train, model.predict_proba(X_train))
            val_loss = log_loss(y_val, model.predict_proba(X_val))
            if not epoch%10:
                print(
                    f"Epoch: {epoch:02d} | "
                    f"train_loss: {train_loss:.5f}, "
                    f"val_loss: {val_loss:.5f}"
                )

        # Threshold
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)
        args.threshold = np.quantile(
            [y_prob[i][j] for i, j in enumerate(y_pred)], q=0.25)  # Q1

        # Evaluation
        other_index = label_encoder.class_to_index["other"]
        y_prob = model.predict_proba(X_test)
        y_pred = predict.custom_predict(y_prob=y_prob, threshold=args.threshold, index=other_index)
        performance = evaluate.get_metrics(
            y_true=y_test, y_pred=y_pred, classes=label_encoder.classes, df=test_df
        )

        return {
            "args": args,
            "label_encoder": label_encoder,
            "vectorizer": vectorizer,
            "model": model,
            "performance": performance,
        }
    ```

    This `train()` function calls two external functions (`predict.custom_predict()` from `predict.py` and `evaluate.get_metrics()` from `evaluate.py`):

    ```python linenums="1"
    # tagifai/predict.py
    import numpy as np

    def custom_predict(y_prob, threshold, index):
        """Custom predict function that defaults
        to an index if conditions are not met."""
        y_pred = [np.argmax(p) if max(p) > threshold else index for p in y_prob]
        return np.array(y_pred)
    ```

    ```python linenums="1"
    # tagifai/evaluate.py
    import numpy as np
    from sklearn.metrics import precision_recall_fscore_support
    from snorkel.slicing import PandasSFApplier
    from snorkel.slicing import slicing_function

    @slicing_function()
    def nlp_cnn(x):
        """NLP Projects that use convolution."""
        nlp_projects = "natural-language-processing" in x.tag
        convolution_projects = "CNN" in x.text or "convolution" in x.text
        return (nlp_projects and convolution_projects)

    @slicing_function()
    def short_text(x):
        """Projects with short titles and descriptions."""
        return len(x.text.split()) < 8  # less than 8 words

    def get_slice_metrics(y_true, y_pred, slices):
        """Generate metrics for slices of data."""
        metrics = {}
        for slice_name in slices.dtype.names:
            mask = slices[slice_name].astype(bool)
            if sum(mask):
                slice_metrics = precision_recall_fscore_support(
                    y_true[mask], y_pred[mask], average="micro"
                )
                metrics[slice_name] = {}
                metrics[slice_name]["precision"] = slice_metrics[0]
                metrics[slice_name]["recall"] = slice_metrics[1]
                metrics[slice_name]["f1"] = slice_metrics[2]
                metrics[slice_name]["num_samples"] = len(y_true[mask])
        return metrics

    def get_metrics(y_true, y_pred, classes, df=None):
        """Performance metrics using ground truths and predictions."""
        # Performance
        metrics = {"overall": {}, "class": {}}

        # Overall metrics
        overall_metrics = precision_recall_fscore_support(y_true, y_pred, average="weighted")
        metrics["overall"]["precision"] = overall_metrics[0]
        metrics["overall"]["recall"] = overall_metrics[1]
        metrics["overall"]["f1"] = overall_metrics[2]
        metrics["overall"]["num_samples"] = np.float64(len(y_true))

        # Per-class metrics
        class_metrics = precision_recall_fscore_support(y_true, y_pred, average=None)
        for i, _class in enumerate(classes):
            metrics["class"][_class] = {
                "precision": class_metrics[0][i],
                "recall": class_metrics[1][i],
                "f1": class_metrics[2][i],
                "num_samples": np.float64(class_metrics[3][i]),
            }

        # Slice metrics
        if df is not None:
            slices = PandasSFApplier([nlp_cnn, short_text]).apply(df)
            metrics["slices"] = get_slice_metrics(
                y_true=y_true, y_pred=y_pred, slices=slices)

        return metrics
    ```

    Install required packages and add to `requirements.txt`:

    ```bash
    pip install imbalanced-learn==0.8.1 snorkel==0.9.8
    ```

    ```bash
    # Add to requirements.txt
    imbalanced-learn==0.8.1
    snorkel==0.9.8
    ```

    Commands to train a model:

    ```python linenums="1"
    from config import config
    from tagifai import main
    args_fp = Path(config.CONFIG_DIR, "args.json")
    main.train_model(args_fp)
    ```

    <pre class="output">
    Epoch: 00 | train_loss: 1.16783, val_loss: 1.20177
    Epoch: 10 | train_loss: 0.46262, val_loss: 0.62612
    Epoch: 20 | train_loss: 0.31599, val_loss: 0.51986
    Epoch: 30 | train_loss: 0.25191, val_loss: 0.47544
    Epoch: 40 | train_loss: 0.21720, val_loss: 0.45176
    Epoch: 50 | train_loss: 0.19610, val_loss: 0.43770
    Epoch: 60 | train_loss: 0.18221, val_loss: 0.42857
    Epoch: 70 | train_loss: 0.17291, val_loss: 0.42246
    Epoch: 80 | train_loss: 0.16643, val_loss: 0.41818
    Epoch: 90 | train_loss: 0.16160, val_loss: 0.41528
    {
      "overall": {
        "precision": 0.8990934378802025,
        "recall": 0.8194444444444444,
        "f1": 0.838280325954406,
        "num_samples": 144.0
      },
      "class": {
        "computer-vision": {
          "precision": 0.975,
          "recall": 0.7222222222222222,
          "f1": 0.8297872340425532,
          "num_samples": 54.0
        },
        "mlops": {
          "precision": 0.9090909090909091,
          "recall": 0.8333333333333334,
          "f1": 0.8695652173913043,
          "num_samples": 12.0
        },
        "natural-language-processing": {
          "precision": 0.9803921568627451,
          "recall": 0.8620689655172413,
          "f1": 0.9174311926605505,
          "num_samples": 58.0
        },
        "other": {
          "precision": 0.4523809523809524,
          "recall": 0.95,
          "f1": 0.6129032258064516,
          "num_samples": 20.0
        }
      },
      "slices": {
        "nlp_cnn": {
          "precision": 1.0,
          "recall": 1.0,
          "f1": 1.0,
          "num_samples": 1
        },
        "short_text": {
          "precision": 0.8,
          "recall": 0.8,
          "f1": 0.8000000000000002,
          "num_samples": 5
        }
      }
    }
    </pre>


### Optimize

??? quote "Optimize args"

    Now that we can train one model, we're ready to train many models to optimize our hyperparameters:

    ```python linenums="1"
    # tagifai/main.py
    import mlflow
    from numpyencoder import NumpyEncoder
    import optuna
    from optuna.integration.mlflow import MLflowCallback

    def optimize(study_name, num_trials):
        """Optimize hyperparameters."""
        # Load labeled data
        df = pd.read_csv(Path(config.DATA_DIR, "labeled_projects.csv"))

        # Optimize
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
        study = optuna.create_study(study_name="optimization", direction="maximize", pruner=pruner)
        mlflow_callback = MLflowCallback(
            tracking_uri=mlflow.get_tracking_uri(), metric_name="f1")
        study.optimize(
            lambda trial: train.objective(args, df, trial),
            n_trials=num_trials,
            callbacks=[mlflow_callback])

        # Best trial
        trials_df = study.trials_dataframe()
        trials_df = trials_df.sort_values(["user_attrs_f1"], ascending=False)
        utils.save_dict({**args.__dict__, **study.best_trial.params}, args_fp, cls=NumpyEncoder)
        print(f"\nBest value (f1): {study.best_trial.value}")
        print(f"Best hyperparameters: {json.dumps(study.best_trial.params, indent=2)}")
    ```

    We'll define the `objective()` function inside `train.py`:

    ```python linenums="1"
    # tagifai/train.py
    def objective(args, df, trial):
        """Objective function for optimization trials."""
        # Parameters to tune
        args.analyzer = trial.suggest_categorical("analyzer", ["word", "char", "char_wb"])
        args.ngram_max_range = trial.suggest_int("ngram_max_range", 3, 10)
        args.learning_rate = trial.suggest_loguniform("learning_rate", 1e-2, 1e0)
        args.power_t = trial.suggest_uniform("power_t", 0.1, 0.5)

        # Train & evaluate
        artifacts = train(args=args, df=df, trial=trial)

        # Set additional attributes
        overall_performance = artifacts["performance"]["overall"]
        print(json.dumps(overall_performance, indent=2))
        trial.set_user_attr("precision", overall_performance["precision"])
        trial.set_user_attr("recall", overall_performance["recall"])
        trial.set_user_attr("f1", overall_performance["f1"])

        return overall_performance["f1"]
    ```

    Recall that in our notebook, we modified the `train()` function to include information about trials during optimization for pruning:

    ```python linenums="1"
    # tagifai/train.py
    import optuna

    def train():
        ...
        # Training
        for epoch in range(args.num_epochs):
            ...
            # Pruning (for optimization in next section)
            if trial:
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
    ```

    Since we're using the `MLflowCallback` here with Optuna, we can either allow all our experiments to be stored under the default `mlruns` directory that MLflow will create or we can configure that location:

    ```python linenums="1"
    # config/config.py
    import mlflow
    STORES_DIR = Path(BASE_DIR, "stores")
    MODEL_REGISTRY = Path(STORES_DIR, "model")
    MODEL_REGISTRY.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri("file://" + str(MODEL_REGISTRY.absolute()))
    ```

    Install required packages and add to `requirements.txt`:

    ```bash
    pip install mlflow==1.23.1 optuna==2.10.0 numpyencoder==0.3.0
    ```

    ```bash
    # Add to requirements.txt
    mlflow==1.23.1
    numpyencoder==0.3.0
    optuna==2.10.0
    ```

    Commands to optimize hyperparameters:

    ```python linenums="1"
    from config import config
    from tagifai import main
    args_fp = Path(config.CONFIG_DIR, "args.json")
    main.optimize(args_fp, study_name="optimization", num_trials=20)
    ```

    <pre class="output">
    A new study created in memory with name: optimization
    ...
    Best value (f1): 0.8497010532479641
    Best hyperparameters: {
        "analyzer": "char_wb",
        "ngram_max_range": 6,
        "learning_rate": 0.8616849162496086,
        "power_t": 0.21283622300887173
    }
    </pre>

    We should see our experiment in our model registry, located at `stores/model/`:

    ```bash
    stores/model/
    └── 0/
    ```

### Experiment tracking

??? quote "Experiment tracking"

    Now that we have our optimized hyperparameters, we can train a model and store it's artifacts via experiment tracking. We'll start by modifying the `train()` operation in our `main.py` script:

    ```python linenums="1"
    # tagifai/main.py
    import joblib
    import tempfile

    def train_model(args_fp, experiment_name, run_name):
        """Train a model given arguments."""
        # Load labeled data
        df = pd.read_csv(Path(config.DATA_DIR, "labeled_projects.csv"))

        # Train
        args = Namespace(**utils.load_dict(filepath=args_fp))
        mlflow.set_experiment(experiment_name=experiment_name)
        with mlflow.start_run(run_name=run_name):
            run_id = mlflow.active_run().info.run_id
            print(f"Run ID: {run_id}")
            artifacts = train.train(df=df, args=args)
            performance = artifacts["performance"]
            print(json.dumps(performance, indent=2))

            # Log metrics and parameters
            performance = artifacts["performance"]
            mlflow.log_metrics({"precision": performance["overall"]["precision"]})
            mlflow.log_metrics({"recall": performance["overall"]["recall"]})
            mlflow.log_metrics({"f1": performance["overall"]["f1"]})
            mlflow.log_params(vars(artifacts["args"]))

            # Log artifacts
            with tempfile.TemporaryDirectory() as dp:
                artifacts["label_encoder"].save(Path(dp, "label_encoder.json"))
                joblib.dump(artifacts["vectorizer"], Path(dp, "vectorizer.pkl"))
                joblib.dump(artifacts["model"], Path(dp, "model.pkl"))
                utils.save_dict(performance, Path(dp, "performance.json"))
                mlflow.log_artifacts(dp)

        # Save to config
        open(Path(config.CONFIG_DIR, "run_id.txt"), "w").write(run_id)
        utils.save_dict(performance, Path(config.CONFIG_DIR, "performance.json"))
    ```

    There's a lot more happening inside our `train_model()` function but it's necessary in order to store all the metrics, parameters and artifacts. We're also going to update the `train()` function inside `train.py` so that the intermediate metrics are captured:

    ```python linenums="1"
    # tagifai/train.py
    import mlflow

    def train():
        ...
        # Training
        for epoch in range(args.num_epochs):
            ...
            # Log
            if not trial:
                mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)
    ```

    Commands to train a model with experiment tracking:

    ```python linenums="1"
    from config import config
    from tagifai import main
    args_fp = Path(config.CONFIG_DIR, "args.json")
    main.train_model(args_fp, experiment_name="baselines", run_name="sgd")
    ```

    <pre class="output">
    Run ID: d91d9760b2e14a5fbbae9f3762f0afaf
    Epoch: 00 | train_loss: 0.74266, val_loss: 0.83335
    Epoch: 10 | train_loss: 0.21884, val_loss: 0.42853
    Epoch: 20 | train_loss: 0.16632, val_loss: 0.39420
    Epoch: 30 | train_loss: 0.15108, val_loss: 0.38396
    Epoch: 40 | train_loss: 0.14589, val_loss: 0.38089
    Epoch: 50 | train_loss: 0.14358, val_loss: 0.37992
    Epoch: 60 | train_loss: 0.14084, val_loss: 0.37977
    Epoch: 70 | train_loss: 0.14025, val_loss: 0.37828
    Epoch: 80 | train_loss: 0.13983, val_loss: 0.37699
    Epoch: 90 | train_loss: 0.13841, val_loss: 0.37772
    {
      "overall": {
        "precision": 0.9026155077984347,
        "recall": 0.8333333333333334,
        "f1": 0.8497010532479641,
        "num_samples": 144.0
      },
      "class": {
        "computer-vision": {
          "precision": 0.975609756097561,
          "recall": 0.7407407407407407,
          "f1": 0.8421052631578947,
          "num_samples": 54.0
        },
        "mlops": {
          "precision": 0.9090909090909091,
          "recall": 0.8333333333333334,
          "f1": 0.8695652173913043,
          "num_samples": 12.0
        },
        "natural-language-processing": {
          "precision": 0.9807692307692307,
          "recall": 0.8793103448275862,
          "f1": 0.9272727272727272,
          "num_samples": 58.0
        },
        "other": {
          "precision": 0.475,
          "recall": 0.95,
          "f1": 0.6333333333333334,
          "num_samples": 20.0
        }
      },
      "slices": {
        "nlp_cnn": {
          "precision": 1.0,
          "recall": 1.0,
          "f1": 1.0,
          "num_samples": 1
        },
        "short_text": {
          "precision": 0.8,
          "recall": 0.8,
          "f1": 0.8000000000000002,
          "num_samples": 5
        }
      }
    }
    </pre>

    Our configuration directory should now have a `performance.json` and a `run_id.txt` file. We're saving these so we can quickly access this metadata of the latest successful training. If we were considering several models as once, we could manually set the run_id of the run we want to deploy or programmatically identify the best across experiments.

    ```bash
    config/
    ├── args.json         - arguments
    ├── config.py         - configuration setup
    ├── performance.json  - performance metrics
    └── run_id.txt        - ID of latest successful run
    ```

    And we should see this specific experiment and run in our model registry:

    ```bash
    stores/model/
    ├── 0/
    └── 1/
    ```

### Predict

??? quote "Predict texts"

    We're finally ready to use our trained model for inference. We'll add the operation to predict a tag to `main.py`:

    ```python linenums="1"
    # tagifai/main.py
    from tagifai import data, predict, train, utils

    def predict_tag(text, run_id=None):
        """Predict tag for text."""
        if not run_id:
            run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
        artifacts = load_artifacts(run_id=run_id)
        prediction = predict.predict(texts=[text], artifacts=artifacts)
        print(json.dumps(prediction, indent=2))
        return prediction
    ```

    This involves creating the `load_artifacts()` function inside our `main.py` script:

    ```python linenums="1"
    # tagifai/main.py
    def load_artifacts(run_id):
        """Load artifacts for a given run_id."""
        # Locate specifics artifacts directory
        experiment_id = mlflow.get_run(run_id=run_id).info.experiment_id
        artifacts_dir = Path(config.MODEL_REGISTRY, experiment_id, run_id, "artifacts")

        # Load objects from run
        args = Namespace(**utils.load_dict(filepath=Path(artifacts_dir, "args.json")))
        vectorizer = joblib.load(Path(artifacts_dir, "vectorizer.pkl"))
        label_encoder = data.LabelEncoder.load(fp=Path(artifacts_dir, "label_encoder.json"))
        model = joblib.load(Path(artifacts_dir, "model.pkl"))
        performance = utils.load_dict(filepath=Path(artifacts_dir, "performance.json"))

        return {
            "args": args,
            "label_encoder": label_encoder,
            "vectorizer": vectorizer,
            "model": model,
            "performance": performance
        }
    ```

    and defining the `predict()` function inside `predict.py`:

    ```python linenums="1"
    def predict(texts, artifacts):
        """Predict tags for given texts."""
        x = artifacts["vectorizer"].transform(texts)
        y_pred = custom_predict(
            y_prob=artifacts["model"].predict_proba(x),
            threshold=artifacts["args"].threshold,
            index=artifacts["label_encoder"].class_to_index["other"])
        tags = artifacts["label_encoder"].decode(y_pred)
        predictions = [
            {
                "input_text": texts[i],
                "predicted_tags": tags[i],
            }
            for i in range(len(tags))
        ]
        return predictions
    ```

    Commands to predict the tag for text:

    ```python linenums="1"
    text = "Transfer learning with transformers for text classification."
    run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
    predict_tag(text=text, run_id=run_id)
    ```

    <pre class="output">
    [
      {
        "input_text": "Transfer learning with transformers for text classification.",
        "predicted_tag": "natural-language-processing"
      }
    ]
    </pre>

> Don't worry about formatting our functions and classes just yet. We'll be covering how to properly do this in the [documentation](documentation.md){:target="_blank"} lesson.

!!! question "So many functions and classes..."
    As we migrated from notebooks to scripts, we had to define so many functions and classes. How can we improve this?

    ??? quote "Show answer"

        As we work on more projects, we may find it useful to contribute our generalized functions and classes to a central repository. Provided that all the code is [tested](testing.md){:target="_blank"} and [documented](documentation.md){:target="_blank"}, this can reduce boilerplate code and redundant efforts. To make this central repository available for everyone, we can [package](https://packaging.python.org/tutorials/packaging-projects/){:target="_blank"} it and share it publicly or keep it private with a PyPI mirror, etc.
        ```bash
        # Ex. installing our public repo
        pip install git+https://github.com/GokuMohandas/mlops-course#egg=tagifai
        ```

<!-- Course signup -->
{% include "templates/course-signup.md" %}

<!-- Citation -->
{% include "templates/cite.md" %}
