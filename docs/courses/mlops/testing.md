---
template: lesson.html
title: "Testing Machine Learning Systems: Code, Data and Models"
description: Learn how to test ML models (and their code and data) to ensure consistent behavior in our ML systems.
keywords: testing, testing ml, pytest, great expectations, unit test, parametrize, fixtures, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/mlops.png
repository: https://github.com/GokuMohandas/mlops-course
notebook: https://github.com/GokuMohandas/testing-ml/blob/main/testing.ipynb
---

{% include "styles/lesson.md" %}

## Intuition

In this lesson, we'll learn how to test code, data and models to construct a machine learning system that we can reliably iterate on. Tests are a way for us to ensure that something works as intended. We're incentivized to implement tests and discover sources of error as early in the development cycle as possible so that we can decrease [downstream costs](https://assets.deepsource.io/39ed384/images/blog/cost-of-fixing-bugs/chart.jpg){:target="_blank"} and wasted time. Once we've designed our tests, we can automatically execute them every time we change or add to our codebase.

!!! tip
    We highly recommend that you explore this lesson *after* completing the previous lessons since the topics (and code) are iteratively developed. We did, however, create the :fontawesome-brands-github:{ .github } [testing-ml](https://github.com/GokuMohandas/testing-ml){:target="_blank"} repository for a quick overview with an interactive notebook.

### Types of tests

There are four majors types of tests which are utilized at different points in the development cycle:

1. `#!js Unit tests`: tests on individual components that each have a [single responsibility](https://en.wikipedia.org/wiki/Single-responsibility_principle){:target="_blank"} (ex. function that filters a list).
2. `#!js Integration tests`: tests on the combined functionality of individual components (ex. data processing).
3. `#!js System tests`: tests on the design of a system for expected outputs given inputs (ex. training, inference, etc.).
4. `#!js Acceptance tests`: tests to verify that requirements have been met, usually referred to as User Acceptance Testing (UAT).
5. `#!js Regression tests`: tests based on errors we've seen before to ensure new changes don't reintroduce them.

While ML systems are probabilistic in nature, they are composed of many deterministic components that can be tested in a similar manner as traditional software systems. The distinction between testing ML systems begins when we move from testing code to testing the [data](testing.md#data){:target="_blank"} and [models](testing.md#models){:target="_blank"}.

<div class="ai-center-all">
    <img width="700" src="/static/images/mlops/testing/tests.png" alt="types of tests">
</div>

> There are many other types of functional and non-functional tests as well, such as smoke tests (quick health checks), performance tests (load, stress), security tests, etc. but we can generalize all of these under the system tests above.

### How should we test?

The framework to use when composing tests is the [Arrange Act Assert](http://wiki.c2.com/?ArrangeActAssert){:target="_blank"} methodology.

- `#!js Arrange`: set up the different inputs to test on.
- `#!js Act`: apply the inputs on the component we want to test.
- `#!js Assert`: confirm that we received the expected output.

> `#!js Cleaning` is an unofficial fourth step to this methodology because it's important to not leave remnants of a previous test which may affect subsequent tests. We can use packages such as [pytest-randomly](https://github.com/pytest-dev/pytest-randomly){:target="_blank"} to test against state dependency by executing tests randomly.

In Python, there are many tools, such as [unittest](https://docs.python.org/3/library/unittest.html){:target="_blank"}, [pytest](https://docs.pytest.org/en/stable/){:target="_blank"}, etc. that allow us to easily implement our tests while adhering to the *Arrange Act Assert* framework. These tools come with powerful built-in functionality such as parametrization, filters, and more, to test many conditions at scale.

### What should we test?

When *arranging* our inputs and *asserting* our expected outputs, what are some aspects of our inputs and outputs that we should be testing for?

- **inputs**: data types, format, length, edge cases (min/max, small/large, etc.)
- **outputs**: data types, formats, exceptions, intermediary and final outputs

> 👉 &nbsp;We'll cover specific details pertaining to what to test for regarding our [data](testing.md#data) and [models](testing.md#models) below.

## Best practices
Regardless of the framework we use, it's important to strongly tie testing into the development process.

- `#!js atomic`: when creating functions and classes, we need to ensure that they have a [single responsibility](https://en.wikipedia.org/wiki/Single-responsibility_principle){:target="_blank"} so that we can easily test them. If not, we'll need to split them into more granular components.
- `#!js compose`: when we create new components, we want to compose tests to validate their functionality. It's a great way to ensure reliability and catch errors early on.
- `#!js reuse`: we should maintain central repositories where core functionality is tested at the source and reused across many projects. This significantly reduces testing efforts for each new project's code base.
- `#!js regression`: we want to account for new errors we come across with a regression test so we can ensure we don't reintroduce the same errors in the future.
- `#!js coverage`: we want to ensure [100% coverage](#coverage) for our codebase. This doesn't mean writing a test for every single line of code but rather accounting for every single line.
- `#!js automate`: in the event we forget to run our tests before committing to a repository, we want to auto run tests when we make changes to our codebase. We'll learn how to do this locally using [pre-commit hooks](pre-commit.md){:target="_blank"} and remotely via [GitHub actions](cicd.md#github-actions){:target="_blank"} in subsequent lessons.

## Test-driven development

[Test-driven development (TDD)](https://en.wikipedia.org/wiki/Test-driven_development){:target="_blank"} is the process of writing a test before writing the functionality to ensure that tests are always written. This is in contrast to writing functionality first and then composing tests afterwards. Here are our thoughts on this:

- good to write tests as we progress, but it does signify 100% correctness.
- initial time should be spent on design before ever getting into the code or tests.

Perfect coverage doesn't mean that our application is error free if those tests aren't meaningful and don't encompass the field of possible inputs, intermediates and outputs. Therefore, we should work towards better design and agility when facing errors, quickly resolving them and writing test cases around them to avoid next time.

## Application

In our [application](https://github.com/GokuMohandas/mlops-course){:target="_blank"}, we'll be testing the code, data and models. We'll start by creating a separate `tests` directory with `code` subdirectory for testing our `tagifai` scripts. We'll create subdirectories for testing [data](#🔢nbsp-data) and [models](#🤖nbsp-models) soon below.

```bash
mkdir tests
cd tests
mkdir app config model tagifai
touch <SCRIPTS>
cd ../
```

```bash
tests/
└── code/
│   ├── test_data.py
│   ├── test_evaluate.py
│   ├── test_main.py
│   ├── test_predict.py
│   └── test_utils.py
```

Feel free to write the tests and organize them in these scripts *after* learning about all the concepts in this lesson. We suggest using our [`tests`](https://github.com/GokuMohandas/mlops-course/tree/main/tests){:target="_blank"} directory on GitHub as a reference.

> Notice that our `tagifai/train.py` script does not have it's respective `tests/code/test_train.py`. Some scripts have large functions (ex. `train.train()`, `train.optimize()`, `predict.predict()`, etc.) with dependencies (ex. artifacts) and it makes sense to test them via `tests/code/test_main.py`.

## 🧪&nbsp; Pytest

We're going to be using [pytest](https://docs.pytest.org/en/stable/){:target="_blank"} as our testing framework for it's powerful builtin features such as [parametrization](#parametrize), [fixtures](#fixtures), [markers](#markers) and more.

```bash
pip install pytest==7.1.2
```

Since this testing package is not integral to the core machine learning operations, let's create a separate list in our `setup.py` and add it to our `extras_require`:

```python linenums="1" hl_lines="10"
# setup.py
test_packages = [
    "pytest==7.1.2",
]

# Define our package
setup(
    ...
    extras_require={
        "dev": docs_packages + style_packages + test_packages,
        "docs": docs_packages,
        "test": test_packages,
    },
)
```

We created an explicit `test` option because a user will want to only download the testing packages. We'll see this in action when we use [CI/CD workflows](cicd.md){:target="_blank"} to run tests via GitHub Actions.

### Configuration
Pytest expects tests to be organized under a `tests` directory by default. However, we can also add to our existing `pyproject.toml` file to configure any other test directories as well. Once in the directory, pytest looks for python scripts starting with `tests_*.py` but we can configure it to read any other file patterns as well.

```toml linenums="1"
# Pytest
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
```

### Assertions

Let's see what a sample test and it's results look like. Assume we have a simple function that determines whether a fruit is crisp or not:

```python linenums="1"
# food/fruits.py
def is_crisp(fruit):
    if fruit:
        fruit = fruit.lower()
    if fruit in ["apple", "watermelon", "cherries"]:
        return True
    elif fruit in ["orange", "mango", "strawberry"]:
        return False
    else:
        raise ValueError(f"{fruit} not in known list of fruits.")
    return False
```

To test this function, we can use [assert statements](https://docs.pytest.org/en/stable/assert.html){:target="_blank"} to map inputs with expected outputs. The statement following the word `assert` must return True.

```python linenums="1"
# tests/food/test_fruits.py
def test_is_crisp():
    assert is_crisp(fruit="apple")
    assert is_crisp(fruit="Apple")
    assert not is_crisp(fruit="orange")
    with pytest.raises(ValueError):
        is_crisp(fruit=None)
        is_crisp(fruit="pear")
```

> We can also have assertions about [exceptions](https://docs.pytest.org/en/stable/assert.html#assertions-about-expected-exceptions){:target="_blank"} like we do in lines 6-8 where all the operations under the with statement are expected to raise the specified exception.

??? quote "Example of using `assert` in our project"

    ```python linenums="1"
    # tests/code/test_evaluate.py
    def test_get_metrics():
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])
        classes = ["a", "b"]
        performance = evaluate.get_metrics(y_true=y_true, y_pred=y_pred, classes=classes, df=None)
        assert performance["overall"]["precision"] == 2/4
        assert performance["overall"]["recall"] == 2/4
        assert performance["class"]["a"]["precision"] == 1/2
        assert performance["class"]["a"]["recall"] == 1/2
        assert performance["class"]["b"]["precision"] == 1/2
        assert performance["class"]["b"]["recall"] == 1/2
    ```

### Execution

We can execute our tests above using several different levels of granularity:

```bash
python3 -m pytest                                           # all tests
python3 -m pytest tests/food                                # tests under a directory
python3 -m pytest tests/food/test_fruits.py                 # tests for a single file
python3 -m pytest tests/food/test_fruits.py::test_is_crisp  # tests for a single function
```

Running our specific test above would produce the following output:
```bash
python3 -m pytest tests/food/test_fruits.py::test_is_crisp
```
<pre class="output">
tests/food/test_fruits.py::test_is_crisp <span style="color: #5A9C4B; font-weight: 600;">.           [100%]</span>
</pre>

Had any of our assertions in this test failed, we would see the failed assertions, along with the expected and actual output from our function.

<pre class="output">
tests/food/test_fruits.py <span style="color: #F50071; font-weight: 600;">F                          [100%]</span>

    def test_is_crisp():
>       assert is_crisp(fruit="orange")
<span style="color: #F50071; font-weight: 600;">E       AssertionError: assert False</span>
<span style="color: #F50071; font-weight: 600;">E        +  where False = is_crisp(fruit='orange')</span>
</pre>

!!! tip
    It's important to test for the variety of inputs and expected outputs that we outlined [above](#how-should-we-test) and to **never assume that a test is trivial**. In our example above, it's important that we test for both "apple" and "Apple" in the event that our function didn't account for casing!

### Classes

We can also test classes and their respective functions by creating test classes. Within our test class, we can optionally define [functions](https://docs.pytest.org/en/stable/xunit_setup.html){:target="_blank"} which will automatically be executed when we setup or teardown a class instance or use a class method.

- `#!js setup_class`: set up the state for any class instance.
- `#!js teardown_class`: teardown the state created in setup_class.
- `#!js setup_method`: called before every method to setup any state.
- `#!js teardown_method`: called after every method to teardown any state.

```python linenums="1"
class Fruit(object):
    def __init__(self, name):
        self.name = name

class TestFruit(object):
    @classmethod
    def setup_class(cls):
        """Set up the state for any class instance."""
        pass

    @classmethod
    def teardown_class(cls):
        """Teardown the state created in setup_class."""
        pass

    def setup_method(self):
        """Called before every method to setup any state."""
        self.fruit = Fruit(name="apple")

    def teardown_method(self):
        """Called after every method to teardown any state."""
        del self.fruit

    def test_init(self):
        assert self.fruit.name == "apple"
```

We can execute all the tests for our class by specifying the class name:

```bash
python3 -m pytest tests/food/test_fruits.py::TestFruit
```
<pre class="output">
tests/food/test_fruits.py::TestFruit <span style="color: #5A9C4B; font-weight: 600;">.           [100%]</span>
</pre>

??? quote "Example of testing a `#!python class` in our project"

    ```python linenums="1"
    # tests/code/test_data.py
    class TestLabelEncoder:
    @classmethod
    def setup_class(cls):
        """Called before every class initialization."""
        pass

    @classmethod
    def teardown_class(cls):
        """Called after every class initialization."""
        pass

    def setup_method(self):
        """Called before every method."""
        self.label_encoder = data.LabelEncoder()

    def teardown_method(self):
        """Called after every method."""
        del self.label_encoder

    def test_empty_init(self):
        label_encoder = data.LabelEncoder()
        assert label_encoder.index_to_class == {}
        assert len(label_encoder.classes) == 0

    def test_dict_init(self):
        class_to_index = {"apple": 0, "banana": 1}
        label_encoder = data.LabelEncoder(class_to_index=class_to_index)
        assert label_encoder.index_to_class == {0: "apple", 1: "banana"}
        assert len(label_encoder.classes) == 2

    def test_len(self):
        assert len(self.label_encoder) == 0

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as dp:
            fp = Path(dp, "label_encoder.json")
            self.label_encoder.save(fp=fp)
            label_encoder = data.LabelEncoder.load(fp=fp)
            assert len(label_encoder.classes) == 0

    def test_str(self):
        assert str(data.LabelEncoder()) == "<LabelEncoder(num_classes=0)>"

    def test_fit(self):
        label_encoder = data.LabelEncoder()
        label_encoder.fit(["apple", "apple", "banana"])
        assert "apple" in label_encoder.class_to_index
        assert "banana" in label_encoder.class_to_index
        assert len(label_encoder.classes) == 2

    def test_encode_decode(self):
        class_to_index = {"apple": 0, "banana": 1}
        y_encoded = [0, 0, 1]
        y_decoded = ["apple", "apple", "banana"]
        label_encoder = data.LabelEncoder(class_to_index=class_to_index)
        label_encoder.fit(["apple", "apple", "banana"])
        assert np.array_equal(label_encoder.encode(y_decoded), np.array(y_encoded))
        assert label_encoder.decode(y_encoded) == y_decoded
    ```

### Parametrize

So far, in our tests, we've had to create individual assert statements to validate different combinations of inputs and expected outputs. However, there's a bit of redundancy here because the inputs always feed into our functions as arguments and the outputs are compared with our expected outputs. To remove this redundancy, pytest has the [`@pytest.mark.parametrize`](https://docs.pytest.org/en/stable/parametrize.html){:target="_blank"} decorator which allows us to represent our inputs and outputs as parameters.

```python linenums="1"
@pytest.mark.parametrize(
    "fruit, crisp",
    [
        ("apple", True),
        ("Apple", True),
        ("orange", False),
    ],
)
def test_is_crisp_parametrize(fruit, crisp):
    assert is_crisp(fruit=fruit) == crisp
```

<pre class="output">
python3 -m pytest tests/food/test_is_crisp_parametrize.py <span style="color: #5A9C4B; font-weight: 600;">...</span>   [100%]
</pre>

1. `#!js [Line 2]`: define the names of the parameters under the decorator, ex. "fruit, crisp" (note that this is one string).
2. `#!js [Lines 3-7]`: provide a list of combinations of values for the parameters from Step 1.
3. `#!js [Line 9]`: pass in parameter names to the test function.
4. `#!js [Line 10]`: include necessary assert statements which will be executed for each of the combinations in the list from Step 2.

Similarly, we could pass in an exception as the expected result as well:

```python linenums="1"
@pytest.mark.parametrize(
    "fruit, exception",
    [
        ("pear", ValueError),
    ],
)
def test_is_crisp_exceptions(fruit, exception):
    with pytest.raises(exception):
        is_crisp(fruit=fruit)
```

??? quote "Example of `parametrize` in our project"

    ```python linenums="1"
    # tests/code/test_data.py
    from tagifai import data
    @pytest.mark.parametrize(
        "text, lower, stem, stopwords, cleaned_text",
        [
            ("Hello worlds", False, False, [], "Hello worlds"),
            ("Hello worlds", True, False, [], "hello worlds"),
            ...
        ],
    )
    def test_preprocess(text, lower, stem, stopwords, cleaned_text):
        assert (
            data.clean_text(
                text=text,
                lower=lower,
                stem=stem,
                stopwords=stopwords,
            )
            == cleaned_text
        )
    ```

### Fixtures

Parametrization allows us to reduce redundancy inside test functions but what about reducing redundancy across different test functions? For example, suppose that different functions all have a dataframe as an input. Here, we can use pytest's builtin [fixture](https://docs.pytest.org/en/stable/fixture.html){:target="_blank"}, which is a function that is executed before the test function.

```python linenums="1"
@pytest.fixture
def my_fruit():
    fruit = Fruit(name="apple")
    return fruit

def test_fruit(my_fruit):
    assert my_fruit.name == "apple"
```

> Note that the name of fixture and the input to the test function are identical (`my_fruit`).

We can apply fixtures to classes as well where the fixture function will be invoked when any method in the class is called.

```python linenums="1"
@pytest.mark.usefixtures("my_fruit")
class TestFruit:
    ...
```

??? quote "Example of `fixtures` in our project"
    In our project, we use fixtures to efficiently pass a set of inputs (ex. Pandas DataFrame) to different testing functions that require them (cleaning, splitting, etc.).

    ```python linenums="1"
    # tests/code/test_data.py
    @pytest.fixture(scope="module")
    def df():
        data = [
            {"title": "a0", "description": "b0", "tag": "c0"},
            {"title": "a1", "description": "b1", "tag": "c1"},
            {"title": "a2", "description": "b2", "tag": "c1"},
            {"title": "a3", "description": "b3", "tag": "c2"},
            {"title": "a4", "description": "b4", "tag": "c2"},
            {"title": "a5", "description": "b5", "tag": "c2"},
        ]
        df = pd.DataFrame(data * 10)
        return df


    @pytest.mark.parametrize(
        "labels, unique_labels",
        [
            ([], ["other"]),  # no set of approved labels
            (["c3"], ["other"]),  # no overlap b/w approved/actual labels
            (["c0"], ["c0", "other"]),  # partial overlap
            (["c0", "c1", "c2"], ["c0", "c1", "c2"]),  # complete overlap
        ],
    )
    def test_replace_oos_labels(df, labels, unique_labels):
        replaced_df = data.replace_oos_labels(
            df=df.copy(), labels=labels, label_col="tag", oos_label="other"
        )
        assert set(replaced_df.tag.unique()) == set(unique_labels)
    ```

    Note that we don't use the `df` fixture directly (we pass in `df.copy()`) inside our parametrized test function. If we did, then we'd be changing `df`'s values after each parametrization.

    !!! tip
        When creating fixtures around datasets, it's best practice to create a simplified version that still adheres to the same schema. For example, in our dataframe fixture above, we're creating a smaller dataframe that still has the same column names as our actual dataframe. While we could have loaded our actual dataset, it can cause issues as our dataset changes over time (new labels, removed labels, very large dataset, etc.)

Fixtures can have different scopes depending on how we want to use them. For example our `df` fixture has the module scope because we don't want to keep recreating it after every test but, instead, we want to create it just once for all the tests in our module (`tests/test_data.py`).

- `function`: fixture is destroyed after every test. `#!js [default]`
- `class`: fixture is destroyed after the last test in the class.
- `module`: fixture is destroyed after the last test in the module (script).
- `package`: fixture is destroyed after the last test in the package.
- `session`: fixture is destroyed after the last test of the session.

Functions are lowest level scope while [sessions](https://docs.pytest.org/en/6.2.x/fixture.html#scope-sharing-fixtures-across-classes-modules-packages-or-session){:target="_blank"} are the highest level. The highest level scoped fixtures are executed first.

> Typically, when we have many fixtures in a particular test file, we can organize them all in a `fixtures.py` script and invoke them as needed.

### Markers

We've been able to execute our tests at various levels of granularity (all tests, script, function, etc.) but we can create custom granularity by using [markers](https://docs.pytest.org/en/stable/mark.html){:target="_blank"}. We've already used one type of marker (parametrize) but there are several other [builtin markers](https://docs.pytest.org/en/stable/mark.html#mark){:target="_blank"} as well. For example, the [`skipif`](https://docs.pytest.org/en/stable/skipping.html#id1){:target="_blank"} marker allows us to skip execution of a test if a condition is met. For example, supposed we only wanted to test training our model if a GPU is available:

```python linenums="1"
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Full training tests require a GPU."
)
def test_training():
    pass
```

We can also create our own custom markers with the exception of a few [reserved](https://docs.pytest.org/en/stable/reference.html#marks){:target="_blank"} marker names.

```python linenums="1"
@pytest.mark.fruits
def test_fruit(my_fruit):
    assert my_fruit.name == "apple"
```

We can execute them by using the `-m` flag which requires a (case-sensitive) marker expression like below:

```bash
pytest -m "fruits"      #  runs all tests marked with `fruits`
pytest -m "not fruits"  #  runs all tests besides those marked with `fruits`
```

!!! tip
    The proper way to use markers is to explicitly list the ones we've created in our [pyproject.toml](https://github.com/GokuMohandas/mlops-course/blob/main/pyproject.toml){:target="_blank"} file. Here we can specify that all markers must be defined in this file with the `--strict-markers` flag and then declare our markers (with some info about them) in our `markers` list:

    ```python linenums="1"
    @pytest.mark.training
    def test_train_model():
        assert ...
    ```

    ```toml linenums="1" hl_lines="6-8"
    # Pytest
    [tool.pytest.ini_options]
    testpaths = ["tests"]
    python_files = "test_*.py"
    addopts = "--strict-markers --disable-pytest-warnings"
    markers = [
        "training: tests that involve training",
    ]
    ```
    Once we do this, we can view all of our existing list of markers by executing `#!bash pytest --markers` and we'll receive an error when we're trying to use a new marker that's not defined here.

### Coverage

As we're developing tests for our application's components, it's important to know how well we're covering our code base and to know if we've missed anything. We can use the [Coverage](https://coverage.readthedocs.io/){:target="_blank"} library to track and visualize how much of our codebase our tests account for. With pytest, it's even easier to use this package thanks to the [pytest-cov](https://pytest-cov.readthedocs.io/){:target="_blank"} plugin.

```bash
pip install pytest-cov==2.10.1
```

And we'll add this to our `setup.py` script:

```python linenums="1"
# setup.py
test_packages = [
    "pytest==7.1.2",
    "pytest-cov==2.10.1"
]
```

```bash
python3 -m pytest --cov tagifai --cov-report html
```

<div class="ai-center-all">
    <img width="600" src="/static/images/mlops/testing/pytest.png" alt="pytest">
</div>

Here we're asking for coverage for all the code in our tagifai and app directories and to generate the report in HTML format. When we run this, we'll see the tests from our tests directory executing while the coverage plugin is keeping tracking of which lines in our application are being executed. Once our tests are complete, we can view the generated report (default is `htmlcov/index.html`) and click on individual files to see which parts were not covered by any tests. This is especially useful when we forget to test for certain conditions, exceptions, etc.

<div class="ai-center-all">
    <img width="500" src="/static/images/mlops/testing/coverage.png" alt="test coverage">
</div>

!!! warning
    Though we have 100% coverage, this does not mean that our application is perfect. Coverage only indicates that a piece of code executed in a test, not necessarily that every part of it was tested, let alone thoroughly tested. Therefore, coverage should **never** be used as a representation of correctness. However, it is very useful to maintain coverage at 100% so we can know when new functionality has yet to be tested. In our CI/CD lesson, we'll see how to use GitHub actions to make 100% coverage a requirement when pushing to specific branches.

### Exclusions

Sometimes it doesn't make sense to write tests to cover every single line in our application yet we still want to account for these lines so we can maintain 100% coverage. We have two levels of purview when applying exclusions:

1. Excusing lines by adding this comment `# pragma: no cover, <MESSAGE>`
```python linenums="1"
if trial:  # pragma: no cover, optuna pruning
    trial.report(val_loss, epoch)
    if trial.should_prune():
        raise optuna.TrialPruned()
```

2. Excluding files by specifying them in our `pyproject.toml` configuration:

```toml linenums="1"
# Pytest coverage
[tool.coverage.run]
omit = ["app/gunicorn.py"]
```

> The main point is that we were able to add justification to these exclusions through comments so our team can follow our reasoning.

Now that we have a foundation for testing traditional software, let's dive into testing our data and models in the context of machine learning systems.

## 🔢&nbsp; Data

So far, we've used unit and integration tests to test the functions that interact with our data but we haven't tested the validity of the data itself. We're going to use the [great expectations](https://github.com/great-expectations/great_expectations){:target="_blank"} library to test what our data is expected to look like. It's a library that allows us to create expectations as to what our data should look like in a standardized way. It also provides modules to seamlessly connect with backend data sources such as local file systems, S3, databases, etc. Let's explore the library by implementing the expectations we'll need for our application.

> 👉 &nbsp; Follow along interactive notebook in the :fontawesome-brands-github:{ .github } [**testing-ml**](https://github.com/GokuMohandas/testing-ml){:target="_blank"} repository as we implement the concepts below.

```bash
pip install great-expectations==0.15.15
```

And we'll add this to our `setup.py` script:

```python linenums="1"
# setup.py
test_packages = [
    "pytest==7.1.2",
    "pytest-cov==2.10.1",
    "great-expectations==0.15.15"
]
```

First we'll load the data we'd like to apply our expectations on. We can load our data from a variety of [sources](https://docs.greatexpectations.io/docs/guides/connecting_to_your_data/connect_to_data_overview){:target="_blank"} (filesystem, database, cloud etc.) which we can then wrap around a [Dataset module](https://legacy.docs.greatexpectations.io/en/latest/autoapi/great_expectations/dataset/index.html){:target="_blank"} (Pandas / Spark DataFrame, SQLAlchemy).

```python linenums="1"
import great_expectations as ge
import json
import pandas as pd
from urllib.request import urlopen
```

```python linenums="1"
# Load labeled projects
projects = pd.read_csv("https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/projects.csv")
tags = pd.read_csv("https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/tags.csv")
df = ge.dataset.PandasDataset(pd.merge(projects, tags, on="id"))
print (f"{len(df)} projects")
df.head(5)
```
<pre class="output">
<div class="output_subarea output_html rendered_html output_result" dir="auto"><div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>created_on</th>
      <th>title</th>
      <th>description</th>
      <th>tag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>2020-02-20 06:43:18</td>
      <td>Comparison between YOLO and RCNN on real world...</td>
      <td>Bringing theory to experiment is cool. We can ...</td>
      <td>computer-vision</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>2020-02-20 06:47:21</td>
      <td>Show, Infer &amp; Tell: Contextual Inference for C...</td>
      <td>The beauty of the work lies in the way it arch...</td>
      <td>computer-vision</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>2020-02-24 16:24:45</td>
      <td>Awesome Graph Classification</td>
      <td>A collection of important graph embedding, cla...</td>
      <td>graph-learning</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15</td>
      <td>2020-02-28 23:55:26</td>
      <td>Awesome Monte Carlo Tree Search</td>
      <td>A curated list of Monte Carlo tree search papers...</td>
      <td>reinforcement-learning</td>
    </tr>
    <tr>
      <th>4</th>
      <td>19</td>
      <td>2020-03-03 13:54:31</td>
      <td>Diffusion to Vector</td>
      <td>Reference implementation of Diffusion2Vec (Com...</td>
      <td>graph-learning</td>
    </tr>
  </tbody>
</table>
</div></div>
</pre>

### Expectations

When it comes to creating expectations as to what our data should look like, we want to think about our entire dataset and all the features (columns) within it.

```python
# Presence of specific features
df.expect_table_columns_to_match_ordered_list(
    column_list=["id", "created_on", "title", "description", "tag"]
)
```

```python
# Unique combinations of features (detect data leaks!)
df.expect_compound_columns_to_be_unique(column_list=["title", "description"])
```

```python
# Missing values
df.expect_column_values_to_not_be_null(column="tag")
```

```python
# Unique values
df.expect_column_values_to_be_unique(column="id")
```

```python
# Type adherence
df.expect_column_values_to_be_of_type(column="title", type_="str")
```

```python
# List (categorical) / range (continuous) of allowed values
tags = ["computer-vision", "graph-learning", "reinforcement-learning",
        "natural-language-processing", "mlops", "time-series"]
df.expect_column_values_to_be_in_set(column="tag", value_set=tags)
```

Each of these expectations will create an output with details about success or failure, expected and observed values, expectations raised, etc. For example, the expectation ```#!python df.expect_column_values_to_be_of_type(column="title", type_="str")``` would produce the following if successful:

```json linenums="1" hl_lines="7"
{
  "exception_info": {
    "raised_exception": false,
    "exception_traceback": null,
    "exception_message": null
  },
  "success": true,
  "meta": {},
  "expectation_config": {
    "kwargs": {
      "column": "title",
      "type_": "str",
      "result_format": "BASIC"
    },
    "meta": {},
    "expectation_type": "_expect_column_values_to_be_of_type__map"
  },
  "result": {
    "element_count": 955,
    "missing_count": 0,
    "missing_percent": 0.0,
    "unexpected_count": 0,
    "unexpected_percent": 0.0,
    "unexpected_percent_nonmissing": 0.0,
    "partial_unexpected_list": []
  }
}
```

and if we have a failed expectation (ex. ```#!python  df.expect_column_values_to_be_of_type(column="title", type_="int")```), we'd receive this output(notice the counts and examples for what caused the failure):
```json linenums="1" hl_lines="2 24"
{
  "success": false,
  "exception_info": {
    "raised_exception": false,
    "exception_traceback": null,
    "exception_message": null
  },
  "expectation_config": {
    "meta": {},
    "kwargs": {
      "column": "title",
      "type_": "int",
      "result_format": "BASIC"
    },
    "expectation_type": "_expect_column_values_to_be_of_type__map"
  },
  "result": {
    "element_count": 955,
    "missing_count": 0,
    "missing_percent": 0.0,
    "unexpected_count": 955,
    "unexpected_percent": 100.0,
    "unexpected_percent_nonmissing": 100.0,
    "partial_unexpected_list": [
      "How to Deal with Files in Google Colab: What You Need to Know",
      "Machine Learning Methods Explained (+ Examples)",
      "OpenMMLab Computer Vision",
      "...",
    ]
  },
  "meta": {}
}
```

There are just a few of the different expectations that we can create. Be sure to explore all the [expectations](https://greatexpectations.io/expectations/), including [custom expectations](https://docs.greatexpectations.io/docs/guides/expectations/creating_custom_expectations/overview/). Here are some other popular expectations that don't pertain to our specific dataset but are widely applicable:

- feature value relationships with other feature values → `expect_column_pair_values_a_to_be_greater_than_b`
- row count (exact or range) of samples → `expect_table_row_count_to_be_between`
- value statistics (mean, std, median, max, min, sum, etc.) → `expect_column_mean_to_be_between`

### Organization

When it comes to organizing expectations, it's recommended to start with table-level ones and then move on to individual feature columns.

#### Table expectations
```python linenums="1"
# columns
df.expect_table_columns_to_match_ordered_list(
    column_list=["id", "created_on", "title", "description", "tag"])

# data leak
df.expect_compound_columns_to_be_unique(column_list=["title", "description"])
```

#### Column expectations
```python linenums="1"
# id
df.expect_column_values_to_be_unique(column="id")

# created_on
df.expect_column_values_to_not_be_null(column="created_on")
df.expect_column_values_to_match_strftime_format(
    column="created_on", strftime_format="%Y-%m-%d %H:%M:%S")

# title
df.expect_column_values_to_not_be_null(column="title")
df.expect_column_values_to_be_of_type(column="title", type_="str")

# description
df.expect_column_values_to_not_be_null(column="description")
df.expect_column_values_to_be_of_type(column="description", type_="str")

# tag
df.expect_column_values_to_not_be_null(column="tag")
df.expect_column_values_to_be_of_type(column="tag", type_="str")
```

We can group all the expectations together to create an [Expectation Suite](https://docs.greatexpectations.io/en/latest/reference/core_concepts/expectations/expectations.html#expectation-suites) object which we can use to validate any Dataset module.

```python linenums="1"
# Expectation suite
expectation_suite = df.get_expectation_suite(discard_failed_expectations=False)
print(df.validate(expectation_suite=expectation_suite, only_return_failures=True))
```
```json linenums="1"
{
  "success": true,
  "results": [],
  "statistics": {
    "evaluated_expectations": 11,
    "successful_expectations": 11,
    "unsuccessful_expectations": 0,
    "success_percent": 100.0
  },
  "evaluation_parameters": {}
}
```

### Projects
So far we've worked with the Great Expectations library at the adhoc script / notebook level but we can further organize our expectations by creating a Project.

```bash
cd tests
great_expectations init
```
This will set up a `tests/great_expectations` directory with the following structure:
```bash
tests/great_expectations/
├── checkpoints/
├── expectations/
├── plugins/
├── uncommitted/
├── .gitignore
└── great_expectations.yml
```

#### Data source

The first step is to establish our `datasource` which tells Great Expectations where our data lives:

```bash
great_expectations datasource new
```
```bash
What data would you like Great Expectations to connect to?
    1. Files on a filesystem (for processing with Pandas or Spark) 👈
    2. Relational database (SQL)
```
```bash
What are you processing your files with?
1. Pandas 👈
2. PySpark
```
```bash
Enter the path of the root directory where the data files are stored: ../data
```

Run the cells in the generated notebook and change the `datasource_name` to `local_data`. After we run the cells, we can close the notebook (and end the process on the terminal with ++ctrl++ + c) and we can see the Datasource being added to `great_expectations.yml`.


#### Suites
Create expectations manually, interactively or automatically and save them as suites (a set of expectations for a particular data asset).
```bash
great_expectations suite new
```
```bash
How would you like to create your Expectation Suite?
    1. Manually, without interacting with a sample batch of data (default)
    2. Interactively, with a sample batch of data 👈
    3. Automatically, using a profiler
```
```bash
Which data asset (accessible by data connector "default_inferred_data_connector_name") would you like to use?
    1. labeled_projects.csv
    2. projects.csv 👈
    3. tags.csv
```
```bash
Name the new Expectation Suite [projects.csv.warning]: projects
```
This will open up an interactive notebook where we can add expectations. Copy and paste the expectations below and run all the cells. Repeat this step for `tags.csv` and `labeled_projects.csv`.

<div class="ai-center-all">
    <img width="700" src="/static/images/mlops/testing/suite.png" alt="great expectations suite">
</div>

??? quote "Expectations for `projects.csv`"
    Table expectations
    ```python linenums="1"
    # Presence of features
    validator.expect_table_columns_to_match_ordered_list(
        column_list=["id", "created_on", "title", "description"])
    validator.expect_compound_columns_to_be_unique(column_list=["title", "description"])  # data leak
    ```

    Column expectations:
    ```python linenums="1"
    # id
    validator.expect_column_values_to_be_unique(column="id")

    # create_on
    validator.expect_column_values_to_not_be_null(column="created_on")
    validator.expect_column_values_to_match_strftime_format(
        column="created_on", strftime_format="%Y-%m-%d %H:%M:%S")

    # title
    validator.expect_column_values_to_not_be_null(column="title")
    validator.expect_column_values_to_be_of_type(column="title", type_="str")

    # description
    validator.expect_column_values_to_not_be_null(column="description")
    validator.expect_column_values_to_be_of_type(column="description", type_="str")
    ```
??? quote "Expectations for `tags.csv`"
    Table expectations
    ```python linenums="1"
    # Presence of features
    validator.expect_table_columns_to_match_ordered_list(column_list=["id", "tag"])
    ```

    Column expectations:
    ```python linenums="1"
    # id
    validator.expect_column_values_to_be_unique(column="id")

    # tag
    validator.expect_column_values_to_not_be_null(column="tag")
    validator.expect_column_values_to_be_of_type(column="tag", type_="str")
    ```
??? quote "Expectations for `labeled_projects.csv`"
    Table expectations
    ```python linenums="1"
    # Presence of features
    validator.expect_table_columns_to_match_ordered_list(
        column_list=["id", "created_on", "title", "description", "tag"])
    validator.expect_compound_columns_to_be_unique(column_list=["title", "description"])  # data leak
    ```

    Column expectations:
    ```python linenums="1"
    # id
    validator.expect_column_values_to_be_unique(column="id")

    # create_on
    validator.expect_column_values_to_not_be_null(column="created_on")
    validator.expect_column_values_to_match_strftime_format(
        column="created_on", strftime_format="%Y-%m-%d %H:%M:%S")

    # title
    validator.expect_column_values_to_not_be_null(column="title")
    validator.expect_column_values_to_be_of_type(column="title", type_="str")

    # description
    validator.expect_column_values_to_not_be_null(column="description")
    validator.expect_column_values_to_be_of_type(column="description", type_="str")

    # tag
    validator.expect_column_values_to_not_be_null(column="tag")
    validator.expect_column_values_to_be_of_type(column="tag", type_="str")
    ```

All of these expectations have been saved under `great_expectations/expectations`:

```bash
great_expectations/
├── expectations/
│   ├── labeled_projects.csv
│   ├── projects.csv
│   └── tags.csv
```

And we can also list the suites with:
```bash
great_expectations suite list
```
<pre class="output">
Using v3 (Batch Request) API
3 Expectation Suites found:
 - labeled_projects
 - projects
 - tags
</pre>

To edit a suite, we can execute the follow CLI command:
```bash
great_expectations suite edit <SUITE_NAME>
```

#### Checkpoints

Create Checkpoints where a Suite of Expectations are applied to a specific data asset. This is a great way of programmatically applying checkpoints on our existing and new data sources.
```bash
cd tests
great_expectations checkpoint new CHECKPOINT_NAME
```
So for our project, it would be:
```
great_expectations checkpoint new projects
great_expectations checkpoint new tags
great_expectations checkpoint new labeled_projects
```
Each of these checkpoint creation calls will launch a notebook where we can define which suites to apply this checkpoint to. We have to change the lines for `data_asset_name` (which data asset to run the checkpoint suite on) and `expectation_suite_name` (name of the suite to use). For example, the `projects` checkpoint would use the `projects.csv` data asset and the `projects` suite.

> Checkpoints can share the same suite, as long the schema and validations are applicable.

```python linenums="1" hl_lines="12 15"
my_checkpoint_name = "projects"  # This was populated from your CLI command.

yaml_config = f"""
name: {my_checkpoint_name}
config_version: 1.0
class_name: SimpleCheckpoint
run_name_template: "%Y%m%d-%H%M%S-my-run-name-template"
validations:
  - batch_request:
      datasource_name: local_data
      data_connector_name: default_inferred_data_connector_name
      data_asset_name: projects.csv
      data_connector_query:
        index: -1
    expectation_suite_name: projects
"""
print(yaml_config)
```

!!! warning "Validate autofills"
    Be sure to ensure that the `datasource_name`, `data_asset_name` and `expectation_suite_name` are all what we want them to be (Great Expectations autofills those with assumptions which may not always be accurate).

Repeat these same steps for the `tags` and `labeled_projects` checkpoints and then we're ready to execute them:
```bash
great_expectations checkpoint run projects
great_expectations checkpoint run tags
great_expectations checkpoint run labeled_projects
```

<div class="ai-center-all">
    <img width="650" src="/static/images/mlops/testing/checkpoint.png" alt="great expectations checkpoint">
</div>

At the end of this lesson, we'll create a target in our `Makefile` that run all these tests (code, data and models) and we'll automate their execution in our [pre-commit lesson](pre-commit.md){:target="_blank"}.

!!! note
    We've applied expectations on our source dataset but there are many other key areas to test the data as well. For example, the intermediate outputs from processes such as cleaning, augmentation, splitting, preprocessing, tokenization, etc.

### Documentation

When we create expectations using the CLI application, Great Expectations automatically generates documentation for our tests. It also stores information about validation runs and their results. We can launch the generate data documentation with the following command: ```#!bash great_expectations docs build```

<div class="ai-center-all">
    <img width="700" src="/static/images/mlops/testing/docs.png" alt="data documentation">
</div>

> By default, Great Expectations stores our expectations, results and metrics locally but for production, we'll want to set up remote [metadata stores](https://docs.greatexpectations.io/docs/guides/setup/#metadata-stores){:target="_blank"}.

### Production

The advantage of using a library such as great expectations, as opposed to isolated assert statements is that we can:

- reduce redundant efforts for creating tests across data modalities
- automatically create testing [checkpoints](https://madewithml.com/courses/mlops/testing#checkpoints){:target="_blank} to execute as our dataset grows
- automatically generate [documentation](https://madewithml.com/courses/mlops/testing#documentation){:target="_blank} on expectations and report on runs
- easily connect with backend data sources such as local file systems, S3, databases, etc.

Many of these expectations will be executed when the data is extracted, loaded and transformed during our [DataOps workflows](orchestration.md#dataops){:target="_blank"}. Typically, the data will be extracted from a source ([database](data-stack.md#database){:target="_blank"}, [API](api.md){:target="_blank"}, etc.) and loaded into a data system (ex. [data warehouse](data-stack.md#data-warehouse){:target="_blank"}) before being transformed there (ex. using [dbt](https://www.getdbt.com/){:target="_blank"}) for downstream applications. Throughout these tasks, Great Expectations checkpoint validations can be run to ensure the validity of the data and the changes applied to it. We'll see a simplified version of when data validation should occur in our data workflows in the [orchestration lesson](orchestration.md#dataops){:target="_blank"}.

<div class="ai-center-all mb-4">
    <img width="650" src="/static/images/mlops/testing/production.png" alt="ELT pipelines in production">
</div>

> Learn more about different data systems in our [data stack lesson](data-stack.md){:target="_blank"} if you're not familiar with them.

## 🤖&nbsp; Models

The final aspect of testing ML systems involves testing our models during training, evaluation, inference and deployment.

### Training

We want to write tests iteratively while we're developing our training pipelines so we can catch errors quickly. This is especially important because, unlike traditional software, ML systems can run to completion without throwing any exceptions / errors but can produce incorrect systems. We also want to catch errors quickly to save on time and compute.

- Check shapes and values of model output
```python linenums="1"
assert model(inputs).shape == torch.Size([len(inputs), num_classes])
```
- Check for decreasing loss after one batch of training
```python linenums="1"
assert epoch_loss < prev_epoch_loss
```
- Overfit on a batch
```python linenums="1"
accuracy = train(model, inputs=batches[0])
assert accuracy == pytest.approx(0.95, abs=0.05) # 0.95 ± 0.05
```
- Train to completion (tests early stopping, saving, etc.)
```python linenums="1"
train(model)
assert learning_rate >= min_learning_rate
assert artifacts
```
- On different devices
```python linenums="1"
assert train(model, device=torch.device("cpu"))
assert train(model, device=torch.device("cuda"))
```

!!! note
    You can mark the compute intensive tests with a pytest marker and only execute them when there is a change being made to system affecting the model.
    ```python linenums="1"
    @pytest.mark.training
    def test_train_model():
        ...
    ```

### Behavioral testing

Behavioral testing is the process of testing input data and expected outputs while treating the model as a black box (model agnostic evaluation). They don't necessarily have to be adversarial in nature but more along the types of perturbations we may expect to see in the real world once our model is deployed. A landmark paper on this topic is [Beyond Accuracy: Behavioral Testing of NLP Models with CheckList](https://arxiv.org/abs/2005.04118){:target="_blank"} which breaks down behavioral testing into three types of tests:

- `#!js invariance`: Changes should not affect outputs.
```python linenums="1"
# INVariance via verb injection (changes should not affect outputs)
tokens = ["revolutionized", "disrupted"]
texts = [f"Transformers applied to NLP have {token} the ML field." for token in tokens]
predict.predict(texts=texts, artifacts=artifacts)
```
<pre class="output">
['natural-language-processing', 'natural-language-processing']
</pre>
- `#!js directional`: Change should affect outputs.
```python linenums="1"
# DIRectional expectations (changes with known outputs)
tokens = ["text classification", "image classification"]
texts = [f"ML applied to {token}." for token in tokens]
predict.predict(texts=texts, artifacts=artifacts)
```
<pre class="output">
['natural-language-processing', 'computer-vision']
</pre>
- `#!js minimum functionality`: Simple combination of inputs and expected outputs.
```python linenums="1"
# Minimum Functionality Tests (simple input/output pairs)
tokens = ["natural language processing", "mlops"]
texts = [f"{token} is the next big wave in machine learning." for token in tokens]
predict.predict(texts=texts, artifacts=artifacts)
```
<pre class="output">
['natural-language-processing', 'mlops']
</pre>

!!! tip "Adversarial testing"
    Each of these types of tests can also include adversarial tests such as testing with common biased tokens or noisy tokens, etc.

    ```python linenums="1"
    texts = [
        "CNNs for text classification.",  # CNNs are typically seen in computer-vision projects
        "This should not produce any relevant topics."  # should predict `other` label
    ]
    predict.predict(texts=texts, artifacts=artifacts)
    ```
    <pre class="output">
        ['natural-language-processing', 'other']
    </pre>


And we can convert these tests into systematic parameterized tests:

```bash
mkdir tests/model
touch tests/model/test_behavioral.py
```

```python linenums="1"
# tests/model/test_behavioral.py
from pathlib import Path
import pytest
from config import config
from tagifai import main, predict

@pytest.fixture(scope="module")
def artifacts():
    run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
    artifacts = main.load_artifacts(run_id=run_id)
    return artifacts

@pytest.mark.parametrize(
    "text_a, text_b, tag",
    [
        (
            "Transformers applied to NLP have revolutionized machine learning.",
            "Transformers applied to NLP have disrupted machine learning.",
            "natural-language-processing",
        ),
    ],
)
def test_inv(text_a, text_b, tag, artifacts):
    """INVariance via verb injection (changes should not affect outputs)."""
    tag_a = predict.predict(texts=[text_a], artifacts=artifacts)[0]["predicted_tag"]
    tag_b = predict.predict(texts=[text_b], artifacts=artifacts)[0]["predicted_tag"]
    assert tag_a == tag_b == tag
```

??? quote "View `tests/model/test_behavioral.py`"
    ```python linenums="1"
    from pathlib import Path

    import pytest

    from config import config
    from tagifai import main, predict


    @pytest.fixture(scope="module")
    def artifacts():
        run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
        artifacts = main.load_artifacts(run_id=run_id)
        return artifacts


    @pytest.mark.parametrize(
        "text, tag",
        [
            (
                "Transformers applied to NLP have revolutionized machine learning.",
                "natural-language-processing",
            ),
            (
                "Transformers applied to NLP have disrupted machine learning.",
                "natural-language-processing",
            ),
        ],
    )
    def test_inv(text, tag, artifacts):
        """INVariance via verb injection (changes should not affect outputs)."""
        predicted_tag = predict.predict(texts=[text], artifacts=artifacts)[0]["predicted_tag"]
        assert tag == predicted_tag


    @pytest.mark.parametrize(
        "text, tag",
        [
            (
                "ML applied to text classification.",
                "natural-language-processing",
            ),
            (
                "ML applied to image classification.",
                "computer-vision",
            ),
            (
                "CNNs for text classification.",
                "natural-language-processing",
            )
        ],
    )
    def test_dir(text, tag, artifacts):
        """DIRectional expectations (changes with known outputs)."""
        predicted_tag = predict.predict(texts=[text], artifacts=artifacts)[0]["predicted_tag"]
        assert tag == predicted_tag


    @pytest.mark.parametrize(
        "text, tag",
        [
            (
                "Natural language processing is the next big wave in machine learning.",
                "natural-language-processing",
            ),
            (
                "MLOps is the next big wave in machine learning.",
                "mlops",
            ),
            (
                "This should not produce any relevant topics.",
                "other",
            ),
        ],
    )
    def test_mft(text, tag, artifacts):
        """Minimum Functionality Tests (simple input/output pairs)."""
        predicted_tag = predict.predict(texts=[text], artifacts=artifacts)[0]["predicted_tag"]
        assert tag == predicted_tag
    ```

### Inference

When our model is deployed, most users will be using it for inference (directly / indirectly), so it's very important that we test all aspects of it.

#### Loading artifacts
This is the first time we're not loading our components from in-memory so we want to ensure that the required artifacts (model weights, encoders, config, etc.) are all able to be loaded.

```python linenums="1"
artifacts = main.load_artifacts(run_id=run_id)
assert isinstance(artifacts["label_encoder"], data.LabelEncoder)
...
```

#### Prediction
Once we have our artifacts loaded, we're readying to test our prediction pipelines. We should test samples with just one input, as well as a batch of inputs (ex. padding can have unintended consequences sometimes).
```python linenums="1"
# test our API call directly
data = {
    "texts": [
        {"text": "Transfer learning with transformers for text classification."},
        {"text": "Generative adversarial networks in both PyTorch and TensorFlow."},
    ]
}
response = client.post("/predict", json=data)
assert response.status_code == HTTPStatus.OK
assert response.request.method == "POST"
assert len(response.json()["data"]["predictions"]) == len(data["texts"])
...
```

## Makefile
Let's create a target in our `Makefile` that will allow us to execute all of our tests with one call:
```bash
# Test
.PHONY: test
test:
	pytest -m "not training"
	cd tests && great_expectations checkpoint run projects
	cd tests && great_expectations checkpoint run tags
	cd tests && great_expectations checkpoint run labeled_projects
```

```bash
make test
```

## Testing vs. monitoring

We'll conclude by talking about the similarities and distinctions between testing and [monitoring](monitoring.md){:target="_blank"}. They're both integral parts of the ML development pipeline and depend on each other for iteration. Testing is assuring that our system (code, data and models) passes the expectations that we've established offline. Whereas, monitoring involves that these expectations continue to pass online on live production data while also ensuring that their data distributions are [comparable](monitoring.md#measuring-drift){:target="_blank"} to the reference window (typically subset of training data) through $t_n$. When these conditions no longer hold true, we need to inspect more closely (retraining may not always fix our root problem).

With [monitoring](monitoring.md){:target="_blank"}, there are quite a few distinct concerns that we didn't have to consider during testing since it involves (live) data we have yet to see.

- features and prediction distributions (drift), typing, schema mismatches, etc.
- determining model performance (rolling and window metrics on overall and slices of data) using indirect signals (since labels may not be readily available).
- in situations with large data, we need to know which data points to label and upsample for training.
- identifying anomalies and outliers.

> We'll cover all of these concepts in much more depth (and code) in our [monitoring](monitoring.md){:target="_blank"} lesson.

## Resources
- [Great Expectations](https://github.com/great-expectations/great_expectations){:target="_blank"}
- [The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/aad9f93b86b7addfea4c419b9100c6cdd26cacea.pdf){:target="_blank"}
- [Beyond Accuracy: Behavioral Testing of NLP Models with CheckList](https://arxiv.org/abs/2005.04118){:target="_blank"}
- [Robustness Gym: Unifying the NLP Evaluation Landscape](https://arxiv.org/abs/2101.04840){:target="_blank"}

<!-- Citation -->
{% include "styles/cite.md" %}