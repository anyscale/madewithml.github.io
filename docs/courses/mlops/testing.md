---
template: lesson.html
title: "Testing ML Systems: Code, Data and Models"
description: Testing code, data and models to ensure consistent behavior in ML systems.
keywords: testing, pytest, unit test, parametrize, fixtures, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning, great expectations
image: https://madewithml.com/static/images/mlops.png
repository: https://github.com/GokuMohandas/follow/tree/testing
---

{% include "styles/lesson.md" %}

## Intuition

Tests are a way for us to ensure that something works as intended. We're incentivized to implement tests and discover sources of error as early in the development cycle as possible so that we can reduce [increasing downstream costs](https://assets.deepsource.io/39ed384/images/blog/cost-of-fixing-bugs/chart.jpg){:target="_blank"} and wasted time. Once we've designed our tests, we can automatically execute them every time we implement a change to our system and continue to build on them over time.

### Types of tests

There are many four majors types of tests which are utilized at different points in the development cycle:

1. `#!js Unit tests`: tests on individual components that each have a [single responsibility](https://en.wikipedia.org/wiki/Single-responsibility_principle){:target="_blank"} (ex. function that filters a list).
2. `#!js Integration tests`: tests on the combined functionality of individual components (ex. data processing).
3. `#!js System tests`: tests on the design of a system for expected outputs given inputs (ex. training, inference, etc.).
4. `#!js Acceptance tests`: tests to verify that requirements have been met, usually referred to as User Acceptance Testing (UAT).
5. `#!js Regression tests`: testing errors we've seen before to ensure new changes don't reintroduce them.

!!! warning
    These types of test are not specific to machine learning. However, while ML systems are probabilistic in nature, they are composed of many deterministic components that can be tested in a similar manner as traditional software systems. The distinction between testing ML systems begins when we move from testing code to testing the [data](testing.md#data){:target="_blank"} and [models](testing.md#models){:target="_blank"}.

<div class="ai-center-all">
    <img width="700" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/testing/tests.png">
</div>

> There are many other types of functional and non-functional tests as well, such as smoke tests (quick health checks), performance tests (load, stress), security tests, etc. but we can generalize these under the system tests above.

### How should we test?

The framework to use when composing tests is the [Arrange Act Assert](http://wiki.c2.com/?ArrangeActAssert){:target="_blank"} methodology.

- `#!js Arrange`: set up the different inputs to test on.
- `#!js Act`: apply the inputs on the component we want to test.
- `#!js Assert`: confirm that we received the expected output.

> `#!js Cleaning` is an unofficial fourth step to this methodology because it's important to not leave remnants of a previous state which may affect subsequent tests. We can use packages such as [pytest-randomly](https://github.com/pytest-dev/pytest-randomly){:target="_blank"} to test against state dependency by executing tests randomly.

In Python, there are many tools, such as [unittest](https://docs.python.org/3/library/unittest.html){:target="_blank"}, [pytest](https://docs.pytest.org/en/stable/){:target="_blank"}, etc., that allow us to easily implement our tests while adhering to the *Arrange Act Assert* framework above. These tools come with powerful built-in functionality such as parametrization, filters, and more, to test many conditions at scale.

!!! question "What should we be testing for?"
    When *arranging* our inputs and *asserting* our expected outputs, what are some aspects of our inputs and outputs that we should be testing for?

    ??? quote "Show answer"

        - **inputs**: data types, format, length, edge cases (min/max, small/large, etc.)
        - **outputs**: data types, formats, exceptions, intermediary and final outputs

## Best practices
Regardless of the framework we use, it's important to strongly tie testing into the development process.

- `#!js atomic`: when creating unit components, we need to ensure that they have a [single responsibility](https://en.wikipedia.org/wiki/Single-responsibility_principle){:target="_blank"} so that we can easily test them. If not, we'll need to split them into more granular units.
- `#!js compose`: when we create new components, we want to compose tests to validate their functionality. It's a great way to ensure reliability and catch errors early on.
- `#!js regression`: we want to account for new errors we come across with a regression test so we can ensure we don't reintroduce the same errors in the future.
- `#!js coverage`: we want to ensure that 100% of our codebase has been accounter for. This doesn't mean writing a test for every single line of code but rather accounting for every single line (more on this in the [coverage section](#coverage) below).
- `#!js automate`: in the event we forget to run our tests before committing to a repository, we want to auto run tests for every commit. We'll learn how to do this locally using [pre-commit hooks](git.md#pre-commit){:target="_blank"} and remotely (ie. main branch) via GitHub actions in subsequent lessons.

## Test-driven development

[Test-driven development (TDD)](https://en.wikipedia.org/wiki/Test-driven_development){:target="_blank"} is the process where you write a test before completely writing the functionality to ensure that tests are always written. This is in contrast to writing functionality first and then composing tests afterwards. Here are my thoughts on this:

- good to write tests as we progress, but it's not the representation of correctness.
- initial time should be spent on design before ever getting into the code or tests.
- using a test as guide doesn't mean that our functionality is error free.

Perfect coverage doesn't mean that our application is error free if those tests aren't meaningful and don't encompass the field of possible inputs, intermediates and outputs. Therefore, we should work towards better design and agility when facing errors, quickly resolving them and writing test cases around them to avoid them next time.

> This topic is still highly debated and I'm only reflecting on [my experience](https://linkedin.com/in/goku){:target="_blank"} and what's worked well for my teams. What's most important is that the team is producing reliable systems that can be tested and improved upon.

## Application

In our [application](https://github.com/GokuMohandas/MLOps){:target="_blank"}, we'll be testing the code, data and models. Be sure to look inside each of the different testing scripts after reading through the components below.

```bash linenums="1"
tests/
├── app/
|   └── test_api.py
├── config/
|   └── test_config.py
├── great_expectations/
|   ├── expectations/
|   |   ├── projects.json
|   |   └── tags.json
|   ├── ...
├── model/
|   └── test_behavioral.py
└── tagifai/
|   ├── test_data.py
|   ├── test_eval.py
|   ├── test_main.py
|   ├── test_models.py
|   ├── test_train.py
|   └── test_utils.py
```

> Alternatively, we could've organized our tests by types of tests as well (unit, integration, etc.) but I find it more intuitive for navigation by organizing by how our application is set up. But regardless of how we organize our tests, we can use [markers](#markers) to  allow us to run any subset of tests by specifying filters.

## 🧪&nbsp; Pytest

We're going to be using [pytest](https://docs.pytest.org/en/stable/){:target="_blank"} as our testing framework for it's powerful builtin features such as [parametrization](#parametrize), [fixtures](#fixtures), [markers](#markers), etc.

### Configuration
Pytest expects tests to be organized under a `tests` directory by default. However, we can also use our [pyproject.toml](https://github.com/GokuMohandas/MLOps/blob/main/pyproject.toml){:target="_blank"} file to configure any other test path directories as well. Once in the directory, pytest looks for python scripts starting with `tests_*.py` but we can configure it to read any other file patterns as well.

```toml linenums="1"
# Pytest
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
```

### Assertions

Let's see what a sample test and it's results look like. Assume we have a simple function that determines whether a fruit is crisp or not (notice: [single responsibility](https://en.wikipedia.org/wiki/Single-responsibility_principle){:target="_blank"}):

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

To test this function, we can use [assert statements](https://docs.pytest.org/en/stable/assert.html){:target="_blank"} to map inputs with expected outputs:

```python linenums="1"
# tests/food/test_fruits.py
def test_is_crisp():
    assert is_crisp(fruit="apple") #  or == True
    assert is_crisp(fruit="Apple")
    assert not is_crisp(fruit="orange")
    with pytest.raises(ValueError):
        is_crisp(fruit=None)
        is_crisp(fruit="pear")
```

> We can also have assertions about [exceptions](https://docs.pytest.org/en/stable/assert.html#assertions-about-expected-exceptions){:target="_blank"} like we do in lines 6-8 where all the operations under the with statement are expected to raise the specified exception.

### Execution

We can execute our tests above using several different levels of granularity:

```bash linenums="1"
pytest                                           # all tests
pytest tests/food                                # tests under a directory
pytest tests/food/test_fruits.py                 # tests for a single file
pytest tests/food/test_fruits.py::test_is_crisp  # tests for a single function
```

Running our specific test above would produce the following output:
<pre class="output">
tests/food/test_fruits.py::test_is_crisp <span style="color: #5A9C4B; font-weight: 600;">PASSED</span>      [100%]
</pre>
Had any of our assertions in this test failed, we would see the failed assertions as well as the expected output and the output we received from our function.

> It's important to test for the variety of inputs and expected outputs that we outlined [above](#how-should-we-test) and to **never assume that a test is trivial**. In our example above, it's important that we test for both "apple" and "Apple" in the event that our function didn't account for casing!

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

```bash linenums="1"
tests/food/test_fruits.py::TestFruit .  [100%]
```

> We use test classes to test all of our class modules such as `LabelEncoder`, `Tokenizer`, `CNN`, etc.

### Interfaces

We can also easily test our CLI and API interfaces using their built-in test clients.

```python linenums="1"
from typer.testing import CliRunner
from tagifai.main import app

# Initialize runner
runner = CliRunner()

def test_cli_command():
    result = runner.invoke(app, <CLI_COMMAND>)
    assert result.exit_code == 0
    assert ...
```

and similarly for the API:

```python linenums="1"
from fastapi.testclient import TestClient

# Initialize client
client = TestClient(app)

def test_api_command():
    response = client.get(<ENDPOINT_PATH>)
    assert response.status_code == HTTPStatus.OK
    assert ...
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
pytest tests/food/test_is_crisp_parametrize.py <span style="color: #5A9C4B; font-weight: 600;">...</span>   [100%]
</pre>

1. `#!js [Line 2]`: define the names of the parameters under the decorator, ex. "fruit, crisp" (note that this is one string).
2. `#!js [Lines 3-7]`: provide a list of combinations of values for the parameters from Step 1.
3. `#!js [Line 9]`: pass in parameter names to the test function.
4. `#!js [Line 10]`: include necessary assert statements which will be executed for each of the combinations in the list from Step 2.


> In our application, we use parametrization to test components that require varied sets of inputs and expected outputs such as preprocessing, filtering, etc.

!!! note
    We could pass in an exception as the expected result as well:
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

### Fixtures

Parametrization allows us to efficiently reduce redundancy inside test functions but what about its inputs? Here, we can use pytest's builtin [fixture](https://docs.pytest.org/en/stable/fixture.html){:target="_blank"}, which is a function that is executed before the test function. This significantly reduces redundancy when multiple test functions want to use the same inputs.

```python linenums="1"
@pytest.fixture
def my_fruit():
    fruit = Fruit(name="apple")
    return fruit


def test_fruit(my_fruit):
    assert my_fruit.name == "apple"
```

We can apply fixtures to classes as well where the fixture function will be invoked when any method in the class is called.

```python linenums="1"
@pytest.mark.usefixtures("my_fruit")
class TestFruit:
    ...
```

We use fixtures to efficiently pass a set of inputs (ex. Pandas DataFrame) to different testing functions that require them (cleaning, splitting, etc.).

```python linenums="1"
@pytest.fixture(scope="module")
def df():
    projects_fp = Path(config.DATA_DIR, "projects.json")
    projects_dict = utils.load_dict(filepath=projects_fp)
    df = pd.DataFrame(projects_dict)
    return df


def test_split(df):
    splits = split_data(df=df)
    ...
```

> Typically, when we have too many fixtures in a particular test file, we can organize them all in a `fixtures.py` script and invoke them as needed.

!!! note
    Fixtures can have different scopes depending on how we want to use them. For example our `df` fixture has the module scope because we don't want to keep recreating it after every test but, instead, we want to create it just once for all the tests in our module ([tests/test_data.py](https://github.com/GokuMohandas/MLOps/blob/main/tests/test_data.py){:target="_blank"}).

    - `function`: fixture is destroyed after every test. `#!js [default]`
    - `class`: fixture is destroyed after the last test in the class.
    - `module`: fixture is destroyed after the last test in the module (script).
    - `package`: fixture is destroyed after the last test in the package.
    - `session`: fixture is destroyed after the last test of the session.

    Functions are lowest level scope while [sessions](https://docs.pytest.org/en/6.2.x/fixture.html#scope-sharing-fixtures-across-classes-modules-packages-or-session){:target="_blank"} are the highest level. The highest level scoped fixtures are executed first.


### Markers

We've been able to execute our tests at various levels of granularity (all tests, script, function, etc.) but we can create custom granularity by using [markers](https://docs.pytest.org/en/stable/mark.html){:target="_blank"}. We've already used one type of marker (parametrize) but there are several other [builtin markers](https://docs.pytest.org/en/stable/mark.html#mark){:target="_blank"} as well. For example, the [`skipif`](https://docs.pytest.org/en/stable/skipping.html#id1){:target="_blank"} marker allows us to skip execution of a test if a condition is met.

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

```bash linenums="1"
pytest -m "fruits"      #  runs all tests marked with `fruits`
pytest -m "not fruits"  #  runs all tests besides those marked with `fruits`
```

The proper way to use markers is to explicitly list the ones we've created in our [pyproject.toml](https://github.com/GokuMohandas/MLOps/blob/main/pyproject.toml){:target="_blank"} file. Here we can specify that all markers must be defined in this file with the `--strict-markers` flag and then declare our markers (with some info about them) in our `markers` list:
```toml linenums="1"
# Pytest
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
addopts = "--strict-markers --disable-pytest-warnings"
markers = [
    "training: tests that involve training",
]
```

Once we do this, we can view all of our existing list of markers by executing `#!bash pytest --markers` and we'll also receive an error when we're trying to use a new marker that's not defined here.

> We use custom markers to label which of our test functions involve training so we can separate long running tests from everything else.
```python linenums="1"
@pytest.mark.training
def test_train_model():
    experiment_name = "test_experiment"
    run_name = "test_run"
    result = runner.invoke()
    ...
```

> Another way to run custom tests is to use the `-k` flag when running pytest. The [k expression](https://docs.pytest.org/en/stable/example/markers.html#using-k-expr-to-select-tests-based-on-their-name){:target="_blank"} is much less strict compared to the marker expression where we can define expressions even based on names.


### Coverage

As we're developing tests for our application's components, it's important to know how well we're covering our code base and to know if we've missed anything. We can use the [Coverage](https://coverage.readthedocs.io/){:target="_blank"} library to track and visualize how much of our codebase our tests account for. With pytest, it's even easier to use this package thanks to the [pytest-cov](https://pytest-cov.readthedocs.io/){:target="_blank"} plugin.

```bash linenums="1"
pytest --cov tagifai --cov app --cov-report html
```

<div class="ai-center-all">
    <img width="650" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/testing/pytest.png" style="border-radius: 7px;">
</div>

Here we're asking for coverage for all the code in our tagifai and app directories and to generate the report in HTML format. When we run this, we'll see the tests from our tests directory executing while the coverage plugin is keeping tracking of which lines in our application are being executed. Once our tests are complete, we can view the generated report (default is `htmlcov/index.html`) and click on individual files to see which parts were not covered by any tests. This is especially useful when we forget to test for certain conditions, exceptions, etc.

<div class="ai-center-all">
    <img width="500" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/testing/coverage.png">
</div>

!!! warning
    Though we have 100% coverage, this does not mean that our application is perfect. Coverage only indicates that a piece of code executed in a test, not necessarily that every part of it was tested, let alone thoroughly tested. Therefore, coverage should **never** be used as a representation of correctness. However, it is very useful to maintain coverage at 100% so we can know when new functionality has yet to be tested. In our CI/CD lesson, we'll see how to use GitHub actions to make 100% coverage a requirement when pushing to specific branches.

#### Exclusions

Sometimes it doesn't make sense to write tests to cover every single line in our application yet we still want to account for these lines so we can maintain 100% coverage. We have two levels of purview when applying exclusions:

1. Excusing lines by adding this comment `# pragma: no cover, <MESSAGE>`
```python linenums="1"
if self.trial.should_prune():  # pragma: no cover, optuna pruning
    pass
```

2. Excluding files by specifying them in our [pyproject.toml](https://github.com/GokuMohandas/MLOps/blob/main/pyproject.toml){:target="_blank"} configuration, such as our [gunicorn.py](https://github.com/GokuMohandas/MLOps/blob/main/app/gunicorn.py){:target="_blank"} script since it's from a trusted template. We could, however, compose some tests to use this script but for now, we'll omit it.

```toml linenums="1"
# Pytest coverage
[tool.coverage.run]
omit = ["app/gunicorn.py"]
```

> The main point is that we were able to add justification to these exclusions through comments so our team can follow our reasoning.

## Machine learning

Now that we have a foundation for testing traditional software, let's dive into testing our data and models in the context of machine learning systems.

## 🔢&nbsp; Data

We've already tested the functions that act on our data through unit and integration tests but we haven't tested the validity of the data itself. Once we define what our data should look like, we can use, expand and adapt these expectations as our dataset grows.

### Expectations

There are many dimensions to what our data is expected to look like. We'll briefly talk about a few of them, including ones that may not directly be applicable to our task but, nonetheless, are very important to be aware of.

- `#!js rows / cols`: the most basic expectation is validating the presence of samples (rows) and features (columns). These can help identify mismatches between upstream backend database schema changes, upstream UI form changes, etc.

    !!! question "Rows/cols"
        What are aspects of rows and cols in our dataset that we should test for?

        ??? quote "Show answer"
            - presence of specific features
            - row count (exact or range) of samples

- `#!js individual values`: we can also have expectations about the individual values of specific features.

    !!! question "Individual"
        What are aspects of individual values that we should test for?

        ??? quote "Show answer"
            - missing values
            - type adherence (ex. feature values are all `float`)
            - values must be unique or from a predefined set
            - list (categorical) / range (continuous) of allowed values
            - feature value relationships with other feature values (ex. column 1 values must always be great that column 2)

- `#!js aggregate values`: we can also set expectations about all the values of specific features.

    !!! question "Aggregate"
        What are aspects of aggregate values that we should test for?

        ??? quote "Show answer"
            - value statistics (mean, std, median, max, min, sum, etc.)
            - distribution shift by comparing current values to previous values (useful for detecting drift)

To implement these expectations, we could compose assert statements or we could leverage the open-source library called [Great Expectations](https://github.com/great-expectations/great_expectations){:target="_blank"}. It's a fantastic library that already has many of these expectations builtin (map, aggregate, multi-column, distributional, etc.) and allows us to create custom expectations as well. It also provides modules to seamlessly connect with backend data sources such as local file systems, S3, databases and even DAG runners. Let's explore the library by implementing the expectations we'll need for our application.

> Though Great Expectations has all the data validation functionality we need, there are several other production-grade data validation options available as well, such as [TFX](https://www.tensorflow.org/tfx/data_validation/get_started){:target="_blank"}, [AWS Deequ](https://github.com/awslabs/deequ){:target="_blank"}, etc.

First we'll load the data we'd like to apply our expectations on. We can load our data from a variety of [sources](https://docs.greatexpectations.io/en/latest/guides/how_to_guides/configuring_datasources.html){:target="_blank"} (filesystem, S3, DB, etc.) which we can then wrap around a [Dataset module](https://docs.greatexpectations.io/en/latest/autoapi/great_expectations/dataset/index.html){:target="_blank"} (Pandas / Spark DataFrame, SQLAlchemy).

```python linenums="1"
from pathlib import Path
import great_expectations as ge
import pandas as pd
from tagifai import config, utils

# Create Pandas DataFrame
projects_fp = Path(config.DATA_DIR, "projects.json")
projects_dict = utils.load_dict(filepath=projects_fp)
df = ge.dataset.PandasDataset(projects_dict)
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
      <th>tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2020-02-17 06:30:41</td>
      <td>Machine Learning Basics</td>
      <td>A practical set of notebooks on machine learni...</td>
      <td>[code, tutorial, keras, pytorch, tensorflow, d...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2020-02-17 06:41:45</td>
      <td>Deep Learning with Electronic Health Record (E...</td>
      <td>A comprehensive look at recent machine learnin...</td>
      <td>[article, tutorial, deep-learning, health, ehr]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2020-02-20 06:07:59</td>
      <td>Automatic Parking Management using computer vi...</td>
      <td>Detecting empty and parked spaces in car parki...</td>
      <td>[code, tutorial, video, python, machine-learni...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2020-02-20 06:21:57</td>
      <td>Easy street parking using region proposal netw...</td>
      <td>Get a text on your phone whenever a nearby par...</td>
      <td>[code, tutorial, python, pytorch, machine-lear...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2020-02-20 06:29:18</td>
      <td>Deep Learning based parking management system ...</td>
      <td>Fastai provides easy to use wrappers to quickl...</td>
      <td>[code, tutorial, fastai, deep-learning, parkin...</td>
    </tr>
  </tbody>
</table>
</div></div>
</pre>


#### Built-in

Once we have our data source wrapped in a Dataset module, we can compose and apply expectations on it. There are many [built-in expectations](https://docs.greatexpectations.io/en/latest/reference/glossary_of_expectations.html#missing-values-unique-values-and-types){:target="blank"} to choose from:

```python linenums="1"
# Presence of features
expected_columns = ["id", "title", "description", "tags"]
df.expect_table_columns_to_match_ordered_list(column_list=expected_columns)

# Unique
df.expect_column_values_to_be_unique(column="id")

# No null values
df.expect_column_values_to_not_be_null(column="created_on")
df.expect_column_values_to_not_be_null(column="title")
df.expect_column_values_to_not_be_null(column="description")
df.expect_column_values_to_not_be_null(column="tags")

# Type
df.expect_column_values_to_be_of_type(column="title", type_="str")
df.expect_column_values_to_be_of_type(column="description", type_="str")
df.expect_column_values_to_be_of_type(column="tags", type_="list")

# Format
df.expect_column_values_to_match_strftime_format(
    column="created_on", strftime_format="%Y-%m-%d %H:%M:%S")

# Data leaks
df.expect_compound_columns_to_be_unique(column_list=["title", "description"])
```

Each of these expectations will create an output with details about success or failure, expected and observed values, expectations raised, etc. For example, the expectation ```#!python df.expect_column_values_to_be_of_type(column="title", type_="str")``` would produce the following if successful:

```json linenums="1" hl_lines="5 18-26"
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
    "element_count": 2032,
    "missing_count": 0,
    "missing_percent": 0.0,
    "unexpected_count": 0,
    "unexpected_percent": 0.0,
    "unexpected_percent_nonmissing": 0.0,
    "partial_unexpected_list": []
  }
}
```

and this output if it failed (notice the counts and examples for what caused the failure):
```json linenums="1" hl_lines="2 17-30"
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
    "element_count": 2032,
    "missing_count": 0,
    "missing_percent": 0.0,
    "unexpected_count": 2032,
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

We can group all the expectations together to create an [Expectation Suite](https://docs.greatexpectations.io/en/latest/reference/core_concepts/expectations/expectations.html#expectation-suites) object which we can use to validate any Dataset module.

```python linenums="1"
# Expectation suite
expectation_suite = df.get_expectation_suite()
print(df.validate(expectation_suite=expectation_suite, only_return_failures=True))
```
```json linenums="1"
{
  "success": true,
  "results": [],
  "statistics": {
    "evaluated_expectations": 9,
    "successful_expectations": 9,
    "unsuccessful_expectations": 0,
    "success_percent": 100.0
  },
  "evaluation_parameters": {}
}
```

#### Custom

Our `tags` feature column is a list of tags for each input. The Great Expectation's library doesn't come equipped to process a list feature but we can easily do so by creating a [custom expectation](https://docs.greatexpectations.io/en/latest/guides/how_to_guides/creating_and_editing_expectations/how_to_create_custom_expectations.html){:target="_blank"}.

1. Create a custom Dataset module to wrap our data around.
2. Define expectation functions that can map to each individual row of the feature column ([map](https://docs.greatexpectations.io/en/latest/autoapi/great_expectations/dataset/dataset/index.html#great_expectations.dataset.dataset.MetaDataset.column_map_expectation){:target="_blank"}) or to the entire feature column ([aggregate](https://docs.greatexpectations.io/en/latest/autoapi/great_expectations/dataset/dataset/index.html#great_expectations.dataset.dataset.MetaDataset.column_aggregate_expectation){:target="_blank"}) by specifying the appropriate decorator.
```python linenums="1"
class CustomPandasDataset(ge.dataset.PandasDataset):
    _data_asset_type = "CustomPandasDataset"

    @ge.dataset.MetaPandasDataset.column_map_expectation
    def expect_column_list_values_to_be_not_null(self, column):
        return column.map(lambda x: None not in x)

    @ge.dataset.MetaPandasDataset.column_map_expectation
    def expect_column_list_values_to_be_unique(self, column):
        return column.map(lambda x: len(x) == len(set(x)))
```
3. Wrap data with the custom Dataset module and use the custom expectations.
```python linenums="1"
df = CustomPandasDataset(projects_dict)
df.expect_column_values_to_not_be_null(column="tags")
df.expect_column_list_values_to_be_unique(column="tags")
```

> There are various levels of [abstraction](https://docs.greatexpectations.io/en/latest/guides/how_to_guides/creating_and_editing_expectations/how_to_create_custom_expectations.html){:target="_blank"} (following a template vs. completely from scratch) available when it comes to creating a custom expectation with Great Expectations.


### Projects
So far we've worked with the Great Expectations library at the Python script level but we can organize our expectations even more by creating a Project.

1. Initialize the Project using ```#!bash great_expectations init -d tests/```. This will interactively walk us through setting up data sources, naming, etc. and set up a [great_expectations](https://github.com/GokuMohandas/MLOps/blob/main/great_expectations){:target="_blank"} directory with the following structure:
```bash linenums="1"
tests/great_expectations/
|   ├── checkpoints/
|   ├── expectations/
|   ├── notebooks/
|   ├── plugins/
|   ├── uncommitted/
|   ├── .gitignore
|   └── great_expectations.yml
```
2. Define our [custom module](https://github.com/GokuMohandas/MLOps/blob/main/great_expectations/plugins/custom_module/custom_dataset.py){:target="_blank"} under the [plugins](https://github.com/GokuMohandas/MLOps/blob/main/great_expectations/plugins){:target="_blank"} directory and use it to define our data sources in our [great_expectations.yml](https://github.com/GokuMohandas/MLOps/blob/main/great_expectations/great_expectations.yml){:target="_blank"} configuration file.
```yaml linenums="1" hl_lines="5-6"
datasources:
  data:
    class_name: PandasDatasource
    data_asset_type:
      module_name: custom_module.custom_dataset
      class_name: CustomPandasDataset
    module_name: great_expectations.datasource
    batch_kwargs_generators:
      subdir_reader:
        class_name: SubdirReaderBatchKwargsGenerator
        base_directory: ../data
```
3. Create expectations using the profiler, which creates automatic expectations based on the data, or we can also create our own expectations. All of this is done interactively via a launched Jupyter notebook and saved under our [great_expectations/expectations](https://github.com/GokuMohandas/MLOps/tree/main/great_expectations/expectations){:target="_blank"} directory.
```bash linenums="1"
cd tests
great_expectations suite scaffold SUITE_NAME  # uses profiler
great_expectations suite new --suite  # no profiler
great_expectations suite edit SUITE_NAME  # add your own custom expectations
```
> When using the automatic profiler, you can choose which feature columns to apply profiling to. Since our tags feature is a list feature, we'll leave it commented and create our own expectations using the `suite edit` command.

4. Create Checkpoints where a Suite of Expectations are applied to a specific Data Asset. This is a great way of programmatically applying checkpoints on our existing and new data sources.
```bash linenums="1"
cd tests
great_expectations checkpoint new CHECKPOINT_NAME SUITE_NAME
great_expectations checkpoint run CHECKPOINT_NAME
```
5. Run checkpoints on new batches of incoming data by adding to our testing pipeline via Makefile, or workflow orchestrator like [Airflow](https://airflow.apache.org/){:target="_blank"}, [KubeFlow Pipelines](https://www.kubeflow.org/docs/components/pipelines/overview/pipelines-overview/){:target="_blank"}, etc. We can also use the [Great Expectations GitHub Action](https://github.com/great-expectations/great_expectations_action){:target="_blank"} to automate validating our data pipeline code when we push a change. More on using these Checkpoints with pipelines in our [Pipelines](pipelines.md){:target="_blank"} lesson.

```makefile linenums="1"
# Test
.PHONY: test
test:
	cd tests && great_expectations checkpoint run projects
	cd tests && great_expectations checkpoint run tags
	pytest -m "not training"
```

> Note that we're only executing the tests with the `training` [marker](testing.md#markers).

<div class="ai-center-all">
    <img width="700" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/testing/ge.png" style="border-radius: 7px;">
</div>


### Data docs
When we create expectations using the CLI application, Great Expectations automatically generates documentation for our tests. It also stores information about validation runs and their results. We can launch the generate data documentation with the following command: ```#!bash great_expectations docs build```

<div class="ai-center-all">
    <img width="700" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/testing/docs.png">
</div>

### Best practices

We've applied expectations on our source dataset but there are many other key areas to test the data as well. Throughout the ML development pipeline, we should test the intermediate outputs from processes such as cleaning, augmentation, splitting, preprocessing, tokenization, etc. We'll use these expectations to monitor new batches of data and before combining them with our existing data assets.

!!! note
    Currently, these data processing steps are tied with our application code but in future lessons, we'll separate these into individual pipelines and use Great Expectation Checkpoints in between to apply all these expectations in an orchestrated fashion.

    <div class="ai-center-all">
        <img width="650" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/testing/pipelines.png">
    </div>
    <div class="ai-center-all">
        <a href="https://docs.greatexpectations.io/en/latest/_images/ge_tutorials_pipeline.png" target="_blank">Pipelines with Great Expectations Checkpoints</a>
    </div>


## 🤖&nbsp; Models

The other half of testing ML systems involves testing our models during training, evaluation, inference and deployment.

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
assert accuracy == pytest.approx(1.0, abs=0.05) # 1.0 ± 0.05
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

### Evaluation

When it comes to testing how well our model performs, we need to first have our priorities in line.

- What metrics are important?
- What tradeoffs are we willing to make?
- Are there certain subsets (slices) of data that are important?

#### Metrics

##### Overall
We want to ensure that our key metrics on the overall dataset improves with each iteration of our model. Overall metrics include accuracy, precision, recall, f1, etc. and we should define what counts as a performance regression. For example, is a higher precision at the expensive of recall an improvement or a regression? Usually, a team of developers and domain experts will establish what the key metric(s) are while also specifying the lowest regression tolerance for other metrics.

```python linenums="1"
assert precision > prev_precision  # most important, cannot regress
assert recall >= best_prev_recall - 0.03  # recall cannot regress > 3%
```

##### Per-class

We can perform similar assertions for class specific metrics as well.

```python linenums="1"
assert metrics["class"]["data_augmentation"]["f1"] > prev_data_augmentation_f1  # priority class
```

##### Slices

In the same vain, we can create assertions for certain key [slices](evaluation.md#slices){:target="_blank"} of our data as well. This can be very simple test to ensure that our high priority slices of data continue to improve in performance.

```python linenums="1"
assert metrics["slices"]["class"]["cv_transformers"]["f1"] > prev_cv_transformers_f1  # priority slice
```

#### Behavioral testing

Besides just looking at metrics, we also want to conduct some behavior sanity tests. Behavioral testing is the process of testing input data and expected outputs while treating the model as a black box. They don't necessarily have to be adversarial in nature but more along the types of perturbations we'll see in the real world once our model is deployed. A landmark paper on this topic is [Beyond Accuracy: Behavioral Testing of NLP Models with CheckList](https://arxiv.org/abs/2005.04118){:target="_blank"} which breaks down behavioral testing into three types of tests:

- `#!js invariance`: Changes should not affect outputs.
```python linenums="1"
# INVariance via verb injection (changes should not affect outputs)
tokens = ["revolutionized", "disrupted"]
tags = [["transformers"], ["transformers"]]
texts = [f"Transformers have {token} the ML field." for token in tokens]
```
- `#!js directional`: Change should affect outputs.
```python linenums="1"
# DIRectional expectations (changes with known outputs)
tokens = ["PyTorch", "Huggingface"]
tags = [
    ["pytorch", "transformers"],
    ["huggingface", "transformers"],
]
texts = [f"A {token} implementation of transformers." for token in tokens]
```
- `#!js minimum functionality`: Simple combination of inputs and expected outputs.
```python linenums="1"
# Minimum Functionality Tests (simple input/output pairs)
tokens = ["transformers", "graph neural networks"]
tags = [["transformers"], ["graph-neural-networks"]]
texts = [f"{token} have revolutionized machine learning." for token in tokens]
```

And we can easily convert these behavioral tests into parameterized pytest functions.

```python linenums="1"
@pytest.mark.parametrize(
    "text, tags",
    [
        ("Transformers have revolutionized machine learning.", ["transformers"]),
        ("GNNs have revolutionized machine learning.", ["graph-neural-networks"]),
    ],
)
def test_mft(text, tags, artifacts):
    """# Minimum Functionality Tests (simple input/output pairs)."""
    results = predict.predict(texts=[text], artifacts=artifacts)
    assert [set(result["predicted_tags"]).issubset(set(tags)) for result in results]
```

> If not all behavioral tests are meant to be passed, then we could compute a behavioral score and use that as part of our evaluation criteria.

!!! note
    Be sure to explore the [NLP Checklist](https://github.com/marcotcr/checklist){:target="_blank"} package which simplifies and augments the creation of these behavioral tests via functions, templates, pretrained models and interactive GUIs in Jupyter notebooks.

    <div class="ai-center-all">
        <img width="600" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/testing/checklist.gif">
    </div>
    <div class="ai-center-all mt-2">
        <a href="https://github.com/marcotcr/checklist" target="_blank">NLP Checklist</a>
    </div>

### Inference

When our model is deployed, most users will be using it for inference (directly / indirectly), so it's very important that we test all aspects of it.

#### Loading artifacts
This is the first time we're not loading our components from in-memory so we want to ensure that the required artifacts (model weights, encoders, config, etc.) are all able to be loaded.

```python linenums="1"
artifacts = main.load_artifacts(run_id=run_id, device=torch.device("cpu"))
assert isinstance(artifacts["model"], nn.Module)
...
```

#### Prediction
Once we have our artifacts loaded, we're readying to test our prediction pipelines. We should test samples with just one input, as well as a batch of inputs (ex. padding can have unintended consequences sometimes).
```python linenums="1"
# tests/app/test_api.py | test_best_predict()
data = {
    "run_id": "",
    "texts": [
        {"text": "Transfer learning with transformers for self-supervised learning."},
        {"text": "Generative adversarial networks in both PyTorch and TensorFlow."},
    ],
}
response = client.post("/predict", json=data)
assert response.status_code == HTTPStatus.OK
assert response.request.method == "POST"
assert len(response.json()["data"]["predictions"]) == len(data["texts"])
...
```

### Deployment

Once we've tested our model's ability to perform in the production environment (**offline tests**), we can run several types of **online tests** to determine the quality of that performance. These test are a mix of system and acceptance tests because we want to see how our system behaves as part of the larger product and to ensure that it meets certain acceptance criteria (ex. latency).

#### AB tests
AB tests involve sending production traffic to the different systems that we're evaluating and then using statistical hypothesis testing to decide which system is better. There are several common issues with AB testing such as accounting for different sources of bias, such as the novelty effect of showing some users the new system. We also need to ensure that the same users continue to interact with the same systems so we can compare the results without contamination. In many cases, if we're simply trying to compare the different versions for a certain metric, multi-armed bandits will be a better approach.

<div class="ai-center-all">
    <img width="500" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/infrastructure/ab.png">
</div>

#### Canary tests
Canary tests involve sending most of the production traffic to the currently deployed system but sending traffic from a small cohort of users to the new system we're trying to evaluate. Again we need to make sure that the same users continue to interact with the same system as we gradually roll out the new system.

<div class="ai-center-all">
    <img width="500" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/infrastructure/canary.png">
</div>

#### Shadow tests
Shadow testing involves sending the same production traffic to the different systems. We don't have to worry about system contamination and it's very safe compared to the previous approaches since the new system's results are not served.

!!! question "Why can go wrong?"
    If shadow tests allow us to test our updated system without having to actually serve the new results, why doesn't everyone adopt it?

    ??? quote "Show answer"
        We need to ensure that we're replicating as much of the production system as possible so we can catch issues that are unique to production early on. This is rarely possible because. while your ML system may be a standalone microservice, it ultimately interacts with an intricate production environment that has *many* dependencies.

But overall, shadow testing is easy to monitor, validate operational consistency, etc.

<div class="ai-center-all">
    <img width="500" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/infrastructure/shadow.png">
</div>

## Testing vs. monitoring

We'll conclude by talking about the similarities and distinctions between testing and [monitoring](monitoring.md){:target="_blank"}. They're both integral parts of the ML development pipeline and depend on each other for iteration. Testing is assuring that our system (code, data and models) passes the expectations that we've established at $t_0$. Whereas, monitoring involves that these expectations continue to pass on live production data while also ensuring that their data distributions are [comparable](monitoring.md#measuring-drift){:target="_blank"} to the reference window (typically subset of training data) through $t_n$. When these conditions no longer hold true, we need to inspect more closely (retraining may not always fix our root problem).

With [monitoring](monitoring.md){:target="_blank"}, there are quite a few distinct concerns that we didn't have to consider during testing since it involves (live) data we have yet to see.

- features and prediction distributions (drift), typing, schema mismatches, etc.
- determining model performance (rolling and window metrics on overall and slices of data) using indirect signals (since labels may not be readily available).
- in situations with large data, we need to know which data points to label and upsample for training.
- identifying anomalies and outliers.

> We'll cover all of these concepts in much more depth (and code) in our [monitoring](monitoring.md){:target="_blank"} lesson and cover some data-centric approaches in our [data-driven development lesson](data-driven-development.md){:target="_blank"}.

## Resources
- [Great Expectations](https://github.com/great-expectations/great_expectations){:target="_blank"}
- [The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/aad9f93b86b7addfea4c419b9100c6cdd26cacea.pdf){:target="_blank"}
- [Beyond Accuracy: Behavioral Testing of NLP Models with CheckList](https://arxiv.org/abs/2005.04118){:target="_blank"}
- [A Recipe for Training Neural Networks](http://karpathy.github.io/2019/04/25/recipe/){:target="_blank"}
- [Effective testing for machine learning systems](https://www.jeremyjordan.me/testing-ml/){:target="_blank"}

<!-- Citation -->
{% include "cite.md" %}