---
template: lesson.html
title: "Testing Machine Learning Systems: Code, Data and Models"
description: Testing code, data and models to ensure consistent behavior in ML systems.
keywords: testing, pytest, unit test, parametrize, fixtures, applied ml, mlops, machine learning, ml in production, machine learning in production, applied machine learning, great expectations
image: https://madewithml.com/static/images/applied_ml.png
repository: https://github.com/GokuMohandas/applied-ml
---

{% include "styles/lesson.md" %}

## Intuition

Tests are a way for us to ensure that something works as intended. We're incentivized to implement tests and discover sources of error as early in the development cycle as possible so that we can reduce [increasing downstream costs](https://assets.deepsource.io/39ed384/images/blog/cost-of-fixing-bugs/chart.jpg){:target="_blank"} and wasted time. Once we've designed our tests, we can automatically execute them every time we implement a change to our system, and continue to build on them.

### Types of tests

There are many four majors types of tests which are utilized at different points in the development cycle:

1. `#!js Unit tests`: tests on individual components that each have a [single responsibility](https://en.wikipedia.org/wiki/Single-responsibility_principle){:target="_blank"} (ex. function that filters a list).
2. `#!js Integration tests`: tests on the combined functionality of individual components (ex. data processing).
3. `#!js System tests`: tests on the design of a system for expected outputs given inputs (ex. training, inference, etc.).
4. `#!js Acceptance tests`: tests to verify that requirements have been met, usually referred to as User Acceptance Testing (UAT).
5. `#!js Regression tests`: testing errors we've seen before to ensure new changes don't reintroduce them.

<div class="ai-center-all">
    <img width="700" src="https://raw.githubusercontent.com/GokuMohandas/madewithml/main/images/applied-ml/testing/tests.png">
</div>

!!! note
    There are many other types of functional and non-functional tests as well, such as smoke tests (quick health checks), performance tests (load, stress), security tests, etc. but we can generalize these under the system tests above.

### How should we test?

The framework to use when composing tests is the [Arrange Act Assert](http://wiki.c2.com/?ArrangeActAssert){:target="_blank"} methodology.

- `#!js Arrange`: set up the different inputs to test on.
- `#!js Act`: apply the inputs on the component we want to test.
- `#!js Assert`: confirm that we received the expected output.

!!! tip
    `#!js Cleaning` is an unofficial fourth step to this methodology because it's important to not leave remnants of a previous state which may affect subsequent tests. We can use packages such as [pytest-randomly](https://github.com/pytest-dev/pytest-randomly){:target="_blank"} to test against state dependency by executing tests randomly.

In Python, there are many tools, such as [unittest](https://docs.python.org/3/library/unittest.html){:target="_blank"}, [pytest](https://docs.pytest.org/en/stable/){:target="_blank"}, etc., that allow us to easily implement our tests while adhering to the *Arrange Act Assert* framework above. These tools come with powerful built-in functionality such as parametrization, filters, and more, to test many conditions at scale.

!!! note
    When *arranging* our inputs and *asserting* our expected outputs, it's important to test across the entire gambit of inputs and outputs:

    - **inputs**: data types, format, length, edge cases (min/max, small/large, etc.)
    - **outputs**: data types, formats, exceptions, intermediary and final outputs

## Best practices
Regardless of the framework we use, it's important to strongly tie testing into the development process.

- `#!js atomic`: when creating unit components, we need to ensure that they have a [single responsibility](https://en.wikipedia.org/wiki/Single-responsibility_principle){:target="_blank"} so that we can easily test them. If not, we'll need to split them into more granular units.
- `#!js compose`: when we create new components, we want to compose tests to validate their functionality. It's a great way to ensure reliability and catch errors early on.
- `#!js regression`: we want to account for new errors we come across with a regression test so we can ensure we don't reintroduce the same errors in the future.
- `#!js coverage`: we want to ensure that 100% of our codebase has been accounter for. This doesn't mean writing a test for every single line of code but rather accounting for every single line (more on this in the [coverage section](#coverage) below).
- `#!js automate`: in the event we forget to run our tests before committing to a repository, we want to auto run tests for every commit. We'll learn how to do this locally using Precommit and remotely (ie. main branch) via GitHub actions in subsequent lessons.

## Test-driven development

[Test-driven development (TDD)](https://en.wikipedia.org/wiki/Test-driven_development){:target="_blank"} is the process where you write a test before completely writing the functionality to ensure that tests are always written. This is in contrast to writing functionality first and then composing tests afterwards. Here are my thoughts on this:

- good to write tests as we progress, but it's not the representation of correctness.
- initial time should be spent on design before ever getting into the code or tests.
- using a test as guide doesn't mean that our functionality is error free.

Perfect coverage doesn't mean that our application is error free if those tests aren't meaningful and don't encompass the field of possible inputs, intermediates and outputs. Therefore, we should work towards better design and agility when facing errors, quickly resolving them and writing test cases around them to avoid them next time.

!!! warning
    This topic is still highly debated and I'm only reflecting on my experience and what's worked well for me at a large company (Apple), very early stage startup and running a company of my own. What's most important is that the team is producing reliable systems that can be tested and improved upon.

## Application

In our [application](https://github.com/GokuMohandas/applied-ml){:target="_blank"}, we'll be testing the code, data and models. Be sure to look inside each of the different testing scripts after reading through the components below.

```bash linenums="1"
great_expectations/           # data tests
|   ├── expectations/
|   |   ├── projects.json
|   |   └── tags.json
|   ├── ...
tagifai/
|   ├── eval.py               # model tests
tests/                        # code tests
├── app/
|   ├── test_api.py
|   └── test_cli.py
└── tagifai/
|   ├── test_config.py
|   ├── test_data.py
|   ├── test_eval.py
|   ├── test_models.py
|   ├── test_train.py
|   └── test_utils.py
```

!!! note
    Alternatively, we could've organized our tests by types of tests as well (unit, integration, etc.) but I find it more intuitive for navigation by organizing by how our application is set up. We'll learn about [markers](#markers) below which will allow us to run any subset of tests by specifying filters.

## 🧪&nbsp; Pytest

We're going to be using [pytest](https://docs.pytest.org/en/stable/){:target="_blank"} as our testing framework for it's powerful builtin features such as [parametrization](#parametrize), [fixtures](#fixtures), [markers](#markers), etc.

### Configuration
Pytest expects tests to be organized under a `tests` directory by default. However, we can also use our [pyproject.toml](https://github.com/GokuMohandas/applied-ml/blob/main/pyproject.toml){:target="_blank"} file to configure any other test path directories as well. Once in the directory, pytest looks for python scripts starting with `tests_*.py` but we can configure it to read any other file patterns as well.

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

!!! note
    We can also have assertions about [exceptions](https://docs.pytest.org/en/stable/assert.html#assertions-about-expected-exceptions){:target="_blank"} like we do in lines 6-8 where all the operations under the with statement are expected to raise the specified exception.

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

!!! note
    It's important to test for the variety of inputs and expected outputs that we outlined [above](#how-should-we-test) and to never assume that a test is trivial. In our example above, it's important that we test for both "apple" and "Apple" in the event that our function didn't account for casing!

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

Parametrization allows us to efficiently reduce redundancy inside test functions but what about its inputs? Here, we can use pytest's builtin [fixture](https://docs.pytest.org/en/stable/fixture.html){:target="_blank"} which is a function that is executed before the test function. This significantly reduces redundancy when multiple test functions require the same inputs.

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

> We use fixtures to efficiently pass a set of inputs (ex. Pandas DataFrame) to different testing functions that require them (cleaning, splitting, etc.).
```python linenums="1"
@pytest.fixture
def df():
    projects_fp = Path(config.DATA_DIR, "projects.json")
    projects_dict = utils.load_dict(filepath=projects_fp)
    df = pd.DataFrame(projects_dict)
    return df


def test_split(df):
    splits = split_data(df=df)
    ...
```

!!! note
    Typically, when we have too many fixtures in a particular test file, we can organize them all in a `fixtures.py` script and invoke them as needed.


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

The proper way to use markers is to explicitly list the ones we've created in our [pyproject.toml](https://github.com/GokuMohandas/applied-ml/blob/main/pyproject.toml){:target="_blank"} file. Here we can specify that all markers must be defined in this file with the `--strict-markers` flag and then declare our markers (with some info about them) in our `markers` list:
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

!!! note
    Another way to run custom tests is to use the `-k` flag when running pytest. The [k expression](https://docs.pytest.org/en/stable/example/markers.html#using-k-expr-to-select-tests-based-on-their-name){:target="_blank"} is much less strict compared to the marker expression where we can define expressions even based on names.


### Coverage

As we're developing tests for our application's components, it's important to know how well we're covering our code base and to know if we've missed anything. We can use the [Coverage](https://coverage.readthedocs.io/){:target="_blank"} library to track and visualize how much of our codebase our tests account for. With pytest, it's even easier to use this package thanks to the [pytest-cov](https://pytest-cov.readthedocs.io/){:target="_blank"} plugin.

```bash linenums="1"
pytest --cov tagifai --cov app --cov-report html
```

<div class="ai-center-all">
    <img width="650" src="https://raw.githubusercontent.com/GokuMohandas/madewithml/main/images/applied-ml/testing/pytest.png" style="border-radius: 7px;">
</div>

Here we're asking for coverage for all the code in our tagifai and app directories and to generate the report in HTML format. When we run this, we'll see the tests from our tests directory executing while the coverage plugin is keeping tracking of which lines in our application are being executed. Once our tests are complete, we can view the generated report (default is `htmlcov/index.html`) and click on individual files to see which parts were not covered by any tests. This is especially useful when we forget to test for certain conditions, exceptions, etc.

<div class="ai-center-all">
    <img width="500" src="https://raw.githubusercontent.com/GokuMohandas/madewithml/main/images/applied-ml/testing/coverage.png">
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

2. Excluding files by specifying them in our [pyproject.toml](https://github.com/GokuMohandas/applied-ml/blob/main/pyproject.toml){:target="_blank"} configuration.
```toml linenums="1"
# Pytest coverage
[tool.coverage.run]
omit = ["app/main.py"]  #  sample API calls
```

The key here is that we were able to add justification to these exclusions through comments so our team can follow our reasoning.

## Machine learning

Now that we have a foundation for testing traditional software, let's dive into testing our data and models in the context of machine learning systems.

## 🔢&nbsp; Data

We've already tested the functions that act on our data through unit and integration tests but we haven't tested the validity of the data itself. Once we define what our data should look like, we can use (and add to) these expectations as our dataset grows.

### Expectations

There are many dimensions to what our data is expected to look like. We'll briefly talk about a few of them, including ones that may not directly be applicable to our task but, nonetheless, are very important to be aware of.

- `#!js rows / cols`: the most basic expectation is validating the presence of samples (rows) and features (columns). These can help identify mismatches between upstream backend database schema changes, upstream UI form changes, etc.
    - presence of specific features
    - row count (exact or range) of samples
- `#!js individual values`: we can also have expectations about the individual values of specific features.
    - missing values
    - type adherence (ex. feature values are all `float`)
    - values must be unique or from a predefined set
    - list (categorical) / range (continuous) of allowed values
    - feature value relationships with other feature values (ex. column 1 values must always be great that column 2)
- `#!js aggregate values`: we can also expectations about all the values of specific features.
    - value statistics (mean, std, median, max, min, sum, etc.)
    - distribution shift by comparing current values to previous values (useful for detecting drift)

To implement these expectations, we could compose assert statements or we could leverage the open-source library called [Great Expectations](https://github.com/great-expectations/great_expectations){:target="_blank"}. It's a fantastic library that already has many of these expectations builtin (map, aggregate, multi-column, distributional, etc.) and allows us to create custom expectations as well. It also provides modules to seamlessly connect with backend data sources such as local file systems, S3, databases and even DAG runners. Let's explore the library by implementing the expectations we'll need for our application.

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
      <th>title</th>
      <th>description</th>
      <th>tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2438</td>
      <td>How to Deal with Files in Google Colab: What Y...</td>
      <td>How to supercharge your Google Colab experienc...</td>
      <td>[article, google-colab, colab, file-system]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2437</td>
      <td>Rasoee</td>
      <td>A powerful web and mobile application that ide...</td>
      <td>[api, article, code, dataset, paper, research,...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2436</td>
      <td>Machine Learning Methods Explained (+ Examples)</td>
      <td>Most common techniques used in data science pr...</td>
      <td>[article, deep-learning, machine-learning, dim...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2435</td>
      <td>Top “Applied Data Science” Papers from ECML-PK...</td>
      <td>Explore the innovative world of Machine Learni...</td>
      <td>[article, deep-learning, machine-learning, adv...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2434</td>
      <td>OpenMMLab Computer Vision</td>
      <td>MMCV is a python library for CV research and s...</td>
      <td>[article, code, pytorch, library, 3d, computer...</td>
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
df.expect_column_values_to_not_be_null(column="title")
df.expect_column_values_to_not_be_null(column="description")
df.expect_column_values_to_not_be_null(column="tags")

# Type
df.expect_column_values_to_be_of_type(column="title", type_="str")
df.expect_column_values_to_be_of_type(column="description", type_="str")
df.expect_column_values_to_be_of_type(column="tags", type_="list")

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

!!! note
    There are various levels of [abstraction](https://docs.greatexpectations.io/en/latest/guides/how_to_guides/creating_and_editing_expectations/how_to_create_custom_expectations.html){:target="_blank"} (following a template vs. completely from scratch) available when it comes to creating a custom expectation with Great Expectations.


### Projects
So far we've worked with the Great Expectations library at the Python script level but we can organize our expectations even more by creating a Project.

1. Initialize the Project using ```#!bash great_expectations init```. This will interactively walk us through setting up data sources, naming, etc. and set up a [great_expectations](https://github.com/GokuMohandas/applied-ml/blob/main/great_expectations){:target="_blank"} directory with the following structure:
```bash linenums="1"
great_expectations/
|   ├── checkpoints/
|   ├── expectations/
|   ├── notebooks/
|   ├── plugins/
|   ├── uncommitted/
|   ├── .gitignore
|   └── great_expectations.yml
```
2. Define our [custom module](https://github.com/GokuMohandas/applied-ml/blob/main/great_expectations/plugins/custom_module/custom_dataset.py){:target="_blank"} under the [plugins](https://github.com/GokuMohandas/applied-ml/blob/main/great_expectations/plugins){:target="_blank"} directory and use it to define our data sources in our [great_expectations.yml](https://github.com/GokuMohandas/applied-ml/blob/main/great_expectations/great_expectations.yml){:target="_blank"} configuration file.
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
        base_directory: ../assets/data
```
3. Create expectations using the profiler, which creates automatic expectations based on the data, or we can also create our own expectations. All of this is done interactively via a launched Jupyter notebook and saved under our [great_expectations/expectations](https://github.com/GokuMohandas/applied-ml/main/great_expectations/expectations){:target="_blank"} directory.
```bash linenums="1"
great_expectations suite scaffold SUITE_NAME  # uses profiler
great_expectations suite new --suite  # no profiler
great_expectations suite edit SUITE_NAME  # add your own custom expectations
```
> When using the automatic profiler, you can choose which feature columns to apply profiling to. Since our tags feature is a list feature, we'll leave it commented and create our own expectations using the `suite edit` command.

4. Create Checkpoints where a Suite of Expectations are applied to a specific Data Asset. This is a great way of programmatically applying checkpoints on our existing and new data sources.
```bash linenums="1"
great_expectations checkpoint new CHECKPOINT_NAME SUITE_NAME
great_expectations checkpoint run CHECKPOINT_NAME
```
5. Run checkpoints on new batches of incoming data by adding to our testing pipeline via Makefile, or workflow orchestrator like [Airflow](https://airflow.apache.org/){:target="_blank"}, etc. We can also use the [Great Expectations GitHub Action](https://github.com/great-expectations/great_expectations_action){:target="_blank"} to automate validating our data pipeline code when we push a change. More on using these Checkpoints with pipelines in our [workflows](workflows.md){:target="_blank"} lesson.

<div class="ai-center-all">
    <img width="700" src="https://raw.githubusercontent.com/GokuMohandas/madewithml/main/images/applied-ml/testing/ge.png" style="border-radius: 7px;">
</div>


### Data docs
When we create expectations using the CLI application, Great Expectations automatically generates documentation for our tests. It also stores information about validation runs and their results. We can launch the generate data documentation with the following command: ```#!bash great_expectations docs build```

<div class="ai-center-all">
    <img width="700" src="https://raw.githubusercontent.com/GokuMohandas/madewithml/main/images/applied-ml/testing/docs.png">
</div>

### Best practices

We've applied expectations on our source dataset but there are many other key areas to test the data as well. Throughout the ML development pipeline, we should test the intermediate outputs from processes such as cleaning, augmentation, splitting, preprocessing, tokenization, etc. We'll use these expectations to monitor new batches of data and before combining them with our existing data assets.

!!! note
    Currently, these data processing steps are tied with our application code but in future lessons, we'll separate these into individual pipelines and use Great Expectation Checkpoints in between to apply all these expectations in an orchestrated fashion.

    <div class="ai-center-all">
        <img width="650" src="https://raw.githubusercontent.com/GokuMohandas/madewithml/main/images/applied-ml/testing/pipelines.png">
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

#### Overall

We want to ensure that our key metrics on the overall dataset improves with each iteration of our model. Overall metrics include accuracy, precision, recall, f1, etc. and we should define what counts as a performance regression. For example, is a higher precision at the expensive of recall an improvement or a regression? Usually, a team of developers and domain experts will establish what the key metric(s) are while also specifying the lowest regression tolerance for other metrics.

```python linenums="1"
assert precision > prev_precision  # most important, cannot regress
assert recall >= best_prev_recall - 0.03  # recall cannot regress > 3%
```

#### Slicing

Just inspecting the overall metrics isn't enough to deploy our new version to production. There may be key slices of our dataset that we expect to do really well on (ie. minority groups, large customers, etc.) and we need to ensure that their metrics are also improving. An easy way to create and evaluate slices is to define slicing functions.

```python linenums="1"
# tagifai/eval.py
from snorkel.slicing import slicing_function

@slicing_function()
def cv_transformers(x):
    """Projects with the `computer-vision` and `transformers` tags."""
    return all(tag in x.tags for tag in ["computer-vision", "transformers"])
```

Here we're using Snorkel's [`slicing_function`](https://snorkel.readthedocs.io/en/v0.9.3/packages/_autosummary/slicing/snorkel.slicing.slicing_function.html){:target="blank"} to create our different slices. We can visualize our slices by applying this slicing function to a relevant DataFrame using [`slice_dataframe`](https://snorkel.readthedocs.io/en/v0.9.3/packages/_autosummary/slicing/snorkel.slicing.slice_dataframe.html){:target="_blank"}.

```python linenums="1"
from snorkel.slicing import slice_dataframe

test_df = pd.DataFrame({"text": X_test, "tags": label_encoder.decode(y_test)})
cv_transformers_df = slice_dataframe(test_df, cv_transformers)
cv_transformers_df[["text", "tags"]].head()
```
<pre class="output">
<div class="output_subarea output_html rendered_html"><div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>text</th>
      <th>tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>vedastr vedastr open source scene text recogni...</td>
      <td>[computer-vision, natural-language-processing,...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15</td>
      <td>hugging captions generate realistic instagram ...</td>
      <td>[computer-vision, huggingface, language-modeli...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>49</td>
      <td>transformer ocr rectification free ocr using s...</td>
      <td>[attention, computer-vision, natural-language-...</td>
    </tr>
  </tbody>
</table>
</div></div>
</pre>

We can define even more slicing functions and create a slices record array using the [`PandasSFApplier`](https://snorkel.readthedocs.io/en/v0.9.6/packages/_autosummary/slicing/snorkel.slicing.PandasSFApplier.html){:target="_blank"}. The slices array has N (# of data points) items and each item has S (# of slicing functions) items, indicating whether that data point is part of that slice. Think of this record array as a masking layer for each slicing function on our data.

```python linenums="1"
# tagifai/eval.py | get_performance()
from snorkel.slicing import PandasSFApplier

slicing_functions = [cv_transformers, short_text]
applier = PandasSFApplier(slicing_functions)
slices = applier.apply(df)
print (slices)
```
<pre class="output">
[(0, 0) (0, 1) (0, 0) (1, 0) (0, 0) (0, 0) (0, 0) (0, 0) (0, 0) (0, 0)
 (1, 0) (0, 0) (0, 1) (0, 0) (0, 0) (1, 0) (0, 0) (0, 0) (0, 1) (0, 0)
 ...
 (0, 0) (1, 0) (0, 0) (0, 0) (0, 0) (0, 0) (0, 0) (0, 0) (0, 0) (0, 0)
 (0, 0) (0, 0) (1, 0) (0, 0) (0, 0) (0, 0) (1, 0)]
</pre>

One we have our slices record array, we can compute the performance metrics for each slice.

```python linenums="1"
# tagifai/eval.py | get_performance()
for slice_name in slices.dtype.names:
    mask = slices[slice_name].astype(bool)
    metrics = precision_recall_fscore_support(y_true[mask], y_pred[mask], average="micro")
```

!!! note
    Snorkel comes with a builtin [slice scorer](https://snorkel.readthedocs.io/en/v0.9.1/packages/_autosummary/analysis/snorkel.analysis.Scorer.html){:target="_blank"} but we had to implemented a naive version since our task involves multi-label classification.

We can add these slice performance metrics to our larger performance report to analyze downstream when choosing which model to deploy.

```json linenums="1" hl_lines="23"
{
  "overall": {
    "precision": 0.8050380552475824,
    "recall": 0.603411513859275,
    "f1": 0.6674448998627966,
    "num_samples": 207.0
  },
  "class": {
    "attention": {
      "precision": 0.6923076923076923,
      "recall": 0.5,
      "f1": 0.5806451612903226,
      "num_samples": 18.0
    },
    ...
    "unsupervised-learning": {
      "precision": 0.8,
      "recall": 0.5,
      "f1": 0.6153846153846154,
      "num_samples": 8.0
    }
  },
  "slices": {
    "f1": 0.7395604395604396,
    "cv_transformers": {
      "precision": 1.0,
      "recall": 0.5384615384615384,
      "f1": 0.7000000000000001,
      "num_samples": 3
    },
    "short_text": {
      "precision": 0.8333333333333334,
      "recall": 0.7142857142857143,
      "f1": 0.7692307692307692,
      "num_samples": 4
    }
  }
}
```

#### Extensions

We've explored user generated slices but there is currently quite a bit of research on automatically generated slices and overall model robustness. A notable toolkit is the [Robustness Gym](https://arxiv.org/abs/2101.04840){:target="_blank"} which programmatically builds slices, performs adversarial attacks, rule-based data augmentation, benchmarking, reporting and much more.

<div class="ai-center-all">
    <img width="600" src="https://raw.githubusercontent.com/GokuMohandas/madewithml/main/images/applied-ml/testing/gym.png">
</div>
<div class="ai-center-all mt-2">
    <a href="https://arxiv.org/abs/2101.04840" target="_blank">Robustness Gym slice builders</a>
</div>

Instead of passively observing slice performance, we could try and improve them. Usually, a slice may exhibit poor performance when there are too few samples and so a natural approach is to oversample. However, these methods change the underlying data distribution and can cause issues with overall / other slices. It's also not scalable to train a separate model for each unique slice and combine them via [Mixture of Experts (MoE)](https://www.cs.toronto.edu/~hinton/csc321/notes/lec15.pdf){:target="_blank"}. To combat all of these technical challenges and more, the Snorkel team introduced the [Slice Residual Attention Modules (SRAMs)](https://arxiv.org/abs/1909.06349){:target="_blank"}, which can sit on any backbone architecture (ie. our CNN feature extractor) and learn slice-aware representations for the class predictions.

<div class="ai-center-all">
    <img width="600" src="https://raw.githubusercontent.com/GokuMohandas/madewithml/main/images/applied-ml/testing/sram.png">
</div>
<div class="ai-center-all mt-3">
    <a href="https://arxiv.org/abs/1909.06349" target="_blank">Slice Residual Attention Modules (SRAMs)</a>
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
assert response.json()["status-code"] == HTTPStatus.OK
assert response.json()["method"] == "POST"
assert len(response.json()["data"]["predictions"]) == len(data["texts"])
...
```

#### Behavioral testing

Besides just testing if the prediction pipelines work, we also want to ensure that they work well. Behavioral testing is the process of testing input data and expected outputs while treating the model as a black box. They don't necessarily have to be adversarial in nature but more along the types of perturbations we'll see in the real world once our model is deployed. A landmark paper on this topic is [Beyond Accuracy: Behavioral Testing of NLP Models with CheckList](https://arxiv.org/abs/2005.04118){:target="_blank"} which breaks down behavioral testing into three types of tests:

- `#!js invariance`: Changes should not affect outputs.
```python linenums="1"
# INVariance via verb injection (changes should not affect outputs)
tokens = ["revolutionized", "disrupted"]
tags = [["transformers"], ["transformers"]]
texts = [f"Transformers have {token} the ML field." for token in tokens]
compare_tags(texts=texts, tags=tags, artifacts=artifacts, test_type="INV")
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
compare_tags(texts=texts, tags=tags, artifacts=artifacts, test_type="DIR")
```
- `#!js minimum functionality`: Simple combination of inputs and expected outputs.
```python linenums="1"
# Minimum Functionality Tests (simple input/output pairs)
tokens = ["transformers", "graph neural networks"]
tags = [["transformers"], ["graph-neural-networks"]]
texts = [f"{token} have revolutionized machine learning." for token in tokens]
compare_tags(texts=texts, tags=tags, artifacts=artifacts, test_type="MFT")
```

!!! note
    Be sure to explore the [NLP Checklist](https://github.com/marcotcr/checklist){:target="_blank"} package which simplifies and augments the creation of these behavioral tests via functions, templates, pretrained models and interactive GUIs in Jupyter notebooks.

    <div class="ai-center-all">
        <img width="600" src="https://raw.githubusercontent.com/GokuMohandas/madewithml/main/images/applied-ml/testing/checklist.gif">
    </div>
    <div class="ai-center-all mt-2">
        <a href="https://github.com/marcotcr/checklist" target="_blank">NLP Checklist</a>
    </div>

We combine all of these behavioral tests to create a behavioral report (`tagifai.eval.get_behavioral_report()`) which quantifies how many of these tests are passed by a particular instance of a trained model. This report is then saved along with the run's artifacts so we can use this information when choosing which model(s) to deploy to production.

```json linenums="1" hl_lines="2 4 41"
{
  "score": 1.0,
  "results": {
    "passed": [
      {
        "input": {
          "text": "Transformers have revolutionized the ML field.",
          "tags": [
            "transformers"
          ]
        },
        "prediction": {
          "input_text": "Transformers have revolutionized the ML field.",
          "preprocessed_text": "transformers revolutionized ml field",
          "predicted_tags": [
            "natural-language-processing",
            "transformers"
          ]
        },
        "type": "INV"
      },
      ...
      {
        "input": {
          "text": "graph neural networks have revolutionized machine learning.",
          "tags": [
            "graph-neural-networks"
          ]
        },
        "prediction": {
          "input_text": "graph neural networks have revolutionized machine learning.",
          "preprocessed_text": "graph neural networks revolutionized machine learning",
          "predicted_tags": [
            "graph-neural-networks",
            "graphs"
          ]
        },
        "type": "MFT"
      }
    ],
    "failed": []
  }
}
```

!!! warning
    When you create additional behavioral tests, be sure to reevaluate all the models you're considering on the new set of tests so their scores can be compared. We can do this since behavioral tests are not dependent on data or model versions and are simply tests that treat the model as a black box.
    ```bash linenums="1"
    tagifai behavioral-reevaluation --experiment-name=best --all-runs  # update all runs in experiment
    tagifai behavioral-reevaluation --run-id=0deb534  # update specific run
    ```

### Sorted runs

We can combine our overall / slice metrics and our behavioral tests to create a holistic evaluation report for each model run. We can then use this information to choose which model(s) to deploy to production.

<div class="animated-code">

    ```console
    # Get sorted runs
    $ tagifai get-sorted-runs --experiment-name=best
    +--------+--------------+----------+-----------+------------------+
    | run_id | data_version |       f1 | slices_f1 | behavioral_score |
    |--------+--------------+----------+-----------+------------------|
    | 37055e |        0.0.1 | 0.713325 |      0.78 |                1 |
    | 89fd3j |        0.0.1 | 0.697445 |      0.77 |            0.875 |
    | njk34d |        0.0.1 | 0.689365 |      0.73 |              0.9 |
    +--------+--------------+----------+-----------+------------------+
    ```

</div>
<script src="../../../static/js/termynal.js"></script>

### Deployment

There are also a whole class of model tests that are beyond metrics or behavioral testing and focus on the system as a whole. Many of them involve testing and benchmarking the [tradeoffs](baselines.md#tradeoffs){:target="_blank"} (ex. latency, compute, etc.) we discussed from the [baselines](baselines.md){:target="_blank"} lesson. These tests also need to performed across the different systems (ex. devices) that our model may be on. For example, development may happen on a CPU but the deployed model may be loaded on a GPU and there may be incompatible components (ex. reparametrization) that may cause errors. As a rule of thumb, we should test with the system specifications that our production environment utilizes.

!!! note
    We'll automate tests on different devices in our [CI/CD lesson](cicd.cd){:target="_blank"} where we'll use GitHub Actions to spin up our application with Docker Machine on cloud compute instances (we'll also use this for training).

Once we've tested our model's ability to perform in the production environment (**offline tests**), we can run several types of **online tests** to determine the quality of that performance.

- `#!js AB tests`:
    - sending production traffic to the different systems.
    - involves statistical hypothesis testing to decide which system is better.
    - need to account for different sources of bias (ex. novelty effect).
    - multiarmed bandits might be better if optimizing on a certain metric.
- `#!js Shadow tests`:
    - sending the same production traffic to the different systems.
    - safe online evaluation as the new system's results are not served.
    - easy to monitor, validate operational consistency, etc.

## Testing vs. monitoring

We'll conclude by talking about the similarities and distinctions between testing and monitoring. They're both integral parts of the ML development pipeline and depend on each other for iteration. Testing is assuring that our system (code, data and models) behaves the way we intend at the current time $t_0$. Whereas, monitoring is ensuring that the conditions (ie. distributions) during development are maintained and also that the tests that passed at $t_0$ continue to hold true post deployment through $t_n$. When this is no longer true, we need to inspect more closely (retraining may not always fix our root problem).

With monitoring, there are quite a few distinct concerns that we didn't have to consider during testing since it involves (live) data we have yet to see.

- features and prediction distributions (drift), typing, schema mismatches, etc.
- determining model performance (rolling and window metrics on overall and slices of data) using indirect signals (since labels may not be readily available).
- in situations with large data, we need to know which data points to label and upsample for training.
- identifying anomalies and outliers.

We'll cover all of these concepts in much more depth (and code) in our [monitoring](monitoring.md){:target="_blank"} lesson.

## Resources
- [Great Expectations](https://github.com/great-expectations/great_expectations){:target="_blank"}
- [The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/aad9f93b86b7addfea4c419b9100c6cdd26cacea.pdf){:target="_blank"}
- [Beyond Accuracy: Behavioral Testing of NLP Models with CheckList](https://arxiv.org/abs/2005.04118){:target="_blank"}
- [A Recipe for Training Neural Networks](http://karpathy.github.io/2019/04/25/recipe/){:target="_blank"}
- [Effective testing for machine learning systems](https://www.jeremyjordan.me/testing-ml/){:target="_blank"}
- [Slice-based Learning: A Programming Model for Residual Learning in Critical Data Slices](https://papers.nips.cc/paper/2019/file/351869bde8b9d6ad1e3090bd173f600d-Paper.pdf){:target="_blank"}
- [Robustness Gym: Unifying the NLP Evaluation Landscape](https://arxiv.org/abs/2101.04840){:target="_blank"}

<!-- Citation -->
{% include "cite.md" %}