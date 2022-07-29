---
template: lesson.html
title: Workflow Orchestration for Machine Learning
description: Create, schedule and monitor workflows by creating scalable pipelines.
keywords: airflow, prefect, dagster, workflows, pipelines, orchestration, dataops, data warehouse, database, great expectations, data validation, spark, ci/cd, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/mlops.png
repository: https://github.com/GokuMohandas/mlops-course
---

{% include "styles/lesson.md" %}
`
## Intuition

So far we've implemented our DataOps (ETL, preprocessing, validation, etc.) and MLOps (optimization, training, evaluation, etc.) workflows as Python function calls. This has worked well since our dataset is static and small. But happens when we need to:

- **schedule** these workflows as new data arrives?
- **scale** these workflows as our data grows?
- **share** these workflows to downstream consumers?
- **monitor** these workflows?

We'll need to break down our end-to-end ML pipeline into individual workflows that be orchestrated as needed. There are several tools that can help us so this such as [Airflow](http://airflow.apache.org/){:target="_blank"}, [Prefect](https://www.prefect.io/){:target="_blank"}, [Dagster](https://dagster.io/){:target="_blank"}, [Luigi](https://luigi.readthedocs.io/en/stable/){:target="_blank"} and even some ML focused options such as [Metaflow](){}, [Flyte](){}, [KubeFlow Pipelines](https://www.kubeflow.org/docs/components/pipelines/overview/pipelines-overview/){:target="_blank"}, [Vertex pipelines](https://cloud.google.com/vertex-ai/docs/pipelines/introduction){:target="_blank"}, etc. We'll be creating our workflows using AirFlow for its:

- wide adoption of the open source platform in industry
- Python based software development kit (SDK)
- integration with the ecosystem (data ingestion, processing, etc.)
- ability to run locally and scale easily
- maturity over the years and part of the apache ecosystem

> We'll be running Airflow locally but we can easily scale it by running on a managed cluster platform where we can run Python, Hadoop, Spark, etc. on large batch processing jobs (AWS [EMR](https://aws.amazon.com/emr/){:target="_blank"}, Google Cloud's [Dataproc](https://cloud.google.com/dataproc){:target="_blank"}, on-prem hardware, etc.).

## Airflow

Before we create our specific pipelines, let's understand and implement [Airflow](https://airflow.apache.org/){:target="_blank"}'s overarching concepts that will allow us to "author, schedule, and monitor workflows".

## Install

To install and run Airflow, we can either do so [locally](https://airflow.apache.org/docs/apache-airflow/stable/start/local.html){:target="_blank"} or with [Docker](https://airflow.apache.org/docs/apache-airflow/stable/start/docker.html){:target="_blank"}. If using `docker-compose` to run Airflow inside Docker containers, we'll want to allocate at least 4 GB in memory.

```bash
# Configurations
export AIRFLOW_HOME=${PWD}/airflow
AIRFLOW_VERSION=2.3.3
PYTHON_VERSION="$(python --version | cut -d " " -f 2 | cut -d "." -f 1-2)"
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

# Install Airflow (may need to upgrade pip)
pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"

# Initialize DB (SQLite by default)
airflow db init
```

This will create an `airflow` directory with the following components:

```bash
airflow/
├── logs/
└── airflow.cfg
├── airflow.db
├── unittests.cfg
└── webserver_config.py
```

We're going to edit the [airflow.cfg](https://github.com/GokuMohandas/mlops-course/blob/main/airflow/airflow.cfg){:target="_blank"} file to best fit our needs:
```bash
# Inside airflow.cfg
enable_xcom_pickling = True  # needed for Great Expectations airflow provider
load_examples = False  # don't clutter webserver with examples
```

And we'll perform a reset to implement these configuration changes.

```bash
airflow db reset -y
```

Now we're ready to initialize our database with an admin user, which we'll use to login to access our workflows in the webserver.

```bash
# We'll be prompted to enter a password
airflow users create \
    --username admin \
    --firstname FIRSTNAME \
    --lastname LASTNAME \
    --role Admin \
    --email EMAIL
```

## Webserver

Once we've created a user, we're ready to launch the webserver and log in using our credentials.

```bash
# Launch webserver
export AIRFLOW_HOME=${PWD}/airflow
airflow webserver --port 8080  # http://localhost:8080
```

The webserver allows us to run and inspect workflows, establish connections to external data storage, manager users, etc. through a UI. Similarly, we could also use Airflow's [REST API](https://airflow.apache.org/docs/apache-airflow/stable/stable-rest-api-ref.html){:target="_blank"} or [Command-line interface (CLI)](https://airflow.apache.org/docs/apache-airflow/stable/cli-and-env-variables-ref.html){:target="_blank"} to perform the same operations. However, we'll be using the webserver because it's convenient to visually inspect our workflows.

<div class="ai-center-all">
    <img src="/static/images/mlops/orchestration/webserver.png" width="700" alt="airflow webserver">
</div>

We'll explore the different components of the webserver as we learn about Airflow and implement our workflows.

## Scheduler

Next, we need to launch our scheduler, which will execute and monitor the tasks in our workflows. The schedule executes tasks by reading from the metadata database and ensures the task has what it needs to finish running. We'll go ahead and execute the following commands on the *separate terminal* window:

```bash
# Launch scheduler (in separate terminal)
export AIRFLOW_HOME=${PWD}/airflow
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
airflow scheduler
```

## Executor

As our scheduler reads from the metadata database, the executor determines what worker processes are necessary for the task to run to completion. Since our default database SQLlite, which can't support multiple connections, our default executor is the [Sequential Executor](https://airflow.apache.org/docs/apache-airflow/stable/executor/sequential.html){:target="_blank"}. However, if we choose a more production-grade database option such as PostgresSQL or MySQL, we can choose scalable [Executor backends](https://airflow.apache.org/docs/apache-airflow/stable/executor/index.html#supported-backends){:target="_blank"} Celery, Kubernetes, etc. For example, running [Airflow with Docker](https://airflow.apache.org/docs/apache-airflow/stable/start/docker.html){:target="_blank"} uses PostgresSQL as the database and so uses the Celery Executor backend to run tasks in parallel.


## DAGs

Workflows are defined by directed acyclic graphs (DAGs), whose nodes represent tasks and edges represent the data flow relationship between the tasks. Direct and acyclic implies that workflows can only execute in one direction and a previous, upstream task cannot run again once a downstream task has started.

<div class="ai-center-all">
    <img src="/static/images/mlops/orchestration/basic_dag.png" width="250" alt="basic DAG">
</div>

DAGs can be defined inside Python workflow scripts inside the `airflow/dags` directory and they'll automatically appear (and continuously be updated) on the webserver. Before we start creating our DataOps and MLOps workflows, we'll learn about Airflow's concepts via an example DAG outlined in [airflow/dags/example.py](https://github.com/GokuMohandas/mlops-course/blob/main/airflow/dags/example.py){:target="_blank"}. Execute the following commands in a new (3rd) terminal window:

```bash
mkdir airflow/dags
touch airflow/dags/example.py
```

Inside each workflow script, we can define some default arguments that will apply to all DAGs within that workflow.

```python linenums="1"
# Default DAG args
default_args = {
    "owner": "airflow",
}
```

> There are many [more default arguments](https://airflow.apache.org/docs/apache-airflow/stable/tutorial.html#default-arguments){target="_blank"} and we'll cover them as we go through the concepts.

We can initialize DAGs with many [parameters](https://airflow.apache.org/docs/apache-airflow/stable/_api/airflow/models/dag/index.html#airflow.models.dag.DAG){:target="_blank"} (which will override the same parameters in `default_args`) and in several different ways:

- using a [with statement](https://docs.python.org/3/reference/compound_stmts.html#the-with-statement){target="_blank"}
```python linenums="1"
from airflow import DAG

with DAG(
    dag_id="example",
    description="Example DAG",
    default_args=default_args,
    schedule_interval=None,
    start_date=days_ago(2),
    tags=["example"],
) as example:
    # Define tasks
    pass
```

- using the [dag decorator](https://airflow.apache.org/docs/apache-airflow/stable/concepts.html#dag-decorator){target="_blank"}
```python linenums="1"
from airflow.decorators import dag

@dag(
    dag_id="example",
    description="Example DAG",
    default_args=default_args,
    schedule_interval=None,
    start_date=days_ago(2),
    tags=["example"],
)
def example():
    # Define tasks
    pass
```

> There are many [parameters](https://airflow.apache.org/docs/apache-airflow/stable/_api/airflow/models/dag/index.html#airflow.models.dag.DAG){:target="_blank"} that we can initialize our DAGs with, including a `start_date` and a `schedule_interval`. While we could have our workflows execute on a temporal cadence, many ML workflows are initiated by events, which we can map using [sensors](https://airflow.apache.org/docs/apache-airflow/stable/_api/airflow/sensors/index.html){:target="_blank"} and [hooks](https://airflow.apache.org/docs/apache-airflow/stable/_api/airflow/hooks/index.html){:target="_blank"} to external databases, file systems, etc.

## Tasks

Tasks are the operations that are executed in a workflow and are represented by nodes in a DAG. Each task should be a clearly defined single operation and it should be idempotent, which means we can execute it multiple times and expect the same result and system state. This is important in the event we need to retry a failed task and don't have to worry about resetting the state of our system. Like DAGs, there are several different ways to implement tasks:

- using the [task decorator](https://airflow.apache.org/docs/apache-airflow/stable/concepts.html#concepts-task-decorator){:target="_blank"}
```python linenums="1"
from airflow.decorators import dag, task
from airflow.utils.dates import days_ago

@dag(
    dag_id="example",
    description="Example DAG with task decorators",
    default_args=default_args,
    schedule_interval=None,
    start_date=days_ago(2),
    tags=["example"],
)
def example():
    @task
    def task_1():
        return 1
    @task
    def task_2(x):
        return x+1
```

- using [Operators](https://airflow.apache.org/docs/apache-airflow/stable/_api/airflow/operators/index.html){:target="_blank"}
```python linenums="1"
from airflow.decorators import dag
from airflow.operators.bash_operator import BashOperator
from airflow.utils.dates import days_ago

@dag(
    dag_id="example",
    description="Example DAG with Operators",
    default_args=default_args,
    schedule_interval=None,
    start_date=days_ago(2),
    tags=["example"],
)
def example():
    # Define tasks
    task_1 = BashOperator(task_id="task_1", bash_command="echo 1")
    task_2 = BashOperator(task_id="task_2", bash_command="echo 2")
```

> Though the graphs are directed, we can establish certain [trigger rules](https://airflow.apache.org/docs/apache-airflow/stable/concepts.html#trigger-rules){:target="_blank"} for each task to execute on conditional successes or failures of the parent tasks.

### Operators

The first method of creating tasks involved using Operators, which defines what exactly the task will be doing. Airflow has many built-in Operators such as the [BashOperator](https://airflow.apache.org/docs/apache-airflow/stable/_api/airflow/operators/bash/index.html#airflow.operators.bash.BashOperator){:target="_blank"} or [PythonOperator](https://airflow.apache.org/docs/apache-airflow/stable/_api/airflow/operators/python/index.html#airflow.operators.python.PythonOperator){target="_blank"}, which allow us to execute bash and Python commands respectively.

```python linenums="1"
# BashOperator
from airflow.operators.bash_operator import BashOperator
task_1 = BashOperator(task_id="task_1", bash_command="echo 1")

# PythonOperator
from airflow.operators.python import PythonOperator
task_2 = PythonOperator(
    task_id="task_2",
    python_callable=foo,
    op_kwargs={"arg1": ...})
```

There are also many other Airflow native [Operators](https://airflow.apache.org/docs/apache-airflow/stable/_api/airflow/operators/index.html){:target="_blank"} (email, S3, MySQL, Hive, etc.), as well as [community maintained provider packages](https://airflow.apache.org/docs/apache-airflow-providers/packages-ref.html){:target="_blank"} (Kubernetes, Snowflake, Azure, AWS, Salesforce, Tableau, etc.), to execute tasks specific to certain platforms or tools.

> We can also create our own [custom Operators](https://airflow.apache.org/docs/apache-airflow/stable/howto/custom-operator.html){:target="_blank"} by extending the [BashOperator](https://airflow.apache.org/docs/apache-airflow/stable/_api/airflow/operators/bash/index.html#airflow.operators.bash.BashOperator){:target="_blank"} class.

### Relationships

Once we've defined our tasks using Operators or as decorated functions, we need to define the relationships between them (edges). The way we define the relationships depends on how our tasks were defined:

- using decorated functions
```python linenums="1"
# Task relationships
x = task_1()
y = task_2(x=x)
```

- using Operators
```python linenums="1"
# Task relationships
task_1 >> task_2  # same as task_1.set_downstream(task_2) or
                  # task_2.set_upstream(task_1)
```

In both scenarios, we'll setting `task_2` as the downstream task to `task_1`.

!!! note
    We can even create intricate DAGs by using these notations to define the relationships.

    ```python linenums="1"
    task_1 >> [task_2_1, task_2_2] >> task_3
    task_2_2 >> task_4
    [task_3, task_4] >> task_5
    ```
    <div class="ai-center-all">
        <img src="/static/images/mlops/orchestration/dag.png" width="500" alt="DAG">
    </div>

### XComs

When we use task decorators, we can see how values can be passed between tasks. But, how can we pass values when using Operators? Airflow uses XComs (cross communications) objects, defined with a key, value, timestamp and task_id, to push and pull values between tasks. When we use decorated functions, XComs are being used under the hood but it's abstracted away, allowing us to pass values amongst Python functions seamlessly. But when using Operators, we'll need to explicitly push and pull the values as we need it.

```python linenums="1" hl_lines="3 6 8"
def _task_1(ti):
    x = 2
    ti.xcom_push(key="x", value=x)

def _task_2(ti):
    x = ti.xcom_pull(key="x", task_ids=["task_1"])[0]
    y = x + 3
    ti.xcom_push(key="y", value=y)

@dag(
    dag_id="example",
    description="Example DAG",
    default_args=default_args,
    schedule_interval=None,
    start_date=days_ago(2),
    tags=["example"],
)
def example2():
    # Tasks
    task_1 = PythonOperator(task_id="task_1", python_callable=_task_1)
    task_2 = PythonOperator(task_id="task_2", python_callable=_task_2)
    task_1 >> task_2
```

We can also view our XComs on the webserver by going to **Admin** >> **XComs**:

<div class="ai-center-all">
    <img src="/static/images/mlops/orchestration/xcoms.png" width="700" alt="xcoms">
</div>

!!! warning
    The data we pass between tasks should be small (metadata, metrics, etc.) because Airflow's metadata database is not equipped to hold large artifacts. However, if we do need to store and use the large results of our tasks, it's best to use an external data storage (blog storage, model registry, etc.).


## DAG runs

Once we've defined the tasks and their relationships, we're ready to run our DAGs. We'll start defining our DAG like so:
```python linenums="1"
# Run DAGs
example1_dag = example_1()
example2_dag = example_2()
```

If we refresh our webserver page ([http://localhost:8080/](http://localhost:8080/){:target="_blank"}), the new DAG will have appeared.

### Manual
Our DAG is initially paused since we specified `dags_are_paused_at_creation = True` inside our [airflow.cfg](https://github.com/GokuMohandas/mlops-course/blob/main/airflow/airflow.cfg){:target="_blank"} configuration, so we'll have to manually execute this DAG by clicking on it > unpausing it (toggle) > triggering it (button). To view the logs for any of the tasks in our DAG run, we can click on the task > Log.

<div class="ai-center-all">
    <img src="/static/images/mlops/orchestration/trigger.png" width="700" alt="triggering a DAG">
</div>

!!! note
    We could also use Airflow's [REST API](https://airflow.apache.org/docs/apache-airflow/stable/stable-rest-api-ref.html){:target="_blank"} (will [configured authorization](https://airflow.apache.org/docs/apache-airflow/stable/security/api.html){:target="_blank"}) or [Command-line interface (CLI)](https://airflow.apache.org/docs/apache-airflow/stable/cli-and-env-variables-ref.html){:target="_blank"} to inspect and trigger workflows (and a whole lot more). Or we could even use the [`trigger_dagrun`](https://airflow.apache.org/docs/apache-airflow/stable/_api/airflow/operators/trigger_dagrun/index.html){:target="_blank"} Operator to trigger DAGs from within another workflow.

    ```bash
    # CLI to run dags
    airflow dags trigger <DAG_ID>
    ```

### Interval
Had we specified a `start_date` and `schedule_interval` when defining the DAG, it would have have automatically executed at the appropriate times. For example, the DAG below will have started two days ago and will be triggered at the start of every day.

```python linenums="1"
from airflow.decorators import dag
from airflow.utils.dates import days_ago
from datetime import timedelta

@dag(
    dag_id="example",
    default_args=default_args,
    schedule_interval=timedelta(days=1),
    start_date=days_ago(2),
    tags=["example"],
    catch_up=False,
)
```

!!! warning
    Depending on the `start_date` and `schedule_interval`, our workflow should have been triggered several times and Airflow will try to [catchup](https://airflow.apache.org/docs/apache-airflow/stable/dag-run.html#catchup){:target="_blank"} to the current time. We can avoid this by setting `catchup=False` when defining the DAG. We can also set this configuration as part of the default arguments:

    ```python linenums="1" hl_lines="3"
    default_args = {
        "owner": "airflow",
        "catch_up": False,
    }
    ```

    However, if we did want to run particular runs in the past, we can manually [backfill](https://airflow.apache.org/docs/apache-airflow/stable/dag-run.html#backfill){:target="_blank"} what we need.


We could also specify a [cron](https://crontab.guru/){:target="_blank"} expression for our `schedule_interval` parameter or even use [cron presets](https://airflow.apache.org/docs/apache-airflow/stable/dag-run.html#cron-presets){:target="_blank"}.


> Airflow's Scheduler will run our workflows one `schedule_interval` from the `start_date`. For example, if we want our workflow to start on `01-01-1983` and run `@daily`, then the first run will be immediately after `01-01-1983T11:59`.

### Sensors

While it may make sense to execute many data processing workflows on a scheduled interval, machine learning workflows may require more nuanced triggers. We shouldn't be wasting compute by running executing our workflows *just in case* we have new data. Instead, we can use [sensors](https://airflow.apache.org/docs/apache-airflow/stable/_api/airflow/sensors/){:target="_blank"} to trigger workflows when some external condition is met. For example, we can initiate data processing when a new batch of annotated data appears in a database or when a specific file appears in a file system, etc.

> There's so much more to Airflow (monitoring, Task groups, smart senors, etc.) so be sure to explore them as you need them by using the [official documentation](https://airflow.apache.org/docs/apache-airflow/stable/index.html){:target="_blank"}.


## DataOps

Now that we've reviewed Airflow's major concepts, we're ready to create the DataOps workflow for our application. It involves a series of tasks around extraction, transformation, loading, validation, etc. We're going to use a simplified data stack (local file, validation, etc.) as opposed to a production [data stack](data-stack.md){:target="_blank"} but the overall workflows are similar. Instead of extracting data from a source, validating and transforming it and then loading into a data warehouse, we're going to perform ETL from a local file and load the processed data into another local file.

```bash
touch airflow/dags/workflows.py
```
```python linenums="1"
from airflow.decorators import dag, task
from airflow.utils.dates import days_ago

# Default DAG args
default_args = {
    "owner": "airflow",
    "catch_up": False,
}

# Define DAG
@dag(
    dag_id="DataOps",
    description="DataOps tasks.",
    default_args=default_args,
    schedule_interval=None,
    start_date=days_ago(2),
    tags=["dataops"],
)
def dataops():
    pass
```

!!! note "ETL vs. ELT"
    If using a data warehouse (ex. Snowflake), it's common to see ELT (extract-load-transform) workflows to have a permanent location for all historical data. Learn more about the data stack and the different workflow options [here](data-stack.md){:target="_blank"}.

### Extraction

To keep things simple, we'll continue to keep our data as a local file but in a real production setting, our data can come from a wide variety of [data management systems](infrastructure.md#data-management-sytems){:target="_blank"}.

```python linenums="1"
def _extract():
    # Extract from source (ex. DB, API, etc.)
    projects = utils.load_json_from_url(url=config.PROJECTS_URL)  # NOQA: F841 (assigned by unused)
    tags = utils.load_json_from_url(url=config.TAGS_URL)  # NOQA: F841 (assigned by unused)

@dag(...)
def dataops():
    extract = PythonOperator(task_id="extract", python_callable=_extract)
```

> Typically we'll use [sensors](https://airflow.apache.org/docs/apache-airflow/stable/_api/airflow/sensors/){:target="_blank"} to trigger workflows when a condition is met or trigger them directly from the external source via API calls, etc. Our workflows can communicate with the different platforms by establishing a [connection](https://airflow.apache.org/docs/apache-airflow/stable/howto/connection.html){:target="_blank"} and then using [hooks](https://airflow.apache.org/docs/apache-airflow/stable/_api/airflow/hooks/index.html){:target="_blank"} to interface with the database, data warehouse, etc.


### Validation

The specific process of where and how we extract our data can be bespoke but what's important is that we have a continuous integration to execute our workflows. A key aspect to trusting this continuous integration is validation at every step of the way. We'll once again use [Great Expectations](https://greatexpectations.io/){:target="_blank"}, as we did in our [testing lesson](testing.md#data){:target="_blank"}, to [validate](testing.md#expectations){:target="_blank"} our incoming data before transforming it.

With the Airflow concepts we've learned so far, there are many ways to use our data validation library to validate our data. Regardless of what data validation tool we use (ex. [Great Expectations](https://greatexpectations.io/){:target="_blank"}, [TFX](https://www.tensorflow.org/tfx/data_validation/get_started){:target="_blank"}, [AWS Deequ](https://github.com/awslabs/deequ){:target="_blank"}, etc.) we could use the BashOperator, PythonOperator, etc. to run our tests. However, Great Expectations has a [Airflow Provider package](https://github.com/great-expectations/airflow-provider-great-expectations){:target="_blank"} to make it even easier to validate our data. This package contains a `GreatExpectationsOperator` which we can use to execute specific checkpoints as tasks.

Recall from our testing lesson that we used the following CLI commands to perform our data validation tests:

```bash
great_expectations checkpoint run projects
great_expectations checkpoint run tags
```

We can perform the same operations as Airflow tasks within our DataOps workflow, either with:

- bash commands using the [BashOperator](https://airflow.apache.org/docs/apache-airflow/stable/_api/airflow/operators/bash/index.html#airflow.operators.bash.BashOperator){:target="_blank"}:

```python linenums="1"
from airflow.operators.bash_operator import BashOperator
validate_projects = BashOperator(task_id="validate_projects", bash_command="great_expectations checkpoint run projects")
validate_tags = BashOperator(task_id="validate_tags", bash_command="great_expectations checkpoint run tags")
```

- with the [custom Great Expectations operator](https://github.com/great-expectations/airflow-provider-great-expectations){:target="_blank"}:

```bash
pip install airflow-provider-great-expectations==0.1.1
```

```python linenums="1"
from great_expectations_provider.operators.great_expectations import GreatExpectationsOperator

# Validate data
validate_projects = GreatExpectationsOperator(
    task_id="validate_projects",
    checkpoint_name="projects",
    data_context_root_dir="tests/great_expectations",
    fail_task_on_validation_failure=True,
)
validate_tags = GreatExpectationsOperator(
    task_id="validate_tags",
    checkpoint_name="tags",
    data_context_root_dir="tests/great_expectations",
    fail_task_on_validation_failure=True,
)
```

And we want both tasks to pass so we set the `fail_task_on_validation_failure` parameter to `True` so that downstream tasks don't execute if either fail.

!!! note
    Reminder that we previously set the following configuration in our [airflow.cfg](https://github.com/GokuMohandas/mlops-course/blob/main/airflow/airflow.cfg){:target="_blank"} file since the output of the GreatExpectationsOperator is not JSON serializable.
    ```bash
    # Inside airflow.cfg
    enable_xcom_pickling = True
    ```

### Load

Once we've validated our data, we're ready to load it into our data system (ex. data warehouse). This will be the primary system that potential downstream applications will depend on for current and future versions of data.

```python linenums="1"
def _load():
    # Load into data system (ex. warehouse)
    projects_fp = Path(config.DATA_DIR, "projects.json")
    df = pd.DataFrame(get_projects())
    utils.save_dict(d=df.to_dict(orient="records"), filepath=projects_fp)
    tags_fp = Path(config.DATA_DIR, "tags.json")
    utils.save_dict(d=get_tags(), filepath=tags_fp)

@dag(...)
def dataops():
    ...
    load = PythonOperator(task_id="load", python_callable=_load)
```

### Transform

Once we have validated and loaded our data, we're ready to transform it. Our DataOps workflows are not specific to any particular downstream consumer so the transformation must be globally relevant (ex. cleaning missing date, aggregation, etc.).  We have a wide variety of Operators to choose from depending on the tools we're using for compute (ex. [Python](https://airflow.apache.org/docs/apache-airflow/stable/_api/airflow/operators/python/index.html#airflow.operators.python.PythonOperator){target="_blank"}, [Spark](https://airflow.apache.org/docs/apache-airflow-providers-apache-spark/stable/operators.html){:target="_blank"}, [DBT](https://github.com/gocardless/airflow-dbt){:target="_blank"}, etc.). Many of these options have the advantage of directly performing the transformations in our data warehouse.

```python linenums="1"
def _transform():
    # Transform (ex. using DBT inside DWH)
    df = pd.DataFrame(get_projects())
    df = df[df.tag.notnull()]  # drop rows w/ no tag
    projects_fp = Path(config.DATA_DIR, "projects.json")
    utils.save_dict(d=df.to_dict(orient="records"), filepath=projects_fp)

@dag(...)
def dataops():
    ...
    transform = PythonOperator(task_id="transform", python_callable=_transform)
    validate_transforms = GreatExpectationsOperator(
        task_id="validate_transforms",
        checkpoint_name="projects",
        data_context_root_dir="tests/great_expectations",
        fail_task_on_validation_failure=True,
    )
```

<hr>

```python linenums="1"
# Define DAG
(
    extract
    >> [validate_projects, validate_tags]
    >> load
    >> transform
    >> validate_transforms
)
```

<div class="ai-center-all">
    <img src="/static/images/mlops/orchestration/dataops.png" width="1000" alt="dataops workflow">
</div>

## MLOps

Once we have our features in our feature store, we can use them for MLOps tasks responsible for model creating such as optimization, training, validation, serving, etc.

<div class="ai-center-all">
    <img src="/static/images/mlops/orchestration/model.png" width="1000" alt="mlops model training workflow">
</div>

### Extract data

The first step is to extract the relevant historical features from our feature store/DB/DWH that our DataOps workflow cached to.

```python linenums="1"
# Extract features
extract_features = PythonOperator(
    task_id="extract_features",
    python_callable=main.get_historical_features,
)
```

We'd normally need to provide a set of entities and the timestamp to extract point-in-time features for each of them but here's what a simplified version to extract the most up-to-date features for a set of entities would look like:

```python linenums="1" hl_lines="9 14"
# tagifai/main.py
@app.command()
def get_historical_features():
    """Retrieve historical features for training."""
    # Entities to pull data for (should dynamically read this from somewhere)
    project_ids = [1, 2, 3]
    now = datetime.now()
    timestamps = [datetime(now.year, now.month, now.day)] * len(project_ids)
    entity_df = pd.DataFrame.from_dict({"id": project_ids, "event_timestamp": timestamps})

    # Get historical features
    store = FeatureStore(repo_path=Path(config.BASE_DIR, "features"))
    training_df = store.get_historical_features(
        entity_df=entity_df,
        feature_refs=["project_details:text", "project_details:tags"],
    ).to_df()
    logger.info(training_df.head())
    return training_df
```

### Training

Once we have our features, we can use them to optimize and train the best models. Since these tasks can require lots of compute, we would typically run this entire pipeline in a managed cluster platform which can scale up as our data and models grow.

```python linenums="1"
# Optimization
optimization = BashOperator(
    task_id="optimization",
    bash_command="tagifai optimize",
)
```
```python linenums="1"
# Training
train_model = BashOperator(
    task_id="train_model",
    bash_command="tagifai train-model",
)
```

### Evaluation

It's imperative that we evaluate our trained models so we can be confident in it's ability. We've extensively covered [model evaluation](testing.md#evaluation){:target="_blank"} in our [testing lesson](testing.md){:target="_blank"}, so here we'll talk about what happens after evaluation. We want to execute a chain of tasks depending on whether the model improved or regressed. To do this, we're using a [BranchPythonOperator](https://airflow.apache.org/docs/apache-airflow/stable/concepts.html?highlight=branch#branching){:target="_blank"} for the evaluation task so it can return the appropriate task id.

```python linenums="1"
from airflow.operators.python import BranchPythonOperator

# Evaluate
evaluate_model = BranchPythonOperator(  # BranchPythonOperator returns a task_id or [task_ids]
    task_id="evaluate_model",
    python_callable=_evaluate_model,
)
```

This Operator will execute a function whose return response will be a single (or a list) task id.

```python linenums="1"
def _evaluate_model():
    if improvement_criteria():
        return "improved"  # improved is a task id
    else:
        return "regressed"  # regressed is a task id
```

```python linenums="1"
# Improved or regressed
improved = BashOperator(
    task_id="improved",
    bash_command="echo IMPROVED",
)
regressed = BashOperator(
    task_id="regressed",
    bash_command="echo REGRESSED",
)
```

The returning task ids can correspond to tasks that are simply used to direct the workflow towards a certain set of tasks based on upstream results. In our case, we want to serve the improved model or log a report in the event of a regression.

### Serving

If our model passed our evaluation criteria then we can deploy and serve our model. Again, there are many different options here such as using our CI/CD Git workflows to deploy the model wrapped as a scalable microservice or for more streamlined deployments, we can use a purpose-build [model server](api.md#model-server){:target="_blank"} to seamlessly inspect, update, serve, rollback, etc. multiple versions of models.

```python linenums="1"
# Serve model(s)
serve = BashOperator(
    task_id="serve_model",
    bash_command="echo served model",
)
```

<hr>

```python linenums="1"
# Task relationships
etl_data >> optimization >> train >> evaluate >> [improved, regressed]
improved >> serve
regressed >> report
```

## MLOps (update)

Once we've validated and served our model, how do we know *when* and *how* it needs to be updated? We'll need to compose a set of workflows that reflect the update policies we want to set in place.

<div class="ai-center-all">
    <img src="/static/images/mlops/orchestration/update.png" width="1000" alt="mlops model update workflow">
</div>

### Monitoring
Our inference application will receive requests from our user-facing application and we'll use our versioned model artifact to return inference responses. All of these inputs, predictions and other values will also be sent (batch/real-time) to a monitoring workflow that ensures the health of our system. We have already covered the foundations of monitoring in our [monitoring lesson](monitoring.md){:target="_blank"} but here we'll look at how triggered alerts fit with the overall operational workflow.

### Policies
Based on the metrics we're monitoring using various thresholds, window sizes, frequencies, etc., we'll be able to trigger events based on our update policy engine.

- `#!js continue`: with the currently deployed model without any updates. However, an alert was raised so it should analyzed later to reduce false positive alerts.
- `#!js improve`: by retraining the model to avoid performance degradation causes by meaningful drift (data, target, concept, etc.).
- `#!js inspect`: to make a decision. Typically expectations are reassessed, schemas are reevaluated for changes, slices are reevaluated, etc.
- `#!js rollback`: to a previous version of the model because of an issue with the current deployment. Typically these can be avoided using robust deployment strategies (ex. dark canary).

### Retraining
If we need to improve on the existing version of the model, it's not just the matter of fact of rerunning the model creation workflow on the new dataset. We need to carefully compose the training data in order to avoid issues such as catastrophic forgetting (forget previously learned patterns when presented with new data).

- `#!js labeling`: new incoming data may need to be properly labeled before being used (we cannot just depend on proxy labels).
- `#!js active learning`: we may not be able to explicitly label every single new data point so we have to leverage [active learning](labeling.md#active-learning){:target="_blank"} workflows to complete the labeling process.
- `#!js QA`: quality assurance workflows to ensure that labeling is accurate, especially for known false positives/negatives and historically poorly performing slices of data.
- `#!js augmentation`: increasing our training set with [augmented data](augmentation.md){:target="_blank"} that's representative of the original dataset.
- `#!js sampling`: upsampling and downsampling to address imbalanced data slices.
- `#!js evaluation`:  creation of an evaluation dataset that's representative of what the model will encounter once deployed.

Once we have the proper dataset for retraining, we can kickoff the featurization and model training workflows where a new model will be trained and evaluated before being deployed and receiving new inference requests. In the [next lesson](continual-learning.md){:target="_blank"}, we'll discuss how to combine these pipelines together to create a continual learning system.

<hr>

```python linenums="1"
# Task relationships
monitoring >> update_policy_engine >> [_continue, inspect, improve, rollback]
improve >> compose_retraining_dataset >> retrain
```

## References
- [Airflow official documentation](https://airflow.apache.org/docs/apache-airflow/stable/index.html){:target="_blank"}

<!-- Citation -->
{% include "cite.md" %}