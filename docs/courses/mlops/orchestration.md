---
template: lesson.html
title: Workflow Orchestration for Machine Learning
description: Create, schedule and monitor workflows by creating scalable pipelines.
keywords: airflow, prefect, dagster, workflows, pipelines, orchestration, dataops, data warehouse, database, great expectations, data validation, spark, ci/cd, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/mlops.png
repository: https://github.com/GokuMohandas/mlops-course
---

{% include "styles/lesson.md" %}

!!! danger
    I'm currently making major changes to this lesson, I'll announce all the updates on [Twitter](https://twitter.com/GokuMohandas){:target="_blank"} and [LinkedIn](https://linkedin.com/in/goku){:target="_blank"} in a few days!

## Intuition

So far we've implemented our DataOps (ETL, preprocessing, validation, etc.) and MLOps (optimization, training, evaluation, etc.) workflows as Python function calls. This has worked well since our dataset is static and small. But happens when we need to:

- **schedule** these workflows as new data arrives?
- **scale** these workflows as our data grows?
- **share** these workflows to downstream consumers?
- **monitor** these workflows?

We'll need to break down our end-to-end ML pipeline into individual workflows that be orchestrated as needed. There are several tools that can help us so this such as [Airflow](http://airflow.apache.org/){:target="_blank"}, [Prefect](https://www.prefect.io/){:target="_blank"}, [Dagster](https://dagster.io/){:target="_blank"}, [Luigi](https://luigi.readthedocs.io/en/stable/){:target="_blank"} and even some ML focused options such as [Metaflow](https://metaflow.org/){:target="_blank"}, [Flyte](https://flyte.org/){:target="_blank"}, [KubeFlow Pipelines](https://www.kubeflow.org/docs/components/pipelines/overview/pipelines-overview/){:target="_blank"}, [Vertex pipelines](https://cloud.google.com/vertex-ai/docs/pipelines/introduction){:target="_blank"}, etc. We'll be creating our workflows using AirFlow for its:

- wide adoption of the open source platform in industry
- Python based software development kit (SDK)
- integration with the ecosystem (data ingestion, processing, etc.)
- ability to run locally and scale easily
- maturity over the years and part of the apache ecosystem

> We'll be running Airflow locally but we can easily scale it by running on a managed cluster platform where we can run Python, Hadoop, Spark, etc. on large batch processing jobs (AWS [EMR](https://aws.amazon.com/emr/){:target="_blank"}, Google Cloud's [Dataproc](https://cloud.google.com/dataproc){:target="_blank"}, on-prem hardware, etc.).

## Airflow

Before we create our specific pipelines, let's understand and implement [Airflow](https://airflow.apache.org/){:target="_blank"}'s overarching concepts that will allow us to "author, schedule, and monitor workflows".

### Set up

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

We're going to edit the [airflow.cfg](https://github.com/GokuMohandas/data-engineering/blob/main/airflow/airflow.cfg){:target="_blank"} file to best fit our needs:
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

### Webserver

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

### Scheduler

Next, we need to launch our scheduler, which will execute and monitor the tasks in our workflows. The schedule executes tasks by reading from the metadata database and ensures the task has what it needs to finish running. We'll go ahead and execute the following commands on the *separate terminal* window:

```bash
# Launch scheduler (in separate terminal)
export AIRFLOW_HOME=${PWD}/airflow
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
airflow scheduler
```

### Executor

As our scheduler reads from the metadata database, the executor determines what worker processes are necessary for the task to run to completion. Since our default database SQLlite, which can't support multiple connections, our default executor is the [Sequential Executor](https://airflow.apache.org/docs/apache-airflow/stable/executor/sequential.html){:target="_blank"}. However, if we choose a more production-grade database option such as PostgresSQL or MySQL, we can choose scalable [Executor backends](https://airflow.apache.org/docs/apache-airflow/stable/executor/index.html#supported-backends){:target="_blank"} Celery, Kubernetes, etc. For example, running [Airflow with Docker](https://airflow.apache.org/docs/apache-airflow/stable/start/docker.html){:target="_blank"} uses PostgresSQL as the database and so uses the Celery Executor backend to run tasks in parallel.


## DAGs

Workflows are defined by directed acyclic graphs (DAGs), whose nodes represent tasks and edges represent the data flow relationship between the tasks. Direct and acyclic implies that workflows can only execute in one direction and a previous, upstream task cannot run again once a downstream task has started.

<div class="ai-center-all">
    <img src="/static/images/mlops/orchestration/basic_dag.png" width="250" alt="basic DAG">
</div>

DAGs can be defined inside Python workflow scripts inside the `airflow/dags` directory and they'll automatically appear (and continuously be updated) on the webserver. Before we start creating our DataOps and MLOps workflows, we'll learn about Airflow's concepts via an example DAG outlined in [airflow/dags/example.py](https://github.com/GokuMohandas/data-engineering/blob/main/airflow/dags/example.py){:target="_blank"}. Execute the following commands in a new (3rd) terminal window:

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
    The data we pass between tasks should be small (metadata, metrics, etc.) because Airflow's metadata database is not equipped to hold large artifacts. However, if we do need to store and use the large results of our tasks, it's best to use an external data storage (blog storage, model registry, etc.) and perform heavy processing using Spark or inside data systems like a data warehouse.


## DAG runs

Once we've defined the tasks and their relationships, we're ready to run our DAGs. We'll start defining our DAG like so:
```python linenums="1"
# Run DAGs
example1_dag = example_1()
example2_dag = example_2()
```

If we refresh our webserver page ([http://localhost:8080/](http://localhost:8080/){:target="_blank"}), the new DAG will have appeared.

### Manual
Our DAG is initially paused since we specified `dags_are_paused_at_creation = True` inside our [airflow.cfg](https://github.com/GokuMohandas/data-engineering/blob/main/airflow/airflow.cfg){:target="_blank"} configuration, so we'll have to manually execute this DAG by clicking on it > unpausing it (toggle) > triggering it (button). To view the logs for any of the tasks in our DAG run, we can click on the task > Log.

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

<div class="ai-center-all">
    <img width="650" src="/static/images/mlops/testing/production.png" alt="ETL pipelines in production">
</div>

!!! note
    We'll be breaking apart our `elt_data()` function from our `tagifai/main.py` script so that we can show what the proper data validation tasks look like in production workflows.

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

### Extraction

To keep things simple, we'll continue to keep our data as a local file but in a real production setting, our data can come from a wide variety of [data systems](data-stack.md){:target="_blank"}.

!!! note
    Ideally, the [data labeling](labeling.md){:target="_blank"} workflows would have occurred prior to the DataOps workflows. Depending on the task, it may involve natural labels, where the event that occurred is the label. Or there may be explicit manual labeling workflows that need to be inspected and approved.

```python linenums="1"
def _extract(ti):
    """Extract from source (ex. DB, API, etc.)
    Our simple ex: extract data from a URL
    """
    projects = utils.load_json_from_url(url=config.PROJECTS_URL)
    tags = utils.load_json_from_url(url=config.TAGS_URL)
    ti.xcom_push(key="projects", value=projects)
    ti.xcom_push(key="tags", value=tags)

@dag(...)
def dataops():
    extract = PythonOperator(task_id="extract", python_callable=_extract)
```

!!! warning
    [XComs](#xoms) should be used to share small metadata objects and not large data assets like this. But we're doing so only to simulate a pipeline where these assets would be prior to validation and loading.

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

- a [BashOperator](https://airflow.apache.org/docs/apache-airflow/stable/_api/airflow/operators/bash/index.html#airflow.operators.bash.BashOperator){:target="_blank"}:

```python linenums="1"
from airflow.operators.bash_operator import BashOperator
validate_projects = BashOperator(task_id="validate_projects", bash_command="great_expectations checkpoint run projects")
validate_tags = BashOperator(task_id="validate_tags", bash_command="great_expectations checkpoint run tags")
```

- a [PythonOperator](https://airflow.apache.org/docs/apache-airflow/stable/howto/operator/python.html){:target="_blank"}:

```bash
great_expectations checkpoint script <CHECKPOINT_NAME>
```

This will generate a python script under `great_expectations/uncommitted/run_<CHECKPOINT_NAME>.py` which you can wrap in a function to call using a `PythonOperator`.

- a [custom Great Expectations operator](https://github.com/great-expectations/airflow-provider-great-expectations){:target="_blank"}:

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
    Reminder that we previously set the following configuration in our [airflow.cfg](https://github.com/GokuMohandas/data-engineering/blob/main/airflow/airflow.cfg){:target="_blank"} file since the output of the GreatExpectationsOperator is not JSON serializable.
    ```bash
    # Inside airflow.cfg
    enable_xcom_pickling = True
    ```

### Load

Once we've validated our data, we're ready to load it into our data system (ex. data warehouse). This will be the primary system that potential downstream applications will depend on for current and future versions of data.

```python linenums="1"
def _load(ti):
    """Load into data system (ex. warehouse)
    Our simple ex: load extracted data into a local file
    """
    projects = ti.xcom_pull(key="projects", task_ids=["extract"])[0]
    tags = ti.xcom_pull(key="tags", task_ids=["extract"])[0]
    utils.save_dict(d=projects, filepath=Path(config.DATA_DIR, "projects.csv"))
    utils.save_dict(d=tags, filepath=Path(config.DATA_DIR, "tags.csv"))

@dag(...)
def dataops():
    ...
    load = PythonOperator(task_id="load", python_callable=_load)
```

### Transform

Once we have validated and loaded our data, we're ready to transform it. Our DataOps workflows are not specific to any particular downstream consumer so the transformation must be globally relevant (ex. cleaning missing date, aggregation, etc.).  We have a wide variety of Operators to choose from depending on the tools we're using for compute (ex. [Python](https://airflow.apache.org/docs/apache-airflow/stable/_api/airflow/operators/python/index.html#airflow.operators.python.PythonOperator){target="_blank"}, [Spark](https://airflow.apache.org/docs/apache-airflow-providers-apache-spark/stable/operators.html){:target="_blank"}, [DBT](https://github.com/gocardless/airflow-dbt){:target="_blank"}, etc.). Many of these options have the advantage of directly performing the transformations in our data warehouse.

```python linenums="1"
def _transform(ti):
    """Transform (ex. using DBT inside DWH)
    Our simple ex: using pandas to remove missing data samples
    """
    projects = ti.xcom_pull(key="projects", task_ids=["extract"])[0]
    df = pd.DataFrame(projects)
    df = df[df.tag.notnull()]  # drop rows w/ no tag
    utils.save_dict(d=df.to_dict(orient="records"), filepath=Path(config.DATA_DIR, "projects.json"))

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

## MLOps

Once we have our data prepared, we're ready to create one of the many downstream applications that will consume it. We'll set up our MLOps pipeline inside our `airflow/dags/workflows.py` script:

```python linenums="1"
# airflow/dags/workflows.py
@dag(
    dag_id="MLOps",
    description="MLOps tasks.",
    default_args=default_args,
    schedule_interval=None,
    start_date=days_ago(2),
    tags=["mlops"],
)
def mlops():
    pass
```

```python linenums="1"
# Define DAG
(
    prepare
    >> validate_prepared_data
    >> optimize
    >> train
    >> offline_evaluation
    >> online_evaluation
    >> [deploy, inspect]
)
```

<div class="ai-center-all">
    <img src="/static/images/mlops/orchestration/mlops.png" width="1000" alt="dataops workflow">
</div>

### Prepare

First, we'll need to extract our data that was prepared in the DataOps workflows.

```python linenums="1"
prepare = PythonOperator(
    task_id="prepare",
    python_callable=main.label_data,
    op_kwargs={"args_fp": Path(config.CONFIG_DIR, "args.json")},
)
```
Note that we're one on many potential downstream consumers of the prepared data so we'll want to execute further validation to ensure that the data is appropriate for our application.

```python linenums="1"
validate_prepared_data = GreatExpectationsOperator(
        task_id="validate_labeled_data",
        checkpoint_name="labeled_projects",
        data_context_root_dir=GE_ROOT_DIR,
        fail_task_on_validation_failure=True,
    )
```

### Training

Once we have our data prepped, we can use them to optimize and train the best models. Since these tasks can require lots of compute, we would typically run this entire pipeline in a managed cluster platform which can scale up as our data and models grow.

```python linenums="1"
# Optimization
optimize = PythonOperator(
    task_id="optimize",
    python_callable=main.optimize,
    op_kwargs={
        "args_fp": Path(config.CONFIG_DIR, "args.json"),
        "study_name": "optimization",
        "num_trials": 1,
    },
)
```
```python linenums="1"
# Training
train = PythonOperator(
    task_id="train",
    python_callable=main.train_model,
    op_kwargs={
        "args_fp": Path(config.CONFIG_DIR, "args.json"),
        "experiment_name": "baselines",
        "run_name": "sgd",
    },
)
```

### Offline evaluation

It's imperative that we evaluate our trained models so that we can trust it. We've extensively covered [offline evaluation](evaluation.md){:target="_blank"} before, so here we'll talk about how the evaluation is used. Tt won't always be a simple decision where all metrics/slices are performing better than the previous version. In these scenarios, it's important to know what our main priorities are and where we can have some leeway:

- What criteria are most important?
- What criteria can/cannot regress?
- How much of a regression can be tolerated?

```python linenums="1"
assert precision > prev_precision  # most important, cannot regress
assert recall >= best_prev_recall - 0.03  # recall cannot regress > 3%
assert metrics["class"]["nlp"]["f1"] > prev_nlp_f1  # priority class
assert metrics["slices"]["class"]["nlp_cnn"]["f1"] > prev_nlp_cnn_f1  # priority slice
```

and of course, there are some components, such as [behavioral testing](testing.md#behavioral-testing){:target="_blank"} our model's behavior of our models, that should always pass. We can in corporate this business logic into a function and determine if a newly trained version of the system is *better* than the current version.

```python linenums="1"
def _offline_evaluation():
    """Compare offline evaluation report
    (overall, fine-grained and slice metrics).
    And ensure model behavioral tests pass.
    """
    return True
```
```python linenums="1"
offline_evaluation = PythonOperator(
    task_id="offline_evaluation",
    python_callable=_offline_evaluation,
)
```

### Online evaluation

Once our system has passed offline evaluation criteria, we're ready to evaluate it in the [online setting](evaluation.md#online-evaluation){:target="_blank"}. Here we are using the [](https://airflow.apache.org/docs/apache-airflow/1.10.6/concepts.html?highlight=branch%20operator#branching){:target="_blank"} to execute different actions based on the results of online evaluation.

```python linenums="1"
from airflow.operators.python import BranchPythonOperator
```

This Operator will execute a function whose return response will be a single (or a list) task id.

```python linenums="1"
def _online_evaluation():
    """Run online experiments (AB, shadow, canary) to
    determine if new system should replace the current.
    """
    passed = True
    if passed:
        return "deploy"
    else:
        return "inspect"
```

```python linenums="1"
online_evaluation = BranchPythonOperator(
    task_id="online_evaluation",
    python_callable=_online_evaluation,
)
```

The returning task ids can correspond to tasks that are simply used to direct the workflow towards a certain set of tasks based on upstream results. In our case, we want to deploy the improved model or inspect it if it failed online evaluation requirements.

### Deploy

If our model passed our evaluation criteria then we can deploy and serve our model. Again, there are many different options here such as using our CI/CD Git workflows to deploy the model wrapped as a scalable microservice or for more streamlined deployments, we can use a purpose-build [model server](api.md#model-server){:target="_blank"} to seamlessly inspect, update, serve, rollback, etc. multiple versions of models.

```python linenums="1"
deploy = BashOperator(
    task_id="deploy",
    bash_command="echo update model endpoint w/ new artifacts",
)
```

## Continual learning

The DataOps and MLOps workflows connect to create an ML system that's capable of continually learning. Such a system will guide us with when to update, what exactly to update and how to update it (easily).

> We use the word continual (repeat with breaks) instead of continuous (repeat without interruption / intervention) because we're not trying to create a system that will automatically update with new incoming data without human intervention.

### Monitoring

Our production system is live and [monitored](monitoring.md){:target="_blank"}. When an event of interest occurs (ex. [drift](monitoring.md#drift){:target="_blank"}), one of several events needs to be triggered:

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

Once we have the proper dataset for retraining, we can kickoff the workflows to update our system!

## References
- [Airflow official documentation](https://airflow.apache.org/docs/apache-airflow/stable/index.html){:target="_blank"}

<!-- Citation -->
{% include "cite.md" %}