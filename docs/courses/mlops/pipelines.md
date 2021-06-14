---
template: lesson.html
title: Pipelines
description: Create, schedule and monitor workflows by creating scalable pipelines.
keywords: airflow, workflows, pipelines, orchestration, dataops, data warehouse, database, great expectations, data validation, spark, ci/cd, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/mlops.png
repository: https://github.com/GokuMohandas/MLOps
---

{% include "styles/lesson.md" %}

## Intuition

So far we've implemented the components of our DataOps (cleaning, feature engineering, preprocessing, etc.) and MLOps (optimization, training, evaluation, etc.) workflows as end-to-end Python function calls. This has worked well since our dataset is not large and the fact that we're only dealing with one version of data. But happens when we need to:

- **trigger** or **schedule** these workflows as new data arrives?
- **scale** these workflows as our data grows?
- **share** these workflows so others can use their outputs?
- **monitor** these workflows separately?

We'll need to break down our end-to-end ML pipeline into in it's constituent DataOps and MLOps pipelines that be orchestrated and scaled as needed. There are several tools that can help us create these pipelines and orchestrate our workflows such as [Airflow](http://airflow.apache.org/){:target="_blank"}, [Luigi](https://luigi.readthedocs.io/en/stable/){:target="_blank"} and even some ML focused options such as [KubeFlow Pipelines](https://www.kubeflow.org/docs/components/pipelines/overview/pipelines-overview/){:target="_blank"} and [Dagster](https://dagster.io/){:target="_blank"}. We'll be creating our pipelines using AirFlow because of it's:

- wide adoption in industry and open source
- Python based software development kit (SDK)
- integration with the ecosystem (data ingestion, processing, etc.)
- ability to run locally and scale easily

!!! note
    We'll be running Airflow locally but we can easily scale it by running on a managed cluster platform where we can run Python, Hadoop, Spark, etc. on large batch processing jobs (AWS' [EMR](https://aws.amazon.com/emr/){:target="_blank"}, Google Cloud's [Dataproc](https://cloud.google.com/dataproc){:target="_blank"}, on-prem hardware, etc.)

## Airflow

Before we create our specific pipelines, let's understand and implement [Airflow](https://airflow.apache.org/){:target="_blank"}'s overarching concepts that will allow us to "author, schedule, and monitor workflows".

## Install

To install and run Airflow, we can either do so [locally](https://airflow.apache.org/docs/apache-airflow/stable/start/local.html){:target="_blank"} or with [Docker](https://airflow.apache.org/docs/apache-airflow/stable/start/docker.html){:target="_blank"} and [set up a database backend](https://airflow.apache.org/docs/apache-airflow/stable/howto/set-up-database.html){:target="_blank"} (default is SQLite) and/or establish [connections](https://airflow.apache.org/docs/apache-airflow/stable/howto/connection.html){:target="_blank"}.

!!! warning
    If you do decide to use docker-compose to run Airflow inside Docker containers, you'll want to allocate at least 4 GB in memory.

```bash linenums="1"
# Configurations
export AIRFLOW_HOME=${PWD}/airflow
AIRFLOW_VERSION=2.0.1
PYTHON_VERSION="$(python --version | cut -d " " -f 2 | cut -d "." -f 1-2)"
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

# Install Airflow (may need to upgrade pip)
pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"

# Initialize DB (SQLite by default)
airflow db init
```

This will create an `airflow` directory with the following components:

```bash linenums="1"
airflow/
├── logs/
└── airflow.cfg
├── airflow.db
├── unittests.cfg
└── webserver_config.py
```

We're going to edit the [airflow.cfg](https://github.com/GokuMohandas/MLOps/blob/main/airflow/airflow.cfg){:target="_blank"} file to best fit our needs:
```bash
# Inside airflow.cfg
enable_xcom_pickling = True  # needed for Great Expectations airflow provider
load_examples = False  # don't clutter webserver with examples
```

And we'll perform a reset to account for the configuration changes.

```bash linenums="1"
airflow db reset
```

Now we're ready to initialize our database with an admin user, which we'll use to login to access our workflows in the webserver.

```bash linenums="1"
# We'll be prompted to enter a password
airflow users create \
    --username admin \
    --firstname Goku \
    --lastname Mohandas \
    --role Admin \
    --email hello@madewithml.com
```

## Webserver

Once we've created a user, we're ready to launch the webserver and log in using our credentials.

```bash
# Launch webserver
export AIRFLOW_HOME=${PWD}/airflow
airflow webserver --port 8080  # http://localhost:8080
```

The webserver allows us to run and inspect workflows, establish connections to external data storage, manager users, etc. through a UI. Similarly, we could also use Airflow's [REST API](https://airflow.apache.org/docs/apache-airflow/stable/stable-rest-api-ref.html){:target="_blank"} or [Command-line interface (CLI)](https://airflow.apache.org/docs/apache-airflow/stable/cli-and-env-variables-ref.html){:target="_blank"} to perform the same operations. However, we'll largely be using the webserver because it's convenient to visually inspect our workflows.

<div class="ai-center-all">
    <img src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/pipelines/webserver.png" width="700" alt="pivot">
</div>

We'll explore the different components of the webserver as we learn about Airflow and implement our workflows.

## Scheduler

Next, we need to launch our scheduler, which will execute and monitor the tasks in our workflows. The schedule executes tasks by reading from the metadata database and ensures the task has what it needs to finish running.

```bash
# Launch scheduler (in separate terminal)
export AIRFLOW_HOME=${PWD}/airflow
airflow scheduler
```

## Executor

As our scheduler reads from the metadata database, the executor determines what worker processes are necessary for the task to run to completion. Since our default database SQLlite, which can't support multiple connections, our default executor is the [Sequential Executor](https://airflow.apache.org/docs/apache-airflow/stable/executor/sequential.html){:target="_blank"}. However, if we choose a more production grade DB option such as PostgresSQL or MySQL, we can choose scalable [Executor backends](https://airflow.apache.org/docs/apache-airflow/stable/executor/index.html#supported-backends){:target="_blank"} Celery, Kubernetes, etc. For example, running [Airflow with Docker](https://airflow.apache.org/docs/apache-airflow/stable/start/docker.html){:target="_blank"} uses PostgresSQL as the database and so uses the Celery Executor backend to run tasks in parallel.


## DAGs

Workflows are defined by directed acyclic graphs (DAGs), whose nodes represent tasks and edges represent the relationship between the tasks. Direct and acyclic implies that workflows can only execute in one direction and a previous, upstream task cannot run again once a downstream task has started.

<div class="ai-center-all">
    <img src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/pipelines/basic-dag.png" width="250" alt="pivot">
</div>

DAGs can be defined inside Python workflow scripts inside the `airflow/dags` directory and they'll automatically appear (and continuously be updated) on the webserver. Inside each workflow script, we can define some default arguments that will apply to all DAGs within that workflow.

```python linenums="1"
# Default DAG args
default_args = {
    "owner": "airflow",
}
```

!!! note
    There are many [more default arguments](https://airflow.apache.org/docs/apache-airflow/stable/tutorial.html#default-arguments){target="_blank"} and we'll cover them as we go through the concepts.

We can initialize DAGs with many [parameters](https://airflow.apache.org/docs/apache-airflow/stable/_api/airflow/models/dag/index.html#airflow.models.dag.DAG){:target="_blank"} (which will override the same parameters in `default_args`) and in several different ways:

- using a [with statement](https://docs.python.org/3/reference/compound_stmts.html#the-with-statement){target="_blank"}
```python linenums="1"
from airflow import DAG

with DAG(
    dag_id="dataops",
    description="Data related operations.",
    default_args=default_args,
    tags=["dataops"],
) as dag:
    # Define tasks
    pass
```

- using the [dag decorator](https://airflow.apache.org/docs/apache-airflow/stable/concepts.html#dag-decorator){target="_blank"}
```python linenums="1"
from airflow.decorators import dag

@dag(
    dag_id="dataops",
    description="Data related operations.",
    default_args=default_args,
    tags=["dataops"],
)
def dataops():
    # Define tasks
    pass
```

!!! note
    There are many [parameters](https://airflow.apache.org/docs/apache-airflow/stable/_api/airflow/models/dag/index.html#airflow.models.dag.DAG){:target="_blank"} that we can initialize our DAGs with, including a `start_date` and a `schedule_interval`. While we could have our workflows execute on a temporal cadence, many ML workflows are initiated by events, which we can map using [sensors](https://airflow.apache.org/docs/apache-airflow/stable/_api/airflow/sensors/index.html){:target="_blank"} and [hooks](https://airflow.apache.org/docs/apache-airflow/stable/_api/airflow/hooks/index.html){:target="_blank"} to external databases, file systems, etc.


## Tasks

Tasks are the operations that are executed in a workflow and are represented by nodes in a DAG. Each task should be a clearly defined single operation and it should be idempotent, which means we can execute it multiple times and expect the same result. This is important in the event we need to retry a failed task and don't have to worry about resetting the state of our system. Like DAGs, there are several different ways to implement tasks:

- Using [Operators](https://airflow.apache.org/docs/apache-airflow/stable/_api/airflow/operators/index.html){:target="_blank"}
```python linenums="1"
from airflow.operators.bash_operator import BashOperator

@dag(
    dag_id="example",
    default_args=default_args,
    schedule_interval=None,
    tags=["example"],
)
def example():
    # Define tasks
    task_1 = BashOperator(task_id="task_1", bash_command="echo 1")
    task_2 = BashOperator(task_id="task_2", bash_command="echo 1")
```

- Using the [tag decorator](https://airflow.apache.org/docs/apache-airflow/stable/concepts.html#concepts-task-decorator){:target="_blank"}
```python linenums="1"
@dag(
    dag_id="example",
    default_args=default_args,
    schedule_interval=None,
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

!!! note
    Though the graphs are directed, we can establish certain [trigger rules](https://airflow.apache.org/docs/apache-airflow/stable/concepts.html#trigger-rules){:target="_blank"} for each task to execute on conditional successes or failures of the parent tasks.

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

!!! note
    We can also create our own [custom Operators](https://airflow.apache.org/docs/apache-airflow/stable/howto/custom-operator.html){:target="_blank"} by extending the [BashOperator](https://airflow.apache.org/docs/apache-airflow/stable/_api/airflow/operators/bash/index.html#airflow.operators.bash.BashOperator){:target="_blank"} class.

### Relationships

Once we've defined our tasks using Operators or as decorated functions, we need to define the relationships between them (edges). The way we define the relationships depends on how our tasks were defined:

- if defined using Operators
```python linenums="1"
# Task relationships
task_1 >> task_2  # same as task_1.set_downstream(task_2) or
                  # task_2.set_upstream(task_1)
```

- if defined using decorated functions
```python linenums="1"
# Task relationships
x = task_1()
y = task_2(x=x)
```

In both scenarios, we'll setting `task_2` as the upstream task to `task_1`. We can create intricate DAGs by using these notations to define the relationships.

```python linenums="1"
task_1 >> [task_2_1, task_2_2] >> task_3
task_2_2 >> task_4
[task_3, task_4] >> task_5
```
<div class="ai-center-all">
    <img src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/pipelines/dag.png" width="500" alt="pivot">
</div>

### XComs

When we use decorated functions, we can see how values can be passed between tasks. But, how can we pass values when using Operators? Airflow uses XComs (cross communications) objects, defined with a key, value, timestamp and task_id, to push and pull values between tasks. When we use decorated functions, XComs are being used under the hood but it's abstracted away, allowing us to pass values amongst Python functions seamlessly. But when using Operators, we'll need to explicitly push and pull the values as we need it.

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
    default_args=default_args,
    schedule_interval=None,
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
    <img src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/pipelines/xcoms.png" width="700" alt="pivot">
</div>

!!! warning
    The data we pass between tasks should be small (metadata, metrics, etc.) because Airflow's metadata database is not equipped to hold large artifacts. However, if we do need to store and use the large results of our tasks, it's best to use an external data storage (blog storage, databases, etc.) by creating an interface via [hooks](https://airflow.apache.org/docs/apache-airflow/stable/concepts.html#hooks){target="_blank"}.


## DAG runs

Once we've defined the tasks and their relationships, we're ready to run our DAGs. We'll start defining our DAG like so:
```python linenums="1"
# Define DAG
example_dag = example()
```

If we refresh our webserver page (http://localhost:8080/), the new DAG will have appeared.

### Manual
Our DAG is initially paused since we specified `dags_are_paused_at_creation = True` inside our [airflow.cfg](https://github.com/GokuMohandas/MLOps/blob/main/airflow/airflow.cfg){:target="_blank"} configuration, so we'll have to manually execute this DAG by clicking on it >> unpausing it (toggle) >> triggering it (button). To view the logs for any of the tasks in our DAG run, we can click on the task >> Log.

<div class="ai-center-all">
    <img src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/pipelines/trigger.png" width="700" alt="pivot">
</div>

!!! note
    We could also use Airflow's [REST API](https://airflow.apache.org/docs/apache-airflow/stable/stable-rest-api-ref.html){:target="_blank"} (will [configured authorization](https://airflow.apache.org/docs/apache-airflow/stable/security/api.html){:target="_blank"}) or [Command-line interface (CLI)](https://airflow.apache.org/docs/apache-airflow/stable/cli-and-env-variables-ref.html){:target="_blank"} to inspect and trigger workflows (and a whole lot more). Or we could even use the [`trigger_dagrun`](https://airflow.apache.org/docs/apache-airflow/stable/_api/airflow/operators/trigger_dagrun/index.html){:target="_blank"} Operator to trigger DAGs from within another workflow.

    ```bash
    # CLI to run dags
    airflow dags trigger dataops
    airflow dags trigger mlops
    ```

### Interval
Had we specified a `start_date` and `schedule_interval` when defining the DAG, it would have have automatically executed at the appropriate times. For example, the DAG below will have started two days ago and will be triggered at the start of every day.

```python linenums="1"
from datetime import timedelta
from airflow.utils.dates import days_ago

@dag(
    dag_id="example",
    default_args=default_args,
    schedule_interval=timedelta(days=1),
    start_date=days_ago(2),
    tags=["example"],
)
```

!!! warning
    Depending on the `start_date` and `schedule_interval`, our workflow may have been triggered several times and Airflow will try to [catchup](https://airflow.apache.org/docs/apache-airflow/stable/dag-run.html#catchup){:target="_blank"} to the current time. We can avoid this by setting `catchup=False` when defining the DAG. However, if we did want to run particular runs in the past, we can [backfill](https://airflow.apache.org/docs/apache-airflow/stable/dag-run.html#backfill){:target="_blank"} what we need.


We could also specify a [cron](https://crontab.guru/){:target="_blank"} expression for our `schedule_interval` parameter or even use [cron presets](https://airflow.apache.org/docs/apache-airflow/stable/dag-run.html#cron-presets){:target="_blank"}.


!!! warning
    Airflow's Scheduler will run our workflows one `schedule_interval` from the `start_date`. For example, if we want our workflow to start on `01-01-1983` and run `@daily`, then the first run will be immediately after `01-01-1983T11:59`.

### Sensors

While it may make sense to execute many data processing workflows on a scheduled interval, machine learning workflows may require more nuanced triggers. We shouldn't be wasting compute by running executing our DataOps and MLOps pipelines *just in case* we have new data. Instead, we can use [sensors](https://airflow.apache.org/docs/apache-airflow/stable/_api/airflow/sensors/){:target="_blank"} to trigger workflows when some external condition is met. For example, we can initiate data processing when annotated data appears in a database, or when a specific file appears in a file system.

!!! note
    There's so much more to Airflow (monitoring, Task groups, smart senors, etc.) so be sure to explore them as you need them by using the [official documentation](https://airflow.apache.org/docs/apache-airflow/stable/index.html){:target="_blank"}.


## DataOps

Now that we've reviewed Airflow's major concepts, we're ready to create the DataOps pipeline for our application. It involves a series of tasks, starting from extracting the data, validating it and storing it at the right place for others to use for downstream workflows and applications.

<div class="ai-center-all">
    <img src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/pipelines/dataops.png" width="1000" alt="pivot">
</div>

```python linenums="1"
# Task relationships
extract_data >> [validate_projects, validate_tags] >> compute_features >> cache_to_feature_store
```

To keep things simple, we'll continue to keep our data as a local file:

```python linenums="1"
# Extract data from DWH, blog storage, etc.
extract_data = BashOperator(
    task_id="extract_data",
    bash_command=f"cd {config.BASE_DIR} && dvc pull",
)
```

... but in a real production setting, our data can come from a wide variety of sources. For example, we have a dataset prepared for the purposes of this course but where does this data originate from and where does it end up before we're ready to use it for machine learning?

### Data

When we talk about data that we want to process on in our DataOps pipelines, we need to think about both it's origin and where it lives.

- **Origin**: most applications (including the [old Made With ML](https://madewithml.com/pivot/){:target="_blank"} that our dataset comes from) have a database to store and read information from. Typical choices are [PostgreSQL](https://www.postgresql.org/){:target="_blank"}, [MySQL](https://www.mysql.com/){:target="_blank"}, [MongoDB](https://www.mongodb.com/){:target="_blank"}, [Cassandra](https://cassandra.apache.org/){:target="_blank"}, etc. The specific choice depends on the schemas, data scale, etc. We can also have auxilliary data coming from other services around our main application such as user analytics (ex. [Google Analytics](https://analytics.google.com/analytics/web/){:target="_blank"}), financial transactions (ex. [Stripe](https://stripe.com/){:target="_blank"}), etc.

- **Warehouse**: data is often moved from the database to a data warehouse (DWH) for the added benefits of a powerful analytics engine, front-end clients, etc. to make it very easy for downstream developers to efficiently use the data at scale. Typical choices are [Google BigQuery](https://cloud.google.com/bigquery){:target="_blank"}, [Amazon RedShift](https://aws.amazon.com/redshift/){:target="_blank"}, [SnowFlake](https://www.snowflake.com/){:target="_blank"}, [Hive](https://hive.apache.org/){:target="_blank"}, etc.

Typically we'll use [sensors](https://airflow.apache.org/docs/apache-airflow/stable/_api/airflow/sensors/){:target="_blank"} to trigger workflows when a condition is met or trigger them directly from the external source via API calls, etc. Our workflows can communicate with the different platforms by establishing a [connection](https://airflow.apache.org/docs/apache-airflow/stable/howto/connection.html){:target="_blank"} and then using [hooks](https://airflow.apache.org/docs/apache-airflow/stable/_api/airflow/hooks/index.html){:target="_blank"} to interface with the database, data warehouse, etc.

### Validation

The specific process of where and how we extract our data can be bespoke but what's important is that we have a continuous integration to execute our workflows. A key aspect to trusting this continuous integration is validation at every step of the way. We'll once again use [Great Expectations](https://greatexpectations.io/){:target="_blank"}, as we did in our [testing lesson](../testing/#data){:target="_blank"}, to validate our incoming data before processing it.

With the Airflow concepts we've learned so far, there are many ways to use our data validation library to validate our data. Regardless of what data validation tool we use (ex. [Great Expectations](https://greatexpectations.io/){:target="_blank"}, [TFX](https://www.tensorflow.org/tfx/data_validation/get_started){:target="_blank"}, [AWS Deequ](https://github.com/awslabs/deequ){:target="_blank"}, etc.) we could use the BashOperator, PythonOperator, etc. to run our tests. Luckily, Great Expectations has a [recommended](https://docs.greatexpectations.io/en/stable/guides/workflows_patterns/deployment_airflow.html){:target="_blank"} [Airflow Provider package](https://github.com/great-expectations/airflow-provider-great-expectations){:target="_blank"}. This package contains a `GreatExpectationsOperator` which we can use to execute specific checkpoints as tasks.

Recall from our testing lesson that we used the following CLI commands to perform our data validation tests:

```bash linenums="1"
great_expectations checkpoint run projects
great_expectations checkpoint run tags
```

We can perform the same operations as Airflow tasks within our DataOps workflow, either as the bash commands above using the [BashOperator](https://airflow.apache.org/docs/apache-airflow/stable/_api/airflow/operators/bash/index.html#airflow.operators.bash.BashOperator){:target="_blank"} or with the [custom Great Expectations operator](https://github.com/great-expectations/airflow-provider-great-expectations){:target="_blank"} like below:

```python linenums="1"
from great_expectations_provider.operators.great_expectations import GreatExpectationsOperator

# Validate projects
validate_projects = GreatExpectationsOperator(
    task_id="validate_projects",
    checkpoint_name="projects",
    data_context_root_dir="great_expectations",
    fail_task_on_validation_failure=True,
)

# Validate tags
validate_tags = GreatExpectationsOperator(
    task_id="validate_tags",
    checkpoint_name="tags",
    data_context_root_dir="great_expectations",
    fail_task_on_validation_failure=True,
)
```

And we want both tasks to pass so we set the `fail_task_on_validation_failure` parameter to `True` so that downstream tasks don't execute if they fail.

!!! note
    Reminder that we previous set the following configuration in our [airflow.cfg](https://github.com/GokuMohandas/MLOps/blob/main/airflow/airflow.cfg){:target="_blank"} file since the output of the GreatExpectationsOperator is not JSON serializable.
    ```bash
    # Inside airflow.cfg
    enable_xcom_pickling = True
    ```

### Compute

Once we have validated our data, we're ready to compute features. We have a wide variety of Operators to choose from depending on the tools we're using for compute (ex. [Python](https://airflow.apache.org/docs/apache-airflow/stable/_api/airflow/operators/python/index.html#airflow.operators.python.PythonOperator){target="_blank"}, [Spark](https://airflow.apache.org/docs/apache-airflow-providers-apache-spark/stable/operators.html){:target="_blank"}, [DBT](https://github.com/gocardless/airflow-dbt){:target="_blank"}, etc.). And of course we can easily scale all of this by running on a managed cluster platform (AWS' [EMR](https://aws.amazon.com/emr/){:target="_blank"}, Google Cloud's [Dataproc](https://cloud.google.com/dataproc){:target="_blank"}, on-prem hardware, etc.). For our task, we'll just need a PythonOperator to execute our feature engineering CLI command:

```python linenums="1"
# Compute features
compute_features = PythonOperator(
    task_id="compute_features",
    python_callable=cli.compute_features,
    op_kwargs={"params_fp": Path(config.CONFIG_DIR, "params.json")}
)
```

### Cache

When we establish our DataOps pipeline, it's not something that's specific to any one application. Instead it's its own repository that's responsible for extracting, transforming and loading (ETL) data for downstream pipelines who are dependent on it for their own unique applications. This is one of the most important benefits of not doing an end-to-end ML application because it allows for true continued collaboration. And so we need to cache our computed features to a central [feature store](feature-store.md){:target="_blank"}. This way, downstream developers can easily access features and use them to build applications without having to worry about doing much heavy lifting with data processing.

```python linenums="1"
# Feature store
cache_to_feature_store = BashOperator(
    task_id="cache_to_feature_store",
    bash_command=f"cd {config.BASE_DIR}/features && feast materialize-incremental {END_TS}",
)
```

!!! note
    Learn all about what features stores are, why we need them and how to implement them in our [feature stores lesson](feature-store.md){:target="_blank"}.


## MLOps

Once we have our features in our feature store, we can use them for MLOps tasks such as optimization, training, validation, serving, etc.

<div class="ai-center-all">
    <img src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/pipelines/mlops.png" width="1000" alt="pivot">
</div>

```python linenums="1"
# Task relationships
optimization >> train_model >> evaluate_model >> [improved, regressed]
improved >> [set_monitoring_references, deploy_model, notify_teams]
regressed >> [notify_teams, file_report]
```

### Extract features

The first step is to extract the relevant historical features from our feature store that our DataOps workflow ended with. We'd normally need to provide a set of entities and the timestamp to extract point-in-time features for each of them.

```python linenums="1"
# Extract features
extract_features = PythonOperator(
    task_id="extract_features",
    python_callable=cli.get_historical_features,
)
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

When we're ready to evaluate our trained models, we want to execute a chain of tasks depending on whether the model improved or regressed. To do this, we're using a [BranchPythonOperator](https://airflow.apache.org/docs/apache-airflow/stable/concepts.html?highlight=branch#branching){:target="_blank"} for the evaluation task so it can return the appropriate task id.

```python linenums="1"
from airflow.operators.python import BranchPythonOperator

# Evaluate
evaluate_model = BranchPythonOperator(  # BranchPythonOperator returns a task_id or [task_ids]
    task_id="evaluate_model",
    python_callable=_evaluate_model,
)
```

This Operator will execute a function whose return response will be a single (or a list) task id.

```python
def _evaluate_model():
    if improvement_criteria():
        return "improved"  # improved is a task id
    else:
        return "regressed"  # regressed is a task id
```

The actual task ids that the callable python function returns don't necessarily have to have any functions.

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

Instead they're simply used to direct the workflow towards a certain set of tasks based on upstream results. In our case, we want to notify teams regardless of improvement or regression but we have some unique tasks for each scenario as well (deploying the improved model, resetting distribution references, filing a regression report, etc.).

```python linenums="1"
# Task relationships
extract_features >> optimization >> train_model >> evaluate_model >> [improved, regressed]
improved >> [set_monitoring_references, deploy_model, notify_teams]
regressed >> [notify_teams, file_report]
```

!!! note
    Using these workflows for continuous iteration *doesn't* have to mean that it's all fully automated without human intervention. Instead it's a system that can iterate and improve with proper validation along the way and complete transparency and traceability.

## References
- [Airflow official documentation](https://airflow.apache.org/docs/apache-airflow/stable/index.html){:target="_blank"}

<!-- Citation -->
{% include "cite.md" %}