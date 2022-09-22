---
template: lesson.html
title: Orchestration for Machine Learning
description: Create, schedule and monitor workflows by creating scalable pipelines.
keywords: airflow, prefect, dagster, workflows, pipelines, orchestration, dataops, data warehouse, database, great expectations, data validation, spark, ci/cd, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/mlops.png
repository: https://github.com/GokuMohandas/data-engineering
---

{% include "styles/lesson.md" %}

## Intuition

So far we've implemented our DataOps (ELT, validation, etc.) and MLOps (optimization, training, evaluation, etc.) workflows as Python function calls. This has worked well since our dataset is static and small. But happens when we need to:

- **schedule** these workflows as new data arrives?
- **scale** these workflows as our data grows?
- **share** these workflows to downstream applications?
- **monitor** these workflows?

We'll need to break down our end-to-end ML pipeline into individual workflows that can be orchestrated as needed. There are several tools that can help us so this such as [Airflow](http://airflow.apache.org/){:target="_blank"}, [Prefect](https://www.prefect.io/){:target="_blank"}, [Dagster](https://dagster.io/){:target="_blank"}, [Luigi](https://luigi.readthedocs.io/en/stable/){:target="_blank"}, [Orchest](https://www.orchest.io/){:target="_blank"} and even some ML focused options such as [Metaflow](https://metaflow.org/){:target="_blank"}, [Flyte](https://flyte.org/){:target="_blank"}, [KubeFlow Pipelines](https://www.kubeflow.org/docs/components/pipelines/overview/pipelines-overview/){:target="_blank"}, [Vertex pipelines](https://cloud.google.com/vertex-ai/docs/pipelines/introduction){:target="_blank"}, etc. We'll be creating our workflows using AirFlow for its:

- wide adoption of the open source platform in industry
- Python based software development kit (SDK)
- integration with the ecosystem (data ingestion, processing, etc.)
- ability to run locally and scale easily
- maturity over the years and part of the apache ecosystem

> We'll be running Airflow locally but we can easily scale it by running on a managed cluster platform where we can run Python, Hadoop, Spark, etc. on large batch processing jobs (AWS [EMR](https://aws.amazon.com/emr/){:target="_blank"}, Google Cloud's [Dataproc](https://cloud.google.com/dataproc){:target="_blank"}, on-prem hardware, etc.).

## Airflow

Before we create our specific pipelines, let's understand and implement [Airflow](https://airflow.apache.org/){:target="_blank"}'s overarching concepts that will allow us to "author, schedule, and monitor workflows".

!!! note "Separate repository"
    Our work in this lesson will live in a separate repository so create a new directory (outside our `mlops-course` repository) called `data-engineering`. All the work in this lesson can be found in our :fontawesome-brands-github:{ .github } [data-engineering](https://github.com/GokuMohandas/data-engineering){:target="_blank"} repository.

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
├── airflow.cfg
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
source venv/bin/activate
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
source venv/bin/activate
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

> Typically, our DAGs are not the only ones running in an Airflow cluster. However, it can be messy and sometimes impossible to execute different workflows when they require different resources, package versions, etc. For teams with multiple projects, it’s a good idea to use something like the [KubernetesPodOperator](https://airflow.apache.org/docs/apache-airflow-providers-cncf-kubernetes){:target=“_blank”} to execute each job using an isolated docker image.

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

The new DAG will have appeared when we refresh our [Airflow webserver](http://localhost:8080/){:target="_blank"}.

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

Now that we've reviewed Airflow's major concepts, we're ready to create the DataOps workflows. It's the exact same workflow we defined in our [data stack lesson](data-stack.md){:target="_blank"} -- extract, load and transform -- but this time we'll be doing everything programmatically and orchestrating it with Airflow.

<div class="ai-center-all mb-4">
    <img width="650" src="/static/images/mlops/testing/production.png" alt=ELT pipelines in production">
</div>

We'll start by creating the script where we'll define our workflows:

```bash
touch airflow/dags/workflows.py
```

```python linenums="1"
from pathlib import Path
from airflow.decorators import dag
from airflow.utils.dates import days_ago

# Default DAG args
default_args = {
    "owner": "airflow",
    "catch_up": False,
}
BASE_DIR = Path(__file__).parent.parent.parent.absolute()

@dag(
    dag_id="dataops",
    description="DataOps workflows.",
    default_args=default_args,
    schedule_interval=None,
    start_date=days_ago(2),
    tags=["dataops"],
)
def dataops():
    """DataOps workflows."""
    pass

# Run DAG
do = dataops()
```

In two separate terminals, activate the virtual environment and spin up the Airflow webserver and scheduler:

<div class="row">
    <div class="col-md-6">
<div class="highlight"><pre id="__code_2"><span></span><button class="md-clipboard md-icon" title="Copy to clipboard" data-clipboard-target="#__code_2 > code"></button><code tabindex="0"><span class="c1"># Airflow webserver</span>
<span class="nb">source</span> venv/bin/activate
<span class="nb">export</span> <span class="nv">AIRFLOW_HOME</span><span class="o">=</span><span class="si">${</span><span class="nv">PWD</span><span class="si">}</span>/airflow
<span class="nb">export</span> <span class="nv">GOOGLE_APPLICATION_CREDENTIALS</span><span class="o">=</span>/Users/goku/Downloads/made-with-ml-XXXXXX-XXXXXXXXXXXX.json <span class="c1"># REPLACE</span>
airflow webserver --port <span class="m">8080</span>
<span class="c1"># Go to http://localhost:8080</span>
</code></pre></div>
    </div>
    <div class="col-md-6">
<div class="highlight"><pre id="__code_3"><span></span><button class="md-clipboard md-icon" title="Copy to clipboard" data-clipboard-target="#__code_3 > code"></button><code tabindex="0"><span class="c1"># Airflow scheduler</span>
<span class="nb">source</span> venv/bin/activate
<span class="nb">export</span> <span class="nv">AIRFLOW_HOME</span><span class="o">=</span><span class="si">${</span><span class="nv">PWD</span><span class="si">}</span>/airflow
<span class="nb">export</span> <span class="nv">OBJC_DISABLE_INITIALIZE_FORK_SAFETY</span><span class="o">=</span>YES
<span class="nb">export</span> <span class="nv">GOOGLE_APPLICATION_CREDENTIALS</span><span class="o">=</span>~/Downloads/made-with-ml-XXXXXX-XXXXXXXXXXXX.json <span class="c1"># REPLACE</span>
airflow scheduler
</code></pre></div>
    </div>
</div>

### Extract and load

We're going to use the Airbyte connections we set up in our [data-stack lesson](data-stack.md){:target="_blank"} but this time we're going to programmatically trigger the data syncs with Airflow. First, let's ensure that Airbyte is running on a separate terminal in it's repository:

```bash
git clone https://github.com/airbytehq/airbyte.git  # skip if already create in data-stack lesson
cd airbyte
docker-compose up
```

Next, let's install the required packages and establish the connection between Airbyte and Airflow:

```bash
pip install apache-airflow-providers-airbyte==3.1.0
```

1. Go to the [Airflow webserver](http://localhost:8080/){:target="_blank"} and click `Admin` > `Connections` > ➕
2. Add the connection with the following details:
```yaml
Connection ID: airbyte
Connection Type: HTTP
Host: localhost
Port: 8000
```

> We could also establish connections [programmatically](https://airflow.apache.org/docs/apache-airflow/stable/howto/connection.html#connection-cli){:target=“_blank”} but it’s good to use the UI to understand what’s happening under the hood.

In order to execute our extract and load data syncs, we can use the [`AirbyteTriggerSyncOperator`](https://airflow.apache.org/docs/apache-airflow-providers-airbyte/stable/operators/airbyte.html){:target="_blank"}:

```python linenums="1"
@dag(...)
def dataops():
    """Production DataOps workflows."""
    # Extract + Load
    extract_and_load_projects = AirbyteTriggerSyncOperator(
        task_id="extract_and_load_projects",
        airbyte_conn_id="airbyte",
        connection_id="XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX",  # REPLACE
        asynchronous=False,
        timeout=3600,
        wait_seconds=3,
    )
    extract_and_load_tags = AirbyteTriggerSyncOperator(
        task_id="extract_and_load_tags",
        airbyte_conn_id="airbyte",
        connection_id="XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX",  # REPLACE
        asynchronous=False,
        timeout=3600,
        wait_seconds=3,
    )

    # Define DAG
    extract_and_load_projects
    extract_and_load_tags
```

We can find the `connection_id` for each Airbyte connection by:

1. Go to our [Airbyte webserver](http://localhost:8000/){:target="_blank"} and click `Connections` on the left menu.
2. Click on the specific connection we want to use and the URL should be like this:
```bash
https://demo.airbyte.io/workspaces/<WORKSPACE_ID>/connections/<CONNECTION_ID>/status
```
3. The string in the `CONNECTION_ID` position is the connection's id.

We can trigger our DAG right now and view the extracted data be loaded into our BigQuery data warehouse but we'll continue developing and execute our DAG once the entire DataOps workflow has been defined.

### Validate

The specific process of where and how we extract our data can be bespoke but what's important is that we have validation at every step of the way. We'll once again use [Great Expectations](https://greatexpectations.io/){:target="_blank"}, as we did in our [testing lesson](testing.md#data){:target="_blank"}, to [validate](testing.md#expectations){:target="_blank"} our extracted and loaded data before transforming it.

With the Airflow concepts we've learned so far, there are many ways to use our data validation library to validate our data. Regardless of what data validation tool we use (ex. [Great Expectations](https://greatexpectations.io/){:target="_blank"}, [TFX](https://www.tensorflow.org/tfx/data_validation/get_started){:target="_blank"}, [AWS Deequ](https://github.com/awslabs/deequ){:target="_blank"}, etc.) we could use the BashOperator, PythonOperator, etc. to run our tests. However, Great Expectations has a [Airflow Provider package](https://github.com/great-expectations/airflow-provider-great-expectations){:target="_blank"} to make it even easier to validate our data. This package contains a [`GreatExpectationsOperator`](https://registry.astronomer.io/providers/great-expectations/modules/greatexpectationsoperator){:target="_blank"} which we can use to execute specific checkpoints as tasks.

```bash
pip install airflow-provider-great-expectations==0.1.1 great-expectations==0.15.19
great_expectations init
```

This will create the following directory within our data-engineering repository:

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

But first, before we can create our tests, we need to define a new `datasource` within Great Expectations for our Google BigQuery data warehouse. This will require several packages and exports:

```bash
pip install pybigquery==0.10.2 sqlalchemy_bigquery==1.4.4
export GOOGLE_APPLICATION_CREDENTIALS=/Users/goku/Downloads/made-with-ml-XXXXXX-XXXXXXXXXXXX.json  # REPLACE
```

```bash
great_expectations datasource new
```
```bash
What data would you like Great Expectations to connect to?
    1. Files on a filesystem (for processing with Pandas or Spark)
    2. Relational database (SQL) 👈
```
```bash
What are you processing your files with?
1. MySQL
2. Postgres
3. Redshift
4. Snowflake
5. BigQuery 👈
6. other - Do you have a working SQLAlchemy connection string?
```

This will open up an interactive notebook where we can fill in the following details:
```yaml
datasource_name = “dwh"
connection_string = “bigquery://made-with-ml-359923/mlops_course”
```

#### Suite

Next, we can create a [suite of expectations](testing.md#suites){:target="_blank"} for our data assets:

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
Select a datasource
    1. dwh 👈
```
```bash
Which data asset (accessible by data connector "default_inferred_data_connector_name") would you like to use?
    1. mlops_course.projects 👈
    2. mlops_course.tags
```
```bash
Name the new Expectation Suite [mlops.projects.warning]: projects
```

This will open up an interactive notebook where we can define our expectations. Repeat the same for creating a suite for our tags data asset as well.

??? quote "Expectations for `mlops_course.projects`"
    Table expectations
    ```python linenums="1"
    # data leak
    validator.expect_compound_columns_to_be_unique(column_list=["title", "description"])
    ```

    Column expectations:
    ```python linenums="1"
    # id
    validator.expect_column_values_to_be_unique(column="id")

    # create_on
    validator.expect_column_values_to_not_be_null(column="created_on")

    # title
    validator.expect_column_values_to_not_be_null(column="title")
    validator.expect_column_values_to_be_of_type(column="title", type_="STRING")

    # description
    validator.expect_column_values_to_not_be_null(column="description")
    validator.expect_column_values_to_be_of_type(column="description", type_="STRING")
    ```
??? quote "Expectations for `mlops_course.tags`"
    Column expectations:
    ```python linenums="1"
    # id
    validator.expect_column_values_to_be_unique(column="id")

    # tag
    validator.expect_column_values_to_not_be_null(column="tag")
    validator.expect_column_values_to_be_of_type(column="tag", type_="STRING")
    ```

#### Checkpoints

Once we have our suite of expectations, we're ready to check [checkpoints](testing.md#checkpoints){:target="_blank"} to execute these expectations:

```bash
great_expectations checkpoint new projects
```

This will, of course, open up an interactive notebook. Just ensure that the following information is correct (the default values may not be):
```yaml
datasource_name: dwh
data_asset_name: mlops_course.projects
expectation_suite_name: projects
```

And repeat the same for creating a checkpoint for our tags suite.

#### Tasks

With our checkpoints defined, we're ready to apply them to our data assets in our warehouse.

```python linenums="1"
GE_ROOT_DIR = Path(BASE_DIR, "great_expectations")

@dag(...)
def dataops():
    ...
    validate_projects = GreatExpectationsOperator(
        task_id="validate_projects",
        checkpoint_name="projects",
        data_context_root_dir=GE_ROOT_DIR,
        fail_task_on_validation_failure=True,
    )
    validate_tags = GreatExpectationsOperator(
        task_id="validate_tags",
        checkpoint_name="tags",
        data_context_root_dir=GE_ROOT_DIR,
        fail_task_on_validation_failure=True,
    )

    # Define DAG
    extract_and_load_projects >> validate_projects
    extract_and_load_tags >> validate_tags
```

### Transform

Once we've validated our extracted and loaded data, we're ready to [transform](data-stack.md#transform){:target="_blank"} it. Our DataOps workflows are not specific to any particular downstream application so the transformation must be globally relevant (ex. cleaning missing data, aggregation, etc.). Just like in our [data stack lesson](data-stack.md){:target="_blank"}, we're going to use [dbt](https://www.getdbt.com/){:target="_blank"} to transform our data. However, this time, we're going to do everything programmatically using the open-source [dbt-core](https://github.com/dbt-labs/dbt-core){:target="_blank"} package.

In the root of our data-engineering repository, initialize our dbt directory with the following command:
```bash
dbt init dbf_transforms
```
```bash
Which database would you like to use?
[1] bigquery 👈
```
```bash
Desired authentication method option:
[1] oauth
[2] service_account 👈
```
```yaml
keyfile: /Users/goku/Downloads/made-with-ml-XXXXXX-XXXXXXXXXXXX.json  # REPLACE
project (GCP project id): made-with-ml-XXXXXX  # REPLACE
dataset: mlops_course
threads: 1
job_execution_timeout_seconds: 300
```
```bash
Desired location option:
[1] US  👈  # or what you picked when defining your dataset in Airbyte DWH destination setup
[2] EU
```

#### Models

We'll prepare our dbt models as we did using the [dbt Cloud IDE](data-stack.md#dbt-cloud){:target="_blank"} in the previous lesson.

```bash
cd dbt_transforms
rm -rf models/example
mkdir models/labeled_projects
touch models/labeled_projects/labeled_projects.sql
touch models/labeled_projects/schema.yml
```

and add the following code to our model files:

```sql linenums="1"
-- models/labeled_projects/labeled_projects.sql
SELECT p.id, created_on, title, description, tag
FROM `made-with-ml-XXXXXX.mlops_course.projects` p  -- REPLACE
LEFT JOIN `made-with-ml-XXXXXX.mlops_course.tags` t  -- REPLACE
ON p.id = t.id
```

```yaml linenums="1"
# models/labeled_projects/schema.yml

version: 2

models:
    - name: labeled_projects
      description: "Tags for all projects"
      columns:
          - name: id
            description: "Unique ID of the project."
            tests:
                - unique
                - not_null
          - name: title
            description: "Title of the project."
            tests:
                - not_null
          - name: description
            description: "Description of the project."
            tests:
                - not_null
          - name: tag
            description: "Labeled tag for the project."
            tests:
                - not_null

```

And we can use the BashOperator to execute our dbt commands like so:

```python linenums="1"
DBT_ROOT_DIR = Path(BASE_DIR, "dbt_transforms")

@dag(...)
def dataops():
    ...
    # Transform
    transform = BashOperator(task_id="transform", bash_command=f"cd {DBT_ROOT_DIR} && dbt run && dbt test")

    # Define DAG
    extract_and_load_projects >> validate_projects
    extract_and_load_tags >> validate_tags
    [validate_projects, validate_tags] >> transform
```

!!! note "Programmatically using dbt Cloud"
    While we developed locally, we could just as easily use Airflow’s [dbt cloud provider](https://airflow.apache.org/docs/apache-airflow-providers-dbt-cloud/){:target=“blank”} to connect to our dbt cloud and use the different operators to schedule jobs. This is recommended for production because we can design jobs with proper environment, authentication, schemas, etc.

    - Connect Airflow with dbt Cloud:

    Go to Admin > Connections > +
    ```yaml
    Connection ID: dbt_cloud_default
    Connection Type: dbt Cloud
    Account ID: View in URL of https://cloud.getdbt.com/
    API Token: View in https://cloud.getdbt.com/#/profile/api/
    ```

    - Transform

    ```bash
    pip install apache-airflow-providers-dbt-cloud==2.1.0
    ```
    ```python linenums="1"
    from airflow.providers.dbt.cloud.operators.dbt import DbtCloudRunJobOperator
    transform = DbtCloudRunJobOperator(
        task_id="transform",
        job_id=118680,  # Go to dbt UI > click left menu > Jobs > Transform > job_id in URL
        wait_for_termination=True,  # wait for job to finish running
        check_interval=10,  # check job status
        timeout=300,  # max time for job to execute
    )
    ```

#### Validate

And of course, we'll want to validate our transformations beyond dbt's built-in methods, using great expectations. We'll create a suite and checkpoint as we did above for our projects and tags data assets.
```bash
great_expectations suite new  # for mlops_course.labeled_projects
```
??? quote "Expectations for `mlops_course.labeled_projects`"
    Table expectations
    ```python linenums="1"
    # data leak
    validator.expect_compound_columns_to_be_unique(column_list=["title", "description"])
    ```

    Column expectations:
    ```python linenums="1"
    # id
    validator.expect_column_values_to_be_unique(column="id")

    # create_on
    validator.expect_column_values_to_not_be_null(column="created_on")

    # title
    validator.expect_column_values_to_not_be_null(column="title")
    validator.expect_column_values_to_be_of_type(column="title", type_="STRING")

    # description
    validator.expect_column_values_to_not_be_null(column="description")
    validator.expect_column_values_to_be_of_type(column="description", type_="STRING")

    # tag
    validator.expect_column_values_to_not_be_null(column="tag")
    validator.expect_column_values_to_be_of_type(column="tag", type_="STRING")
    ```
```bash
great_expectations checkpoint new labeled_projects
```
```yaml
datasource_name: dwh
data_asset_name: mlops_course.labeled_projects
expectation_suite_name: labeled_projects
```

and just like how we added the validation task for our extracted and loaded data, we can do the same for our transformed data in Airflow:

```python linenums="1"
@dag(...)
def dataops():
    ...
    # Transform
    transform = BashOperator(task_id="transform", bash_command=f"cd {DBT_ROOT_DIR} && dbt run && dbt test")
    validate_transforms = GreatExpectationsOperator(
        task_id="validate_transforms",
        checkpoint_name="labeled_projects",
        data_context_root_dir=GE_ROOT_DIR,
        fail_task_on_validation_failure=True,
    )

    # Define DAG
    extract_and_load_projects >> validate_projects
    extract_and_load_tags >> validate_tags
    [validate_projects, validate_tags] >> transform >> validate_transforms
```

<hr>

Now we have our entire DataOps DAG define and executing it will prepare our data from extraction to loading to transformation (and with validation at every step of the way) for [downstream applications](data-stack.md#applications){:target="_blank"}.

<div class="ai-center-all">
    <img src="/static/images/mlops/orchestration/dataops.png" width="700" alt="dataops">
</div>

> Typically we'll use [sensors](https://airflow.apache.org/docs/apache-airflow/stable/_api/airflow/sensors/){:target="_blank"} to trigger workflows when a condition is met or trigger them directly from the external source via API calls, etc. For our ML use cases, this could be at regular intervals or when labeling or monitoring workflows trigger retraining, etc.

## MLOps

Once we have our data prepared, we're ready to create one of the many potential downstream applications that will depend on it. Let's head back to our `mlops-course` project and follow the same [set up instructions](#set-up) for Airflow (you can stop the Airflow webserver and scheduler from our data-engineering project since we'll reuse PORT 8000).

<div class="row">
    <div class="col-md-6">
<div class="highlight"><pre id="__code_2"><span></span><button class="md-clipboard md-icon" title="Copy to clipboard" data-clipboard-target="#__code_2 > code"></button><code tabindex="0"><span class="c1"># Airflow webserver</span>
<span class="nb">source</span> venv/bin/activate
<span class="nb">export</span> <span class="nv">AIRFLOW_HOME</span><span class="o">=</span><span class="si">${</span><span class="nv">PWD</span><span class="si">}</span>/airflow
<span class="nb">export</span> <span class="nv">GOOGLE_APPLICATION_CREDENTIALS</span><span class="o">=</span>/Users/goku/Downloads/made-with-ml-XXXXXX-XXXXXXXXXXXX.json <span class="c1"># REPLACE</span>
airflow webserver --port <span class="m">8080</span>
<span class="c1"># Go to http://localhost:8080</span>
</code></pre></div>
    </div>
    <div class="col-md-6">
<div class="highlight"><pre id="__code_3"><span></span><button class="md-clipboard md-icon" title="Copy to clipboard" data-clipboard-target="#__code_3 > code"></button><code tabindex="0"><span class="c1"># Airflow scheduler</span>
<span class="nb">source</span> venv/bin/activate
<span class="nb">export</span> <span class="nv">AIRFLOW_HOME</span><span class="o">=</span><span class="si">${</span><span class="nv">PWD</span><span class="si">}</span>/airflow
<span class="nb">export</span> <span class="nv">OBJC_DISABLE_INITIALIZE_FORK_SAFETY</span><span class="o">=</span>YES
<span class="nb">export</span> <span class="nv">GOOGLE_APPLICATION_CREDENTIALS</span><span class="o">=</span>~/Downloads/made-with-ml-XXXXXX-XXXXXXXXXXXX.json <span class="c1"># REPLACE</span>
airflow scheduler
</code></pre></div>
    </div>
</div>

```bash
touch airflow/dags/workflows.py
```

```python linenums="1"
# airflow/dags/workflows.py
from pathlib import Path
from airflow.decorators import dag
from airflow.utils.dates import days_ago

# Default DAG args
default_args = {
    "owner": "airflow",
    "catch_up": False,
}

@dag(
    dag_id="mlops",
    description="MLOps tasks.",
    default_args=default_args,
    schedule_interval=None,
    start_date=days_ago(2),
    tags=["mlops"],
)
def mlops():
    """MLOps workflows."""
    pass

# Run DAG
ml = mlops()
```

### Dataset

We already had an `tagifai.elt_data` function defined to prepare our data but if we want to leverage the data inside our data warehouse, we'll want to connect to it.

```bash
pip install google-cloud-bigquery==1.21.0
```

```python linenums="1"
# airflow/dags/workflows.py
from google.cloud import bigquery
from google.oauth2 import service_account

PROJECT_ID = "made-with-ml-XXXXX" # REPLACE
SERVICE_ACCOUNT_KEY_JSON = "/Users/goku/Downloads/made-with-ml-XXXXXX-XXXXXXXXXXXX.json"  # REPLACE

def _extract_from_dwh():
    """Extract labeled data from
    our BigQuery data warehouse and
    save it locally."""
    # Establish connection to DWH
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_KEY_JSON)
    client = bigquery.Client(credentials=credentials, project=PROJECT_ID)

    # Query data
    query_job = client.query("""
        SELECT *
        FROM mlops_course.labeled_projects""")
    results = query_job.result()
    results.to_dataframe().to_csv(Path(config.DATA_DIR, "labeled_projects.csv"), index=False)

@dag(
    dag_id="mlops",
    description="MLOps tasks.",
    default_args=default_args,
    schedule_interval=None,
    start_date=days_ago(2),
    tags=["mlops"],
)
def mlops():
    """MLOps workflows."""
    extract_from_dwh = PythonOperator(
        task_id="extract_data",
        python_callable=_extract_from_dwh,
    )

    # Define DAG
    extract_from_dwh
```

### Validate

Next, we'll use Great Expectations to validate our data. Even though we've already validated our data, it's a best practice to test for data quality whenever there is a hand-off of data from one place to another. We've already created a checkpoint for our `labeled_projects` in our [testing lesson](testing.md#checkpoints){:target="_blank"} so we'll just leverage that inside our MLOps DAG.

```bash
pip install airflow-provider-great-expectations==0.1.1 great-expectations==0.15.19
```

```python linenums="1"
from great_expectations_provider.operators.great_expectations import GreatExpectationsOperator
from config import config

GE_ROOT_DIR = Path(config.BASE_DIR, "tests", "great_expectations")

@dag(...)
def mlops():
    """MLOps workflows."""
    extract_from_dwh = PythonOperator(
        task_id="extract_data",
        python_callable=_extract_from_dwh,
    )
    validate = GreatExpectationsOperator(
        task_id="validate",
        checkpoint_name="labeled_projects",
        data_context_root_dir=GE_ROOT_DIR,
        fail_task_on_validation_failure=True,
    )

    # Define DAG
    extract_from_dwh >> validate
```

### Train

Finally, we'll optimize and train a model using our validated data.

```python linenums="1"
from airflow.operators.python_operator import PythonOperator
from config import config
from tagifai import main

@dag(...)
def mlops():
    """MLOps workflows."""
    ...
    optimize = PythonOperator(
        task_id="optimize",
        python_callable=main.optimize,
        op_kwargs={
            "args_fp": Path(config.CONFIG_DIR, "args.json"),
            "study_name": "optimization",
            "num_trials": 1,
        },
    )
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

<hr>

And with that we have our MLOps workflow defined that uses the prepared data from our DataOps workflow. At this point, we can add additional tasks for offline/online evaluation, deployment, etc. with the same process as above.

<div class="ai-center-all">
    <img src="/static/images/mlops/orchestration/mlops.png" width="700" alt="mlops">
</div>

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

<!-- Course signup -->
{% include "templates/course-signup.md" %}

<!-- Citation -->
{% include "templates/cite.md" %}