---
template: lesson.html
title: Feature Store
description: Using a feature store to connect the DataOps and MLOps pipelines to enable collaborative teams to develop efficiently.
keywords: feature stores, feast, point-in-time correctness, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/mlops.png
repository: https://github.com/GokuMohandas/mlops-course
notebook: https://colab.research.google.com/github/GokuMohandas/mlops-course/blob/main/notebooks/feature_store.ipynb
---

## Intuition

Let's motivate the need for a feature store by chronologically looking at what challenges developers face in their current workflows. Suppose we had a task where we needed to predict something for an entity (ex. user) using their features.

1. **Isolation**: feature development in isolation (for each unique ML application) can lead to duplication of efforts (setting up ingestion pipelines, feature engineering, etc.).
    - `#!js Solution`: create a central feature repository where the entire team contributes maintained features that anyone can use for any application.
2. **Skew**: we may have different pipelines for generating features for training and serving which can introduce skew through the subtle differences.
    - `#!js Solution`: create features using a unified pipeline and store them in a central location that the training and serving pipelines pull from.
3. **Values**: once we set up our data pipelines, we need to ensure that our input feature values are up-to-date so we aren't working with stale data, while maintaining point-in-time correctness so we don't introduce data leaks.
    - `#!js Solution`: retrieve input features for the respective outcomes by pulling what's available when a prediction would be made.

Point-in-time correctness refers to mapping the appropriately up-to-date input feature values to an observed outcome at $t_{n+1}$. This involves knowing the time ($t_n$) that a prediction is needed so we can collect feature values ($X$) at that time.
<div class="ai-center-all">
    <img src="/static/images/mlops/feature_store/point_in_time.png" width="700" alt="point-in-time correctness">
</div>

When actually constructing our feature store, there are several core components we need to have to address these challenges:

- **data ingestion**: ability to ingest data from various sources (databases, data warehouse, etc.) and keep them updated.
- **feature definitions**: ability to define entities and corresponding features
- **historical features**: ability to retrieve historical features to use for training.
- **online features**: ability to retrieve features from a low latency origin for inference.

Each of these components is fairly easy to set up but connecting them all together requires a managed service, SDK layer for interactions, etc. Instead of building from scratch, it's best to leverage one of the production-ready, feature store options such as [Feast](https://feast.dev/){:target="_blank"}, [Hopsworks](https://www.hopsworks.ai/){:target="_blank"}, [Tecton](https://www.tecton.ai/){:target="_blank"}, [Rasgo](https://www.rasgoml.com/){:target="_blank"}, etc. And of course, the large cloud providers have their own feature store options as well (Amazon's [SageMaker Feature Store](https://docs.aws.amazon.com/sagemaker/latest/dg/feature-store.html){:target="_blank"}, Google's [Vertex AI](https://cloud.google.com/vertex-ai/docs/featurestore){:target="_blank"}, etc.)

## Over-engineering

Not all machine learning tasks require a feature store. In fact, our use case is a perfect example of a test that *does not* benefit from a feature store. All of our data points are independent and stateless and there is no entity that has changing features over time. The real utility of a feature store shines when we need to have up-to-date features for an entity that we continually generate predictions for. For example, a user's behavior (clicks, purchases, etc.) on an e-commerce platform or the deliveries a food runner recently made today, etc.

!!! question "When do I need a feature store?"
    Once we've determined if our task can benefit from a feature store, do we build one right away?

    ??? quote "Show answer"
        As usual, it depends.

        Use it from the very beginning if:

        - someone on our team has set one up before
        - we have time to focus on infrastructure

        Delay using it until later if:

        - no one on the team has set one up before
        - we need to iterate on product first (ex. early-stage startup)
        - we want to motivate the need for each advantage of a feature store

        However, If we follow the delayed approach, we need to consider adopting a feature store if we find ourselves repeating feature preparation and serving steps repeatedly for every new project. The time wasted through these repeated efforts will be significantly more than the time needed to set up a feature store (not to mention improving the developer's experience).

## Feast

We're going to leverage [Feast](https://feast.dev/){:target="_blank"} as the feature store for our application for it's ease of local setup, SDK for training/serving, etc.

```bash
# Install Feast and dependencies
pip install feast==0.10.5 PyYAML==5.3.1 -q
```

> All the code accompanying this lesson can be found in this [notebook](https://colab.research.google.com/github/GokuMohandas/mlops-course/blob/main/notebooks/feature_store.ipynb){:target="_blank"}.

### Set up

We're going to create a feature repository at the root of our project. [Feast](https://feast.dev/) will create a configuration file for us and we're going to add an additional [features.py](https://github.com/GokuMohandas/mlops-course/blob/main/features/features.py){:target="_blank"} file to define our features.

> Traditionally, the feature repository would be it's own isolated repository that other services will use to read/write features from.

```bash
mkdir -p stores/feature
mkdir -p data
feast init --minimal --template local features
cd features
touch features.py
```

<pre class="output">
Creating a new Feast repository in /content/features.
</pre>

The initialized feature repository (with the additional file we've added) will include:

```bash
features/
├── feature_store.yaml  - configuration
└── features.py         - feature definitions
```

We're going to configure the locations for our registry and online store (SQLite) in our `feature_store.yaml` file.

<div class="ai-center-all">
    <img src="/static/images/mlops/feature_store/batch.png" width="1000" alt="batch processing">
</div>

- **registry**: contains information about our feature repository, such as data sources, feature views, etc. Since it's in a DB, instead of a Python file, it can very quickly be accessed in production.
- **online store**: DB (SQLite for local) that stores the (latest) features for defined entities to be used for online inference.

If all our [feature definitions](#feature-definitions) look valid, Feast will sync the metadata about Feast objects to the registry. The registry is a tiny database storing most of the same information you have in the feature repository. This step is necessary because the production feature serving infrastructure won't be able to access Python files in the feature repository at run time, but it will be able to efficiently and securely read the feature definitions from the registry.

> When we run Feast locally, the offline store is effectively represented via Pandas point-in-time joins. Whereas, in production, the offline store can be something more robust like [Google BigQuery](https://cloud.google.com/bigquery){:target="_blank"}, [Amazon RedShift](https://aws.amazon.com/redshift/){:target="_blank"}, etc.


We'll go ahead and paste this into our `features/feature_store.yaml` file (the [notebook](https://colab.research.google.com/github/GokuMohandas/mlops-course/blob/main/notebooks/feature_store.ipynb){:target="_blank"} cell is automatically do this):

```yaml
project: features
registry: ../stores/feature/registry.db
provider: local
online_store:
    path: ../stores/feature/online_store.db
```

### Data ingestion

The first step is to establish connections with our data sources (databases, data warehouse, etc.). Feast requires it's [data sources](https://github.com/feast-dev/feast/blob/master/sdk/python/feast/data_source.py){:target="_blank"} to either come from a file ([Parquet](https://databricks.com/glossary/what-is-parquet){:target="_blank"}), data warehouse ([BigQuery](https://cloud.google.com/bigquery){:target="_blank"}) or data stream ([Kafka](https://kafka.apache.org/){:target="_blank"} / [Kinesis](https://aws.amazon.com/kinesis/){:target="_blank"}). We'll convert our generated features file from the DataOps pipeline (`features.json`) into a Parquet file, which is a column-major data format that allows fast feature retrieval and caching benefits (contrary to row-major data formats such as CSV where we have to traverse every single row to collect feature values).


```python linenums="1"
import os
import json
import pandas as pd
from pathlib import Path
from urllib.request import urlopen
```

```python linenums="1"
# Load projects
url = "https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/projects.json"
projects = json.loads(urlopen(url).read())
df = pd.DataFrame(projects)
df["text"] = df.title + " " + df.description
df.drop(["title", "description"], axis=1, inplace=True)
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
      <th>tag</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>2020-02-20 06:43:18</td>
      <td>computer-vision</td>
      <td>Comparison between YOLO and RCNN on real world...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>2020-02-20 06:47:21</td>
      <td>computer-vision</td>
      <td>Show, Infer &amp; Tell: Contextual Inference for C...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>2020-02-24 16:24:45</td>
      <td>graph-learning</td>
      <td>Awesome Graph Classification A collection of i...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15</td>
      <td>2020-02-28 23:55:26</td>
      <td>reinforcement-learning</td>
      <td>Awesome Monte Carlo Tree Search A curated list...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>19</td>
      <td>2020-03-03 13:54:31</td>
      <td>graph-learning</td>
      <td>Diffusion to Vector Reference implementation o...</td>
    </tr>
  </tbody>
</table>
</div></div>
</pre>

```python linenums="1"
# Format timestamp
df.created_on = pd.to_datetime(df.created_on)
```

```python linenums="1"
# Convert to parquet
DATA_DIR = Path(os.getcwd(), "data")
df.to_parquet(
    Path(DATA_DIR, "features.parquet"),
    compression=None,
    allow_truncated_timestamps=True,
)
```

### Feature definitions

Now that we have our data source prepared, we can define our features for the feature store.

```python linenums="1"
from datetime import datetime
from pathlib import Path
from feast import Entity, Feature, FeatureView, ValueType
from feast.data_source import FileSource
from google.protobuf.duration_pb2 import Duration
```

The first step is to define the location of the features (FileSource in our case) and the timestamp column for each data point.

```python linenums="1"
# Read data
START_TIME = "2020-02-17"
project_details = FileSource(
    path=str(Path(DATA_DIR, "features.parquet")),
    event_timestamp_column="created_on",
)
```

Next, we need to define the main entity that each data point pertains to. In our case, each project has a unique ID with features such as text and tags.

```python linenums="1"
# Define an entity
project = Entity(
    name="id",
    value_type=ValueType.INT64,
    description="project id",
)
```

Finally, we're ready to create a [FeatureView](https://docs.feast.dev/concepts/feature-views){:target="_blank"} that loads specific features (`features`), of various [value types](https://api.docs.feast.dev/python/feast.html?highlight=valuetype#feast.value_type.ValueType){:target="_blank"}, from a data source (`input`) for a specific period of time (`ttl`).

```python linenums="1" hl_lines="5 8 13"
# Define a Feature View for each project
project_details_view = FeatureView(
    name="project_details",
    entities=["id"],
    ttl=Duration(
        seconds=(datetime.today() - datetime.strptime(START_TIME, "%Y-%m-%d")).days * 24 * 60 * 60
    ),
    features=[
        Feature(name="text", dtype=ValueType.STRING),
        Feature(name="tag", dtype=ValueType.STRING),
    ],
    online=True,
    input=project_details,
    tags={},
)
```

So let's go ahead and define our feature views by moving this code into our `features/features.py` script (the [notebook](https://colab.research.google.com/github/GokuMohandas/mlops-course/blob/main/notebooks/feature_store.ipynb){:target="_blank"} cell is automatically do this):

??? quote "Show code"
    ```python
    from datetime import datetime
    from pathlib import Path

    from feast import Entity, Feature, FeatureView, ValueType
    from feast.data_source import FileSource
    from google.protobuf.duration_pb2 import Duration


    # Read data
    START_TIME = "2020-02-17"
    project_details = FileSource(
        path="/content/data/features.parquet",
        event_timestamp_column="created_on",
    )

    # Define an entity for the project
    project = Entity(
        name="id",
        value_type=ValueType.INT64,
        description="project id",
    )

    # Define a Feature View for each project
    # Can be used for fetching historical data and online serving
    project_details_view = FeatureView(
        name="project_details",
        entities=["id"],
        ttl=Duration(
            seconds=(datetime.today() - datetime.strptime(START_TIME, "%Y-%m-%d")).days * 24 * 60 * 60
        ),
        features=[
            Feature(name="text", dtype=ValueType.STRING),
            Feature(name="tag", dtype=ValueType.STRING),
        ],
        online=True,
        input=project_details,
        tags={},
    )
    ```

Once we've defined our feature views, we can `apply` it to push a version controlled definition of our features to the registry for fast access. It will also configure our registry and online stores that we've defined in our `feature_store.yaml`.

```bash
cd features
feast apply
```

<pre output="class">
Registered <span style="color: #39BC70; font-weight: 700;">entity id</span>
Registered feature view <span style="color: #39BC70; font-weight: 700;">project_details</span>
Deploying infrastructure for <span style="color: #39BC70; font-weight: 700;">project_details</span>
</pre>

### Historical features

Once we've registered our feature definition, along with the data source, entity definition, etc., we can use it to fetch historical features. This is done via joins using the provided timestamps using pandas for our local setup or BigQuery, Hive, etc. as an offline DB for production.

```python linenums="1"
import pandas as pd
from feast import FeatureStore
```

```python linenums="1"
# Identify entities
project_ids = df.id[0:3].to_list()
now = datetime.now()
timestamps = [datetime(now.year, now.month, now.day)]*len(project_ids)
entity_df = pd.DataFrame.from_dict({"id": project_ids, "event_timestamp": timestamps})
entity_df.head()
```

<div class="output_subarea output_html rendered_html ai-center-all"><div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>event_timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>2022-06-23</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>2022-06-23</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>2022-06-23</td>
    </tr>
  </tbody>
</table>
</div></div>

```python linenums="1"
# Get historical features
store = FeatureStore(repo_path="features")
training_df = store.get_historical_features(
    entity_df=entity_df,
    feature_refs=["project_details:text", "project_details:tag"],
).to_df()
training_df.head()
```

<div class="output_subarea output_html rendered_html ai-center-all"><div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>event_timestamp</th>
      <th>id</th>
      <th>project_details__text</th>
      <th>project_details__tag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2022-06-23 00:00:00+00:00</td>
      <td>6</td>
      <td>Comparison between YOLO and RCNN on real world...</td>
      <td>computer-vision</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2022-06-23 00:00:00+00:00</td>
      <td>7</td>
      <td>Show, Infer &amp; Tell: Contextual Inference for C...</td>
      <td>computer-vision</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2022-06-23 00:00:00+00:00</td>
      <td>9</td>
      <td>Awesome Graph Classification A collection of i...</td>
      <td>graph-learning</td>
    </tr>
  </tbody>
</table>
</div></div>

### Materialize

For online inference, we want to retrieve features very quickly via our online store, as opposed to fetching them from slow joins. However, the features are not in our online store just yet, so we'll need to [materialize](https://docs.feast.dev/quickstart#4-materializing-features-to-the-online-store) them first.

```bash
cd features
CURRENT_TIME=$(date -u +"%Y-%m-%dT%H:%M:%S")
feast materialize-incremental $CURRENT_TIME
```

<pre class="output">
Materializing 1 feature views to 2022-06-23 19:16:05+00:00 into the sqlite online store.
project_details from 2020-02-17 19:16:06+00:00 to 2022-06-23 19:16:05+00:00:
100%|██████████████████████████████████████████████████████████| 955/955 [00:00<00:00, 10596.97it/s]
</pre>

This has moved the features for all of our projects into the online store since this was first time materializing to the online store. When we subsequently run the [`materialize-incremental`](https://docs.feast.dev/getting-started/load-data-into-the-online-store#2-b-materialize-incremental-alternative){:target="_blank"} command, Feast keeps track of previous materializations and so we'll only materialize the new data since the last attempt.

### Online features

Once we've materialized the features (or directly sent to the online store in the stream scenario), we can use the online store to retrieve features.

```python linenums="1"
# Get online features
store = FeatureStore(repo_path="features")
feature_vector = store.get_online_features(
    feature_refs=["project_details:text", "project_details:tag"],
    entity_rows=[{"id": 6}],
).to_dict()
feature_vector
```

```python linenums="1"
{'id': [6],
 'project_details__tag': ['computer-vision'],
 'project_details__text': ['Comparison between YOLO and RCNN on real world videos Bringing theory to experiment is cool. We can easily train models in colab and find the results in minutes.']}
```

## Architecture

### Batch processing

The feature store we implemented above assumes that our task requires [batch processing](../infrastructure/#batch-processing){:target="_blank"}. This means that inference requests on specific entity instances can use features that have been materialized from the offline store. Note that they may not be the most recent feature values for that entity.

<div class="ai-center-all">
    <img src="/static/images/mlops/feature_store/batch.png" width="1000" alt="batch processing">
</div>

1. Application data is stored in a database and/or a data warehouse, etc. And it goes through the [DataOps pipeline](../pipelines/#dataops){:target="_blank"} to validate the data and engineer the features.
2. These features are written to the offline store which can then be used to retrieve [historical training data](#historical-features) to train a model with. In our local set up, this was join via Pandas DataFrame joins for given timestamps and entity IDs but in a production setting, something like Google BigQuery or Hive would receive the feature requests.
3. Once we have our training data, we can start the workflows to optimize, train and validate a model.
4. We can incrementally [materialize](#online-features) features to the online store so that we can retrieve an entity's feature values with low latency. In our local set up, this was join via SQLite for a given set of entities but in a production setting, something like Redis or DynamoDB would be used.
5. These online features are passed on to the deployed model to generate predictions that would be used downstream.

!!! warning
    Had our entity (projects) had features that change over time, we would materialize them to the online store incrementally. But since they don't, this would be considered over engineering but it's important to know how to leverage a feature store for entities with changing features over time.

### Stream processing

Some applications may require [stream processing](../infrastructure/#stream-processing){:target="_blank"} where we require near real-time feature values to deliver up-to-date predictions at low latency. While we'll still utilize an offline store for retrieving historical data, our application's real-time event data will go directly through our data streams to an online store for serving. An example where stream processing would be needed is when we want to retrieve real-time user session behavior (clicks, purchases) in an e-commerce platform so that we can recommend the appropriate items from our catalog.

<div class="ai-center-all">
    <img src="/static/images/mlops/feature_store/stream.png" width="1000" alt="stream processing">
</div>

1. Real-time event data enters our running data streams ([Kafka](https://kafka.apache.org/){:target="_blank"} / [Kinesis](https://aws.amazon.com/kinesis/){:target="_blank"}, etc.) where they can be processed to generate features.
2. These features are written to the online store which can then be used to retrieve [online features](#online-features) for serving at low latency. In our local set up, this was join via SQLite for a given set of entities but in a production setting, something like Redis or DynamoDB would be used.
3. Streaming features are also written from the data stream to the batch data source (data warehouse, db, etc.) to be processed for generating training data later on.
4. Historical data will be validated and used to generate features for training a model. This cadence for how often this happens depends on whether there are data annotation lags, compute constraints, etc.

> There are a few more components we're not visualizing here such as the unified ingestion layer (Spark), that connects data from the varied data sources (warehouse, DB, etc.) to the offline/online stores, or low latency serving (<10 ms). We can read more about all of these in the official [Feast Documentation](https://docs.feast.dev/){:target="_blank"}, which also has [guides](https://docs.feast.dev/how-to-guides/feast-gcp-aws){:target="_blank"} to set up a feature store with Feast with AWS, GCP, etc.


## Additional functionality

Additional functionality that many feature store providers are currently (or recently) trying to integrate within the feature store platform include:

- **transform**: ability to directly apply global preprocessing or feature engineering on top of raw data extracted from data sources.
    - `#!js Current solution`: apply transformations as a separate Spark, Python, etc. workflow task before writing to the feature store.
- **validate**: ability to assert [expectations](../testing/#expectations){:target="_blank"} and identify [data drift](../monitoring/#data-drift){:target="_blank"} on the feature values.
    - `#!js Current solution`: apply data testing and monitoring as upstream workflow tasks before they are written to the feature store.
- **discover**: ability for anyone in our team to easily discover features that they can leverage for their application.
    - `#!js Current solution`: add a data discovery engine, such as [Amundsen](https://www.amundsen.io/){:target="_blank"}, on top of our feature store to enable others to search for features.

## Reproducibility

Though we could continue to [version](versioning.md){:target="blank"} our training data with [DVC](https://dvc.org/){:target="_blank"} whenever we release a version of the model, it might not be necessary. When we pull data from source or compute features, should they save the data itself or just the operations?

- **Version the data**
    - This is okay if (1) the data is manageable, (2) if our team is small/early stage ML or (3) if changes to the data are infrequent.
    - But what happens as data becomes larger and larger and we keep making copies of it.
- **Version the operations**
    - We could keep snapshots of the data (separate from our projects) and provided the operations and timestamp, we can execute operations on those snapshots of the data to recreate the precise data artifact used for training. Many data systems use [time-travel](https://docs.snowflake.com/en/user-guide/data-time-travel.html){:target="blank"} to achieve this efficiently.
    - But eventually this also results in data storage bulk. What we need is an *append-only* data source where all changes are kept in a log instead of directly changing the data itself. So we can use the data system with the logs to produce versions of the data as they were without having to store separate snapshots of the the data itself.

Regardless of the choice above, feature stores are very useful here. Instead of coupling data pulls and feature compute with the time of modeling, we can separate these two processes so that features are up-to-date when we need them. And we can still achieve reproducibility via efficient point-in-time correctness, low latency snapshots, etc. This essentially creates the ability to work with any version of the dataset at any point in time.


## Resources

- [Feast Documentation](https://docs.feast.dev/){:target="_blank"}
- [Feature Store for ML](https://www.featurestore.org/){:target="_blank"}
- [Understanding & Using Time Travel](https://docs.snowflake.com/en/user-guide/data-time-travel.html){:target="_blank"}