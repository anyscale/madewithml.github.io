---
template: lesson.html
title: Data Stack for Machine Learning
description: Learn data engineering fundamentals by constructing a modern data stack for analytics and machine learning applications.
keywords: data engineering, data stack, modern data stack, data warehouse, snowflake, fivetran, airbyte, dbt, systems design, mlops, machine learning
image: https://madewithml.com/static/images/mlops.png
repository: https://github.com/GokuMohandas/data-engineering
notebook: https://github.com/GokuMohandas/data-engineering/blob/main/extract_from_dwh.ipynb
---

{% include "styles/lesson.md" %}

## Intuition

So far we've had the convenience of using local CSV files as data source but in reality, our data can come from many disparate sources. Additionally, our processes around transforming and testing our data should ideally be moved upstream so that many different downstream processes can benefit from them. Our ML use case being just one among the many potential downstream applications. To address these shortcomings, we're going to learn about the fundamentals of data engineering and construct a modern data stack that can scale and provide high quality data for our applications.

> View the :fontawesome-brands-github:{ .github } [**data-engineering**](https://github.com/GokuMohandas/data-engineering){:target="_blank"} repository for all the code.

<!-- Youtube Video -->
<!--
<div class="ai-yt-mobile">
    <iframe id="yt-video-mobile" frameborder="0" scrolling="no" marginheight="0" marginwidth="0" width="620" height="348.75" type="text/html" src="" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

<div class="ai-yt-desktop">
    <iframe id="yt-video-desktop" frameborder="0" scrolling="no" marginheight="0" marginwidth="0" width="620" height="348.75" type="text/html" src="" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

<div class="ai-yt-admonition">
    <div class="admonition danger">
        <p class="admonition-title">Viewing options</p>
        <p>You can watch the entire video at once or follow along each section (each header has a link to the corresponding timestamp in the video).</p>
    </div>
</div>

<script>
    var yt_video_id = "_sYrVHGRqPo"
    var yt_video_source = "https://www.youtube.com/embed/" + yt_video_id + "?autoplay=0&fs=1&iv_load_policy=1&showinfo=1&rel=0&cc_load_policy=0&start=120&end=124&vq=hd720"
    document.getElementById("yt-video-mobile").src = yt_video_source;
    document.getElementById("yt-video-desktop").src = yt_video_source;
</script> -->

At a high level, we're going to:

1. [**E**xtract and **L**oad](#extract-and-load) data from [sources](#sources) to [destinations](#destinations).
2. [**T**ransform](#transform) data for downstream applications.

This process is more commonly known as ELT, but there are variants such as ETL and reverse ETL, etc. They are all essentially the same underlying workflows but have slight differences in the order of data flow and where data is processed and stored.

<div class="ai-center-all">
    <img width="800" src="/static/images/mlops/data_stack/data.png" alt="data stack">
</div>

!!! tip "Utility and simplicity"
    It can be enticing to set up a modern data stack in your organization, especially with all the hype. But it's very important to motivate utility and adding additional complexity:

    - Start with a use case that we already have data sources for and has direct impact on the business' bottom line (ex. user churn).
    - Start with the simplest infrastructure (source → database → report) and add complexity (in infrastructure, performance and team) as needed.

## Data systems

<!-- **[:fontawesome-brands-youtube:{ .youtube } 1:10 - 3:15](){:target="_blank"}** -->

Before we start working with our data, it's important to understand the different types of systems that our data can live in. So far in this course we've worked with files, but there are several types of data systems that are widely adopted in industry for different purposes.

<div class="ai-center-all">
    <img width="450" src="/static/images/mlops/data_stack/systems.png" alt="data systems">
</div>

### Data lake

A data lake is a flat data management system that stores raw objects. It's a great option for inexpensive storage and has the capability to hold all types of data (unstructured, semi-structured and structured). Object stores are becoming the standard for data lakes with default options across the popular cloud providers. Unfortunately, because data is stored as objects in a data lake, it's not designed for operating on structured data.

> Popular data lake options include [Amazon S3](https://aws.amazon.com/s3/){:target="_blank"}, [Azure Blob Storage](https://azure.microsoft.com/en-us/services/storage/blobs/){:target="_blank"}, [Google Cloud Storage](https://cloud.google.com/storage){:target="_blank"}, etc.

### Database

Another popular storage option is a database (DB), which is an organized collection of structured data that adheres to either:

- relational schema (tables with rows and columns) often referred to as a Relational Database Management System (RDBMS) or SQL database.
- non-relational (key/value, graph, etc.), often referred to as a non-relational database or NoSQL database.

A database is an [online transaction processing (OLTP)](https://en.wikipedia.org/wiki/Online_transaction_processing){:target="_blank"} system because it's typically used for day-to-day CRUD (create, read, update, delete) operations where typically information is accessed by rows. However, they're generally used to store data from one application and is not designed to hold data from across many sources for the purpose of analytics.

> Popular database options include [PostgreSQL](https://www.postgresql.org/){:target="_blank"}, [MySQL](https://www.mysql.com/){:target="_blank"}, [MongoDB](https://www.mongodb.com/){:target="_blank"}, [Cassandra](https://cassandra.apache.org/){:target="_blank"}, etc.

### Data warehouse

A data warehouse (DWH) is a type of database that's designed for storing structured data from many different sources for downstream analytics and data science. It's an [online analytical processing (OLAP)](https://en.wikipedia.org/wiki/Online_analytical_processing){:target="_blank"} system that's optimized for performing operations across aggregating column values rather than accessing specific rows.

> Popular data warehouse options include [SnowFlake](https://www.snowflake.com/){:target="_blank"}, [Google BigQuery](https://cloud.google.com/bigquery){:target="_blank"}, [Amazon RedShift](https://aws.amazon.com/redshift/){:target="_blank"}, [Hive](https://hive.apache.org/){:target="_blank"}, etc.

## Extract and load

The first step in our data pipeline is to extract data from a source and load it into the appropriate destination. While we could construct custom scripts to do this manually or on a schedule, an ecosystem of data ingestion tools have already standardized the entire process. They all come equipped with connectors that allow for extraction, normalization, cleaning and loading between sources and destinations. And these pipelines can be scaled, monitored, etc. all with very little to no code.

<div class="ai-center-all">
    <img width="600" src="/static/images/mlops/data_stack/pipelines.png" alt="ingestion pipelines">
</div>

> Popular data ingestion tools include [Fivetran](https://www.fivetran.com/){:target="_blank"}, [Airbyte](https://airbyte.com/){:target="_blank"}, [Stitch](https://www.stitchdata.com/){:target="_blank"}, etc.

We're going to use the open-source tool [Airbyte](https://airbyte.com/){:target="_blank"} to create connections between our data sources and destinations. Let's set up Airbyte and define our data sources. As we progress in this lesson, we'll set up our destinations and create connections to extract and load data.

1. Ensure that we still have Docker installed from our [Docker lesson](docker.md){:target="_blank"} but if not, download it [here](https://www.docker.com/products/docker-desktop/){:target="_blank"}. For Windows users, be sure to have these [configurations](https://docs.airbyte.com/deploying-airbyte/local-deployment/#deploy-on-windows){:target="_blank"} enabled.
2. In a parent directory, outside our project directory for the MLOps course, execute the following commands to load the Airbyte repository locally and launch the service.
```bash
git clone https://github.com/airbytehq/airbyte.git
cd airbyte
docker-compose up
```
3. After a few minutes, visit [http://localhost:8000/](http://localhost:8000/){:target="_blank"} to view the launched Airbyte service.

### Sources

Our data sources we want to extract from can be from anywhere. They could come from 3rd party apps, files, user click streams, physical devices, data lakes, databases, data warehouses, etc. But regardless of the source of our data, they type of data should fit into one of these categories:

- `#!js structured`: organized data stored in an explicit structure (ex. tables)
- `#!js semi-structured`: data with some structure but no formal schema or data types (web pages, CSV, JSON, etc.)
- `#!js unstructured`: qualitative data with no formal structure (text, images, audio, etc.)

For our application, we'll define two data sources:

- [projects.csv](https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/projects.csv){:target="_blank"}: data containing projects with their ID, create date, title and description.
- [tags.csv](https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/tags.csv){:target="_blank"}: labels for each of project IDs in projects.csv

> Ideally, these data assets would be retrieved from a database that contains projects that we extracted and perhaps another database that stores labels from our labeling team's workflows. However, for simplicity we'll use CSV files to demonstrate how to define a data source.

#### Define file source in Airbyte

We'll start our ELT process by defining the data source in Airbyte:

1. On our [Airbyte UI](http://localhost:8000/){:target="_blank"}, click on `Sources` on the left menu. Then click the `+ New source` button on the top right corner.
2. Click on the `Source type` dropdown and choose `File`. This will open a view to define our file data source.
```yaml
Name: Projects
URL: https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/projects.csv
File Format: csv
Storage Provider: HTTPS: Public Web
Dataset Name: projects
```
3. Click the `Set up source` button and our data source will be tested and saved.
4. Repeat steps 1-3 for our tags data source as well:
```yaml
Name: Tags
URL: https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/tags.csv
File Format: csv
Storage Provider: HTTPS: Public Web
Dataset Name: tags
```

<div class="ai-center-all">
    <img width="1000" src="/static/images/mlops/data_stack/sources.png" alt="data sources">
</div>

### Destinations

Once we know the source we want to extract data from, we need to decide the destination to load it. The choice depends on what our downstream applications want to be able to do with the data. And it's also common to store data in one location (ex. data lake) and move it somewhere else (ex. data warehouse) for specific processing.

#### Set up Google BigQuery

Our destination will be a [data warehouse](#data-warehouse) since we'll want to use the data for downstream analytical and machine learning applications. We're going to use [Google BigQuery](https://cloud.google.com/bigquery){:target="_blank"} which is free under Google Cloud's [free tier](https://cloud.google.com/bigquery/pricing#free-tier){:target="_blank"} for up to 10 GB storage and 1TB of queries (which is significantly more than we'll ever need for our purpose).

1. Log into your [Google account](https://accounts.google.com/signin){:target="_blank} and then head over to [Google CLoud](https://cloud.google.com/){:target="_blank"}. If you haven't already used Google Cloud's free trial, you'll have to sign up. It's free and you won't be autocharged unless you manually upgrade your account. Once the trial ends, we'll still have the free tier which is more than plenty for us.
2. Go to the [Google BigQuery page](https://console.cloud.google.com/bigquery){:target="_blank} and click on the `Go to console` button.
3. We can create a new project by following these [instructions](https://cloud.google.com/resource-manager/docs/creating-managing-projects#console){:target="_blank"} which will lead us to the [create project page](https://console.cloud.google.com/projectcreate){:target="_blank"}.
```yaml
Project name: made-with-ml  # Google will append a unique ID to the end of it
Location: No organization
```
4. Once the project has been created, refresh the page and we should see it (along with few other default projects from Google).

```bash
# Google BigQuery projects
├── made-with-ml-XXXXXX   👈 our project
├── bigquery-publicdata
├── imjasonh-storage
└── nyc-tlc
```

!!! tip "Console or code"
    Most cloud providers will allow us to do everything via console but also programmatically via API, Python, etc. For example, we manually create a project but we could've also done so with code as shown [here](https://cloud.google.com/resource-manager/docs/creating-managing-projects#python){:target="_blank"}.

#### Define BigQuery destination in Airbyte

Next, we need to establish the connection between Airbyte and BigQuery so that we can load the extracted data to the destination. In order to authenticate our access to BigQuery with Airbyte, we'll need to create a service account and generate a secret key. This is basically creating an identity with certain access that we can use for verification. Follow these [instructions](https://cloud.google.com/iam/docs/creating-managing-service-account-keys#iam-service-account-keys-create-console){:target="_blank"} to create a service and generate the key file (JSON). Note down the location of this file because we'll be using it throughout this lesson. For example ours is `/Users/goku/Downloads/made-with-ml-XXXXXX-XXXXXXXXXXXX.json`.

1. On our [Airbyte UI](http://localhost:8000/){:target="_blank"}, click on `Destinations` on the left menu. Then click the `+ New destination` button on the top right corner.
2. Click on the `Destination type` dropdown and choose `BigQuery`. This will open a view to define our file data source.
```yaml
Name: BigQuery
Default Dataset ID: mlops_course  # where our data will go inside our BigQuery project
Project ID: made-with-ml-XXXXXX  # REPLACE this with your Google BiqQuery Project ID
Credentials JSON: SERVICE-ACCOUNT-KEY.json  # REPLACE this with your service account JSON location
Dataset location: US  # select US or EU, all other options will not be compatible with dbt later
```
3. Click the `Set up destination` button and our data destination will be tested and saved.

<div class="ai-center-all">
    <img width="1000" src="/static/images/mlops/data_stack/destinations.png" alt="data destinations">
</div>

### Connections

So we've set up our data sources (public CSV files) and destination (Google BigQuery data warehouse) but they haven't been connected yet. To create the connection, we need to think about a few aspects.

#### Frequency

How often do we want to extract data from the sources and load it into the destination?

- `#!js batch`: extracting data in batches, usually following a schedule (ex. daily) or when an event of interest occurs (ex. new data count)
- `#!js streaming`: extracting data in a continuous stream (using tools like [Kafka](https://kafka.apache.org/){:target="_blank"}, [Kinesis](https://aws.amazon.com/kinesis/){:target="_blank"}, etc.)

!!! note "Micro-batch"
    As we keep decreasing the time between batch ingestion (ex. towards 0), do we have stream ingestion? Not exactly. Batch processing is deliberately deciding to extract data from a source at a given interval. As that interval becomes <15 minutes, it's referred to as a micro-batch (many data warehouses allow for batch ingestion every 5 minutes). However, with stream ingestion, the extraction process is continuously on and events will keep being ingested.

!!! tip "Start simple"
    In general, it's a good idea to start with batch ingestion for most applications and slowly add the complexity of streaming ingestion (and additional infrastructure). This was we can prove that downstream applications are finding value from the data source and evolving to streaming later should only improve things.

> We'll learn more about the different system design implications of batch vs. stream in our [systems design lesson](systems-design.md){:target="_blank"}.

#### Connecting File source to BigQuery destination

Now we're ready to create the connection between our sources and destination:

1. On our [Airbyte UI](http://localhost:8000/){:target="_blank"}, click on `Connections` on the left menu. Then click the `+ New connection` button on the top right corner.
2. Under `Select a existing source`, click on the `Source` dropdown and choose `Projects` and click `Use existing source`.
3. Under `Select a existing destination`, click on the `Destination` dropdown and choose `BigQuery` and click `Use existing destination`.
```yaml
Connection name: Projects <> BigQuery
Replication frequency: Manual
Destination Namespace: Mirror source structure
Normalized tabular data: True  # leave this selected
```
4. Click the `Set up connection` button and our connection will be tested and saved.
5. Repeat the same for our `Tags` source with the same `BigQuery` destination.

> Notice that our sync mode is `Full refresh | Overwrite` which means that every time we sync data from our source, it'll overwrite the existing data in our destination. As opposed to `Full refresh | Append` which will add entries from the source to bottom of the previous syncs.

<div class="ai-center-all">
    <img width="1000" src="/static/images/mlops/data_stack/connections.png" alt="data connections">
</div>

#### Data sync

Our replication frequency is `Manual` because we'll trigger the data syncs ourselves:

1. On our [Airbyte UI](http://localhost:8000/){:target="_blank"}, click on `Connections` on the left menu. Then click the `Projects <> BigQuery` connection we set up earlier.
2. Press the `🔄 Sync now` button and once it's completed we'll see that the projects are now in our BigQuery data warehouse.
3. Repeat the same with our `Tags <> BigQuery` connection.

```bash
# Inside our data warehouse
made-with-ml-XXXXXX               - Project
└── mlops_course                  - Dataset
│   ├── _airbyte_raw_projects     - table
│   ├── _airbyte_raw_tags         - table
│   ├── projects                  - table
│   └── tags                      - table
```

>In our [orchestration lesson](orchestration.md){:target="_blank"}, we'll use Airflow to programmatically execute the data sync.

We can easily explore and query this data using SQL directly inside our warehouse:

1. On our BigQuery project page, click on the `🔍 QUERY` button and select `In new tab`.
2. Run the following SQL statement and view the data:
```sql linenums="1"
SELECT *
FROM `made-with-ml-XXXXXX.mlops_course.projects`
LIMIT 1000
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>2020-02-20 06:43:18</td>
      <td>Comparison between YOLO and RCNN on real world...</td>
      <td>Bringing theory to experiment is cool. We can ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>2020-02-20 06:47:21</td>
      <td>Show, Infer &amp; Tell: Contextual Inference for C...</td>
      <td>The beauty of the work lies in the way it arch...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>2020-02-24 16:24:45</td>
      <td>Awesome Graph Classification</td>
      <td>A collection of important graph embedding, cla...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15</td>
      <td>2020-02-28 23:55:26</td>
      <td>Awesome Monte Carlo Tree Search</td>
      <td>A curated list of Monte Carlo tree search papers...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>19</td>
      <td>2020-03-03 13:54:31</td>
      <td>Diffusion to Vector</td>
      <td>Reference implementation of Diffusion2Vec (Com...</td>
    </tr>
  </tbody>
</table>
</div></div>
</pre>

### Best practices

With the advent of cheap storage and cloud SaaS options to manage them, it's become a best practice to store raw data into data lakes. This allows for storage of raw, potentially unstructured, data without having to justify storage with downstream applications. When we do need to transform and process the data, we can move it to a data warehouse so can perform those operations efficiently.

<div class="ai-center-all mt-4">
    <img width="600" src="/static/images/mlops/data_stack/best_practice.png" alt="best practice">
</div>

## Transform

Once we've extracted and loaded our data, we need to transform the data so that it's ready for downstream applications. These transformations are different from the [preprocessing](preprocessing.md#transformations){:target="_blank"} we've seen before but are instead reflective of business logic that's agnostic to downstream applications. Common transformations include defining schemas, filtering, cleaning and joining data across tables, etc. While we could do all of these things with SQL in our data warehouse (save queries as tables or views), dbt delivers production functionality around version control, testing, documentation, packaging, etc. out of the box. This becomes crucial for maintaining observability and high quality data workflows.

<div class="ai-center-all mb-4">
    <img width="500" src="/static/images/mlops/data_stack/transform.png" alt="data transform">
</div>

> Popular transformation tools include [dbt](https://www.getdbt.com/){:target="_blank"}, [Matillion](https://www.matillion.com/){:target="_blank"}, custom jinja templated SQL, etc.

!!! note
    In addition to data transformations, we can also process the data using large-scale analytics engines like [Spark](https://spark.apache.org/){:target="_blank"}, [Flink](https://flink.apache.org/){:target="_blank"}, etc.

### dbt Cloud

Now we're ready to transform our data in our data warehouse using [dbt](https://www.getdbt.com/){:target="_blank"}. We'll be using a developer account on dbt Cloud (free), which provides us with an IDE, unlimited runs, etc.

> We'll learn how to use the [dbt-core](https://github.com/dbt-labs/dbt-core){:target="_blank"} in our [orchestration lesson](orchestration.md){:target="_blank"}. Unlike dbt Cloud, dbt core is completely open-source and we can programmatically connect to our data warehouse and perform transformations.

1. Create a [free account](https://www.getdbt.com/signup/){:target="_blank"} and verify it.
2. Go to [https://cloud.getdbt.com/](https://cloud.getdbt.com/){:target="_blank"} to get set up.
3. Click `continue` and choose `BigQuery` as the database.
4. Click `Upload a Service Account JSON file` and upload our file to autopopulate everything.
5. Click the `Test` > `Continue`.
6. Click `Managed` repository and name it `dbt-transforms` (or anything else you want).
7. Click `Create` > `Continue` > `Skip and complete`.
8. This will open the project page and click `>_ Start Developing` button.
9. This will open the IDE where we can click `🗂 initialize your project`.

Now we're ready to start developing our models:

1. Click the `···` next to the `models` directory on the left menu.
2. Click `New folder` called `models/labeled_projects`.
3. Create a `New file` under `models/labeled_projects` called `labeled_projects.sql`.
4. Repeat for another file under `models/labeled_projects` called `schema.yml`.

```bash
dbt-cloud-XXXXX-dbt-transforms
├── ...
├── models
│   ├── example
│   └── labeled_projects
│   │   ├── labeled_projects.sql
│   │   └── schema.yml
├── ...
└── README.md
```

### Joins

Inside our `models/labeled_projects/labeled_projects.sql` file we'll create a view that joins our project data with the appropriate tags. This will create the labeled data necessary for downstream applications such as machine learning models. Here we're joining based on the matching id between the projects and tags:

```sql linenums="1"
-- models/labeled_projects/labeled_projects.sql
SELECT p.id, created_on, title, description, tag
FROM `made-with-ml-XXXXXX.mlops_course.projects` p  -- REPLACE
LEFT JOIN `made-with-ml-XXXXXX.mlops_course.tags` t  -- REPLACE
ON p.id = t.id
```

We can view the queried results by clicking the `Preview` button and view the data lineage as well.

### Schemas

Inside our `models/labeled_projects/schema.yml` file we'll define the schemas for each of the features in our transformed data. We also define several tests that each feature should pass. View the full list of [dbt tests](https://docs.getdbt.com/docs/building-a-dbt-project/tests){:target="_blank"} but note that we'll use [Great Expectations](https://madewithml.com/courses/mlops/testing/#expectations){:target="_blank"} for more comprehensive tests when we orchestrate all these data workflows in our [orchestration lesson](orchestration.md){:target="_blank"}.


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

### Runs

At the bottom of the IDE, we can execute runs based on the transformations we've defined. We'll run each of the following commands and once they finish, we can see the transformed data inside our data warehouse.

```bash
dbt run
dbt test
```

Once these commands run successfully, we're ready to move our transformations to a production environment where we can insert this view in our data warehouse.

### Jobs

In order to apply these transformations to the data in our data warehouse, it's best practice to create an [Environment](https://docs.getdbt.com/guides/legacy/managing-environments){:target="_blank"} and then define [Jobs](https://docs.getdbt.com/guides/getting-started/building-your-first-project/schedule-a-job){:target="_blank"}:

1. Click `Environments` on the left menu > `New Environment` button (top right corner) and fill out the details:
```yaml
Name: Production
Type: Deployment
...
Dataset: mlops_course
```
2. Click `New Job` with the following details and then click `Save` (top right corner).
```yaml
Name: Transform
Environment: Production
Commands: dbt run
          dbt test
Schedule: uncheck "RUN ON SCHEDULE"
```
3. Click `Run Now` and view the transformed data in our data warehouse under a view called `labeled_projects`.

```bash
# Inside our data warehouse
made-with-ml-XXXXXX               - Project
└── mlops_course                  - Dataset
│   ├── _airbyte_raw_projects     - table
│   ├── _airbyte_raw_tags         - table
│   ├── labeled_projects          - view
│   ├── projects                  - table
│   └── tags                      - table
```

<div class="ai-center-all">
    <img width="800" src="/static/images/mlops/data_stack/dbt_run.png" alt="dbt run">
</div>

> There is so much more to dbt so be sure to check out their [official documentation](https://docs.getdbt.com/docs/building-a-dbt-project/documentation){:target="_blank"} to really customize any workflows. And be sure to check out our [orchestration lesson](orchestration.md){:target="_blank"} where we'll programmatically create and execute our dbt transformations.


## Implementations

Hopefully we created our data stack for the purpose of gaining some actionable insight about our business, users, etc. Because it's these use cases that dictate which sources of data we extract from, how often and how that data is stored and transformed. Downstream applications of our data typically fall into one of these categories:

- `#!js data analytics`: use cases focused on reporting trends, aggregate views, etc. via charts, dashboards, etc.for the purpose of providing operational insight for business stakeholders.
> 🛠&nbsp; Popular tools: [Tableau](https://www.tableau.com/){:target="_blank"}, [Looker](https://www.looker.com/){:target="_blank"}, [Metabase](https://www.metabase.com/){:target="_blank"}, [Superset](https://superset.apache.org/){:target="_blank"}, etc.
- `#!js machine learning`: use cases centered around using the transformed data to construct predictive models (forecasting, personalization, etc.).

While it's very easy to extract data from our data warehouse:

```bash
pip install google-cloud-bigquery==1.21.0
```
```python
from google.cloud import bigquery
from google.oauth2 import service_account

# Replace these with your own values
project_id = "made-with-ml-XXXXXX"  # REPLACE
SERVICE_ACCOUNT_KEY_JSON = "/Users/goku/Downloads/made-with-ml-XXXXXX-XXXXXXXXXXXX.json"  # REPLACE

# Establish connection
credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_KEY_JSON)
client = bigquery.Client(credentials= credentials, project=project_id)

# Query data
query_job = client.query("""
   SELECT *
   FROM mlops_course.labeled_projects""")
results = query_job.result()
results.to_dataframe().head()
```

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
      <td>1994.0</td>
      <td>2020-07-29 04:51:30</td>
      <td>Understanding the Effectivity of Ensembles in ...</td>
      <td>The report explores the ideas presented in Dee...</td>
      <td>computer-vision</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1506.0</td>
      <td>2020-06-19 06:26:17</td>
      <td>Using GitHub Actions for MLOps &amp; Data Science</td>
      <td>A collection of resources on how to facilitate...</td>
      <td>mlops</td>
    </tr>
    <tr>
      <th>2</th>
      <td>807.0</td>
      <td>2020-05-11 02:25:51</td>
      <td>Introduction to Machine Learning Problem Framing</td>
      <td>This course helps you frame machine learning (...</td>
      <td>mlops</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1204.0</td>
      <td>2020-06-05 22:56:38</td>
      <td>Snaked: Classifying Snake Species using Images</td>
      <td>Proof of concept that it is possible to identi...</td>
      <td>computer-vision</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1706.0</td>
      <td>2020-07-04 11:05:28</td>
      <td>PokeZoo</td>
      <td>A deep learning based web-app developed using ...</td>
      <td>computer-vision</td>
    </tr>
  </tbody>
</table>
</div></div>

!!! warning
    Check out our [notebook](https://github.com/GokuMohandas/data-engineering/blob/main/extract_from_dwh.ipynb){:target="_blank"} where we extract the transformed data from our data warehouse. We do this in a separate notebook because it requires the `google-cloud-bigquery` package and until dbt [loosens](https://github.com/dbt-labs/dbt-core/issues/4748){:target="_blank"} it's Jinja versioning constraints... it'll have to be done in a separate environment. However, downstream applications are typically analytics or ML applications which have their own environments anyway so these conflicts are not inhibiting.

many of the analytics (ex. dashboards) and machine learning solutions (ex. feature stores) allow for easy connection to our data warehouses so that workflows can be triggered when an event occurs or on a schedule. We're going to take this a step further in the [next lesson](orchestration.md){:target="_blank"} where we'll use a central orchestration platform to control all these workflows.

!!! tip "Analytics first, then ML"
    It's a good idea for the first several applications to be analytics and reporting based in order to establish a robust data stack. These use cases typically just involve displaying data aggregations and trends, as opposed to machine learning systems that involve additional complex [infrastructure](feature-store.md){:target="_blank"} and [workflows](orchestration.md#mlops){:target="_blank"}.


## Observability

When we create complex data workflows like this, observability becomes a top priority. Data observability is the general  concept of understanding the condition of data in our system and it involves:

- `#!js data quality`: testing and monitoring our [data quality](testing.md#data){:target="_blank"} after every step (schemas, completeness, recency, etc.).
- `#!js data lineage`: mapping the where data comes from and how it's being transformed as it moves through our pipelines.
- `#!js discoverability`: enabling discovery of the different data sources and features for downstream applications.
- `#!js privacy + security`: are the different data assets treated and restricted appropriately amongst the applications?

> Popular observability tools include [Monte Carlo](https://www.montecarlodata.com/){:target="_blank"}, [Bigeye](https://www.bigeye.com/){:target="_blank"}, etc.

## Considerations

The data stack ecosystem to create the robust data workflows is growing and maturing. However, it can be overwhelming when it comes to choosing the best tooling options, especially as needs change over time. Here are a few important factors to consider when making a tooling decision in this space:

- What is the cost per time per employee? Some of the tooling options can rack up quite the bill!
- Does the tool have the proper connectors to integrate with our data sources and the rest of the stack?
- Does the tool fit with our team's technical aptitude (SQL, Spark, Python, etc.)?
- What kind of support does the tool offer (enterprise, community, etc.)?

<!-- Course signup -->
{% include "templates/signup.md" %}

<!-- Citation -->
{% include "templates/cite.md" %}