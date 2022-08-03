---
template: lesson.html
title: Data Stacks for Machine Learning
description: A look at the different data systems available for enabling machine learning applications.
keywords: data stack, modern data stack, data warehouse, snowflake, fivetran, airbyte, dbt, systems design, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/mlops.png
repository: https://github.com/GokuMohandas/mlops-course
---

{% include "styles/lesson.md" %}

## Data systems

We also want to set the scene for where our data could live. So far, we've been using a JSON file (which is great for quick POCs) but for production, we'll need to rely on much more reliable sources of data.

<div class="ai-center-all">
    <img width="1000" src="/static/images/mlops/infrastructure/data.png" alt="data systems">
</div>

1. First, we have our **data sources**, which can be from APIs, users, other databases, etc. and can be:

    - `#!js structured`: organized data stored in an explicit structure (ex. tables)
    - `#!js semi-structured`: data with some structure but no formal schema or data types (web pages, CSV, JSON, etc.)
    - `#!js unstructured`: qualitative data with no formal structure (text, images, audio, etc.)

2. These data sources are usually consumed by a **data lake**, which is a central repository that stores the data in its raw format. Traditionally, **object stores**, which manage as objects, as opposed to files under a certain structure, are used as the storage platform for the data lake.

    🛠&nbsp; [Amazon S3](https://aws.amazon.com/s3/){:target="_blank"}, [Google Cloud Storage](https://cloud.google.com/storage){:target="_blank"}, etc.

3. Raw data from data lakes are moved to more organized storage options via [ETL](https://en.wikipedia.org/wiki/Extract,_transform,_load){:target="_blank"} (extract, transform, load) pipelines.

    🛠&nbsp; [Fivetran](https://www.fivetran.com/){:target="_blank"}, [Airbyte](https://airbyte.com/){:target="_blank"}, etc. + [dbt](https://www.getdbt.com/){:target="_blank"}

4. Data can be loaded into a variety of storage options depending on the types of operations we want do. Some popular options include:

    - **databases** (DB): an organized collection of data that adheres to either:
        - relational schema (tables with rows and columns) often referred to as a Relational Database Management System (RDBMS) or SQL database.
        - non-relational (key/value, graph, etc.), often referred to as a non-relational database or NoSQL database.

        A database is an [online transaction processing (OLTP)](https://en.wikipedia.org/wiki/Online_transaction_processing){:target="_blank"} system because it's typically used for day-to-day CRUD (create, read, update, delete) operations where typically information is accessed by rows.

        🛠&nbsp; [PostgreSQL](https://www.postgresql.org/){:target="_blank"}, [MySQL](https://www.mysql.com/){:target="_blank"}, [MongoDB](https://www.mongodb.com/){:target="_blank"}, [Cassandra](https://cassandra.apache.org/){:target="_blank"}, etc.

    - **data warehouse** (DWH): a type of database that stores transactional data in a way that's efficient for analytics purposes. Here, the downstream tasks are more concerned about aggregating column values (trends) rather than accessing specific rows. It's an [online analytical processing (OLAP)](https://en.wikipedia.org/wiki/Online_analytical_processing){:target="_blank"} system because it's used for ad-hoc querying on aggregate views of the data.

        🛠&nbsp; [SnowFlake](https://www.snowflake.com/){:target="_blank"}, [Google BigQuery](https://cloud.google.com/bigquery){:target="_blank"}, [Amazon RedShift](https://aws.amazon.com/redshift/){:target="_blank"}, [Hive](https://hive.apache.org/){:target="_blank"}, etc.

5. Finally, our **data consumers** can ingest data from these storage options for downstream tasks such as data analytics, machine learning, etc. There may be additional data systems that may further simplify the data pipelines for consumers such as a [feature store](feature-store.md){:target="_blank"} to load the appropriate features for training or inference.

!!! note

    As data scientists or machine learning engineers, we most likely will not be creating and initializing these data systems ourselves but will instead be consuming the data pipelines that the data engineering team has established. However, every team has their own system so it's typical to learn how to interact with them with the help of existing team members, documentation, etc.
