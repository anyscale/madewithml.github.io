---
template: lesson.html
title: Data Stack for Machine Learning
description: An in-depth analysis of the modern data stack for analytics and machine learning applications.
keywords: data stack, modern data stack, data warehouse, snowflake, fivetran, airbyte, dbt, systems design, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/mlops.png
repository: https://github.com/GokuMohandas/mlops-course
---

{% include "styles/lesson.md" %}

Before we continue to learn about advanced production ML topics, we need to take a step back and understand the flow of data. It's very important that we have a way to produce high quality data and to do so in a reproducible and scalable manner. In this lesson, we're going to learn about the different tools and components of the data stack used industry to achieve a quality data foundation for all downstream consumers.

At a high level, we're going to:

1. [**E**xtract](#extract) data from various [sources](#sources).
2. [**L**oad](#load) the data into the appropriate [storage](#data-warehouse).
3. [**T**ransform](#transform) for downstream [consumers](#consumers).

<div class="ai-center-all">
    <img width="800" src="/static/images/mlops/data-stack/data.png" alt="data stack">
</div>

> This process is more commonly known as ELT, but there are variants such as ETL and reverse ETL, etc. They are all essentially the same underlying workflows but have slight differences in the order of data flow and where data is processed and stored.

!!! tip "Utility and simplicity"
    It can be enticing to set up a modern data stack in your organization, especially with all the hype. But it's very important to motivate utility and adding additional complexity:

    - Start with a use case that we already have data sources for and has direct impact on the business' bottom line (ex. user churn).
    - Start with the simplest infrastructure (source → database → report) and add complexity (in infrastructure, performance and team) as needed.


## Extract

The first step in our data pipeline is to extract the raw data from different sources in a standardized and reliable way.

### Sources

Our data sources we want to extract from can be from anywhere. They could come from 3rd party apps, files, databases, user click streams, physical devices, etc. But regardless of the source of our data, they type of data should fit into one of these categories:

- `#!js structured`: organized data stored in an explicit structure (ex. tables)
- `#!js semi-structured`: data with some structure but no formal schema or data types (web pages, CSV, JSON, etc.)
- `#!js unstructured`: qualitative data with no formal structure (text, images, audio, etc.)

### Frequency

Once we've identified the sources of data we want to extract data from, we need to decide at what frequency we want to extract at. The decision depends on the downstream applications and the infrastructure available.

- `#!js batch`: extracting data in batches, usually following a schedule (ex. daily) or when an event of interest occurs (ex. new data count)
- `#!js streaming`: extracting data in a continuous stream (using tools like [Kafka](https://kafka.apache.org/){:target="_blank"}, [Kinesis](https://aws.amazon.com/kinesis/){:target="_blank"}, etc.)

!!! note "Micro-batch"
    As we keep decreasing the time between batch ingestion (ex. towards 0), do we have stream ingestion? Not exactly. Batch processing is deliberately deciding to extract data from a source at a given interval. As that interval becomes <15 minutes, it's referred to as a micro-batch (many data warehouses allow for batch ingestion every 5 minutes). However, with stream ingestion, the extraction process is continuously on and events will keep being ingested.

!!! tip "Start simple"
    In general, it's a good idea to start with batch ingestion for most applications and slowly add the complexity of streaming ingestion (and additional infrastructure). This was we can prove that downstream applications are finding value from the data source and evolving to streaming later should only improve things.

> We'll learn more about the different system design implications of batch vs. stream in our [systems design lesson](systems-deisgn.md){:target="_blank"}.

### Pipelines

Once we have our data sources and we know how often we want to extract data from it, we need to establish the pipelines to enable the extraction. While we could construct custom scripts to extract data from the source and load it into storage (ex. [data warehouse](#data-warehouse)), an ecosystem of data ingestion tools have standardized the entire process. They all come equipped with connectors that allow for extraction, normalization, cleaning and loading to another location. These pipelines can be scaled, monitored, etc. all with very little to no code.

<div class="ai-center-all">
    <img width="500" src="/static/images/mlops/data-stack/pipelines.png" alt="ingestion pipelines">
</div>

> 🛠&nbsp; Popular tools: [Fivetran](https://www.fivetran.com/){:target="_blank"}, [Airbyte](https://airbyte.com/){:target="_blank"}, [Stitch](https://www.stitchdata.com/){:target="_blank"}, [Talend](https://www.talend.com/){:target="_blank"}, etc.

## Load

Once we have our data extracted, we need to load it into the appropriate storage option(s). The choice depends on what our downstream [consumers](#consumers) want to be able to do with the data. It's also common to store data in one location (ex. data lake) and move it somewhere else (ex. data warehouse) for specific processing.

### Data lake

A data lake is a flat data management system that stores raw objects. It's a great option for inexpensive storage and has the capability to hold all types of data (unstructured, semi-structured and structured). Object stores are becoming the standard for data lakes with default options across the popular cloud providers. Unfortunately, because data is stored as objects in a data lake, it's not designed for operating on structured data.

> 🛠&nbsp; Popular tools: [Amazon S3](https://aws.amazon.com/s3/){:target="_blank"}, [Azure Blob Storage](https://azure.microsoft.com/en-us/services/storage/blobs/){:target="_blank"}, [Google Cloud Storage](https://cloud.google.com/storage){:target="_blank"}, etc.

### Database

Another popular storage option is a database (DB), which is an organized collection of structured data that adheres to either:

- relational schema (tables with rows and columns) often referred to as a Relational Database Management System (RDBMS) or SQL database.
- non-relational (key/value, graph, etc.), often referred to as a non-relational database or NoSQL database.

A database is an [online transaction processing (OLTP)](https://en.wikipedia.org/wiki/Online_transaction_processing){:target="_blank"} system because it's typically used for day-to-day CRUD (create, read, update, delete) operations where typically information is accessed by rows. However, they're generally used to store data from one application and is not designed to hold data from across many sources for the purpose of analytics.

> 🛠&nbsp; Popular tools: [PostgreSQL](https://www.postgresql.org/){:target="_blank"}, [MySQL](https://www.mysql.com/){:target="_blank"}, [MongoDB](https://www.mongodb.com/){:target="_blank"}, [Cassandra](https://cassandra.apache.org/){:target="_blank"}, etc.

### Data warehouse

A data warehouse (DWH) is a type of database that's designed for storing structured data from many different sources for downstream analytics and data science. It's an [online analytical processing (OLAP)](https://en.wikipedia.org/wiki/Online_analytical_processing){:target="_blank"} system that's optimized for performing operations across aggregating column values rather than accessing specific rows.

> 🛠&nbsp; Popular tools: [SnowFlake](https://www.snowflake.com/){:target="_blank"}, [Google BigQuery](https://cloud.google.com/bigquery){:target="_blank"}, [Amazon RedShift](https://aws.amazon.com/redshift/){:target="_blank"}, [Hive](https://hive.apache.org/){:target="_blank"}, etc.

!!! note "Data lakehouse"
    There are new data systems introduced constantly, such as the data lakehouse, that offer to combine the best aspects of previous systems. For example, the lakehouse allows for storing both raw and transformed data with the structure of a data warehouse.

### Best practices

With the advent of cheap storage and cloud SaaS options to manage them, it's become a best practice to store raw data into data lakes. This allows for storage of raw, potentially unstructured, data without having to justify storage with downstream applications. When we do need to transform and process the data, we can move it to a data warehouse so can perform those operations efficiently.

<div class="ai-center-all">
    <img width="600" src="/static/images/mlops/data-stack/redundancy.png" alt="redundancy">
</div>

## Transform

Once we've extracted and loaded our data into, for example, a data warehouse, we'd normally need to transform the data so that it's compatible with standards. These transformations are different from the [preprocessing](preprocessing.md#transformations){:target="_blank"} we've seen before but are instead reflective of business logic that's agnostic to downstream applications. Common transformations include defining schemas and ensuring adherence (ex. [star schema](https://docs.microsoft.com/en-us/power-bi/guidance/star-schema){:target="_blank"}), filtering, cleaning and joining data across tables, etc. Additionally, many tools make it easy to transform the data directly inside our data warehouse and come with production functionality around version control, testing, documentation, etc.

<div class="ai-center-all">
    <img width="450" src="/static/images/mlops/data-stack/transform.png" alt="data transform">
</div>

> 🛠&nbsp; Popular tools: [dbt](https://www.getdbt.com/){:target="_blank"}, [Matillion](){:target="_blank"}, custom jinja templated SQL, etc.

!!! note
    In addition to data transformations, we can also process the data using large-scale analytics engines like Spark, Flink, etc. We'll learn more about [processing](systems-design.md#processing){:target="_blank"} in our systems design lesson.

## Consumers

Hopefully we created our data stack for the purpose of gaining some actionable insight about our business, users, etc. Because it's these use cases that dictate which sources of data we extract from, how often and how that data is stored and transformed. Downstream consumers of our data typically fall into one of these categories:

- `#!js data analytics`: use cases focused on reporting trends, aggregate views, etc. via charts, dashboards, etc.for the purpose of providing operational insight for business stakeholders.
> 🛠&nbsp; Popular tools: [Tableau](https://www.tableau.com/){:target="_blank"}, [Looker](https://www.looker.com/){:target="_blank"}, [Metabase](https://www.metabase.com/){:target="_blank"}, [Chartio](https://chartio.com/){:target="_blank"} (now Atlassian), etc.
- `#!js machine learning`: use cases centered around using the transformed data to construct predictive models (forecasting, personalization, etc.).

!!! tip "Analytics first, then ML"
    It's a good idea for the first several consumers to be analytics and reporting based in order to establish a robust data stack. These use cases typically just involve displaying data aggregations and trends, as opposed to machine learning systems that involve additional complex [infrastructure](feature-store.md){:target="_blank"} and [workflows](orchestration.md#mlops){:target="_blank"}.

## Observability

When we create complex data workflows like this, observability becomes a top priority. Data observability is the general  concept of understanding the condition of data in our system and it involves:

- `#!js data quality`: testing and monitoring our data's quality after every step (schemas, completeness, recency, etc.).
- `#!js data lineage`: mapping the where data comes from and how it's being transformed as it moves through our pipelines.
- `#!js discoverability`: enabling discovery of the different data sources and features for downstream applications.
- `#!js privacy + security`: are the different data assets treated and restricted appropriately amongst the consumers?

> We'll learn how to incorporate many of these observability concepts into our Dataops workflow in our [orchestration lesson](orchestration.md){:target="_blank"}.

## Stack considerations

The data stack ecosystem to create the ideal data workflow is growing and maturing. It can be overwhelming when it comes to choosing the best tooling options, especially as needs mature over time. Here are a few important factors to consider when making a tooling decision in this space:

- What is the cost per time per employee? Some of the tooling options can rack up quite the annual bill!
- Does the tool have the proper connectors to integrate with our data sources and the rest of the stack?
- Does the tool fit with our team's technical aptitude (SQL, Spark, Python, etc.)?
- What kind of support does the tool offer (enterprise, community, etc.)?

> We've got a lot more coming for this lesson over the next few months as we work with many of these tooling options to truly simplify the modern data stack for machine learning applications.