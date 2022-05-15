---
template: lesson.html
title: Infrastructure
description: A closer look at the infrastructure needed for deployment and serving of ML applications.
keywords: deployment, serving, online learning, stream processing, batch prediction, real-time prediction, model compression, pruning, quantization, distillation, kubernetes, seldon, kubeflow, kfserving, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning, great expectations
image: https://madewithml.com/static/images/mlops.png
repository: https://github.com/GokuMohandas/MLOps
---

{% include "styles/lesson.md" %}

## Intuition

In this lessons, we’ll discuss the different options we have for processing features, learning from them, experimenting on models and serving them. We'll also talk about the different options for infrastructure to orchestrate and scale them.

## Tasks

Before we talk about the infrastructure needed for ML tasks, we need to talk about the fundamental types of ML tasks.

### Static
A task can involve features that don't change over time. For example if an API classifies uploaded images, all the input features come from the image the user just uploaded. If that same image is uploaded later and the same model is used, the prediction will remain unchanged.

### Dynamic
A task can also involve features that change over time. For example, if you want to predict whether a user would enjoy a movie, you'll want to retrieve the latest available data for that user's behavior. Using the exact same model, your prediction can change as the user's features change over time.

> This subtle difference can drive key architectural choices when it comes how to store, process and retrieve your data (ex. databases, feature stores, streams, etc.).

## Data management systems

We also want to set the scene for where our data could live. So far, we've been using a JSON file (which is great for quick POCs) but for production, we'll need to rely on much more reliable sources of data.

<div class="ai-center-all">
    <img width="1000" src="/static/images/mlops/infrastructure/data.png" alt="data management systems">
</div>

1. First, we have our **data sources**, which can be from APIs, users, other databases, etc. and can be:

    - `#!js structured`: organized data stored in an explicit structure (ex. tables)
    - `#!js semi-structured`: data with some structure but no formal schema or data types (web pages, CSV, JSON, etc.)
    - `#!js unstructured`: qualitative data with no formal structure (text, images, audio, etc.)

2. These data sources are usually consumed by a **data lake**, which is a central repository that stores the data in its raw format. Traditionally, **object stores**, which manage as objects, as opposed to files under a certain structure, are used as the storage platform for the data lake.

    > Common options: [Amazon S3](https://aws.amazon.com/s3/){:target="_blank"}, [Google Cloud Storage](https://cloud.google.com/storage){:target="_blank"}, etc.

3. Raw data from data lakes are moved to more organized storage options via [ETL](https://en.wikipedia.org/wiki/Extract,_transform,_load){:target="_blank"} (extract, transform, load) pipelines.

4. Data can be loaded into a variety of storage options depending on the types of operations we want do. Some popular options include:

    - **databases** (DB): an organized collection of data that adheres to either:
        - relational schema (tables with rows and columns) often referred to as a Relational Database Management System (RDBMS) or SQL database.
        - non-relational (key/value, graph, etc.), often referred to as a non-relational database or NoSQL database.

        A database is an [online transaction processing (OLTP)](https://en.wikipedia.org/wiki/Online_transaction_processing){:target="_blank"} system because it's typically used for day-to-day CRUD (create, read, update, delete) operations where typically information is accessed by rows.

        > Common options: [PostgreSQL](https://www.postgresql.org/){:target="_blank"}, [MySQL](https://www.mysql.com/){:target="_blank"}, [MongoDB](https://www.mongodb.com/){:target="_blank"}, [Cassandra](https://cassandra.apache.org/){:target="_blank"}, etc.

    - **data warehouse** (DWH): a type of database that stores transactional data in a way that's efficient for analytics purposes. Here, the downstream tasks are more concerned about aggregating column values (trends) rather than accessing specific rows. It's an [online analytical processing (OLAP)](https://en.wikipedia.org/wiki/Online_analytical_processing){:target="_blank"} system because it's used for ad-hoc querying on aggregate views of the data.

        > Common options: [Google BigQuery](https://cloud.google.com/bigquery){:target="_blank"}, [Amazon RedShift](https://aws.amazon.com/redshift/){:target="_blank"}, [SnowFlake](https://www.snowflake.com/){:target="_blank"}, [Hive](https://hive.apache.org/){:target="_blank"}, etc.

5. Finally, our **data consumers** can ingest data from these storage options for downstream tasks such as data analytics, machine learning, etc. There may be additional data systems that may further simplify the data pipelines for consumers such as a [feature store](feature-store.md){:target="_blank"} to load the appropriate features for training or inference.

!!! note

    As a data scientist or machine learning engineer, you most likely will not be creating these data management systems yourself but will instead be consuming the data pipelines that the data engineering team. So it's imperative that you know how to integrate your workflows with them.

    > Every company has their own system so you typically learn how to interact with them through existing team members, documentation, etc.


## Serving

The first decision is whether to serve predictions via batches or real-time, which is entirely based on the feature space (finite vs. unbound). We're starting backwards here because this decision will dictate many of the upstream decision around processing, learning and experimentation.

### Batch serving

We can make batch predictions on a finite set of inputs which are then written to a database for low latency inference. When a user or downstream process makes an inference request in real-time, cached results from the database are returned.

<div class="ai-center-all">
    <img width="600" src="/static/images/mlops/infrastructure/batch_serving.png">
</div>

- ✅&nbsp; generate and cache predictions for very fast inference for users.
- ✅&nbsp; the model doesn't need to be spun up as it's own service since it's never used in real-time.
- ❌&nbsp; predictions can become stale if user develops new interests that aren’t captured by the old data that the current predictions are based on.
- ❌&nbsp; input feature space must be finite because we need to generate all the predictions before they're needed for real-time.

!!! question "Batch serving tasks"
    What are some tasks where batch serving is ideal?

    ??? quote "Show answer"
        Recommend content that *existing* users will like based on their viewing history. However, *new* users may just receive some generic recommendations based on their explicit interests until we process their history the next day. And even if we're not doing batch serving, it might still be useful to cache very popular sets of input features (ex. combination of explicit interests > recommended content) so that we can serve those predictions faster.

### Real-time serving

We can also serve live predictions, typically through an HTTPS call with the appropriate input data. This will involve spinning up our ML application as a microservice since users or downstream processes will interact directly with the model.

<div class="ai-center-all">
    <img width="400" src="/static/images/mlops/infrastructure/real_time_serving.png">
</div>

- ✅&nbsp; can yield more up-to-date predictions which may yield a more meaningful user experience, etc.
- ❌&nbsp; requires managed microservices to handle request traffic.
- ❌&nbsp; requires real-time monitoring since input space in unbounded, which could yield erroneous predictions.

!!! question "Real-time serving tasks"
    In our example task for batch serving above, how can real-time serving significantly improve content recommendations?

    ??? quote "Show answer"
        With batch processing, we generate content recommendations for users offline using their history. These recommendations won't change until we process the batch the next day using the updated user features. But what is the user's taste significantly changes during the day (ex. user is searching for horror movies to watch). With real-time serving, we can use these recent features to recommend highly relevant content based on the immediate searches.

> Besides wrapping our model(s) as separate, scalable microservices, we can also have a purpose-built model server to host our models. Model servers, such as [MLFlow](https://mlflow.org/){:target="_blank"}, [TorchServe](https://pytorch.org/serve/){:target="_blank"}, [RedisAI](https://oss.redislabs.com/redisai/){:target="_blank"} or [Nvidia's Triton](https://developer.nvidia.com/nvidia-triton-inference-server){:target="_blank"} inference server, provide a common interface to interact with models for inspection, inference, etc. In fact, modules like RedisAI can even offer added benefits such as data locality for super fast inference.

## Processing

We also have control over the features that we use to generate our real-time predictions.

Our use case doesn't necessarily involve entity features changing over time so it makes sense to just have one processing pipeline. However, not all entities in ML applications work this way. Using our content recommendation example, a given user can have certain features that are updated over time, such as favorite genres, click rate, etc. As we'll see below, we have the option to batch process features for users at a previous time or we could process features in a stream as they become available and use them to make relevant predictions.

### Batch processing

Batch process features for a given entity at a previous point in time, which are later used for generating real-time predictions.

<div class="ai-center-all">
    <img width="600" src="/static/images/mlops/infrastructure/batch_processing.png">
</div>

- ✅&nbsp; can perform heavy feature computations offline and have it ready for fast inference.
- ❌&nbsp; features can become stale since they were predetermined a while ago. This can be a huge disadvantage when your prediction depends on very recent events. (ex. catching fraudulent transactions as quickly as possible).

> We discuss more about different [data management architectures](pipelines.md#extraction){:target="_blank"}, such as databases and data warehouses (DWH) in the [pipelines](pipelines.md){:target="_blank"} lesson.

### Stream processing

Perform inference on a given set of inputs with *near* real-time, streaming, features for a given entity.

<div class="ai-center-all">
    <img width="600" src="/static/images/mlops/infrastructure/stream_processing.png">
</div>

- ✅&nbsp; we can generate better predictions by providing real-time, streaming, features to the model.
- ❌&nbsp; extra infrastructure needed for maintaining data streams ([Kafka](https://kafka.apache.org/){:target="_blank"}, [Kinesis](https://aws.amazon.com/kinesis/){:target="_blank"}, etc.) and for stream processing (Apache [Flink](https://flink.apache.org/){:target="_blank"}, [Beam](https://beam.apache.org/){:target="_blank"}, etc.).

> Recommend content based on the real-time history that the users have generated. Note that the same model is used but the input data can change and grow.

If we infinitely reduce the time between each batch processing event, we’ll [effectively have](https://www.ververica.com/blog/batch-is-a-special-case-of-streaming){:target="_blank"} stream (real-time) processing since the features will always be up-to-date.

!!! tip
    Even if our application requires stream processing, it's a good idea to implement the system with batch processing first if it's technically easier. If our task is high-stakes and requires stream processing even for the initial deployments, we can still experiment with batch processing for internal releases. This can allow us to start collecting feedback, generating more data to label, etc.

## Learning

So far, while we have the option to use batch / streaming features and serve batch / real-time predictions, we've kept the model fixed. This, however, is another decision that we have to make depending on the use case and what our infrastructure allows for.

### Offline learning

The traditional approach is to train our models offline and then deploy them to inference. We may periodically retrain them offline as new data becomes labeled, validated, etc. and deploy them after evaluation. We may also expedite retraining if we discover an issue during [monitoring](monitoring.md){:target="_blank"} such as drift.

<div class="ai-center-all">
    <img width="600" src="/static/images/mlops/infrastructure/offline_learning.png">
</div>

- ✅&nbsp; don't need to worry about provisioning resources for compute since it happens offline.
- ✅&nbsp; no urgency to get recent data immediately labeled and validated.
- ❌&nbsp; the model can become stale and may not adapt to recent changes until some monitoring alerts trigger retraining.

> Learn more about the executing MLOps pipeline tasks using a workflow orchestrator in the [Pipelines lesson](pipelines.md){:target="_blank"}.

### Online learning

In order to truly serve the most informed predictions, we should have a model trained on the most recent data. However, instead of using expensive stateless batch learning, a stateful and incremental learning approach is adopted. Here the model is trained offline, as usual, on the initial dataset but is then stochastically updated at a single instance or mini-batch level as new data becomes available. This removes the compute costs associated with traditional stateless, redundant training on same same past data.

<div class="ai-center-all">
    <img width="400" src="/static/images/mlops/infrastructure/online_learning.png">
</div>

- ✅&nbsp; model is aware of distributional shifts and can quickly adapt to provide highly informed predictions.
- ✅&nbsp; stateful training can significantly lower compute costs and provide faster convergence.
- ✅&nbsp; possible for tasks where the event that occurs is the label (user clicks, time-series, etc.)
- ❌&nbsp; may not be possible for tasks that involve explicit labeling or delayed outcomes.
- ❌&nbsp; prone to catastrophic inference where the model is learning from malicious live production data (mitigated with monitoring and rollbacks).
- ❌&nbsp; models may suffer from [catastrophic forgetting](https://en.wikipedia.org/wiki/Catastrophic_interference){:target="_blank"} as we continue to update it using new data.

!!! question "What about new feature values?"
    With online learning, how can we encode new feature values without retraining from scratch?

    ??? quote "Show answer"
        We can use clever tricks to represent out-of-vocabulary feature values such encoding based on mapped feature values or hashing. For example, we may wan to encode the name of a few restaurant but it's not mapped explicitly by our encoder. Instead we could choose to represent restaurants based on it's location, cuisine, etc. and so any new restaurant who has these feature values can be represented in a similar manner as restaurants we had available during training. Similarly, hashing can map OOV values but keep in mind that this is a one-way encoding (can't reverse the hashing to see what the value was) and we have to choose a hash size large enough to avoid collisions (<10%).

## Experimentation

There are many different experimentation strategies we can use to measure real-time performance before committing to replace our existing version of the system.

### AB tests
AB tests involve sending production traffic to the different systems that we're evaluating and then using statistical hypothesis testing to decide which system is better. There are several common issues with AB testing such as accounting for different sources of bias, such as the novelty effect of showing some users the new system. We also need to ensure that the same users continue to interact with the same systems so we can compare the results without contamination.

<div class="ai-center-all">
    <img width="500" src="/static/images/mlops/infrastructure/ab.png">
</div>

> In many cases, if we're simply trying to compare the different versions for a certain metric, multi-armed bandits will be a better approach.

### Canary tests
Canary tests involve sending most of the production traffic to the currently deployed system but sending traffic from a small cohort of users to the new system we're trying to evaluate. Again we need to make sure that the same users continue to interact with the same system as we gradually roll out the new system.

<div class="ai-center-all">
    <img width="500" src="/static/images/mlops/infrastructure/canary.png">
</div>

### Shadow tests
Shadow testing involves sending the same production traffic to the different systems. We don't have to worry about system contamination and it's very safe compared to the previous approaches since the new system's results are not served. However, we do need to ensure that we're replicating as much of the production system as possible so we can catch issues that are unique to production early on. But overall, shadow testing is easy to monitor, validate operational consistency, etc.

<div class="ai-center-all">
    <img width="500" src="/static/images/mlops/infrastructure/shadow.png">
</div>

!!! question "What can go wrong?"
    If shadow tests allow us to test our updated system without having to actually serve the new results, why doesn't everyone adopt it?

    ??? quote "Show answer"
        With shadow deployment, we'll miss out on any live feedback signals (explicit/implicit) from our users since users are not directly interacting with the product using our new version.

        We also need to ensure that we're replicating as much of the production system as possible so we can catch issues that are unique to production early on. This is rarely possible because, while your ML system may be a standalone microservice, it ultimately interacts with an intricate production environment that has *many* dependencies.

## Methods

The way we process our features and serve predictions dictates how we deploy our application. Depending on the pipeline components, scale, etc. we have several different options for how we deploy.

### Compute engines
Compute engines such as AWS EC2, Google Compute, Azure VM, on-prem, etc. that can launch our application across multiple workers.

- **Pros**: easy to deploy and manage these single instances.
- **Cons**: when we do need to scale, it's not easy to manage these instances individually.

### Container orchestration

Container orchestration via [Kubernetes](https://kubernetes.io/){:target="_blank"} (K8s) for managed deployment, scaling, etc. There are several ML specific platforms to help us **self-manage** K8s via control planes such as [Seldon](https://www.seldon.io/tech/){:target="_blank"}, [KFServing](https://www.kubeflow.org/docs/components/kfserving/kfserving/){:target="_blank"}, etc. However, there are also **fully-managed** solutions, such as [SageMaker](https://aws.amazon.com/sagemaker/){:target="_blank"}, [Cortex](https://www.cortex.dev/){:target="_blank"}, [BentoML](https://www.bentoml.ai/){:target="_blank"}, etc. Many of these tools also come with additional features such as experiment tracking, monitoring, etc.

<div class="ai-center-all">
    <img width="400" src="/static/images/mlops/infrastructure/managed.png">
</div>

- **Pros**: very easy to scale our services since it's all managers with the proper components (load balancers, control planes, etc.)
- **Cons**: can introduce too much complexity overhead.

### Serverless

Serverless options such as [AWS Lambda](https://aws.amazon.com/lambda/){:target="_blank"}, [Google Cloud Functions](https://cloud.google.com/functions){:target="_blank"}, etc.

- **Pros**: no need to manage any servers and it all scale automatically depending on the request traffic.
- **Cons**: size limits on function storage, payload, etc. based on provider and usually no accelerators (GPU, TPU, etc.)

> Be sure to explore the [CI/CD workflows](cicd.md#serving){:target="_blank"} that accompany many of these deployment and serving options so you can have a continuous training, validation and serving process.

## Resources

- [Batch is a special case of streaming](https://www.ververica.com/blog/batch-is-a-special-case-of-streaming){:target="_blank"}
- [Machine learning is going real-time](https://huyenchip.com/2020/12/27/real-time-machine-learning.html){:target="_blank"}

<!-- Citation -->
{% include "cite.md" %}
