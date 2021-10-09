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

We’ve already covered the methods of deployment via our API, Docker, and CI/CD lessons where the ML system is its own application, as opposed to being tied to a monolithic general application. This way we’re able to scale our ML workflows as needed and use it to deliver value across any other applications that maybe interested in using it’s outputs. So in this lesson, we’ll instead talk about the types of tasks, serving and how optimize, orchestrate and scale them. We highly recommend using a framework such as [Metaflow](https://metaflow.org/){:target="_blank"} to seamlessly interact with all the infrastructure that we'll be discussing.

## Tasks

Before we talk about the infrastructure needed for ML tasks, we need to talk about the fundamental types of ML tasks. A task can involve features that don't change over time. For example if an API classifies uploaded images, all the input features come from the image the user just uploaded. If that same image is uploaded later and the same model is used, the prediction will remain unchanged. However, a task can also involve features that change over time. For example, if you want to predict whether a user would enjoy a movie, you'll want to retrieve the latest available data for that user's behavior. Using the exact same model, your prediction can change as the user's features change over time. This subtle difference can drive key architectural choices when it comes how to store, process and retrieve your data (feature stores, data streams, etc.).

## Serving

The first decision to make it to whether serve predictions via batches or real-time, which is entirely based on the feature space (finite vs. unbound).

### Batch serving

We can make batch predictions on a finite set of inputs which are then written to a database for low latency inference. When a user or downstream process makes an inference request in real-time, cached results from the database are returned.

<div class="ai-center-all">
    <img width="600" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/infrastructure/batch_serving.png">
</div>

- ✅&nbsp; generate and cache predictions for very fast inference for users.
- ✅&nbsp; the model doesn't need to be spun up as it's own service since it's never used in real-time.
- ❌&nbsp; predictions can become stale if user develops new interests that aren’t captured by the old data that the current predictions are based on.
- ❌&nbsp; input feature space must be finite because we need to generate all the predictions before they're needed for real-time.

> Recommend content that *existing* users will like based on their viewing history.

New users may just receive some generic recommendations until we process their history the next day. Even if we're not doing batch serving, it might still be useful to cache very popular sets of input features so that we can serve those predictions faster.

### Real-time serving

We can also serve live predictions, typically through an HTTPS call with the appropriate input data. This will involve spinning up our ML application as a microservice since users or downstream processes will interact directly with the model.

<div class="ai-center-all">
    <img width="400" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/infrastructure/real_time_serving.png">
</div>

- ✅&nbsp; can yield more up-to-date predictions which may can yield a more meaningful user experience, etc.
- ❌&nbsp; requires managed microservices to handle request traffic.
- ❌&nbsp; requires real-time monitoring since input space in unbounded, which could yield erroneous predictions.

> Besides wrapping our model(s) as separate, scalable microservices, we can also have a purpose-built model server to host our models. Model servers, such as [MLFlow](https://mlflow.org/){:target="_blank"}, [TorchServe](https://pytorch.org/serve/){:target="_blank"}, [RedisAI](https://oss.redislabs.com/redisai/){:target="_blank"} or [Nvidia's Triton](https://developer.nvidia.com/nvidia-triton-inference-server){:target="_blank"} inference server, provide a common interface to interact with models for inspection, inference, etc. In fact, modules like RedisAI can even offer added benefits such as data locality for super fast inference.

## Processing

We also have control over the features that we use to generate our real-time predictions.

Our use case doesn't necessarily involve entity features changing over time so it makes sense to just have one processing pipeline. However, not all entities in ML applications work this way. Using our Netflix content recommendation example, a given user can have certain features that are updated over time, such as top genres, click rate, etc. As we'll see below, we have the option to batch process features for users at a previous time or we could process features in a stream as they become available to make live predictions.

### Batch processing

Batch process features for a given entity at a previous point in time, which are later used for generating real-time predictions.

<div class="ai-center-all">
    <img width="600" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/infrastructure/batch_processing.png">
</div>

- ✅&nbsp; can perform heavy feature computations offline and have it ready for fast inference.
- ❌&nbsp; features can become stale since they were predetermined a while ago. This can be a huge disadvantage when your prediction depends on very recent events. (ex. catching fraudulent transactions as quickly as possible).

> We'll discuss how these features are stored for training and inference in the [feature stores lesson](feature-store.md){:target="_blank"}.

### Stream processing

Perform inference on a given set of inputs with near real-time, streaming, features for a given entity.

<div class="ai-center-all">
    <img width="600" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/infrastructure/stream_processing.png">
</div>

- ✅&nbsp; we can generate better predictions by providing real-time, streaming, features to the model.
- ❌&nbsp; extra infrastructure needed for maintaining data streams ([Kafka](https://kafka.apache.org/){:target="_blank"}, [Kinesis](https://aws.amazon.com/kinesis/){:target="_blank"}, etc.) and for stream processing (Apache [Flink](https://flink.apache.org/){:target="_blank"}, [Beam](https://beam.apache.org/){:target="_blank"}, etc.).

> Recommend content based on the real-time history that the users have generated. Note that the same model is used but the input data can change and grow.

If we infinitely reduce how often we do batch processing, we’ll [effectively have](https://www.ververica.com/blog/batch-is-a-special-case-of-streaming){:target="_blank"} stream (real-time) processing since the features will always be up-to-date.

## Learning

While we have the option to use batch / streaming features and serve batch / real-time predictions, we've kept the model completely unchanged. This, however, is another decision that we have to make depending on the use case and what our infrastructure allows for.

### Offline learning

The traditional approach is to train our models offline and then deploy them to inference. We may periodically retrain them offline as new data becomes labeled, validated, etc. and deploy them after evaluation. We may also retrain them if we discover an issue during monitoring such as drift.

<div class="ai-center-all">
    <img width="600" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/infrastructure/offline_learning.png">
</div>

- ✅&nbsp; don't need to worry about provisioning resources for compute since it happens offline.
- ✅&nbsp; no urgency to get recent data immediately labeled and validated.
- ❌&nbsp; the model can become stale and may not adapt to recent changes until some monitoring alerts trigger retraining.

> Learn more about the executing MLOps pipeline tasks using a workflow orchestrator in the [Pipelines lesson](pipelines.md){:target="_blank"}.

### Online learning

In order to truly serve the most informed predictions, we should have a model trained on the most recent data.

<div class="ai-center-all">
    <img width="400" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/infrastructure/online_learning.png">
</div>

- ✅&nbsp; model is aware of recent data distributions, trends, etc. which can provide highly informed predictions.
- ❌&nbsp; compute requirements can quickly surge depending on the retraining schedule.
- ❌&nbsp; need to have quick pipelines to deliver labeled and validated data which can become a bottleneck it it involves human intervention.
- ❌&nbsp; might not be able to do more involved model validation, such as AB or shadow testing, with such limited time in between models.

## Testing

Once our application is deployed after our [offline tests](testing.md#evaluation){:taret="_blank"}, several types of **online tests** that we can run to determine the performance quality.

### AB tests
AB tests involve sending production traffic to the different systems that we're evaluating and then using statistical hypothesis testing to decide which system is better. There are several common issues with AB testing such as accounting for different sources of bias, such as the novelty effect of showing some users the new system. We also need to ensure that the same users continue to interact with the same systems so we can compare the results without contamination. In many cases, if we're simply trying to compare the different versions for a certain metric, multi-armed bandits will be a better approach.

<div class="ai-center-all">
    <img width="500" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/infrastructure/ab.png">
</div>

### Canary tests
Canary tests involve sending most of the production traffic to the currently deployed system but sending traffic from a small cohort of users to the new system we're trying to evaluate. Again we need to make sure that the same users continue to interact with the same system as we gradually roll out the new system.

<div class="ai-center-all">
    <img width="500" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/infrastructure/canary.png">
</div>

### Shadow tests
Shadow testing involves sending the same production traffic to the different systems. We don't have to worry about system contamination and it's very safe compared to the previous approaches since the new system's results are not served. However, we do need to ensure that we're replicating as much of the production system as possible so we can catch issues that are unique to production early on. But overall, shadow testing is easy to monitor, validate operational consistency, etc.

<div class="ai-center-all">
    <img width="500" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/infrastructure/shadow.png">
</div>

## Optimization

We’ve discussed [distributed training](baselines.md#distributed-training){:target="_blank"} strategies for when our data or models are too large for training but what about when our models are too large to deploy? The following model compression techniques are commonly used to make large models fit within existing infrastructure:

- **Pruning**: remove weights (unstructured) or entire channels (structured) to reduce the size of the network. The objective is to preserve the model’s performance while increasing its sparsity.
- **Quantization**: reduce the memory footprint of the weights by reducing their precision (ex. 32 bit to 8 bit). We may loose some precision but it shouldn’t affect performance too much.
- **Distillation**: training smaller networks to “mimic” larger networks by having it reproduce the larger network’s layers’ outputs.

There are always new model compression techniques coming out so best keep up to date via [survey papers](https://arxiv.org/abs/1710.09282){:target="_blank"} on them.

## Methods

The way we process our features and serve predictions dictates how we deploy our application. Depending on the pipeline components, scale, etc. we have several different options for how we deploy.

### Compute engines
Compute engines such as AWS EC2, Google Compute, Azure VM, on-prem, etc. that can launch our application across multiple workers.

- **Pros**: easy to deploy and manage these single instances.
- **Cons**: when we do need to scale, it's not easy to manage these instances individually.

### Container orchestration

Container orchestration via [Kubernetes](https://kubernetes.io/){:target="_blank"} (K8s) for managed deployment, scaling, etc. There are several ML specific platforms to help us **self-manage** K8s via control planes such as [Seldon](https://www.seldon.io/tech/){:target="_blank"}, [KFServing](https://www.kubeflow.org/docs/components/kfserving/kfserving/){:target="_blank"}, etc. However, there are also **fully-managed** solutions, such as [SageMaker](https://aws.amazon.com/sagemaker/){:target="_blank"}, [Cortex](https://www.cortex.dev/){:target="_blank"}, [BentoML](https://www.bentoml.ai/){:target="_blank"}, etc. Many of these tools also come with additional features such as experiment tracking, monitoring, etc.

<div class="ai-center-all">
    <img width="400" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/infrastructure/managed.png">
</div>

- **Pros**: very easy to scale our services since it's all managers with the proper components (load balancers, control planes, etc.)
- **Cons**: can introduce too much complexity overhead.

### Serverless

Serverless options such as [AWS Lambda](https://aws.amazon.com/lambda/){:target="_blank"}, [Google Cloud Functions](https://cloud.google.com/functions){:target="_blank"}, etc.

- **Pros**: no need to manage any servers and it all scale automatically depending on the request traffic.
- **Cons**: size limits on function storage, payload, etc. based on provider and usually no accelerators (GPU, TPU, etc.)

> Be sure to explore the [CI/CD workflows](cicd.md#serving){:target="_blank"} that accompany many of these deployment and serving options so you can have a continuous training, validation and serving process.

## Application

For our application, we'll need the following components to best serve our users.

- **real-time serving** since the inputs features (title, description, etc.) are user generated just moments before predicted tags are needed. This is why we've created an API around our models and defined a Dockerfile, which can be used to create and scale our microservice. Since our models and overall Docker images are small, we could use serverless options such as AWS Lambda to deploy our versioned applications. We could do this directly through GitHub actions once the CI/CD workflows have all passed.
- **stream processing** since our predictions are on entities (projects) that have never been seen before. We'll want to have the latest (and only) features for each project so we can predict the most appropriate tags. However, our use case doesn't necessarily involve entity features changing over time so it makes sense to just have one processing pipeline and not have to worry about separate batch and stream pipelines.
- **offline learning** will work just fine for our application since the vocabulary and tag spaces are fairly constrained for a given period of time. However, we will use [monitoring](monitoring.md){:target="_blank"} to ensure that our input and output spaces are as expected and initiate retraining periodically.

## Resources

- [Batch is a special case of streaming](https://www.ververica.com/blog/batch-is-a-special-case-of-streaming){:target="_blank"}
- [Machine learning is going real-time](https://huyenchip.com/2020/12/27/real-time-machine-learning.html){:target="_blank"}

<!-- Citation -->
{% include "cite.md" %}
