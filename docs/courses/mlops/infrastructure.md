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

We’ve already covered the methods of deployment via our API, Docker, and CI/CD lessons where the ML system is its own microservice, as opposed to being tied to a monolithic general application. This way we’re able to scale our ML workflows as needed and use it to deliver value to downstream applications. So in this lesson, we’ll instead discuss the types of tasks, serving strategies and how to optimize, orchestrate and scale them. We highly recommend using a framework such as [Metaflow](https://metaflow.org/){:target="_blank"} to seamlessly interact with all the infrastructure that we'll be discussing.

## Tasks

Before we talk about the infrastructure needed for ML tasks, we need to talk about the fundamental types of ML tasks. A task can involve features that don't change over time. For example if an API classifies uploaded images, all the input features come from the image the user just uploaded. If that same image is uploaded later and the same model is used, the prediction will remain unchanged. However, a task can also involve features that change over time. For example, if you want to predict whether a user would enjoy a movie, you'll want to retrieve the latest available data for that user's behavior. Using the exact same model, your prediction can change as the user's features change over time. This subtle difference can drive key architectural choices when it comes how to store, process and retrieve your data (feature stores, data streams, etc.).

## Serving

The first decision is whether to serve predictions via batches or real-time, which is entirely based on the feature space (finite vs. unbound).

### Batch serving

We can make batch predictions on a finite set of inputs which are then written to a database for low latency inference. When a user or downstream process makes an inference request in real-time, cached results from the database are returned.

<div class="ai-center-all">
    <img width="600" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/infrastructure/batch_serving.png">
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
    <img width="400" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/infrastructure/real_time_serving.png">
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

If we infinitely reduce the time between each batch processing event, we’ll [effectively have](https://www.ververica.com/blog/batch-is-a-special-case-of-streaming){:target="_blank"} stream (real-time) processing since the features will always be up-to-date.

## Learning

So far, while we have the option to use batch / streaming features and serve batch / real-time predictions, we've kept the model fixed. This, however, is another decision that we have to make depending on the use case and what our infrastructure allows for.

### Offline learning

The traditional approach is to train our models offline and then deploy them to inference. We may periodically retrain them offline as new data becomes labeled, validated, etc. and deploy them after evaluation. We may also expedite retraining if we discover an issue during [monitoring](monitoring.md){:target="_blank"} such as drift.

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
- ❌&nbsp; need to have quick pipelines to deliver labeled and validated data which can become a bottleneck if it involves human intervention.
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

!!! question "What can go wrong?"
    If shadow tests allow us to test our updated system without having to actually serve the new results, why doesn't everyone adopt it?

    ??? quote "Show answer"
        With shadow deployment, we'll miss out on any live feedback signals (explicit/implicit) from our users since users are not directly interacting with the product using our new version.

        We also need to ensure that we're replicating as much of the production system as possible so we can catch issues that are unique to production early on. This is rarely possible because, while your ML system may be a standalone microservice, it ultimately interacts with an intricate production environment that has *many* dependencies.

## Optimization

We’ve discussed [distributed training](baselines.md#distributed-training){:target="_blank"} strategies for when our data or models are too large for training but what about when our models are too large to deploy? The following model compression techniques are commonly used to make large models fit within existing infrastructure:

- **Pruning**: remove weights (unstructured) or entire channels (structured) to reduce the size of the network. The objective is to preserve the model’s performance while increasing its sparsity.
- **Quantization**: reduce the memory footprint of the weights by reducing their precision (ex. 32 bit to 8 bit). We may loose some precision but it shouldn’t affect performance too much.
- **Distillation**: training smaller networks to “mimic” larger networks by having it reproduce the larger network’s layers’ outputs.

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

## Resources

- [Batch is a special case of streaming](https://www.ververica.com/blog/batch-is-a-special-case-of-streaming){:target="_blank"}
- [Machine learning is going real-time](https://huyenchip.com/2020/12/27/real-time-machine-learning.html){:target="_blank"}

<!-- Citation -->
{% include "cite.md" %}
