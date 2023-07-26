---
template: lesson.html
title: Experiment Tracking
description: Managing and tracking machine learning experiments.
keywords: experiment tracking, mlflow, weights and biases, mlops, machine learning
image: https://madewithml.com/static/images/mlops.png
repository: https://github.com/GokuMohandas/Made-With-ML
notebook: https://github.com/GokuMohandas/Made-With-ML/blob/main/notebooks/madewithml.ipynb
---

{% include "styles/lesson.md" %}

## Intuition
So far, we've been training and evaluating our different baselines but haven't really been tracking these experiments. We'll fix this but defining a proper process for experiment tracking which we'll use for all future experiments (including hyperparameter optimization). Experiment tracking is the process of managing all the different experiments and their components, such as parameters, metrics, models and other artifacts and it enables us to:

- **Organize** all the necessary components of a specific experiment. It's important to have everything in one place and know where it is so you can use them later.
- **Reproduce** past results (easily) using saved experiments.
- **Log** iterative improvements across time, data, ideas, teams, etc.

## Tools
There are many options for experiment tracking but we're going to use [MLFlow](https://mlflow.org/){:target="_blank"} (100% free and [open-source](https://github.com/mlflow/mlflow){:target="_blank"}) because it has all the functionality we'll need. We can run MLFlow on our own servers and databases so there are no storage cost / limitations, making it one of the most popular options and is used by Microsoft, Facebook, Databricks and others. There are also several popular options such as a [Comet ML](https://www.comet.ml/site/){:target="_blank"} (used by Google AI, HuggingFace, etc.), [Neptune](https://neptune.ai/){:target="_blank"} (used by Roche, NewYorker, etc.), [Weights and Biases](https://www.wandb.com/){:target="_blank"} (used by Open AI, Toyota Research, etc.). These are fully managed solutions that provide features like dashboards, reports, etc.

## Setup

We'll start by setting up our model registry where all of our experiments and their artifacts will be stores.

```python linenums="1"
import mlflow
from pathlib import Path
from ray.air.integrations.mlflow import MLflowLoggerCallback
import time
```

```python linenums="1"
# Config MLflow
MODEL_REGISTRY = Path("/tmp/mlflow")
Path(MODEL_REGISTRY).mkdir(parents=True, exist_ok=True)
MLFLOW_TRACKING_URI = "file://" + str(MODEL_REGISTRY.absolute())
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
print (mlflow.get_tracking_uri())
```

<pre class="output">
file:///tmp/mlflow
</pre>

> On Windows, the tracking URI should have three forwards slashes:
```python linenums="1"
MLFLOW_TRACKING_URI = "file:///" + str(MODEL_REGISTRY.absolute())
```

!!! note
    In this course, our MLflow artifact and backend store will both be on our local machine. In a production setting, these would be remote such as S3 for the artifact store and a database service (ex. PostgreSQL RDS) as our backend store.

## Integration

While we could use MLflow directly to log metrics, artifacts and parameters:

```python linenums="1"
# Example mlflow calls
mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)
mlflow.log_artifacts(dir)
mlflow.log_params(config)
```

We'll instead use Ray to integrate with MLflow. Specifically we'll use the [MLflowLoggerCallback](https://docs.ray.io/en/latest/tune/api/integration.html#mlflow-air-integrations-mlflow){:target="_blank"} which will automatically log all the necessary components of our experiments to the location specified in our `MLFLOW_TRACKING_URI`. We of course can still use MLflow directly if we want to log something that's not automatically logged by the callback. And if we're using other experiment trackers, Ray has [integrations](https://docs.ray.io/en/latest/tune/api/integration.html){:target="_blank"} for those as well.

```python linenums="1"
# MLflow callback
experiment_name = f"llm-{int(time.time())}"
mlflow_callback = MLflowLoggerCallback(
    tracking_uri=MLFLOW_TRACKING_URI,
    experiment_name=experiment_name,
    save_artifact=True)
```

Once we have the callback defined, all we have to do is update our [`RunConfig`](https://docs.ray.io/en/latest/ray-air/api/doc/ray.air.RunConfig.html){:target="_blank"} to include it.

```python linenums="1"
# Run configuration with MLflow callback
run_config = RunConfig(
    callbacks=[mlflow_callback],
    checkpoint_config=checkpoint_config,
)
```

## Training

With our updated [`RunConfig`](https://docs.ray.io/en/latest/ray-air/api/doc/ray.air.RunConfig.html){:target="_blank"}, with the MLflow callback, we can now train our model and all the necessary components will be logged to MLflow. This is the exact same training workflow we've been using so far from the [training lesson](training.md){:target="_blank"}.

```python linenums="1"
# Dataset
ds = load_data()
train_ds, val_ds = stratify_split(ds, stratify="tag", test_size=test_size)

# Preprocess
preprocessor = CustomPreprocessor()
train_ds = preprocessor.fit_transform(train_ds)
val_ds = preprocessor.transform(val_ds)
train_ds = train_ds.materialize()
val_ds = val_ds.materialize()

# Trainer
trainer = TorchTrainer(
    train_loop_per_worker=train_loop_per_worker,
    train_loop_config=train_loop_config,
    scaling_config=scaling_config,
    run_config=run_config,  # uses RunConfig with MLflow callback
    datasets={"train": train_ds, "val": val_ds},
    dataset_config=dataset_config,
    preprocessor=preprocessor,
)

# Train
results = trainer.fit()
```

<div class="output_subarea output_html rendered_html output_result" dir="auto"><div>
<table border="1" class="dataframe">
<thead>
<tr><th>Trial name              </th><th>status    </th><th>loc             </th><th style="text-align: right;">  iter</th><th style="text-align: right;">  total time (s)</th><th style="text-align: right;">  epoch</th><th style="text-align: right;">    lr</th><th style="text-align: right;">  train_loss</th></tr>
</thead>
<tbody>
<tr><td>TorchTrainer_8c960_00000</td><td>TERMINATED</td><td>10.0.18.44:68577</td><td style="text-align: right;">    10</td><td style="text-align: right;">         76.3089</td><td style="text-align: right;">      9</td><td style="text-align: right;">0.0001</td><td style="text-align: right;"> 0.000549661</td></tr>
</tbody>
</table></div></div>

```python linenums="1"
results.metrics_dataframe
```

<div class="output_subarea output_html rendered_html output_result" dir="auto"><div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>epoch</th>
      <th>lr</th>
      <th>train_loss</th>
      <th>val_loss</th>
      <th>timestamp</th>
      <th>time_this_iter_s</th>
      <th>should_checkpoint</th>
      <th>done</th>
      <th>training_iteration</th>
      <th>trial_id</th>
      <th>date</th>
      <th>time_total_s</th>
      <th>pid</th>
      <th>hostname</th>
      <th>node_ip</th>
      <th>time_since_restore</th>
      <th>iterations_since_restore</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.0001</td>
      <td>0.005196</td>
      <td>0.004071</td>
      <td>1689030896</td>
      <td>14.162520</td>
      <td>True</td>
      <td>False</td>
      <td>1</td>
      <td>8c960_00000</td>
      <td>2023-07-10_16-14-59</td>
      <td>14.162520</td>
      <td>68577</td>
      <td>ip-10-0-18-44</td>
      <td>10.0.18.44</td>
      <td>14.162520</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.0001</td>
      <td>0.004033</td>
      <td>0.003898</td>
      <td>1689030905</td>
      <td>8.704429</td>
      <td>True</td>
      <td>False</td>
      <td>2</td>
      <td>8c960_00000</td>
      <td>2023-07-10_16-15-08</td>
      <td>22.866948</td>
      <td>68577</td>
      <td>ip-10-0-18-44</td>
      <td>10.0.18.44</td>
      <td>22.866948</td>
      <td>2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>0.0001</td>
      <td>0.000550</td>
      <td>0.001182</td>
      <td>1689030958</td>
      <td>6.604867</td>
      <td>True</td>
      <td>False</td>
      <td>10</td>
      <td>8c960_00000</td>
      <td>2023-07-10_16-16-01</td>
      <td>76.308887</td>
      <td>68577</td>
      <td>ip-10-0-18-44</td>
      <td>10.0.18.44</td>
      <td>76.308887</td>
      <td>10</td>
    </tr>
  </tbody>
</table></div></div>

We're going to use the [`search_runs`](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.search_runs){:target="_blank"} function from the [MLflow python API](https://mlflow.org/docs/latest/python_api/mlflow.html){:target="_blank"} to identify the best run in our experiment so far (we' only done one run so far so it will be the run from above).

```python linenums="1"
# Sorted runs
sorted_runs = mlflow.search_runs(experiment_names=[experiment_name], order_by=["metrics.val_loss ASC"])
sorted_runs
```

<pre class="output">
run_id                                                           8e473b640d264808a89914e8068587fb
experiment_id                                                                  853333311265913081
status                                                                                   FINISHED
...
tags.mlflow.runName                                                      TorchTrainer_077f9_00000
Name: 0, dtype: object
</pre>


## Dashboard

Once we're done training, we can use the MLflow dashboard to visualize our results. To do so, we'll use the `mlflow server` command to launch the MLflow dashboard and navigate to the experiment we just created.

```bash
mlflow server -h 0.0.0.0 -p 8080 --backend-store-uri /tmp/mlflow/
```

!!! example "View the dashboard"

    === "Anyscale"

        If you're on [Anyscale Workspaces](https://docs.anyscale.com/develop/workspaces/get-started){:target="_blank"}, then we need to first expose the port of the MLflow server. Run the following command on your Anyscale Workspace terminal to generate the public URL to your MLflow server.

        ```bash
        APP_PORT=8080
        echo https://$APP_PORT-port-$ANYSCALE_SESSION_DOMAIN
        ```

    === "Local"

        If you're running this notebook on your local laptop then head on over to [http://localhost:8080/](http://localhost:8080/) to view your MLflow dashboard.

MLFlow creates a main dashboard with all your experiments and their respective runs. We can sort runs by clicking on the column headers.

<img src="/static/images/mlops/experiment_tracking/dashboard.png" width="1000" alt="mlflow runs">

And within each run, we can view metrics, parameters, artifacts, etc.

<img src="/static/images/mlops/experiment_tracking/params.png" width="1000" alt="mlflow params">

And we can even create custom plots to help us visualize our results.

<img src="/static/images/mlops/experiment_tracking/plots.png" width="1000" alt="mlflow plots">

## Loading

After inspection and once we've identified an experiment that we like, we can load the model for evaluation and inference.

```python linenums="1"
from ray.air import Result
from urllib.parse import urlparse
```

We're going to create a small utility function that uses an MLflow run's artifact path to load a Ray [`Result`](https://docs.ray.io/en/latest/tune/api/doc/ray.air.Result.html){:target="_blank"} object. We'll then use the `Result` object to load the best checkpoint.

```python linenums="1"
def get_best_checkpoint(run_id):
    artifact_dir = urlparse(mlflow.get_run(run_id).info.artifact_uri).path  # get path from mlflow
    results = Result.from_path(artifact_dir)
    return results.best_checkpoints[0][0]
```

With a particular run's best checkpoint, we can load the model from it and use it.

```python linenums="1"
# Evaluate on test split
best_checkpoint = get_best_checkpoint(run_id=best_run.run_id)
predictor = TorchPredictor.from_checkpoint(best_checkpoint)
performance = evaluate(ds=test_ds, predictor=predictor)
print (json.dumps(performance, indent=2))
```
<pre class="output">
{
  "precision": 0.9281010510531216,
  "recall": 0.9267015706806283,
  "f1": 0.9269438615952555
}
</pre>

Before we can use our model for inference, we need to load the preprocessor from our predictor and apply it to our input data.

```python linenums="1"
# Preprocessor
preprocessor = predictor.get_preprocessor()
```
```python linenums="1"
# Predict on sample
title = "Transfer learning with transformers"
description = "Using transformers for transfer learning on text classification tasks."
sample_df = pd.DataFrame([{"title": title, "description": description, "tag": "other"}])
predict_with_proba(df=sample_df, predictor=predictor)
```
<pre class="output">
[{'prediction': 'natural-language-processing',
  'probabilities': {'computer-vision': 0.00038025028,
   'mlops': 0.00038209034,
   'natural-language-processing': 0.998792,
   'other': 0.00044562898}}]
</pre>

In the [next lesson](tuning.md){:target="_blank"} we'll learn how to tune our models and use our MLflow dashboard to compare the results.


<!-- Course signup -->
{% include "templates/signup.md" %}

<!-- Citation -->
{% include "templates/cite.md" %}