---
template: lesson.html
title: Hyperparameter Tuning
description: Tuning a set of hyperparameters to optimize our model's performance.
keywords: tuning, hyperparameter tuning, optimization, hyperparameters, optuna, ray, hyperopt, mlops, machine learning
image: https://madewithml.com/static/images/mlops.png
repository: https://github.com/GokuMohandas/Made-With-ML
notebook: https://github.com/GokuMohandas/Made-With-ML/blob/main/notebooks/madewithml.ipynb
---

{% include "styles/lesson.md" %}

## Intuition

Hyperparameter tuning is the process of discovering a set of performant parameter values for our model. It can be a computationally involved process depending on the number of parameters, search space and model architectures. Hyperparameters don't just include the model's parameters but could also include parameters related to preprocessing, splitting, etc. When we look at all the different parameters that can be tuned, it quickly becomes a very large search space. However, just because something is a hyperparameter doesn't mean we need to tune it.

- It's absolutely acceptable to fix some hyperparameters (ex. using lower cased text [`#!python lower=True`] during preprocessing).
- You can initially just tune a small, yet influential, subset of hyperparameters that you believe will yield great results.

We want to optimize our hyperparameters so that we can understand how each of them affects our objective. By running many trials across a reasonable search space, we can determine near ideal values for our different parameters.

## Frameworks

There are many options for hyperparameter tuning ([Ray tune](https://docs.ray.io/en/latest/tune/index.html){:target="_blank"}, [Optuna](https://github.com/optuna/optuna){:target="_blank"}, [Hyperopt](https://github.com/hyperopt/hyperopt){:target="_blank"}, etc.). We'll be using Ray Tune with it's [HyperOpt integration](https://docs.ray.io/en/latest/tune/api/suggestion.html#hyperopt-tune-search-hyperopt-hyperoptsearch){:target="_blank"} for it's simplicity and general popularity. Ray Tune also has a wide variety of support for many [other tune search algorithms](https://docs.ray.io/en/latest/tune/api/suggestion.html){:target="_blank"} (Optuna, Bayesian, etc.).

## Set up

There are many factors to consider when performing hyperparameter tuning. We'll be conducting a small study where we'll tune just a few key hyperparameters across a few trials. Feel free to include additional parameters and to increase the number trials in the tuning experiment.

```python linenums="1"
# Number of trials (small sample)
num_runs = 2
```

We'll start with some the set up, data and model prep as we've done in previous lessons.

```python linenums="1"
from ray import tune
from ray.tune import Tuner
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.hyperopt import HyperOptSearch
```

```python linenums="1"
# Set up
set_seeds()
```
```python linenums="1"
# Dataset
ds = load_data()
train_ds, val_ds = stratify_split(ds, stratify="tag", test_size=test_size)
```
```python linenums="1"
# Preprocess
preprocessor = CustomPreprocessor()
train_ds = preprocessor.fit_transform(train_ds)
val_ds = preprocessor.transform(val_ds)
train_ds = train_ds.materialize()
val_ds = val_ds.materialize()
```
```python linenums="1"
# Trainer
trainer = TorchTrainer(
    train_loop_per_worker=train_loop_per_worker,
    train_loop_config=train_loop_config,
    scaling_config=scaling_config,
    datasets={"train": train_ds, "val": val_ds},
    dataset_config=dataset_config,
    preprocessor=preprocessor,
)
```
```python linenums="1"
# MLflow callback
mlflow_callback = MLflowLoggerCallback(
    tracking_uri=MLFLOW_TRACKING_URI,
    experiment_name=experiment_name,
    save_artifact=True)
```

## Tune configuration

We can think of tuning as training across different combinations of parameters. For this, we'll need to define several configurations around when to stop tuning (stopping criteria), how to define the next set of parameters to train with (search algorithm) and even the different values that the parameters can take (search space).

We'll start by defining our [`CheckpointConfig`](https://docs.ray.io/en/latest/ray-air/api/doc/ray.air.CheckpointConfig.html){:target="_blank"} and [`RunConfig`](https://docs.ray.io/en/latest/ray-air/api/doc/ray.air.RunConfig.html){:target="_blank"} as we did for [training](training.md){:target="_blank"}:

```python linenums="1"
# Run configuration
checkpoint_config = CheckpointConfig(num_to_keep=1, checkpoint_score_attribute="val_loss", checkpoint_score_order="min")
run_config = RunConfig(
    callbacks=[mlflow_callback],
    checkpoint_config=checkpoint_config
)
```

> Notice that we use the same `mlflow_callback` from our [experiment tracking lesson](experiment-tracking.md){:target="_blank"} so all of our runs will be tracked to MLflow automatically.

### Search algorithm

Next, we're going to set the initial parameter values and the search algorithm ([`HyperOptSearch`](https://docs.ray.io/en/latest/tune/api/doc/ray.tune.search.hyperopt.HyperOptSearch.html){:target="_blank"}) for our tuning experiment. We're also going to set the maximum number of trials that can be run concurrently ([`ConcurrencyLimiter`](https://docs.ray.io/en/latest/tune/api/doc/ray.tune.search.ConcurrencyLimiter.html){:target="_blank"}) based on the compute resources we have.

```python linenums="1"
# Hyperparameters to start with
initial_params = [{"train_loop_config": {"dropout_p": 0.5, "lr": 1e-4, "lr_factor": 0.8, "lr_patience": 3}}]
search_alg = HyperOptSearch(points_to_evaluate=initial_params)
search_alg = ConcurrencyLimiter(search_alg, max_concurrent=2)
```

!!! tip
    It's a good idea to start with some initial parameter values that you think might be reasonable. This can help speed up the tuning process and also guarantee at least one experiment that will perform decently well.

### Search space

Next, we're going to define the parameter search space by choosing the parameters, their distribution and range of values. Depending on the parameter type, we have many different [distributions](https://docs.ray.io/en/latest/tune/api/search_space.html#random-distributions-api){:target="_blank"} to choose from.

```python linenums="1"
# Parameter space
param_space = {
    "train_loop_config": {
        "dropout_p": tune.uniform(0.3, 0.9),
        "lr": tune.loguniform(1e-5, 5e-4),
        "lr_factor": tune.uniform(0.1, 0.9),
        "lr_patience": tune.uniform(1, 10),
    }
}
```

### Scheduler

Next, we're going to define a scheduler to prune unpromising trials. We'll be using [`AsyncHyperBandScheduler`](https://docs.ray.io/en/latest/tune/api/doc/ray.tune.schedulers.AsyncHyperBandScheduler.html){:target="_blank"} ([ASHA](https://arxiv.org/abs/1810.05934){:target="_blank"}), which is a very popular and aggressive early-stopping algorithm. Due to our aggressive scheduler, we'll set a `grace_period` to allow the trials to run for at least a few epochs before pruning and a maximum of `max_t` epochs.

```python linenums="1"
# Scheduler
scheduler = AsyncHyperBandScheduler(
    max_t=train_loop_config["num_epochs"],  # max epoch (<time_attr>) per trial
    grace_period=5,  # min epoch (<time_attr>) per trial
)
```

## Tuner

Finally, we're going to define a [`TuneConfig`](https://docs.ray.io/en/latest/tune/api/doc/ray.tune.TuneConfig.html){:target="_blank"} that will combine the `search_alg` and `scheduler` we've defined above.

```python linenums="1"
# Tune config
tune_config = tune.TuneConfig(
    metric="val_loss",
    mode="min",
    search_alg=search_alg,
    scheduler=scheduler,
    num_samples=num_runs,
)
```

And now, we'll pass in our `trainer` object with our configurations to create a [`Tuner`](https://docs.ray.io/en/latest/tune/api/doc/ray.tune.Tuner.html){:target="_blank"} object that we can run.

```python linenums="1"
# Tuner
tuner = Tuner(
    trainable=trainer,
    run_config=run_config,
    param_space=param_space,
    tune_config=tune_config,
)
```

```python linenums="1"
# Tune
results = tuner.fit()
```

<div class="output_subarea output_html rendered_html output_result" dir="auto"><div>
  <table border="1" class="dataframe">
      <thead>
        <tr>
            <th>Trial name</th>
            <th>status</th>
            <th>loc</th>
            <th style="text-align: right;">iter</th>
            <th style="text-align: right;">total time (s)</th>
            <th style="text-align: right;">epoch</th>
            <th style="text-align: right;">lr</th>
            <th style="text-align: right;">train_loss</th>
        </tr>
      </thead>
      <tbody>
        <tr>
            <td>TorchTrainer_8e6e0_00000</td>
            <td>TERMINATED</td>
            <td>10.0.48.210:93017</td>
            <td style="text-align: right;">10</td>
            <td style="text-align: right;">76.2436</td>
            <td style="text-align: right;">9</td>
            <td style="text-align: right;">0.0001</td>
            <td style="text-align: right;">0.0333853</td>
        </tr>
      </tbody>
  </table>
</div></div>

```python linenums="1"
# All trials in experiment
results.get_dataframe()
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
      <th>...</th>
      <th>pid</th>
      <th>hostname</th>
      <th>node_ip</th>
      <th>time_since_restore</th>
      <th>iterations_since_restore</th>
      <th>config/train_loop_config/dropout_p</th>
      <th>config/train_loop_config/lr</th>
      <th>config/train_loop_config/lr_factor</th>
      <th>config/train_loop_config/lr_patience</th>
      <th>logdir</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9</td>
      <td>0.000100</td>
      <td>0.04096</td>
      <td>0.217990</td>
      <td>1689460552</td>
      <td>6.890944</td>
      <td>True</td>
      <td>True</td>
      <td>10</td>
      <td>094e2a7e</td>
      <td>...</td>
      <td>94006</td>
      <td>ip-10-0-48-210</td>
      <td>10.0.48.210</td>
      <td>76.588228</td>
      <td>10</td>
      <td>0.500000</td>
      <td>0.000100</td>
      <td>0.800000</td>
      <td>3.000000</td>
      <td>/home/ray/ray_results/TorchTrainer_2023-07-15_...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0.000027</td>
      <td>0.63066</td>
      <td>0.516547</td>
      <td>1689460571</td>
      <td>14.614296</td>
      <td>True</td>
      <td>True</td>
      <td>1</td>
      <td>4f419368</td>
      <td>...</td>
      <td>94862</td>
      <td>ip-10-0-48-210</td>
      <td>10.0.48.210</td>
      <td>14.614296</td>
      <td>1</td>
      <td>0.724894</td>
      <td>0.000027</td>
      <td>0.780224</td>
      <td>5.243006</td>
      <td>/home/ray/ray_results/TorchTrainer_2023-07-15_...</td>
    </tr>
  </tbody>
</table>
</div></div>

And on our MLflow dashboard, we can create useful plots like a parallel coordinates plot to visualize the different hyperparameters and their values across the different trials.

<div class="ai-center-all">
  <img src="/static/images/mlops/tuning/parallel_coordinates.png" width="800" alt="parallel coordinates plot">
</div>

## Best trial

And from these results, we can extract the best trial and its hyperparameters:

```python linenums="1"
# Best trial's epochs
best_trial = results.get_best_result(metric="val_loss", mode="min")
best_trial.metrics_dataframe
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
      <td>0.582092</td>
      <td>0.495889</td>
      <td>1689460489</td>
      <td>14.537316</td>
      <td>True</td>
      <td>False</td>
      <td>1</td>
      <td>094e2a7e</td>
      <td>2023-07-15_15-34-53</td>
      <td>14.537316</td>
      <td>94006</td>
      <td>ip-10-0-48-210</td>
      <td>10.0.48.210</td>
      <td>14.537316</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.0001</td>
      <td>0.492427</td>
      <td>0.430734</td>
      <td>1689460497</td>
      <td>7.144841</td>
      <td>True</td>
      <td>False</td>
      <td>2</td>
      <td>094e2a7e</td>
      <td>2023-07-15_15-35-00</td>
      <td>21.682157</td>
      <td>94006</td>
      <td>ip-10-0-48-210</td>
      <td>10.0.48.210</td>
      <td>21.682157</td>
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
      <td>0.040960</td>
      <td>0.217990</td>
      <td>1689460552</td>
      <td>6.890944</td>
      <td>True</td>
      <td>True</td>
      <td>10</td>
      <td>094e2a7e</td>
      <td>2023-07-15_15-35-55</td>
      <td>76.588228</td>
      <td>94006</td>
      <td>ip-10-0-48-210</td>
      <td>10.0.48.210</td>
      <td>76.588228</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div></div>

```python linenums="1"
# Best trial's hyperparameters
best_trial.config["train_loop_config"]
```

<pre class="output">
{'dropout_p': 0.5, 'lr': 0.0001, 'lr_factor': 0.8, 'lr_patience': 3.0}
</pre>

And now we'll load the best run from our experiment, which includes all the runs we've done so far (before and including the tuning runs).

```python linenums="1"
# Sorted runs
sorted_runs = mlflow.search_runs(experiment_names=[experiment_name], order_by=["metrics.val_loss ASC"])
sorted_runs
```

<div class="output_subarea output_html rendered_html output_result" dir="auto"><div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>run_id</th>
      <th>experiment_id</th>
      <th>status</th>
      <th>artifact_uri</th>
      <th>start_time</th>
      <th>end_time</th>
      <th>metrics.lr</th>
      <th>metrics.epoch</th>
      <th>metrics.train_loss</th>
      <th>metrics.val_loss</th>
      <th>...</th>
      <th>metrics.config/train_loop_config/num_classes</th>
      <th>params.train_loop_config/dropout_p</th>
      <th>params.train_loop_config/lr_patience</th>
      <th>params.train_loop_config/lr_factor</th>
      <th>params.train_loop_config/lr</th>
      <th>params.train_loop_config/num_classes</th>
      <th>params.train_loop_config/num_epochs</th>
      <th>params.train_loop_config/batch_size</th>
      <th>tags.mlflow.runName</th>
      <th>tags.trial_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>b140fdbc40804c4f94f9aef33e5279eb</td>
      <td>999409133275979199</td>
      <td>FINISHED</td>
      <td>file:///tmp/mlflow/999409133275979199/b140fdbc...</td>
      <td>2023-07-15 22:34:39.108000+00:00</td>
      <td>2023-07-15 22:35:56.260000+00:00</td>
      <td>0.000100</td>
      <td>9.0</td>
      <td>0.040960</td>
      <td>0.217990</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.5</td>
      <td>3.0</td>
      <td>0.8</td>
      <td>0.0001</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>TorchTrainer_094e2a7e</td>
      <td>TorchTrainer_094e2a7e</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9ff8133613604564b0316abadc23b3b8</td>
      <td>999409133275979199</td>
      <td>FINISHED</td>
      <td>file:///tmp/mlflow/999409133275979199/9ff81336...</td>
      <td>2023-07-15 22:33:05.206000+00:00</td>
      <td>2023-07-15 22:34:24.322000+00:00</td>
      <td>0.000100</td>
      <td>9.0</td>
      <td>0.033385</td>
      <td>0.218394</td>
      <td>...</td>
      <td>4.0</td>
      <td>0.5</td>
      <td>3</td>
      <td>0.8</td>
      <td>0.0001</td>
      <td>4</td>
      <td>10</td>
      <td>256</td>
      <td>TorchTrainer_8e6e0_00000</td>
      <td>TorchTrainer_8e6e0_00000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>e4f2d6be9eaa4302b3f697a36ed07d8c</td>
      <td>999409133275979199</td>
      <td>FINISHED</td>
      <td>file:///tmp/mlflow/999409133275979199/e4f2d6be...</td>
      <td>2023-07-15 22:36:00.339000+00:00</td>
      <td>2023-07-15 22:36:15.459000+00:00</td>
      <td>0.000027</td>
      <td>0.0</td>
      <td>0.630660</td>
      <td>0.516547</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.7248940325059469</td>
      <td>5.243006476496198</td>
      <td>0.7802237354477737</td>
      <td>2.7345833037950673e-05</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>TorchTrainer_4f419368</td>
      <td>TorchTrainer_4f419368</td>
    </tr>
  </tbody>
</table>
</div></div>

From this we can load the best checkpoint from the best run and evaluate it on the test split.

```python linenums="1"
# Evaluate on test split
run_id = sorted_runs.iloc[0].run_id
best_checkpoint = get_best_checkpoint(run_id=run_id)
predictor = TorchPredictor.from_checkpoint(best_checkpoint)
performance = evaluate(ds=test_ds, predictor=predictor)
print (json.dumps(performance, indent=2))
```

<pre class="output">
{
  "precision": 0.9487609194455242,
  "recall": 0.9476439790575916,
  "f1": 0.9471734167970421
}
</pre>

And, just as we did in previous lessons, use our model for inference.

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
  'probabilities': {'computer-vision': 0.0003628606,
   'mlops': 0.0002862369,
   'natural-language-processing': 0.99908364,
   'other': 0.0002672623}}]
</pre>

Now that we're tuned our model, in the [next lesson](evaluation.md){:target="_blank"}, we're going to perform a much more intensive evaluation on our model compared to just viewing it's overall metrics on a test set.

<!-- Course signup -->
{% include "templates/signup.md" %}

<!-- Citation -->
{% include "templates/cite.md" %}
