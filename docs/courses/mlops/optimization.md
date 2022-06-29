---
template: lesson.html
title: Optimizing Hyperparameters
description: Optimizing a subset of hyperparameters to achieve an objective.
keywords: optimization, hyperparameters, optuna, ray, hyperopt, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/mlops.png
repository: https://github.com/GokuMohandas/mlops-course
notebook: https://colab.research.google.com/github/GokuMohandas/mlops-course/blob/main/notebooks/tagifai.ipynb
---

{% include "styles/lesson.md" %}

## Intuition

Optimization is the process of fine-tuning the hyperparameters in our experiment to optimize towards a particular objective. It can be a computationally involved process depending on the number of parameters, search space and model architectures. Hyperparameters don't just include the model's parameters but they also include parameters (choices) from preprocessing, splitting, etc. When we look at all the different parameters that can be tuned, it quickly becomes a very large search space. However, just because something is a hyperparameter doesn't mean we need to tune it.

- It's absolutely alright to fix some hyperparameters (ex. `lower=True` during preprocessing) and remove them from the tuning subset. Just be sure to note which parameters you are fixing and your reasoning for doing so.
- You can initially just tune a small, yet influential, subset of hyperparameters that you believe will yield best results.

We want to optimize our hyperparameters so that we can understand how each of them affects our objective. By running many trials across a reasonable search space, we can determine near ideal values for our different parameters. It's also a great opportunity to determine if a smaller parameters yield similar performances as larger ones (efficiency).

## Tools
There are many options for hyperparameter tuning ([Optuna](https://github.com/optuna/optuna){:target="_blank"}, [Ray tune](https://github.com/ray-project/ray/tree/master/python/ray/tune){:target="_blank"}, [Hyperopt](https://github.com/hyperopt/hyperopt){:target="_blank"}, etc.). We'll be using Optuna for it's simplicity, popularity and efficiency though they are all equally so. It really comes down to familiarity and whether a library has a specific implementation readily tested and available.

## Application

There are many factors to consider when performing hyperparameter optimization and luckily Optuna allows us to [implement](https://optuna.readthedocs.io/en/stable/reference/) them with ease. We'll be conducting a small study where we'll tune a set of arguments (we'll do a much more thorough [study](https://optuna.readthedocs.io/en/stable/reference/study.html) of the parameter space when we move our code to Python scripts). Here's the process for the study:

1. Define an objective (metric) and identifying the [direction](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.StudyDirection.html#optuna.study.StudyDirection){:target="_blank"} to optimize.
2. `#!js [OPTIONAL]` Choose a [sampler](https://optuna.readthedocs.io/en/stable/reference/samplers.html){:target="_blank"} for determining parameters for subsequent trials. (default is a tree based sampler).
3. `#!js [OPTIONAL]` Choose a [pruner](https://optuna.readthedocs.io/en/stable/reference/pruners.html){:target="_blank"} to end unpromising trials early.
4. Define the parameters to tune in each [trial](https://optuna.readthedocs.io/en/stable/reference/trial.html){:target="_blank"} and the [distribution](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna-trial-trial){:target="_blank"} of values to sample.

```bash
pip install optuna==2.10.0 numpyencoder==0.3.0 -q
```

```python linenums="1"
import optuna
```

We're going to use the same training function as before since we've added the functionality to prune a specific run if the `trial` argument is not `None`.
```python linenums="1"
# Pruning (inside train() function)
trial.report(val_loss, epoch)
if trial.should_prune():
    raise optuna.TrialPruned()
```

## Objective

We need to define an objective function that will consume a trial and a set of arguments and produce the metric to optimize on (`f1` in our case).

```python linenums="1"
def objective(args, trial):
    """Objective function for optimization trials."""
    # Parameters to tune
    args.analyzer = trial.suggest_categorical("analyzer", ["word", "char", "char_wb"])
    args.ngram_max_range = trial.suggest_int("ngram_max_range", 3, 10)
    args.learning_rate = trial.suggest_loguniform("learning_rate", 1e-2, 1e0)
    args.power_t = trial.suggest_uniform("power_t", 0.1, 0.5)

    # Train & evaluate
    artifacts = train(args=args, df=df, trial=trial)

    # Set additional attributes
    performance = artifacts["performance"]
    print(json.dumps(performance, indent=2))
    trial.set_user_attr("precision", performance["precision"])
    trial.set_user_attr("recall", performance["recall"])
    trial.set_user_attr("f1", performance["f1"])

    return performance["f1"]
```

## Study

We're ready to kick off our study with our [MLFlowCallback](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.integration.MLflowCallback.html){:target="_blank"} so we can track all of the different trials.

```python linenums="1"
from numpyencoder import NumpyEncoder
from optuna.integration.mlflow import MLflowCallback
```
```python linenums="1"
NUM_TRIALS = 20  # small sample for now
```
```python linenums="1"
# Optimize
pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
study = optuna.create_study(study_name="optimization", direction="maximize", pruner=pruner)
mlflow_callback = MLflowCallback(
    tracking_uri=mlflow.get_tracking_uri(), metric_name="f1")
study.optimize(lambda trial: objective(args, trial),
            n_trials=NUM_TRIALS,
            callbacks=[mlflow_callback])
```
<pre class="output">
A new study created in memory with name: optimization
Epoch: 00 | train_loss: 1.34116, val_loss: 1.35091
...
Epoch: 90 | train_loss: 0.32167, val_loss: 0.57661
Stopping early!
Trial 0 finished with value: 0.7703281822265505 and parameters: {'analyzer': 'char', 'ngram_max_range': 10, 'learning_rate': 0.025679294001785473, 'power_t': 0.15046698128066294}. Best is trial 0 with value: 0.7703281822265505.

...

Trial 10 pruned.

...

Epoch: 80 | train_loss: 0.16680, val_loss: 0.43964
Epoch: 90 | train_loss: 0.16134, val_loss: 0.43686
Trial 19 finished with value: 0.8470890576153735 and parameters: {'analyzer': 'char_wb', 'ngram_max_range': 4, 'learning_rate': 0.08452049154544644, 'power_t': 0.39657115651885855}. Best is trial 3 with value: 0.8470890576153735.
</pre>

```python linenums="1"
# Run MLFlow server and localtunnel
get_ipython().system_raw("mlflow server -h 0.0.0.0 -p 8000 --backend-store-uri $PWD/experiments/ &")
!npx localtunnel --port 8000
```
1. Click on the "optimization" experiment on the left side under **Experiments**.
2. Select runs to compare by clicking on the toggle box to the left of each run or by clicking on the toggle box in the header to select all runs in this experiment.
3. Click on the **Compare** button.

<div class="ai-center-all">
    <img src="/static/images/mlops/hyperparameter_optimization/compare.png" width="1000" alt="compare">
</div>

4. In the comparison page, we can then view the results through various lens (contours, parallel coordinates, etc.)

<div class="ai-center-all">
    <img src="/static/images/mlops/hyperparameter_optimization/contour.png" width="1000" alt="contour plots">
</div>
<div class="ai-center-all">
    <img src="/static/images/mlops/hyperparameter_optimization/parallel_coordinates.png" width="1000" alt="parallel coordinates">
</div>

```python linenums="1"
# All trials
trials_df = study.trials_dataframe()
trials_df = trials_df.sort_values(["user_attrs_f1"], ascending=False)  # sort by metric
trials_df.head()
```
<div class="output_subarea output_html rendered_html"><div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>number</th>
      <th>value</th>
      <th>datetime_start</th>
      <th>datetime_complete</th>
      <th>duration</th>
      <th>params_analyzer</th>
      <th>params_learning_rate</th>
      <th>params_ngram_max_range</th>
      <th>params_power_t</th>
      <th>user_attrs_f1</th>
      <th>user_attrs_precision</th>
      <th>user_attrs_recall</th>
      <th>state</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0.847089</td>
      <td>2022-05-18 18:16:58.108105</td>
      <td>2022-05-18 18:17:03.569948</td>
      <td>0 days 00:00:05.461843</td>
      <td>char_wb</td>
      <td>0.088337</td>
      <td>4</td>
      <td>0.118196</td>
      <td>0.847089</td>
      <td>0.887554</td>
      <td>0.833333</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>19</th>
      <td>19</td>
      <td>0.847089</td>
      <td>2022-05-18 18:17:58.219462</td>
      <td>2022-05-18 18:18:00.642571</td>
      <td>0 days 00:00:02.423109</td>
      <td>char_wb</td>
      <td>0.084520</td>
      <td>4</td>
      <td>0.396571</td>
      <td>0.847089</td>
      <td>0.887554</td>
      <td>0.833333</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12</td>
      <td>0.840491</td>
      <td>2022-05-18 18:17:41.845179</td>
      <td>2022-05-18 18:17:45.792068</td>
      <td>0 days 00:00:03.946889</td>
      <td>char_wb</td>
      <td>0.139578</td>
      <td>7</td>
      <td>0.107273</td>
      <td>0.840491</td>
      <td>0.877431</td>
      <td>0.826389</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13</td>
      <td>0.840491</td>
      <td>2022-05-18 18:17:45.862705</td>
      <td>2022-05-18 18:17:49.657014</td>
      <td>0 days 00:00:03.794309</td>
      <td>char_wb</td>
      <td>0.154396</td>
      <td>7</td>
      <td>0.433669</td>
      <td>0.840491</td>
      <td>0.877431</td>
      <td>0.826389</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>15</th>
      <td>15</td>
      <td>0.836255</td>
      <td>2022-05-18 18:17:50.464948</td>
      <td>2022-05-18 18:17:54.446481</td>
      <td>0 days 00:00:03.981533</td>
      <td>char_wb</td>
      <td>0.083253</td>
      <td>7</td>
      <td>0.106982</td>
      <td>0.836255</td>
      <td>0.881150</td>
      <td>0.819444</td>
      <td>COMPLETE</td>
    </tr>
  </tbody>
</table>
</div></div>

```python linenums="1"
# Best trial
print (f"Best value (f1): {study.best_trial.value}")
print (f"Best hyperparameters: {json.dumps(study.best_trial.params, indent=2)}")
```
<pre class="output">
Best value (f1): 0.8535985582060417
Best hyperparameters: {
  "analyzer": "char_wb",
  "ngram_max_range": 4,
  "learning_rate": 0.08981103667371809,
  "power_t": 0.2583427488720579
}
</pre>

```python linenums="1"
# Save best parameter values
args = {**args.__dict__, **study.best_trial.params}
print (json.dumps(args, indent=2, cls=NumpyEncoder))
```
<pre class="output">
{
  "lower": true,
  "stem": false,
  "analyzer": "char_wb",
  "ngram_max_range": 4,
  "alpha": 0.0001,
  "learning_rate": 0.08833689034118489,
  "power_t": 0.1181958972675695
}
</pre>

... and now we're finally ready to move from working in Jupyter notebooks to Python scripts. We'll be revisiting everything we did so far, but this time with proper software engineering principles such as object oriented programming (OOPs), styling, testing, etc. → [https://madewithml.com/#mlops](https://madewithml.com/#mlops)

<!-- Citation -->
{% include "cite.md" %}
