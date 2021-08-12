---
template: lesson.html
title: Optimizing Hyperparameters
description: Optimizing a subset of hyperparameters to achieve an objective.
keywords: optimization, hyperparameters, optuna, ray, hyperopt, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/mlops.png
repository: https://github.com/GokuMohandas/MLOps
notebook: https://colab.research.google.com/github/GokuMohandas/MLOps/blob/main/notebooks/tagifai.ipynb
---

{% include "styles/lesson.md" %}

## Intuition

Optimization is the process of fine-tuning the hyperparameters in our experiment to optimize towards a particular objective. It can be a computationally involved process depending on the number of parameters, search space and model architectures. Hyperparameters don't just include the model's parameters but they also include parameters (choices) from preprocessing, splitting, etc. When we look at all the different parameters that can be tuned, it quickly becomes a very large search space. However, just because something is a hyperparameter doesn't mean we need to tune it.

- It's absolutely alright to fix some hyperparameters (ex. `lower=True` during preprocessing) and remove them from the current tuning subset. Just be sure to note which parameters you are fixing and your reasoning for doing so.
- You can initially just tune a small, yet influential, subset of hyperparameters that you believe will yield best results.

We want to optimize our hyperparameters so that we can understand how each of them affects our objective. By running many trials across a reasonable search space, we can determine near ideal values for our different parameters. It's also a great opportunity to determine if a smaller parameters yield similar performances as larger ones (efficiency).

## Tools
There are many options for hyperparameter tuning ([Optuna](https://github.com/optuna/optuna){:target="_blank"}, [Ray tune](https://github.com/ray-project/ray/tree/master/python/ray/tune){:target="_blank"}, [Hyperopt](https://github.com/hyperopt/hyperopt){:target="_blank"}, etc.). We'll be using Optuna for it's simplicity, popularity and efficiency though they are all equally so. It really comes down to familiarity and whether a library has a specific implementation readily tested and available.

## Application

There are many factors to consider when performing hyperparameter optimization and luckily Optuna allows us to [implement](https://optuna.readthedocs.io/en/stable/reference/) them with ease. We'll be conducting a small study where we'll tune a set of arguments (we'll do a much more thorough [study](https://optuna.readthedocs.io/en/stable/reference/study.html) of the parameter space when we move our code to Python scripts). Here's the process for the study:

1. Define an objective (metric) and identifying the [direction](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.StudyDirection.html#optuna.study.StudyDirection){:target="_blank"} to optimize.
2. `[OPTIONAL]` Choose a [sampler](https://optuna.readthedocs.io/en/stable/reference/samplers.html){:target="_blank"} for determining parameters for subsequent trials. (default is a tree based sampler).
3. `[OPTIONAL]` Choose a [pruner](https://optuna.readthedocs.io/en/stable/reference/pruners.html){:target="_blank"} to end unpromising trials early.
4. Define the parameters to tune in each [trial](https://optuna.readthedocs.io/en/stable/reference/trial.html){:target="_blank"} and the [distribution](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna-trial-trial){:target="_blank"} of values to sample.

!!! note
    There are many more options (multiple objectives, storage options, etc.) to explore but this basic set up will allow us to optimize quite well.

```python linenums="1"
from argparse import Namespace
from numpyencoder import NumpyEncoder

# Specify arguments
params = Namespace(
    char_level=True,
    filter_sizes=list(range(1, 11)),
    batch_size=64,
    embedding_dim=128,
    num_filters=128,
    hidden_dim=128,
    dropout_p=0.5,
    lr=2e-4,
    num_epochs=200,
    patience=10,
)
```

We're going to modify our `Trainer` object to be able to prune unpromising trials based on the trial's validation loss.
```python linenums="1"
class Trainer(object):
    ...
    def train(self, ...):
        ...
        # Pruning based on the intermediate value
        self.trial.report(val_loss, epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned()
        ...
```

??? quote "Code for complete `Trainer` class"
    ```python linenums="1" hl_lines="110-113"
    # Trainer (modified for experiment tracking)
    class Trainer(object):
        def __init__(self, model, device, loss_fn=None,
                    optimizer=None, scheduler=None, trial=None):

            # Set params
            self.model = model
            self.device = device
            self.loss_fn = loss_fn
            self.optimizer = optimizer
            self.scheduler = scheduler
            self.trial = trial

        def train_step(self, dataloader):
            """Train step."""
            # Set model to train mode
            self.model.train()
            loss = 0.0

            # Iterate over train batches
            for i, batch in enumerate(dataloader):
                # Step
                batch = [item.to(self.device) for item in batch]
                inputs, targets = batch[:-1], batch[-1]
                self.optimizer.zero_grad()  # Reset gradients
                z = self.model(inputs)  # Forward pass
                J = self.loss_fn(z, targets)  # Define loss
                J.backward()  # Backward pass
                self.optimizer.step()  # Update weights

                # Cumulative Metrics
                loss += (J.detach().item() - loss) / (i + 1)

            return loss

        def eval_step(self, dataloader):
            """Validation or test step."""
            # Set model to eval mode
            self.model.eval()
            loss = 0.0
            y_trues, y_probs = [], []

            # Iterate over val batches
            with torch.no_grad():
                for i, batch in enumerate(dataloader):

                    # Step
                    batch = [item.to(self.device) for item in batch]  # Set device
                    inputs, y_true = batch[:-1], batch[-1]
                    z = self.model(inputs)  # Forward pass
                    J = self.loss_fn(z, y_true).item()

                    # Cumulative Metrics
                    loss += (J - loss) / (i + 1)

                    # Store outputs
                    y_prob = torch.sigmoid(z).cpu().numpy()
                    y_probs.extend(y_prob)
                    y_trues.extend(y_true.cpu().numpy())

            return loss, np.vstack(y_trues), np.vstack(y_probs)

        def predict_step(self, dataloader):
            """Prediction step."""
            # Set model to eval mode
            self.model.eval()
            y_probs = []

            # Iterate over val batches
            with torch.no_grad():
                for i, batch in enumerate(dataloader):

                    # Forward pass w/ inputs
                    inputs, targets = batch[:-1], batch[-1]
                    y_prob = self.model(inputs)

                    # Store outputs
                    y_probs.extend(y_prob)

            return np.vstack(y_probs)

        def train(self, num_epochs, patience, train_dataloader, val_dataloader):
            best_val_loss = np.inf
            for epoch in range(num_epochs):
                # Steps
                train_loss = self.train_step(dataloader=train_dataloader)
                val_loss, _, _ = self.eval_step(dataloader=val_dataloader)
                self.scheduler.step(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = self.model
                    _patience = patience  # reset _patience
                else:
                    _patience -= 1
                if not _patience:  # 0
                    print("Stopping early!")
                    break

                # Logging
                print(
                    f"Epoch: {epoch+1} | "
                    f"train_loss: {train_loss:.5f}, "
                    f"val_loss: {val_loss:.5f}, "
                    f"lr: {self.optimizer.param_groups[0]['lr']:.2E}, "
                    f"_patience: {_patience}"
                )

                # Pruning based on the intermediate value
                self.trial.report(val_loss, epoch)
                if self.trial.should_prune():
                    raise optuna.TrialPruned()

            return best_model, best_val_loss
    ```

We'll also modify our `train_cnn` function to include information about the trial.
```python linenums="1"
def train_cnn(params, df, trial=None):
    ...
    # Trainer module
    trainer = Trainer(
        model=model, device=device, loss_fn=loss_fn,
        optimizer=optimizer, scheduler=scheduler, trial=trial)
    ...
```

??? quote "Code for complete `train_cnn` function"
    ```python linenums="1" hl_lines="69-72"
    def train_cnn(params, df, trial=None):
        """Train a CNN using specific arguments."""

        # Set seeds
        set_seeds()

        # Get data splits
        preprocessed_df = df.copy()
        preprocessed_df.text = preprocessed_df.text.apply(preprocess, lower=True)
        X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = get_data_splits(preprocessed_df)
        X_test_raw = X_test
        num_classes = len(label_encoder)

        # Set device
        cuda = True
        device = torch.device('cuda' if (
            torch.cuda.is_available() and cuda) else 'cpu')
        torch.set_default_tensor_type('torch.FloatTensor')
        if device.type == 'cuda':
            torch.set_default_tensor_type('torch.cuda.FloatTensor')

        # Tokenize
        tokenizer = Tokenizer(char_level=params.char_level)
        tokenizer.fit_on_texts(texts=X_train)
        vocab_size = len(tokenizer)

        # Convert texts to sequences of indices
        X_train = np.array(tokenizer.texts_to_sequences(X_train))
        X_val = np.array(tokenizer.texts_to_sequences(X_val))
        X_test = np.array(tokenizer.texts_to_sequences(X_test))

        # Class weights
        counts = np.bincount([label_encoder.class_to_index[class_] for class_ in all_tags])
        class_weights = {i: 1.0/count for i, count in enumerate(counts)}

        # Create datasets
        train_dataset = CNNTextDataset(
            X=X_train, y=y_train, max_filter_size=max(params.filter_sizes))
        val_dataset = CNNTextDataset(
            X=X_val, y=y_val, max_filter_size=max(params.filter_sizes))
        test_dataset = CNNTextDataset(
            X=X_test, y=y_test, max_filter_size=max(params.filter_sizes))

        # Create dataloaders
        train_dataloader = train_dataset.create_dataloader(
            batch_size=params.batch_size)
        val_dataloader = val_dataset.create_dataloader(
            batch_size=params.batch_size)
        test_dataloader = test_dataset.create_dataloader(
            batch_size=params.batch_size)

        # Initialize model
        model = CNN(
            embedding_dim=params.embedding_dim, vocab_size=vocab_size,
            num_filters=params.num_filters, filter_sizes=params.filter_sizes,
            hidden_dim=params.hidden_dim, dropout_p=params.dropout_p,
            num_classes=num_classes)
        model = model.to(device)

        # Define loss
        class_weights_tensor = torch.Tensor(np.array(list(class_weights.values())))
        loss_fn = nn.BCEWithLogitsLoss(weight=class_weights_tensor)

        # Define optimizer & scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=5)

        # Trainer module
        trainer = Trainer(
            model=model, device=device, loss_fn=loss_fn,
            optimizer=optimizer, scheduler=scheduler, trial=trial)

        # Train
        best_model, best_val_loss = trainer.train(
            params.num_epochs, params.patience, train_dataloader, val_dataloader)

        # Best threshold for f1
        train_loss, y_true, y_prob = trainer.eval_step(dataloader=train_dataloader)
        precisions, recalls, thresholds = precision_recall_curve(y_true.ravel(), y_prob.ravel())
        threshold = find_best_threshold(y_true.ravel(), y_prob.ravel())

        # Determine predictions using threshold
        test_loss, y_true, y_prob = trainer.eval_step(dataloader=test_dataloader)
        y_pred = np.array([np.where(prob >= threshold, 1, 0) for prob in y_prob])

        # Evaluate
        performance = get_metrics(
            y_true=y_test, y_pred=y_pred, classes=label_encoder.classes)

        return {
            "params": params,
            "tokenizer": tokenizer,
            "label_encoder": label_encoder,
            "model": best_model,
            "performance": performance,
            "best_val_loss": best_val_loss,
            "threshold": threshold,
        }
    ```

## Objective

We need to define an `objective` function that will consume a trial and a set of arguments and produce the metric to optimize on (`f1` in our case).
```python linenums="1"
def objective(trial, params):
    """Objective function for optimization trials."""

    # Paramters (to tune)
    params.embedding_dim = trial.suggest_int("embedding_dim", 128, 512)
    params.num_filters = trial.suggest_int("num_filters", 128, 512)
    params.hidden_dim = trial.suggest_int("hidden_dim", 128, 512)
    params.dropout_p = trial.suggest_uniform("dropout_p", 0.3, 0.8)
    params.lr = trial.suggest_loguniform("lr", 5e-5, 5e-4)

    # Train & evaluate
    artifacts = train_cnn(params=params, df=df, trial=trial)

    # Set additional attributes
    trial.set_user_attr("precision", artifacts["performance"]["overall"]["precision"])
    trial.set_user_attr("recall", artifacts["performance"]["overall"]["recall"])
    trial.set_user_attr("f1", artifacts["performance"]["overall"]["f1"])
    trial.set_user_attr("threshold", artifacts["threshold"])

    return artifacts["performance"]["overall"]["f1"]
```

## Study

We're ready to kick off our study with our [MLFlowCallback](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.integration.MLflowCallback.html){:target="_blank"} so we can track all of the different trials.
```python linenums="1"
from optuna.integration.mlflow import MLflowCallback

# Optimize
NUM_TRIALS = 50 # small sample for now
pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
study = optuna.create_study(study_name="optimization", direction="maximize", pruner=pruner)
mlflow_callback = MLflowCallback(
    tracking_uri=mlflow.get_tracking_uri(), metric_name='f1')
study.optimize(lambda trial: objective(trial, params),
               n_trials=NUM_TRIALS,
               callbacks=[mlflow_callback])
```
<pre class="output">
A new study created in memory with name: optimization
Epoch: 1 | train_loss: 0.00645, val_loss: 0.00314, lr: 3.48E-04, _patience: 10
...
Epoch: 23 | train_loss: 0.00029, val_loss: 0.00175, lr: 3.48E-05, _patience: 1
Stopping early!
Trial 0 finished with value: 0.5999225606985846 and parameters: {'embedding_dim': 508, 'num_filters': 359, 'hidden_dim': 262, 'dropout_p': 0.6008497926241321, 'lr': 0.0003484755175747328}. Best is trial 0 with value: 0.5999225606985846.
INFO: 'optimization' does not exist. Creating a new experiment

...

Trial 10 pruned.

...

Epoch: 25 | train_loss: 0.00029, val_loss: 0.00156, lr: 2.73E-05, _patience: 2
Epoch: 26 | train_loss: 0.00028, val_loss: 0.00152, lr: 2.73E-05, _patience: 1
Stopping early!
Trial 49 finished with value: 0.6220047640997922 and parameters: {'embedding_dim': 485, 'num_filters': 420, 'hidden_dim': 477, 'dropout_p': 0.7984462152799114, 'lr': 0.0002619841505205434}. Best is trial 46 with value: 0.63900047716579.
</pre>
```python linenums="1"
# MLFlow dashboard
get_ipython().system_raw("mlflow server -h 0.0.0.0 -p 5000 --backend-store-uri $PWD/experiments/ &")
ngrok.kill()
ngrok.set_auth_token("")
ngrok_tunnel = ngrok.connect(addr="5000", proto="http", bind_tls=True)
print("MLflow Tracking UI:", ngrok_tunnel.public_url)
```
<pre class="output">
MLflow Tracking UI: https://d19689b7ba4e.ngrok.io
</pre>

You can compare all (or a subset) of the trials in our experiment.
<div class="ai-center-all">
    <img src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/hyperparameter_optimization/compare.png" width="1000" alt="compare experiments">
</div>

We can then view the results through various lens (contours, parallel coordinates, etc.)
<div class="ai-center-all">
    <img src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/hyperparameter_optimization/contour.png" width="1000" alt="contour plot">
</div>
<div class="ai-center-all">
    <img src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/hyperparameter_optimization/parallel_coordinates.png" width="1000" alt="parallel coordinates">
</div>

```python linenums="1"
# All trials
trials_df = study.trials_dataframe()
trials_df = trials_df.sort_values(["value"], ascending=False)  # sort by metric
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
      <th>params_dropout_p</th>
      <th>params_embedding_dim</th>
      <th>params_hidden_dim</th>
      <th>params_lr</th>
      <th>params_num_filters</th>
      <th>user_attrs_f1</th>
      <th>user_attrs_precision</th>
      <th>user_attrs_recall</th>
      <th>user_attrs_threshold</th>
      <th>state</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>46</th>
      <td>46</td>
      <td>0.639000</td>
      <td>2021-01-26 21:29:09.435991</td>
      <td>2021-01-26 21:30:20.637867</td>
      <td>0 days 00:01:11.201876</td>
      <td>0.670784</td>
      <td>335</td>
      <td>458</td>
      <td>0.000298</td>
      <td>477</td>
      <td>0.639000</td>
      <td>0.852947</td>
      <td>0.540094</td>
      <td>0.221352</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>32</th>
      <td>32</td>
      <td>0.638382</td>
      <td>2021-01-26 21:08:27.456865</td>
      <td>2021-01-26 21:09:54.151386</td>
      <td>0 days 00:01:26.694521</td>
      <td>0.485060</td>
      <td>322</td>
      <td>329</td>
      <td>0.000143</td>
      <td>458</td>
      <td>0.638382</td>
      <td>0.860706</td>
      <td>0.535624</td>
      <td>0.285308</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>33</th>
      <td>33</td>
      <td>0.638135</td>
      <td>2021-01-26 21:09:54.182560</td>
      <td>2021-01-26 21:11:14.038009</td>
      <td>0 days 00:01:19.855449</td>
      <td>0.567419</td>
      <td>323</td>
      <td>405</td>
      <td>0.000163</td>
      <td>482</td>
      <td>0.638135</td>
      <td>0.872309</td>
      <td>0.537566</td>
      <td>0.298093</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>39</th>
      <td>39</td>
      <td>0.637652</td>
      <td>2021-01-26 21:18:37.735567</td>
      <td>2021-01-26 21:20:01.271413</td>
      <td>0 days 00:01:23.535846</td>
      <td>0.689044</td>
      <td>391</td>
      <td>401</td>
      <td>0.000496</td>
      <td>512</td>
      <td>0.637652</td>
      <td>0.852757</td>
      <td>0.536279</td>
      <td>0.258009</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>34</th>
      <td>34</td>
      <td>0.634339</td>
      <td>2021-01-26 21:11:14.068099</td>
      <td>2021-01-26 21:12:33.645090</td>
      <td>0 days 00:01:19.576991</td>
      <td>0.592627</td>
      <td>371</td>
      <td>379</td>
      <td>0.000213</td>
      <td>486</td>
      <td>0.634339</td>
      <td>0.863092</td>
      <td>0.531822</td>
      <td>0.263524</td>
      <td>COMPLETE</td>
    </tr>
  </tbody>
</table>
</div></div>

```python linenums="1"
# Best trial
print (f"Best value (val loss): {study.best_trial.value}")
print (f"Best hyperparameters: {study.best_trial.params}")
```
<pre class="output">
Best value (f1): 0.63900047716579
Best hyperparameters: {'embedding_dim': 335, 'num_filters': 477, 'hidden_dim': 458, 'dropout_p': 0.6707843486583486, 'lr': 0.00029782100137454434}
</pre>

!!! note
    Don't forget to save learned parameters (ex. decision threshold) during training which you'll need later for inference.

```python linenums="1"
# Save best parameters
params = {**params.__dict__, **study.best_trial.params}
params["threshold"] = study.best_trial.user_attrs["threshold"]
print (json.dumps(params, indent=2, cls=NumpyEncoder))
```
<pre class="output">
{
  "char_level": true,
  "filter_sizes": [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10
  ],
  "batch_size": 64,
  "embedding_dim": 335,
  "num_filters": 477,
  "hidden_dim": 458,
  "dropout_p": 0.6707843486583486,
  "lr": 0.00029782100137454434,
  "num_epochs": 200,
  "patience": 10,
  "threshold": 0.22135180234909058
}
</pre>

... and now we're finally ready to move from working in Jupyter notebooks to Python scripts. We'll be revisiting everything we did so far, but this time with proper software engineering principles such as object oriented programming (OOPs), testing, styling, etc.

<!-- Citation -->
{% include "cite.md" %}
