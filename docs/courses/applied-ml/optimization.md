---
description: Optimizing a subset of hyperparameters to achieve an objective.
image: https://madewithml.com/static/images/applied_ml.png
---

:octicons-mark-github-16: [Repository](https://github.com/GokuMohandas/applied-ml){:target="_blank"} · :octicons-book-24: [Notebook](https://colab.research.google.com/github/GokuMohandas/applied-ml/blob/main/notebooks/tagifai.ipynb){:target="_blank"}

Optimizing a subset of hyperparameters to achieve an objective.

## Intuition

### What is it?
Optimization is the process of fine-tuning the hyperparameters in our experiment to optimize towards a particular objective. It can be a computationally involved process depending on the number of parameters, search space and model architectures. Hyperparameters don't just include the model's parameters but they also include parameters (choices) from preprocessing, splitting, etc. When we look at all the different parameters that can be tuned, it quickly becomes a very large search space. However, just because something is a hyperparameter doesn't mean we need to tune it.

- It's absolutely alright to fix some hyperparameters (ex. `lower=True` during preprocessing) and remove them from the current tuning subset. Just be sure to note which parameters you are fixing and your reasoning for doing so.
- You can initially just tune a small, yet influential, subset of hyperparameters that you believe will yield best results.

### Why do we need it?
We want to optimize our hyperparameters so that we can understand how each of them affects our objective. By running many trials across a reasonable search space, we can determine near ideal values for our different parameters. It's also a great opportunity to determine if a smaller parameters yield similar performances as larger ones (efficiency).

### How can we do it?
There are many options for hyperparameter tuning ([Optuna](https://github.com/optuna/optuna){:target="_blank"}, [Ray tune](https://github.com/ray-project/ray/tree/master/python/ray/tune){:target="_blank"}, [Hyperopt](https://github.com/hyperopt/hyperopt){:target="_blank"}, etc.). We'll be using Optuna for it's simplicity, popularity and efficiency though they are all equally so. It really comes down to familiarity and whether a library has a specific implementation readily tested and available.

## Application

There are many factors to consider when performing hyperparameter optimization and luckily Optuna allows us to [implement](https://optuna.readthedocs.io/en/stable/reference/) them with ease. We'll be conducting a small study where we'll tune a set of arguments (we'll do a much more thorough [study](https://optuna.readthedocs.io/en/stable/reference/study.html) of the parameter space when we move our code to Python scripts). Here's the process for the study:

1. Define an objective (metric) and identifying the [direction](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.StudyDirection.html#optuna.study.StudyDirection){:target="_blank"} to optimize.
2. `[OPTIONAL]` Choose a [sampler](https://optuna.readthedocs.io/en/stable/reference/samplers.html){:target="_blank"} for determining parameters for subsequent trials. (default is a tree based sampler).
3. `[OPTIONAL]` Choose a [pruner](https://optuna.readthedocs.io/en/stable/reference/pruners.html){:target="_blank"} to end unpromising trials early.
4. Define the parameters to tune in each [trial](https://optuna.readthedocs.io/en/stable/reference/trial.html){:target="_blank"} and the [distribution](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna-trial-trial){:target="_blank"} of values to sample.

!!! note
    There are many more options (multiple objectives, storage options, etc.) to explore but this basic set up will allow us to optimize quite well.

```python
from argparse import Namespace
from numpyencoder import NumpyEncoder
```
```python
# Specify arguments
args = Namespace(
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
```python
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
```python
def train_cnn(args, df, trial=None):
    ...
    # Trainer module
    trainer = Trainer(
        model=model, device=device, loss_fn=loss_fn,
        optimizer=optimizer, scheduler=scheduler, trial=trial)
    ...
```

??? quote "Code for complete `train_cnn` function"
    ```python linenums="1" hl_lines="69-72"
    def train_cnn(args, df, trial=None):
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
        tokenizer = Tokenizer(char_level=args.char_level)
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
            X=X_train, y=y_train, max_filter_size=max(args.filter_sizes))
        val_dataset = CNNTextDataset(
            X=X_val, y=y_val, max_filter_size=max(args.filter_sizes))
        test_dataset = CNNTextDataset(
            X=X_test, y=y_test, max_filter_size=max(args.filter_sizes))

        # Create dataloaders
        train_dataloader = train_dataset.create_dataloader(
            batch_size=args.batch_size)
        val_dataloader = val_dataset.create_dataloader(
            batch_size=args.batch_size)
        test_dataloader = test_dataset.create_dataloader(
            batch_size=args.batch_size)

        # Initialize model
        model = CNN(
            embedding_dim=args.embedding_dim, vocab_size=vocab_size,
            num_filters=args.num_filters, filter_sizes=args.filter_sizes,
            hidden_dim=args.hidden_dim, dropout_p=args.dropout_p,
            num_classes=num_classes)
        model = model.to(device)

        # Define loss
        class_weights_tensor = torch.Tensor(np.array(list(class_weights.values())))
        loss_fn = nn.BCEWithLogitsLoss(weight=class_weights_tensor)

        # Define optimizer & scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=5)

        # Trainer module
        trainer = Trainer(
            model=model, device=device, loss_fn=loss_fn,
            optimizer=optimizer, scheduler=scheduler, trial=trial)

        # Train
        best_model, best_val_loss = trainer.train(
            args.num_epochs, args.patience, train_dataloader, val_dataloader)

        # Best threshold for f1
        train_loss, y_true, y_prob = trainer.eval_step(dataloader=train_dataloader)
        precisions, recalls, thresholds = precision_recall_curve(y_true.ravel(), y_prob.ravel())
        threshold = find_best_threshold(y_true.ravel(), y_prob.ravel())

        # Determine predictions using threshold
        test_loss, y_true, y_prob = trainer.eval_step(dataloader=test_dataloader)
        y_pred = np.array([np.where(prob >= threshold, 1, 0) for prob in y_prob])

        # Evaluate
        performance = get_performance(
            y_true=y_test, y_pred=y_pred, classes=label_encoder.classes)

        return {
            "args": args,
            "tokenizer": tokenizer,
            "label_encoder": label_encoder,
            "model": best_model,
            "performance": performance,
            "best_val_loss": best_val_loss,
            "threshold": threshold,
        }
    ```

### Objective

We need to define an `objective` function that will consume a trial and a set of arguments and produce the metric to optimize on (`best_val_loss` in our case).
```python
def objective(trial, args):
    """Objective function for optimization trials."""

    # Paramters (to tune)
    args.embedding_dim = trial.suggest_int("embedding_dim", 100, 300)
    args.num_filters = trial.suggest_int("num_filters", 100, 300)
    args.hidden_dim = trial.suggest_int("hidden_dim", 128, 256)
    args.dropout_p = trial.suggest_uniform("dropout_p", 0.0, 0.8)
    args.lr = trial.suggest_loguniform("lr", 5e-5, 5e-4)

    # Train & evaluate
    artifacts = train_cnn(args=args, df=df, trial=trial)

    # Set additional attributes
    trial.set_user_attr("precision", artifacts["performance"]["overall"]["precision"])
    trial.set_user_attr("recall", artifacts["performance"]["overall"]["recall"])
    trial.set_user_attr("f1", artifacts["performance"]["overall"]["f1"])
    trial.set_user_attr("threshold", artifacts["threshold"])

    return artifacts["best_val_loss"]
```

### Study

We're ready to kick off our study with our [MLFlowCallback](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.integration.MLflowCallback.html){:target="_blank"} so we can track all of the different trials.
```python
from optuna.integration.mlflow import MLflowCallback
```
```python
NUM_TRIALS = 50 # small sample for now
```
```python
# Optimize
pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
study = optuna.create_study(study_name="optimization", direction="minimize", pruner=pruner)
mlflow_callback = MLflowCallback(
    tracking_uri=mlflow.get_tracking_uri(), metric_name='val_loss')
study.optimize(lambda trial: objective(trial, args),
               n_trials=NUM_TRIALS,
               callbacks=[mlflow_callback])
```
<pre class="output">
[I 2021-01-05 22:24:25,247] A new study created in memory with name: optimization
Epoch: 1 | train_loss: 0.00611, val_loss: 0.00277, lr: 8.10E-05, _patience: 10
...
Epoch: 71 | train_loss: 0.00059, val_loss: 0.00146, lr: 8.10E-07, _patience: 1
Stopping early!
[I 2021-01-05 22:25:43,895] Trial 0 finished with value: 0.0014537220413330942 and parameters: {'embedding_dim': 183, 'num_filters': 250, 'hidden_dim': 163, 'dropout_p': 0.4361347102554463, 'lr': 8.102678985973712e-05}. Best is trial 0 with value: 0.0014537220413330942.
INFO: 'optimization' does not exist. Creating a new experiment

...

[I 2021-01-05 22:31:47,722] Trial 10 pruned.

...

Epoch: 25 | train_loss: 0.00029, val_loss: 0.00156, lr: 2.73E-05, _patience: 2
Epoch: 26 | train_loss: 0.00028, val_loss: 0.00152, lr: 2.73E-05, _patience: 1
Stopping early!
[I 2021-01-05 22:43:38,529] Trial 47 finished with value: 0.0014839080977253616 and parameters: {'embedding_dim': 250, 'num_filters': 268, 'hidden_dim': 189, 'dropout_p': 0.33611557767537004, 'lr': 0.000273415896212767}. Best is trial 28 with value: 0.0013889532128814608.

...

Epoch: 5 | train_loss: 0.00286, val_loss: 0.00236, lr: 3.12E-04, _patience: 10
Epoch: 6 | train_loss: 0.00263, val_loss: 0.00224, lr: 3.12E-04, _patience: 10
[I 2021-01-05 22:43:58,699] Trial 49 pruned.
</pre>
```python
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
    <img src="https://raw.githubusercontent.com/GokuMohandas/madewithml/main/images/applied-ml/hyperparameter_optimization/compare.png" width="1000" alt="pivot">
</div>

We can then view the results through various lens (contours, parallel coordinates, etc.)
<div class="ai-center-all">
    <img src="https://raw.githubusercontent.com/GokuMohandas/madewithml/main/images/applied-ml/hyperparameter_optimization/contour.png" width="1000" alt="pivot">
</div>
<div class="ai-center-all">
    <img src="https://raw.githubusercontent.com/GokuMohandas/madewithml/main/images/applied-ml/hyperparameter_optimization/parallel_coordinates.png" width="1000" alt="pivot">
</div>

```python
# All trials
trials_df = study.trials_dataframe()
trials_df = trials_df.sort_values(["value"], ascending=True)  # sort by metric
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
      <th>28</th>
      <td>28</td>
      <td>0.001389</td>
      <td>2021-01-05 22:38:22.896834</td>
      <td>2021-01-05 22:38:54.515581</td>
      <td>0 days 00:00:31.618747</td>
      <td>0.738263</td>
      <td>230</td>
      <td>216</td>
      <td>0.000500</td>
      <td>238</td>
      <td>0.611860</td>
      <td>0.863897</td>
      <td>0.514516</td>
      <td>0.221920</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>0.001408</td>
      <td>2021-01-05 22:31:47.755110</td>
      <td>2021-01-05 22:32:54.725288</td>
      <td>0 days 00:01:06.970178</td>
      <td>0.572462</td>
      <td>296</td>
      <td>216</td>
      <td>0.000255</td>
      <td>291</td>
      <td>0.608866</td>
      <td>0.872787</td>
      <td>0.497144</td>
      <td>0.328791</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>18</th>
      <td>18</td>
      <td>0.001410</td>
      <td>2021-01-05 22:35:23.967917</td>
      <td>2021-01-05 22:36:04.190109</td>
      <td>0 days 00:00:40.222192</td>
      <td>0.460376</td>
      <td>228</td>
      <td>222</td>
      <td>0.000316</td>
      <td>221</td>
      <td>0.599672</td>
      <td>0.847859</td>
      <td>0.500275</td>
      <td>0.315493</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>22</th>
      <td>22</td>
      <td>0.001426</td>
      <td>2021-01-05 22:36:52.953326</td>
      <td>2021-01-05 22:37:26.792074</td>
      <td>0 days 00:00:33.838748</td>
      <td>0.498631</td>
      <td>205</td>
      <td>208</td>
      <td>0.000288</td>
      <td>246</td>
      <td>0.628976</td>
      <td>0.854277</td>
      <td>0.525414</td>
      <td>0.281477</td>
      <td>COMPLETE</td>
    </tr>
    <tr>
      <th>23</th>
      <td>23</td>
      <td>0.001431</td>
      <td>2021-01-05 22:37:26.822999</td>
      <td>2021-01-05 22:37:56.216911</td>
      <td>0 days 00:00:29.393912</td>
      <td>0.520217</td>
      <td>203</td>
      <td>207</td>
      <td>0.000469</td>
      <td>224</td>
      <td>0.631419</td>
      <td>0.860776</td>
      <td>0.526648</td>
      <td>0.281263</td>
      <td>COMPLETE</td>
    </tr>
  </tbody>
</table>
</div></div>

```python
# Best trial
print (f"Best value (val loss): {study.best_trial.value}")
print (f"Best hyperparameters: {study.best_trial.params}")
```
<pre class="output">
Best value (val loss): 0.0013889532128814608
Best hyperparameters: {'embedding_dim': 230, 'num_filters': 238, 'hidden_dim': 216, 'dropout_p': 0.7382628825813314, 'lr': 0.0004998249007788683}
</pre>

!!! note
    Don't forget to save learned parameters (ex. decision threshold) during training which you'll need later for inference.

```python
# Save best parameters
params = {**args.__dict__, **study.best_trial.params}
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
  "embedding_dim": 230,
  "num_filters": 238,
  "hidden_dim": 216,
  "dropout_p": 0.7382628825813314,
  "lr": 0.0004998249007788683,
  "num_epochs": 200,
  "patience": 10,
  "threshold": 0.22192028164863586
}
</pre>

... and now we're finally ready to move from working in Jupyter notebooks to Python scripts. We'll be revisiting everything we did so far, but this time with proper software engineering principles such as object oriented programming (OOPs), testing, styling, etc.

<!--
```python

```
<pre class="output">

</pre>

<div class="ai-center-all">
    <img src="https://raw.githubusercontent.com/GokuMohandas/madewithml/main/images/applied-ml/solution/suggested_tags.png" width="550" alt="pivot">
</div>
<div class="ai-center-all">
  <small>UX of our hypothetical solution</small>
</div>

{:target="_blank"}
 -->
