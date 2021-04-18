---
template: lesson.html
title: Experiment Tracking
description: Managing and tracking ML experiments and runs.
keywords: experiment tracking, mlflow, weights and biases, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/applied_ml.png
repository: https://github.com/GokuMohandas/mlops
notebook: https://colab.research.google.com/github/GokuMohandas/mlops/blob/main/notebooks/tagifai.ipynb
---

{% include "styles/lesson.md" %}

## Intuition
So far, we've been training and evaluating our different baselines but haven't really been tracking these experiments. We'll fix this but defining a proper process for experiment tracking which we'll use for all future experiments (including hyperparameter optimization). Experiment tracking is the processing of managing all the different experiments and their components, such as parameters, metrics, models and other artifacts and it enables us to:

- *Organize* all the necessary components of a specific experiment. It's important to have everything in one place and know where it is so you can use them later.
- *Reproduce* past results (easily) using saved experiments.
- *Log* iterative improvements across time, data, ideas, teams, etc.

## Tools
There are many options for experiment tracking but we're going to use [MLFlow](https://mlflow.org/){:target="_blank"} (100% free and [open-source](https://github.com/mlflow/mlflow){:target="_blank"}) because it has all the functionality we'll need (and [growing integration support](https://medium.com/pytorch/mlflow-and-pytorch-where-cutting-edge-ai-meets-mlops-1985cf8aa789){:target="_blank"}). You can run MLFlow on your own servers and databases so there are no storage cost / limitations, making it one of the most popular options and is used by Microsoft, Facebook, Databricks and others. You can also set up your own Tracking servers to synchronize runs amongst multiple team members collaborating on the same task.

There are also several popular options such as a [Comet ML](https://www.comet.ml/site/){:target="_blank"} (Used by Google AI, HuggingFace, etc.) and [Weights and Biases](https://www.wandb.com/){:target="_blank"} (Used by Open AI, Toyota Research, etc.). These are fantastic tools that provide features like dashboards, seamless integration, hyperparameter search, reports and even [debugging](https://wandb.ai/latentspace/published-work/The-Science-of-Debugging-with-W-B-Reports--Vmlldzo4OTI3Ng){:target="_blank"}!

!!! note
    Many platforms are leveraging their position as the source for experiment data to provide features that extend into other parts of the ML development pipeline such as versioning, debugging, monitoring, etc.

## Application

We'll start by initializing all the required arguments for our experiment.
```python linenums="1"
from argparse import Namespace
import mlflow
from pathlib import Path
```
```python linenums="1"
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

!!! note
    When we move to Python scripts, we'll use the [Typer](https://typer.tiangolo.com/){:target="_blank"} package instead of argparse for a better CLI experience.

Next, we'll set up our model registry where all the experiments and their respective runs will be stored. We'll load trained models from this registry as well using specific run IDs.
```python linenums="1"
# Model registry
MODEL_REGISTRY = Path(STORES_DIR, "model")
MODEL_REGISTRY.mkdir(parents=True, exist_ok=True)
mlflow.set_tracking_uri("file://" + str(MODEL_REGISTRY.absolute()))
```

!!! note
    When we're collaborating with other team members, this model registry will live on the cloud with some added authentication. Members from our team can connect to it (like above) to save and load trained models. If you don't want to set up and maintain a model registry, this is where platforms like [Comet ML](https://www.comet.ml/site/){:target="_blank"}, [Weights and Biases](https://www.wandb.com/){:target="_blank"} and others offload a lot of technical components.

## Training

Next, we're going to modify our `Trainer` object so that we can log the metrics from each epoch. The only addition are these lines:
```python linenums="1"
class Trainer(object):
    ...
    def train(self, ...):
        ...
        # Tracking
        mlflow.log_metrics(
            {"train_loss": train_loss, "val_loss": val_loss}, step=epoch
        )
        ...
```

??? quote "Code for complete `Trainer` class"
    ```python linenums="1" hl_lines="100-103"
    # Trainer (modified for experiment tracking)
    class Trainer(object):
        def __init__(self, model, device, loss_fn=None,
                    optimizer=None, scheduler=None):

            # Set params
            self.model = model
            self.device = device
            self.loss_fn = loss_fn
            self.optimizer = optimizer
            self.scheduler = scheduler

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

                # Tracking
                mlflow.log_metrics(
                    {"train_loss": train_loss, "val_loss": val_loss}, step=epoch
                )

                # Logging
                print(
                    f"Epoch: {epoch+1} | "
                    f"train_loss: {train_loss:.5f}, "
                    f"val_loss: {val_loss:.5f}, "
                    f"lr: {self.optimizer.param_groups[0]['lr']:.2E}, "
                    f"_patience: {_patience}"
                )

            return best_model, best_val_loss
    ```

And to make things simple, we'll encapsulate all the components for training into one function called `train_cnn` which returns all the artifacts we want to be able to track from our experiment.
```python linenums="1"
def train_cnn(params, df):
    """Train a CNN using specific arguments."""
    ...
    return {
        "params": params,
        "tokenizer": tokenizer,
        "label_encoder": label_encoder,
        "model": best_model,
        "performance": performance,
        "best_val_loss": best_val_loss,
    }
```

The input argument `params`contains all the parameters needed and it's nice to have it all organized under one variable so we can easily log it and tweak it for different experiments (we'll see this when we do hyperparameter optimization).

??? quote "Code for complete `train_cnn` function"
    ```python linenums="1"
    def train_cnn(params, df):
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
            optimizer=optimizer, scheduler=scheduler)

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
        }
    ```

## Tracking
With MLFlow we need to first initialize an experiment and then you can do runs under that experiment.

```python linenums="1"
import tempfile
```
```python linenums="1"
# Set experiment
mlflow.set_experiment(experiment_name="baselines")
```
<pre class="output">
INFO: 'baselines' does not exist. Creating a new experiment
</pre>
```python linenums="1"
def save_dict(d, filepath, cls=None, sortkeys=False):
    with open(filepath, "w") as fp:
        json.dump(d, indent=2, fp=fp, cls=cls, sort_keys=sortkeys)
```
```python linenums="1"
# Tracking
with mlflow.start_run(run_name="cnn") as run:

    # Train & evaluate
    artifacts = train_cnn(params=params, df=df)

    # Log key metrics
    mlflow.log_metrics({"precision": artifacts["performance"]["overall"]["precision"]})
    mlflow.log_metrics({"recall": artifacts["performance"]["overall"]["recall"]})
    mlflow.log_metrics({"f1": artifacts["performance"]["overall"]["f1"]})

    # Log artifacts
    with tempfile.TemporaryDirectory() as dp:
        save_dict(vars(artifacts["params"]), Path(dp, "params.json"))
        save_dict(performance, Path(dp, "performance.json"))
        artifacts["tokenizer"].save(Path(dp, "tokenizer.json"))
        artifacts["label_encoder"].save(Path(dp, "label_encoder.json"))
        torch.save(artifacts["model"].state_dict(), Path(dp, "model.pt"))
        mlflow.log_artifacts(dp)

    # Log parameters
    mlflow.log_params(vars(artifacts["params"]))
```
<pre class="output">
Epoch: 1 | train_loss: 0.00539, val_loss: 0.00301, lr: 2.00E-04, _patience: 10
Epoch: 2 | train_loss: 0.00393, val_loss: 0.00281, lr: 2.00E-04, _patience: 10
Epoch: 3 | train_loss: 0.00345, val_loss: 0.00264, lr: 2.00E-04, _patience: 10
Epoch: 4 | train_loss: 0.00324, val_loss: 0.00259, lr: 2.00E-04, _patience: 10
...
Epoch: 41 | train_loss: 0.00076, val_loss: 0.00158, lr: 2.00E-05, _patience: 4
Epoch: 42 | train_loss: 0.00070, val_loss: 0.00149, lr: 2.00E-05, _patience: 3
Epoch: 43 | train_loss: 0.00068, val_loss: 0.00153, lr: 2.00E-05, _patience: 2
Epoch: 44 | train_loss: 0.00067, val_loss: 0.00149, lr: 2.00E-05, _patience: 1
Stopping early!
</pre>

## Viewing
Let's view what we've tracked from our experiment. MLFlow serves a dashboard for us to view and explore our experiments on a localhost port but since we're inside a notebook, we're going to use public tunnel ([ngrok](https://ngrok.com/){:target="_blank"}) to view it.

```python linenums="1"
from pyngrok import ngrok
```

!!! note
    You may need to rerun the cell below multiple times if the connection times out it is overloaded.

```python linenums="1"
# https://stackoverflow.com/questions/61615818/setting-up-mlflow-on-google-colab
get_ipython().system_raw("mlflow server -h 0.0.0.0 -p 5000 --backend-store-uri $PWD/mlruns/ &")
ngrok.kill()
ngrok.set_auth_token("")
ngrok_tunnel = ngrok.connect(addr="5000", proto="http", bind_tls=True)
print("MLflow Tracking UI:", ngrok_tunnel.public_url)
```
<pre class="output">
MLflow Tracking UI: https://476803694c2e.ngrok.io
</pre>

MLFlow creates a main dashboard with all your experiments and their respective runs. You can sort runs by clicking on the column headers.
<div class="ai-center-all">
    <img src="https://raw.githubusercontent.com/GokuMohandas/madewithml/main/images/mlops/experiment_tracking/dashboard.png" style="width: 100rem;" alt="mlflow dashboard">
</div>
We can click on any of our experiments on the main dashboard to further explore it:
<div class="row">
  <div class="col-6">
    <div class="ai-center-all">
      <img src="https://raw.githubusercontent.com/GokuMohandas/madewithml/main/images/mlops/experiment_tracking/parameters.png" width="400" alt="mlflow dashboard">
    </div>
  </div>
  <div class="col-6">
    <div class="ai-center-all">
      <img src="https://raw.githubusercontent.com/GokuMohandas/madewithml/main/images/mlops/experiment_tracking/metrics.png" width="400" alt="mlflow dashboard">
    </div>
  </div>
</div>
<div class="ai-center-all">
    <img src="https://raw.githubusercontent.com/GokuMohandas/madewithml/main/images/mlops/experiment_tracking/plots.png" width="1000" alt="mlflow dashboard">
</div>

## Loading

We need to be able to load our saved experiment artifacts for inference, retraining, etc.
```python linenums="1"
def load_dict(filepath):
    """Load a dict from a json file."""
    with open(filepath, "r") as fp:
        d = json.load(fp)
    return d
```
```python linenums="1"
# Load components
client = mlflow.tracking.MlflowClient()
experiment_id = mlflow.get_experiment_by_name("baselines").experiment_id
all_runs = mlflow.search_runs(experiment_ids=experiment_id, order_by=["metrics.f1 DESC"])
print (all_runs)
```
<pre class="output">
                             run_id  ... tags.mlflow.runName
0  db54fa3bc6b945f7b7f814843551df36  ...                 cnn

[1 rows x 25 columns]
</pre>
```python linenums="1"
# Best run
device = torch.device("cpu")
best_run_id = all_runs.iloc[0].run_id
best_run = mlflow.get_run(run_id=best_run_id)
with tempfile.TemporaryDirectory() as dp:
    client.download_artifacts(run_id=best_run_id, path="", dst_path=dp)
    tokenizer = Tokenizer.load(fp=Path(dp, "tokenizer.json"))
    label_encoder = LabelEncoder.load(fp=Path(dp, "label_encoder.json"))
    model_state = torch.load(Path(dp, "model.pt"), map_location=device)
    performance = load_dict(filepath=Path(dp, "performance.json"))
```

!!! note
    We can also load a specific run's model artifacts, by using it's run ID, directly from the model registry without having to save them to a temporary directory.
    ```python linenums="1"
    artifact_uri = mlflow.get_run(run_id=run_id).info.artifact_uri.split("file://")[-1]
    params = Namespace(**utils.load_dict(filepath=Path(artifact_uri, "params.json")))
    label_encoder = data.MultiLabelLabelEncoder.load(fp=Path(artifact_uri, "label_encoder.json"))
    tokenizer = data.Tokenizer.load(fp=Path(artifact_uri, "tokenizer.json"))
    model_state = torch.load(Path(artifact_uri, "model.pt"), map_location=device)
    performance = utils.load_dict(filepath=Path(artifact_uri, "performance.json"))
    ```

```python linenums="1"
print (json.dumps(performance["overall"], indent=2))
```
<pre class="output">
{
  "precision": 0.8332201275597738,
  "recall": 0.5110345795419687,
  "f1": 0.6072536294437475,
  "num_samples": 480.0
}
</pre>
```python linenums="1"
# Load artifacts
device = torch.device("cpu")
model = CNN(
    embedding_dim=params.embedding_dim, vocab_size=len(tokenizer),
    num_filters=params.num_filters, filter_sizes=params.filter_sizes,
    hidden_dim=params.hidden_dim, dropout_p=params.dropout_p,
    num_classes=len(label_encoder))
model.load_state_dict(model_state)
model.to(device)
```
<pre class="output">
CNN(
  (embeddings): Embedding(39, 128, padding_idx=0)
  (conv): ModuleList(
    (0): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
    (1): Conv1d(128, 128, kernel_size=(2,), stride=(1,))
    (2): Conv1d(128, 128, kernel_size=(3,), stride=(1,))
    (3): Conv1d(128, 128, kernel_size=(4,), stride=(1,))
    (4): Conv1d(128, 128, kernel_size=(5,), stride=(1,))
    (5): Conv1d(128, 128, kernel_size=(6,), stride=(1,))
    (6): Conv1d(128, 128, kernel_size=(7,), stride=(1,))
    (7): Conv1d(128, 128, kernel_size=(8,), stride=(1,))
    (8): Conv1d(128, 128, kernel_size=(9,), stride=(1,))
    (9): Conv1d(128, 128, kernel_size=(10,), stride=(1,))
  )
  (dropout): Dropout(p=0.5, inplace=False)
  (fc1): Linear(in_features=1280, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=35, bias=True)
)
</pre>
```python linenums="1"
# Initialize trainer
trainer = Trainer(model=model, device=device)
```
```python linenums="1"
# Dataloader
text = "Transfer learning with BERT for self-supervised learning"
X = np.array(tokenizer.texts_to_sequences([preprocess(text)]))
y_filler = label_encoder.encode([np.array([label_encoder.classes[0]]*len(X))])
dataset = CNNTextDataset(
    X=X, y=y_filler, max_filter_size=max(filter_sizes))
dataloader = dataset.create_dataloader(
    batch_size=batch_size)
```
```python linenums="1"
# Inference
y_prob = trainer.predict_step(dataloader)
y_pred = np.array([np.where(prob >= threshold, 1, 0) for prob in y_prob])
label_encoder.decode(y_pred)
```
<pre class="output">
[['natural-language-processing',
  'self-supervised-learning',
  'transfer-learning',
  'transformers']]
</pre>


<!-- Citation -->
{% include "cite.md" %}