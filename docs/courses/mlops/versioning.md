---
template: lesson.html
title: "Versioning Code, Data and Models"
description: Versioning code, data and models to ensure reproducible behavior in ML systems.
keywords: versioning, dvc, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning, great expectations
image: https://madewithml.com/static/images/mlops.png
repository: https://github.com/GokuMohandas/follow/tree/versioning
---

{% include "styles/lesson.md" %}

## Intuition

We learned how to version our code but there are several other very important class of artifacts that we need track and version: config, data and models. It's important that we version everything so that we can reproduce the exact same application anytime. And we're going to do this by using a Git commit as a snapshot of the code, config, data used to produce a specific model. Here are the key elements we'll need to incorporate to make our application entirely reproducible:

- repository should store pointers to large data and model artifacts living in blob storage.
- use commits to store snapshots of the code, config, data and model and be able to update and rollback versions.
- expose configurations so we can see and compare parameters.

<div class="ai-center-all">
    <img width="700" src="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/images/mlops/versioning/versioning.png">
</div>

## Application

There are many tools available for versioning our artifacts ([GitLFS](https://git-lfs.github.com/){:target="_blank"}, [Dolt](https://github.com/dolthub/dolt){:target="_blank"}, [Pachyderm](https://www.pachyderm.com/){:target="_blank"}, etc.) but we'll be using the [Data Version Control (DVC)](https://dvc.org/){:target="_blank"} library for it's simplicity, rich features and most importantly modularity. DVC has lots of other useful [features](https://dvc.org/features){:target="_blank"} (metrics, experiments, etc.) so be sure to explore those as well.

We'll be using DVC to version our datasets and model weights and store them in a local directory which will act as our blob storage. We could use remote blob storage options such as S3, GCP, Google Drive, [DAGsHub](https://dagshub.com/){:target="_blank"}, etc. but we're going to replicate the same actions locally so we can see how the data is stored.

> We'll be using a local directory to act as our blob storage so we can develop and analyze everything locally. We'll continue to do this for other storage components as well such as feature stores and like we have been doing with our local model registry.

## Set up
Let's start by installing DVC and initializing it to create a [.dvc](https://github.com/GokuMohandas/MLOps/tree/main/.dvc){:target="_blank"} directory.
```bash
# Initialization
pip install dvc
pip uninstall dataclasses (Python < 3.8)
dvc init
```

## Remote storage
After initializing DVC, we can establish where our remote storage will be. We be creating and using the `stores/blob` directory which won't be checked into our remote repository.

```python linenums="1" hl_lines="3"
# Local stores
MODEL_REGISTRY = Path(STORES_DIR, "model")
BLOB_STORE = Path(STORES_DIR, "blob")
```

```bash
# Add remote storage
dvc remote add -d storage stores/blob
```
<pre class="output">
Setting 'storage' as a default remote.
</pre>

!!! note
    We can also use remote blob storage options such as S3, GCP, Google Drive, [DAGsHub](https://dagshub.com/){:target="_blank"}, etc. if we're collaborating with other developers. For example, here's how we would set up an S3 bucket to hold our versioned data:
    ```bash
    # Create bucket: https://docs.aws.amazon.com/AmazonS3/latest/userguide/creating-bucket.html
    # Add credentials: https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html
    dvc remote modify storage access_key_id ${AWS_ACCESS_KEY_ID}
    dvc remote modify storage secret_access_key ${AWS_SECRET_ACCESS_KEY}
    dvc remote add -d storage s3://<BUCKET_NAME>
    ```

## Add data
Now we're ready to *add* our data which will create text pointer files for each file.

```bash
# Add artifacts
dvc add data/projects.json
dvc add data/tags.json
dvc add data/features.json
```

```bash
# Pointer files added
📂 data
  📄 .gitignore
  📄 features.json
  📄 features.json.dvc
  📄 projects.json
  📄 projects.json.dvc
  📄 tags.json
  📄 tags.json.dvc
```

Each pointer file will contain the md5 hash, size and the location w.r.t to the directory which we'll be checking into our git repository.

```yaml linenums="1"
# data/projects.json.dvc
outs:
- md5: dafec16f20e07c58af2ab05efe6818ce
  size: 764016
  path: projects.json
```

The data directory containing the files will also have a [.gitignore](https://github.com/GokuMohandas/MLOps/blob/main/data/.gitignore){:target="_blank"} file that includes the actual artifacts so we don't check them into our repository.

```yaml linenums="1"
# data/.gitignore
/projects.json
/tags.json
/features.json
```

> In order for the [.gitignore](https://github.com/GokuMohandas/MLOps/blob/main/data/.gitignore){:target="_blank"} file to be generated inside the data directory, we have to remove it from our root's [.gitignore](https://github.com/GokuMohandas/MLOps/blob/main/.gitignore){:target="_blank"} file. We placed it for earlier branches so that we weren't pushing our data to git but now that we're using DVC we don't have to worry about that.

!!! note
    In terms of versioning our model artifacts, we aren't pushing anything to our blob storage because our model registry already takes care of all that. Instead we expose the run ID, parameters and performance inside the [config directory](https://github.com/GokuMohandas/MLOps/blob/main/config){:target="_blank"} so we can easily view results and compare them with other local runs.

    ```bash
    # Model artifacts
    📂 config
      📄 run_id.txt
      📄 params.json
      📄 performance.json
    ```

    For very large applications or in the case of multiple models in production, these artifacts would be stored in a metadata or evaluation store where they'll be indexed by model run IDs.

## Push
Now we're ready to push our artifacts to our blob store with the *push* command.
```bash
# Push to remote storage
dvc push
```

If we inspect our storage (stores/blob), we'll can see that the data is efficiently stored.
```bash
# Remote storage
📂 stores
  📂 blob
    📂 3e
      📄 173e183b81085ff2d2dc3f137020ba
    📂 72
      📄 2d428f0e7add4b359d287ec15d54ec
    ...
```

!!! note
    In case we forget to add or push our artifacts, we can add it as a pre-commit hook so it happens automatically when we try to commit. If there are no changes to our versioned files, nothing will happen.

    ```yaml linenums="1" hl_lines="9"
    # Makefile
    .PHONY: dvc
    dvc:
        dvc add data/projects.json
        dvc add data/tags.json
        dvc add data/features.json
        dvc push
    ```

    ```yaml linenums="1"
    # Pre-commit hook
    - repo: local
      hooks:
        - id: dvc
          name: dvc
          entry: make
          args: ["dvc"]
          language: system
          pass_filenames: false
    ```

## Pull

When someone else wants to pull updated artifacts or vice verse, we can use the *pull* command to fetch from our remote storage to our local artifact directories. All we need is to first ensure that we have the latest pointer text files (via git pull).

```bash
# Pull from remote storage
dvc pull
```

> We can quickly test this by deleting our data files (the `.json` files not the `.dvc` pointers) and run `#!bash dvc pull` to load the files from our blob store.

## Operations

When we pull data from source or compute features, should they save the data itself or just the operations?

- **Version the data**
    - But what happens as data becomes larger and larger and you keep making copies of it.
    - This is okay if the data is manageable, if your team is small/early stage ML or if changes to the data are infrequent.
    - But what happens when the underlying data changes (labels are fixed, etc.)? Now the same operations result is different data and reproducibility is not possible.
- **Version the operations**
    - We could keep snapshots of the data and provided the operations and timestamp, we can execute operations on those snapshots of the data. Many data systems use [time-travel](https://docs.snowflake.com/en/user-guide/data-time-travel.html){:target="blank"} to achieve this efficiently.
    - But eventually this also results in data storage bulk. What we need is an append-only data source where all changes are kept in a log instead of directly changing the data itself. So we can use the data system with the logs to deterministically produce versions of the data as they were without having to store the data itself!

<!-- Citation -->
{% include "cite.md" %}