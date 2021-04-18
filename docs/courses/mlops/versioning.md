---
template: lesson.html
title: "Versioning Code, Data and Models"
description: Versioning code, data and models to ensure reproducible behavior in ML systems.
keywords: versioning, dvc, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning, great expectations
image: https://madewithml.com/static/images/applied_ml.png
repository: https://github.com/GokuMohandas/mlops
---

{% include "styles/lesson.md" %}

## Intuition

We learned how to version our code but there are several other very important class of artifacts that we need track and version: config, data and models. It's important that we version everything so that we can reproduce the exact same application anytime. And we're going to do this by using a Git commit as a snapshot of the code, config, data used to produce a specific model. Here are the key elements we'll need to incorporate to make our application entirely reproducible:

- repository should store pointers to large data and model artifacts living in blob storage.
- use commits to store snapshots of the code, config, data and model and be able to update and rollback versions.
- expose configurations and performances so we can inspect for improvements and regressions.

<div class="ai-center-all">
    <img width="700" src="https://raw.githubusercontent.com/GokuMohandas/madewithml/main/images/mlops/versioning/versioning.png">
</div>

## Application

There are many tools available for saving and pointing to our large artifacts but we'll be using the [Data Version Control (DVC)](https://dvc.org/){:target="_blank"} library for it's simplicity, rich features and most importantly modularity. DVC has lots of other useful [features](https://dvc.org/features){:target="_blank"} (metrics, experiments, etc.) so be sure to explore those as well.

We'll be using DVC to version our datasets and model weights and store them in a local directory which will act as our blob storage. We could use remote blob storage options such as S3, GCP, Google Drive, [DAGsHub](https://dagshub.com/){:target="_blank"}, etc. but we're going to replicate the same actions locally so we can see how the data is stored.

!!! note
    We'll be using a local directory to act as our blob storage so we can develop and analyze everything locally. We'll continue to do this for other storage components as well such as feature stores and like we have been doing with our local model registry.

### Set up
Let's start by installing DVC and initializing it to create a [.dvc](https://github.com/GokuMohandas/mlops/tree/main/.dvc){:target="_blank"} directory.
```bash
# Initialization
pip install dvc
pip uninstall dataclasses (Python < 3.8)
dvc init
```

### Remote storage
After initializing DVC, we can establish where our remote storage will be. We be using the `stores/blob` directory which won't be checked into our remote repository.
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

### Add data
Now we're ready to *add* our data which will create text pointer files for each file.

```bash
# Add artifacts
dvc add data/projects.json
dvc add data/tags.json
```

```bash
# Pointer files added
📂 data
  📄 .gitignore
  📄 projects.json
  📄 projects.json.dvc
  📄 tags.json
  📄 tags.json.dvc
```

Each pointer file will contain the md5 hash, size and the location w.r.t to the directory which we'll be checking into our git repository.

```yaml
# data/projects.json.dvc
outs:
- md5: dafec16f20e07c58af2ab05efe6818ce
  size: 764016
  path: projects.json
```

The data directory containing the files will also have a .gitignore file that includes the actual artifacts so we don't check them into our repository.

```yaml
# data/.gitignore
/projects.json
/tags.json
```

!!! note
    In terms of versioning our model artifacts, we aren't pushing anything to our blob storage because our model registry already takes care of all that. Instead we expose the run ID so we can load necessary artifacts, [params.json](https://raw.githubusercontent.com/GokuMohandas/mlops/main/model/params.json){:target="_blank"} and [performance.json](https://raw.githubusercontent.com/GokuMohandas/mlops/main/model/performance.json){:target="_blank"}, because we'll be using them to compare different model versions (and they're small enough to version via Git).

    ```bash
    # Model artifacts
    📂 model
      📄 run_id.txt
      📄 params.json
      📄 performance.json
    ```

    For very large applications, these artifacts would be stores in a metadata or evaluation store where they'll be indexed by model run IDs.

### Push
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

    ```yaml hl_lines="9"
    # Makefile
    .PHONY: dvc
    dvc:
        dvc add data/projects.json
        dvc add data/tags.json
        dvc push
    ```

    ```yaml
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

### Pull

When someone else wants to pull updated artifacts or vice verse, we can use the *pull* command to fetch from our remote storage to our local artifact directories. All we need is to first ensure that we have the latest pointer text files (via git pull).

```bash
# Pull from remote storage
dvc pull
```

### Tag
Not every commit is going to involve a new set of data and model artifacts so we can leverage git [tags](https://git-scm.com/book/en/v2/Git-Basics-Tagging){:target="_blank"} to mark our release commits. We can create tags either through the terminal or the online remote interface and this can be done to previous commits as well (in case we forgot).

```bash
# Tags
git tag  # view all existing tags
git tag -a <TAG_NAME> -m "charCNN"  # create a tag
git checkout -b <BRANCH_NAME> <TAG_NAME>  # checkout a specific tag
git tag -d <TAG_NAME>  # delete local tag
git push origin --delete <TAG_NAME>  # delete remote tag
git fetch --all --tags  # fetch all tags from remote
```

!!! note
    Tag names usually adhere to version naming conventions, such as v1.4.2 where the numbers indicate major, minor and bug changes from left to right.


<!-- Citation -->
{% include "cite.md" %}