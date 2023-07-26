---
template: lesson.html
title: "Versioning Code, Data and Models"
description: Versioning code, data and models to ensure reproducible behavior in ML systems.
keywords: versioning, dvc, mlops, machine learning
image: https://madewithml.com/static/images/mlops.png
repository: https://github.com/GokuMohandas/Made-With-ML
---

{% include "styles/lesson.md" %}

## Intuition

In this lesson, we're going to learn how to version our code, data and models to ensure reproducible behavior in our ML systems. It's imperative that we can reproduce our results and track changes to our system so we can debug and improve our application. Without it, it would be difficult to share our work, recreate our models in the event of system failures and fallback to previous versions in the event of regressions.

## Code

To version our code, we'll be using [git](https://git-scm.com/){:target="_blank"}, which is a widely adopted version control system. In fact, when we cloned our repository in the [setup lesson](setup.md){:target="_blank"}, we pulled code from a git repository that we had prepared for you.

```bash
git clone https://github.com/GokuMohandas/Made-With-ML.git .
```

We can then make changes to the code and Git, which is running locally on our computer, will keep track of our files and it's versions as we `add` and `commit` our changes. But it's not enough to just version our code locally, we need to `push` our work to a central location that can be `pull`ed by us and others we want to grant access to. This is where remote repositories like [GitHub](https://github.com/){:target="_blank"}, [GitLab](https://gitlab.com/){:target="_blank"}, [BitBucket](https://bitbucket.org/){:target="_blank"}, etc. provide a remote location to hold our versioned code in.

<div class="ai-center-all">
    <img width="600" src="/static/images/mlops/git/environments.png" alt="git environment">
</div>

Here's a simplified workflow for how we version our code using GitHub:

```bash
[make changes to code]
git add .
git commit -m "message"
git push origin <branch-name>
```

!!! tip
    If you're not familiar with Git, we highly recommend going through our [Git lesson](git.md){:target="_blank"} to learn the basics.

## Artifacts

While Git is ideal for saving our code, it's not ideal for saving artifacts like our datasets (especially unstructured data like text or images) and models. Also, recall that Git stores every version of our files and so large files that change frequently can very quickly take up space. So instead, it would be ideal if we can save locations (pointers) to these large artifacts in our code as opposed to the artifacts themselves. This way, we can version the locations of our artifacts and pull them as they're needed.

<div class="ai-center-all">
    <img width="700" src="/static/images/mlops/versioning/versioning.png" alt="data versioning">
</div>

### Data

While we're saving our dataset on GitHub for easy course access (and because our dataset is small), in a production setting, we would use a remote blob storage like S3 or a data warehouse like Snowflake. There are also many tools available for versioning our data, such as [GitLFS](https://git-lfs.github.com/){:target="_blank"}, [Dolt](https://github.com/dolthub/dolt){:target="_blank"}, [Pachyderm](https://www.pachyderm.com/){:target="_blank"}, [DVC](https://dvc.org/){:target="_blank"}, etc. With any of these solutions, we would be pointing to our remote storage location and versioning the pointer locations (ex. S3 bucket path) to our data instead of the data itself.

### Models

And similarly, we currently store our models locally where the [MLflow](experiment-tracking.md#setup){:target="_blank"} artifact and backend store are local directories.

```python linenums="1"
# Config MLflow
MODEL_REGISTRY = Path("/tmp/mlflow")
Path(MODEL_REGISTRY).mkdir(parents=True, exist_ok=True)
MLFLOW_TRACKING_URI = "file://" + str(MODEL_REGISTRY.absolute())
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
print (mlflow.get_tracking_uri())
```

In a production setting, these would be remote such as S3 for the artifact store and a database service (ex. [PostgreSQL RDS](https://aws.amazon.com/rds/postgresql/){:target="_blank"}) as our backend store. This way, our models can be versioned and others, with the appropriate access credentials, can pull the model artifacts and deploy them.

<!-- Course signup -->
{% include "templates/signup.md" %}

<!-- Citation -->
{% include "templates/cite.md" %}