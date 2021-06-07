---
template: lesson.html
title: Feature Stores
description: Using feature stores to connect the DataOps and MLOps pipelines to enable collaborative teams to develop efficiently.
keywords: feature stores, feast, point-in-time correctness, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/mlops.png
repository: https://github.com/GokuMohandas/MLOps
notebook: https://colab.research.google.com/github/GokuMohandas/MLOps/blob/main/notebooks/feature_store.ipynb
---


SQLlite online store
Offline store is just feature tables from the Parquet file

### Creating Parquet files
```python linenums="1"
import pandas as pd
from pathlib import Path
from tagifai import config, utils

# Load features to df
features_fp = Path(config.DATA_DIR, "features.json")
features = utils.load_dict(filepath=features_fp)
df = pd.DataFrame(features)

# Format timestamp
df.created_on = pd.to_datetime(df.created_on)

# Convert to parquet
df.to_parquet(
    "features.parquet",
    compression=None,
    allow_truncated_timestamps=True,
    )
```

Had our entity (projects) had features that change over time, we would materialize them to the online store incrementally. And of course, if we need stream processing, we would retrieve the features directly from the online store, whose features would have been logged from the data stream (ex. Kafka) to the batch data source (Snowflake, BigQuery, etc.).

- feast has plans to incorporate validation (GE) on features

