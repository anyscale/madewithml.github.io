---
template: lesson.html
title: Dashboard
description: Creating an interactive dashboard to visually inspect our application using Streamlit.
keywords: dashboard, visualization, streamlit, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning, great expectations
image: https://madewithml.com/static/images/mlops.png
repository: https://github.com/GokuMohandas/MLOps
---

{% include "styles/lesson.md" %}

## Intuition

When developing an application, there are a lot of technical decisions and results (preprocessing, performance, etc.) that are integral to our system. How can we **effectively communicate** this to other developers and business stakeholders? One option is a Jupyter [notebook](../foundations/notebooks.md){:target="_blank"} but it's often cluttered with code and isn't very easy for non-technical team members to access and run. We need to create a dashboard that can be accessed without any technical prerequisites and effectively communicates key findings. It would be even more useful if our dashboard was interactive such that it provides utility even for the technical developers.

## Streamlit

There are some great tooling options, such as [Dash](https://plotly.com/dash/){:target="_blank"}, [Gradio](https://gradio.app/){:target="_blank"}, [Streamlit](https://streamlit.io/){:target="_blank"}, [Tableau](https://www.tableau.com/){:target="_blank"}, [Looker](https://looker.com/){:target="_blank"}, etc. for creating dashboards to deliver data oriented insights. Traditionally, interactive dashboards were exclusively created using front-end programming languages such as HTML Javascript, CSS, etc. However, given that many developers working in machine learning are using Python, the tooling landscape has evolved to bridge this gap. These tools now allow ML developers to create interactive dashboards and visualizations in Python while offering full customization via HTML, JS, and CSS. We'll be using [Streamlit](https://streamlit.io/){:target="_blank"} to create our dashboards because of it's intuitive API, sharing capabilities and increasing community adoption.

## Set up
With Streamlit, we can quickly create an empty application and as we develop, the UI will update as well.
```bash
# Setup
pip install streamlit==1.10.0
mkdir streamlit
touch streamlit/app.py
streamlit run streamlit/app.py
```
<pre class="output">
You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://10.0.1.93:8501
</pre>

This will automatically open up the streamlit dashboard for us on [http://localhost:8501](http://localhost:8501){:target="_blank"}.

> Be sure to add this package and version to our `requirements.txt` file.

## API Reference
Before we create a dashboard for our specific application, we need to learn about the different Streamlit components. Instead of going through them all in this lesson, take a few minutes and go through the [API reference](https://docs.streamlit.io/library/api-reference){:target="_blank"}. It's quite short and we promise you'll be amazed at how many UI components (styled text, latex, tables, plots, etc.) you can create using just Python. We'll explore the different components in detail as they apply to creating different interactions for our specific dashboard below.

## Sections
We'll start by outlining the sections we want to have in our dashboard by editing our `streamlit/app.py` script:

```python linenums="1"
import pandas as pd
from pathlib import Path
import streamlit as st

from config import config
from tagifai import main, utils
```

```python linenums="1"
# Title
st.title("TagIfAI · MLOps · Made With ML")

# ToC
st.markdown("🔢 [Data](#data)", unsafe_allow_html=True)
st.markdown("📊 [Performance](#performance)", unsafe_allow_html=True)
st.markdown("🚀 [Inference](#inference)", unsafe_allow_html=True)

# Sections
st.header("🔢 Data")
st.header("📊 Performance")
st.header("🚀 Inference")
```

To see these changes on our dashboard, we can refresh our dashboard page (press `R`) or set it `Always rerun` (press `A`).

<div class="ai-center-all">
    <img width="600" src="/static/images/mlops/dashboard/sections.png">
</div>

### Data

We're going to keep our dashboard simple, so we'll just display the labeled projects.

```python linenums="1"
st.header("Data")
projects_fp = Path(config.DATA_DIR, "labeled_projects.json")
projects = utils.load_dict(filepath=projects_fp)
df = pd.DataFrame(projects)
st.text(f"Projects (count: {len(df)})")
st.write(df)
```

<div class="ai-center-all">
    <img width="700" src="/static/images/mlops/dashboard/data.png">
</div>

### Performance

In this section, we'll display the performance of from our latest trained model. Again, we're going to keep it simple but we could also overlay more information such as improvements or regressions from previous deployments by accessing the model store.

```python linenums="1"
st.header("📊 Performance")
performance_fp = Path(config.CONFIG_DIR, "performance.json")
performance = utils.load_dict(filepath=performance_fp)
st.text("Overall:")
st.write(performance["overall"])
tag = st.selectbox("Choose a tag: ", list(performance["class"].keys()))
st.write(performance["class"][tag])
tag = st.selectbox("Choose a slice: ", list(performance["slices"].keys()))
st.write(performance["slices"][tag])
```

<div class="ai-center-all">
    <img width="700" src="/static/images/mlops/dashboard/performance.png">
</div>

### Inference

With the inference section, we want to be able to quickly predict with the latest trained model.

```python linenums="1"
st.header("🚀 Inference")
text = st.text_input("Enter text:", "Transfer learning with transformers for text classification.")
run_id = st.text_input("Enter run ID:", open(Path(config.CONFIG_DIR, "run_id.txt")).read())
prediction = main.predict_tag(text=text, run_id=run_id)
st.write(prediction)
```

<div class="ai-center-all">
    <img width="700" src="/static/images/mlops/dashboard/inference.png">
</div>

!!! tip
    Our dashboard is quite simple but we can also more comprehensive dashboards that reflect some of the core topics we covered in our [machine learning canvas](/static/templates/ml-canvas.pdf){:target="_blank"}.

    - Display findings from our [labeling](labeling.md){:target="_blank"}, [EDA](exploratory-data-analysis.md){:target="_blank"} and [preprocessing](preprocessing.md){:target="_blank"} stages of development.
    - View [false +/-](evaluation.md#confusion-matrix){:target="_blank"} interactively and connect with annotation pipelines so that changes to the data can be reviewed and incorporated.
    - Compare performances across multiple releases to visualize improvements/regressions over time (using model store, git tags, etc.)

## Caching

Sometimes we may have views that involve computationally heavy operations, such as loading data or model artifacts. It's best practice to cache these operations by wrapping them as a separate function with the [`@st.cache`](https://docs.streamlit.io/library/api-reference/performance/st.cache){:target="_blank"} decorator. This calls for Streamlit to cache the function by the specific combination of it's inputs to deliver the respective outputs when the function is invoked with the same inputs.

```python linenums="1" hl_lines="1"
@st.cache()
def load_data():
    projects_fp = Path(config.DATA_DIR, "labeled_projects.json")
    projects = utils.load_dict(filepath=projects_fp)
    df = pd.DataFrame(projects)
    return df
```

## Deploy

We have several different options for deploying and managing our Streamlit dashboard. We could use Streamlit's [sharing feature](https://blog.streamlit.io/introducing-streamlit-sharing/){:target="_blank"} (beta) which allows us to seamlessly deploy dashboards straight from GitHub. Our dashboard will continue to stay updated as we commit changes to our repository. Another option is to deploy the Streamlit dashboard along with our API service. We can use docker-compose to spin up a separate container or simply add it to the API service's Dockerfile's [ENTRYPOINT](https://docs.docker.com/engine/reference/builder/#entrypoint){:target="_blank"} with the appropriate ports exposed. The later might be ideal, especially if your dashboard isn't meant to be public and it you want added security, performance, etc.


<!-- Citation -->
{% include "cite.md" %}