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

There are some great tooling options, such as [Dash](https://plotly.com/dash/){:target="_blank"}, [Gradio](https://gradio.app/){:target="_blank"}, [Streamlit](https://streamlit.io/){:target="_blank"}, [Tableau](https://www.tableau.com/){:target="_blank"}, [Looker](https://looker.com/){:target="_blank"}, etc. for creating dashboards to deliver data oriented insights. Traditionally, interactive dashboards were exclusively created using front-end programming languages such as HTML Javascript, CSS, etc. However, given that many developers working in machine learning are using Python, the tooling landscape has evolved to bridge this gap. These tools now allow ML developers to create interactive dashboards and visualizations in Python while offering full customization via HTML, JS, and CSS. We'll be using Streamlit to create our dashboards because of it's intuitive API, sharing capabilities and increasing community adoption.

## Set up
With Streamlit, we can quickly create an empty application and as we develop, the UI will update as well.
```bash
# Setup
pip install streamlit
mkdir streamlit
touch streamlit/st_app.py
streamlit run streamlit/st_app.py
```
<pre class="output">
Local URL: http://localhost:8501
</pre>

## Components
Before we create a dashboard for our specific application, we need to learn about the different Streamlit [API components](https://docs.streamlit.io/en/stable/api.html){:target="_blank"}. Instead of going through them all in this lesson, take ten minutes and go through the entire documentation page. It's quite short and we promise you'll be amazed at how many UI components (styled text, latex, tables, plots, etc.) you can create using just Python. We'll explore the different components in detail as they apply to creating different interactions for our specific dashboard below.

## Pages
Our application's [dashboard](https://github.com/GokuMohandas/MLOps/blob/main/streamlit/st_app.py){:target="_blank"} will feature several pages organized by the insight they will provide where the view can choose what via interactive [radio buttons](https://docs.streamlit.io/en/stable/api.html#streamlit.radio){:target="_blank"}.

<div class="ai-center-all">
    <img width="700" src="/static/images/mlops/dashboard/pages.png">
</div>

### Data

The data page contains findings from data [labeling](labeling.md){:target="_blank"}, [exploratory data analysis](exploratory-data-analysis.md){:target="_blank"} and [preprocessing](preprocessing.md){:target="_blank"} but with interactive components.

We start by showing a sample of our different data sources because, for many people, this may be the first time they see the data so it's a good opportunity for them to understand all the different features, formats, etc. For displaying the tags, we don't want to just dump all of them on the dashboard but instead we can use a [selectbox](https://docs.streamlit.io/en/stable/api.html#streamlit.selectbox){:target="_blank"} to allow the user to view them one at a time.

```python linenums="1"
# Load data
projects_fp = Path(config.DATA_DIR, "projects.json")
tags_fp = Path(config.DATA_DIR, "tags.json")
projects = utils.load_dict(filepath=projects_fp)
tags_dict = utils.list_to_dict(utils.load_dict(filepath=tags_fp), key="tag")
col1, col2 = st.beta_columns(2)
with col1:
    st.subheader("Projects (sample)")
    st.write(projects[0])
with col2:
    st.subheader("Tags")
    tag = st.selectbox("Choose a tag", list(tags_dict.keys()))
    st.write(tags_dict[tag])
```

<div class="ai-center-all">
    <img width="700" src="/static/images/mlops/dashboard/data.png">
</div>

We can also show our a snapshot of the loaded DataFrame which has sortable columns that people can play with to explore the data.

```python linenums="1"
# Dataframe
df = pd.DataFrame(projects)
st.text(f"Projects (count: {len(df)}):")
st.write(df)
```

<div class="ai-center-all">
    <img width="700" src="/static/images/mlops/dashboard/df.png">
</div>

We can essentially walk viewers through our entire data phase (EDA, preprocessing, etc.) and allow them (and ourselves) to explore key decisions. For example, we chose to introduce a minimum tag frequency constraint so that we can have enough samples. We can now interactively change that value with a [slider widget](https://docs.streamlit.io/en/stable/api.html#streamlit.slider){:target="_blank"} and see which tags just made and missed the cut.

```python linenums="1"
# Sliders
min_tag_freq = st.slider("min_tag_freq", min_value=1, value=30, step=1)
df, tags_above_freq, tags_below_freq = data.prepare(
    df=df,
    include=list(tags_dict.keys()),
    exclude=config.EXCLUDED_TAGS,
    min_tag_freq=min_tag_freq,
)
col1, col2, col3 = st.beta_columns(3)
with col1:
    st.write("**Most common tags**:")
    for item in tags_above_freq.most_common(5):
        st.write(item)
with col2:
    st.write("**Tags that just made the cut**:")
    for item in tags_above_freq.most_common()[-5:]:
        st.write(item)
with col3:
    st.write("**Tags that just missed the cut**:")
    for item in tags_below_freq.most_common(5):
        st.write(item)
with st.beta_expander("Excluded tags"):
    st.write(config.EXCLUDED_TAGS)
```

<div class="ai-center-all">
    <img width="700" src="/static/images/mlops/dashboard/frequency.png">
</div>

What makes this truly interactive is that when we alter the value here, all the downstream tables and plots will update to reflect that change immediately. This is a great way to explore what constraints to use because we can quickly visualize the impact it can have on our data.

```python linenums="1" hl_lines="2"
# Plots
num_tags_per_project = [len(tags) for tags in df.tags]  # df is dependent on min_tag_freq slider's value
num_tags, num_projects = zip(*Counter(num_tags_per_project).items())
plt.figure(figsize=(10, 3))
ax = sns.barplot(list(num_tags), list(num_projects))
plt.title("Tags per project", fontsize=20)
plt.xlabel("Number of tags", fontsize=16)
ax.set_xticklabels(range(1, len(num_tags) + 1), rotation=0, fontsize=16)
plt.ylabel("Number of projects", fontsize=16)
plt.show()
st.pyplot(plt)
```

<div class="ai-center-all">
    <img width="700" src="/static/images/mlops/dashboard/plots.png">
</div>

On a similar note, we can also interactively view how our preprocessing functions behave. We can alter any of the function's default input arguments, as well as the input text.

```python linenums="1"
# Preprocessing
filters = st.text_input("filters", "[!\"'#$%&()*+,-./:;<=>?@\\[]^_`{|}~]")
lower = st.checkbox("lower", True)
stem = st.checkbox("stem", False)
text = st.text_input("Input text", "Conditional generation using Variational Autoencoders.")
preprocessed_text = data.preprocess(text=text, lower=lower, stem=stem, filters=filters)
st.write("Preprocessed text", preprocessed_text)
```

<div class="ai-center-all">
    <img width="700" src="/static/images/mlops/dashboard/preprocessing.png">
</div>

!!! bug
    In fact, we were able to discover and fix a bug here where the NLTK package automatically lowers text when stemming which we had to override using our `Stemmer` class in our [data script](https://github.com/GokuMohandas/MLOps/blob/main/tagifai/data.py){:target="_blank"}.

### Performance

This page allows us to quickly compare the improvements and regressions of our local system and what's currently in production. We want to provide the key differences in both the performance and parameters used for each system version. We could also use constructs, such as [Git tags](git.md#tags){:target="_blank"}, to visualize these details across multiple previous releases.

<div class="ai-center-all">
    <img width="700" src="/static/images/mlops/dashboard/performance.png">
</div>

### Inference

With the inference page, we want to be able to test our model using various inputs to receive predictions, as well as intermediate outputs (ex. preprocessed text). This is a great way for our team members to quickly play with the latest deployed model.

<div class="ai-center-all">
    <img width="700" src="/static/images/mlops/dashboard/inference.png">
</div>

### Inspection

Our last page will enable a closer inspection on the test split's predictions to identify areas to improve, collect more data, etc. First we offer a quick view of each tag's performance and we could also do the same for specific slices of the data we may care about (high priority, minority, etc.)

<div class="ai-center-all">
    <img width="700" src="/static/images/mlops/dashboard/inspection.png">
</div>

We're also going to inspect the true positive (TP), false positive (FP) and false negative (FN) samples across our different tags. It's a great way to catch issues with labeling (FP), weaknesses (FN), etc.

<div class="ai-center-all">
    <img width="700" src="/static/images/mlops/dashboard/fp.png">
</div>

!!! warning
    Be careful not to make decisions based on predicted probabilities before [calibrating](https://arxiv.org/abs/1706.04599){:target="_blank"} them to reliably use as measures of confidence.

#### Extensions

- Connect inspection pipelines with annotation systems so that changes to the data can be reviewed and incorporated.
- Use false positives to identify potentially mislabeled data or [estimate training data influences (TracIn)](https://arxiv.org/abs/2002.08484){:target="_blank"} on their predictions.
- Inspect the trained model's behavior under various conditions using the [WhatIf](https://pair-code.github.io/what-if-tool/){:target="_blank"} tool.
- Compare performances across multiple releases to visualize improvements/regressions over time.

> Our dashboard can have many other pages as well, especially critical views for [iteration](data-centric-ai.md){:target="_blank"}, such as [active learning](labeling.md#active-learning){:target="_blank"}, [composing retraining datasets](continual-learning.md#retraining){:target="_blank"}, etc.


## Caching

There are a few functions defined at the start of our [st_app.py](https://github.com/GokuMohandas/MLOps/blob/main/streamlit/st_app.py){:target="_blank"} script which have a `@st.cache` decorator. This calls for Streamlit to cache the function by the combination of it's inputs which will significantly improve performance involving computationally heavy functions.

```python linenums="1"
@st.cache()
def load_data():
    # Filepaths
    projects_fp = Path(config.DATA_DIR, "projects.json")
    tags_fp = Path(config.DATA_DIR, "tags.json")
    features_fp = Path(config.DATA_DIR, "features.json")

    # Load data
    projects = utils.load_dict(filepath=projects_fp)
    tags_dict = utils.list_to_dict(utils.load_dict(filepath=tags_fp), key="tag")
    features = utils.load_dict(filepath=features_fp)

    return projects, tags_dict, features
```

## Deploy

We have several different options for deploying and managing our Streamlit dashboard. We could use Streamlit's [sharing feature](https://blog.streamlit.io/introducing-streamlit-sharing/){:target="_blank"} (beta) which allows us to seamlessly deploy dashboards straight from GitHub. Our dashboard will continue to stay updated as we commit changes to our repository. Another option is to deploy the Streamlit dashboard along with our API service. We can use docker-compose to spin up a separate container or simply add it to the API service's Dockerfile's [ENTRYPOINT](https://docs.docker.com/engine/reference/builder/#entrypoint){:target="_blank"} with the appropriate ports exposed. The later might be ideal, especially if your dashboard isn't meant to be public and it you want added security, performance, etc.

## Resources

- [Streamlit API Reference](https://docs.streamlit.io/en/stable/api.html){:target="_blank"}

<!-- Citation -->
{% include "cite.md" %}