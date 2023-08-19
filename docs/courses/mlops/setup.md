---
template: lesson.html
title: Setup
description: Setting up our development environment for local and Anyscale development.
keywords: anyscale, mlops, machine learning
image: https://madewithml.com/static/images/mlops.png
repository: https://github.com/GokuMohandas/Made-With-ML
notebook: https://github.com/GokuMohandas/Made-With-ML/blob/main/notebooks/madewithml.ipynb
---

{% include "styles/lesson.md" %}

In this lesson, we'll setup the development environment that we'll be using in all of our lessons. We'll have instructions for both local laptop and remote scalable clusters ([Anyscale](https://anyscale.com/){:target="_blank"}). While everything will work locally on your laptop, you can sign up to join one of our upcoming live cohorts where we'll provide **live lessons + QA**, **compute (GPUs)** and **community** to learn everything in one day → [sign up here](https://4190urw86oh.typeform.com/madewithml){:target="_blank"}.

## Cluster

We'll start with defining our cluster, which refers to a group of servers that come together to form one system. Our clusters will have a [head node](https://docs.ray.io/en/latest/cluster/key-concepts.html#head-node){:target="_blank"} that manages the cluster and it will be connected to a set of [worker nodes](https://docs.ray.io/en/latest/cluster/key-concepts.html#head-node){:target="_blank"} that will execute workloads for us. These clusters can be fixed in size or [autoscale](https://docs.ray.io/en/latest/cluster/key-concepts.html#cluster-autoscaler){:target="_blank"} based on our application's compute needs, which makes them highly scalable and performant. We'll create our cluster by defining a compute configuration and an environment.

### Environment

We'll start by defining our cluster environment which will specify the software dependencies that we'll need for our workloads.

!!! quote "💻 Local"
    Your personal laptop will need to have Python installed and we highly recommend using Python `3.10`. You can use a tool like [pyenv](https://github.com/pyenv/pyenv) (mac) or [pyenv-win](https://github.com/pyenv-win/pyenv-win) (windows) to easily download and switch between Python versions.

    ```bash
    pyenv install 3.10.11  # install
    pyenv global 3.10.11  # set default
    ```

    Once we have our Python version, we can create a virtual environment to install our dependencies. We'll download our Python dependencies *after* we clone our repository from git [shortly](#git).

    ```bash
    mkdir madewithml
    cd madewithml
    python3 -m venv venv  # create virtual environment
    source venv/bin/activate  # on Windows: venv\Scripts\activate
    python3 -m pip install --upgrade pip setuptools wheel
    ```

!!! example "🚀 Anyscale"
    Our cluster environment will be defined inside a [`cluster_env.yaml`](https://github.com/GokuMohandas/Made-With-ML/blob/main/deploy/cluster_env.yaml){:target="_blank"} file. Here we specify some details around our base image ([anyscale/ray:2.6.0-py310-cu118](https://docs.anyscale.com/reference/base-images/ray-260/py310#ray-2-6-0-py310-cu118){:target="_blank"}) that has our Python version, GPU dependencies, etc.

    ```yaml linenums="1"
    base_image: anyscale/ray:2.6.0-py310-cu118
    env_vars: {}
    debian_packages:
      - curl

    python:
      pip_packages: []
      conda_packages: []

    post_build_cmds:
      - python3 -m pip install --upgrade pip setuptools wheel
      - python3 -m pip install -r https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/requirements.txt
    ```

    We could specify any python packages inside `pip_packages` or `conda_packages` but we're going to use a [`requirements.txt`](https://github.com/GokuMohandas/Made-With-ML/blob/main/requirements.txt){:target="_blank"} file to load our dependencies under `post_build_cmds`.

### Compute

Next, we'll define our compute configuration, which will specify our hardware dependencies (head and worker nodes) that we'll need for our workloads.

!!! quote "💻 Local"
    Your personal laptop (single machine) will act as the cluster, where one CPU will be the head node and some of the remaining CPU will be the worker nodes (no GPUs required). All of the code in this course will work in any personal laptop though it will be slower than executing the same workloads on a larger cluster.

!!! example "🚀 Anyscale"
    Our cluster compute will be defined inside a [`cluster_compute.yaml`](https://github.com/GokuMohandas/Made-With-ML/blob/main/deploy/cluster_compute.yaml){:target="_blank"} file. Here we specify some details around where our compute resources will come from (cloud computing platform like AWS), types of nodes and their counts, etc.

    ```yaml linenums="1"
    cloud: madewithml-us-east-2
    region: us-east2
    head_node_type:
      name: head_node_type
      instance_type: m5.2xlarge  # 8 CPU, 0 GPU, 32 GB RAM
    worker_node_types:
    - name: gpu_worker
      instance_type: g4dn.xlarge  # 4 CPU, 1 GPU, 16 GB RAM
      min_workers: 0
      max_workers: 1
    ...
    ```

    Our worker nodes will be GPU-enabled so we can train our models faster and we set `min_workers` to 0 so that we can autoscale these workers only when they're needed (up to a maximum of `max_workers`). This will help us significantly reduce our compute costs without having to manage the infrastructure ourselves.


## Workspaces

With our compute and environment defined, we're ready to create our cluster workspace. This is where we'll be developing our ML application on top of our compute, environment and storage.

!!! quote "💻 Local"
    Your personal laptop will need to have an interactive development environment (IDE) installed, such as [VS Code](https://code.visualstudio.com/){:target="_blank"}. For bash commands in this course, you're welcome to use the terminal on VSCode or a separate one.

!!! example "🚀 Anyscale"
    We're going to launch an Anyscale [Workspace](https://docs.anyscale.com/develop/workspaces/get-started){:target="_blank"} to do all of our development in. Workspaces allow us to use development tools such as VSCode, Jupyter notebooks, web terminal, etc. on top of our cluster compute, environment and [storage](https://docs.anyscale.com/develop/workspaces/storage){:target="_blank"}. This create an "infinite laptop" experience that feels like a local laptop experience but on a powerful, scalable cluster.

    <div class="ai-center-all">
        <img src="/static/images/mlops/setup/workspaces.png" width="400" alt="Anyscale Workspaces">
    </div>

    We have the option to create our Workspace using a [CLI](https://docs.anyscale.com/reference/anyscale-cli) but we're going to create it using the [web UI](https://console.anyscale.com/o/madewithml/workspaces/add/blank) (you will receive the required credentials during the cohort). On the UI, we can fill in the following information:

    ```md
    - Workspace name: `madewithml`
    - Project: `madewithml`
    - Cluster environment name: `madewithml-cluster-env`
    # Toggle `Select from saved configurations`
    - Compute config: `madewithml-cluster-compute`
    - Click on the **Start** button to launch the Workspace
    ```

    <div class="ai-center-all">
        <img src="/static/images/mlops/setup/configs.png" width="800" alt="Anyscale configs">
    </div>

    We have already saved created our Project, cluster environment and compute config so we can select them from the dropdowns but we could just as easily create new ones / update these using the [CLI](https://docs.anyscale.com/reference/cli/commands){:target="_blank"}.

    ??? info "CLI method"
        ```bash
        # Set credentials
        export ANYSCALE_HOST=https://console.anyscale.com
        export ANYSCALE_CLI_TOKEN=$YOUR_CLI_TOKEN  # retrieved from Anyscale credentials page

        # Create project
        export PROJECT_NAME="madewithml"
        anyscale project create --name $PROJECT_NAME

        # Cluster environment
        export CLUSTER_ENV_NAME="madewithml-cluster-env"
        anyscale cluster-env build deploy/cluster_env.yaml --name $CLUSTER_ENV_NAME

        # Compute config
        export CLUSTER_COMPUTE_NAME="madewithml-cluster-compute"
        anyscale cluster-compute create deploy/cluster_compute.yaml --name $CLUSTER_COMPUTE_NAME
        ```

## Git

With our development workspace all set up, we're ready to start developing. We'll start by following these instructions to create a repository:

1. [Create a new repository](https://github.com/new)
2. name it `Made-With-ML`
3. Toggle `Add a README file` (**very important** as this creates a `main` branch)
4. Scroll down and click `Create repository`

Now we're ready to clone the [Made With ML repository](https://github.com/GokuMohandas/Made-With-ML){:target="_blank"}'s contents from GitHub inside our `madewithml` directory.

```bash
export GITHUB_USERNAME="YOUR_GITHUB_UESRNAME"  # <-- CHANGE THIS to your username
git clone https://github.com/GokuMohandas/Made-With-ML.git .
git remote set-url origin https://github.com/$GITHUB_USERNAME/Made-With-ML.git
git checkout -b dev
export PYTHONPATH=$PYTHONPATH:$PWD  # so we can import modules from our scripts
```

!!! quote "💻 Local"
    Recall that we created our virtual environment earlier but didn't actually load any Python dependencies yet. We'll clone our repository and then install the packages using the `requirements.txt` file.

    ```bash
    python3 -m pip install -r requirements.txt
    ```

    > **Caution**: make sure that we're installing our Python packages inside our virtual environment.

!!! example "🚀 Anyscale"
    Our environment with the appropriate Python version and libraries is already all set for us through the cluster environment we used when setting up our Anyscale Workspace. But if we want to install additional Python packages as we develop, we need to do pip install with the [`--user`](https://pip.pypa.io/en/stable/user_guide/#user-installs){:target="_blank"} flag inside our Workspaces (via terminal) to ensure that our head and all worker nodes receive the package. And then we should also add it to our requirements file so it becomes part of the cluster environment build process next time.

    ```bash
    pip install --user <package_name>:<version>
    ```

## Notebook

Now we're ready to launch our Jupyter notebook to interactively develop our ML application.

!!! quote "💻 Local"
    We already installed jupyter through our [`requirements.txt`](https://github.com/GokuMohandas/Made-With-ML/blob/main/requirements.txt){:target="_blank"} file in the previous step, so we can just launch it.

    ```bash
    jupyter lab notebooks/madewithml.ipynb
    ```

!!! example "🚀 Anyscale"
    Click on the Jupyter icon &nbsp;<img width=15 src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Jupyter_logo.svg/1200px-Jupyter_logo.svg.png">&nbsp; at the top right corner of our Anyscale Workspace page and this will open up our JupyterLab instance in a new tab. Then navigate to the `notebooks` directory and open up the `madewithml.ipynb` notebook.

    <div class="ai-center-all">
        <img src="/static/images/mlops/setup/devtools.png" width="600" alt="Workspace dev tools">
    </div>

## Ray

We'll be using [Ray](https://github.com/project-ray/ray){:target="_blank"} to scale and productionize our ML application. Ray consists of a core distributed runtime along with libraries for scaling ML workloads and has companies like [OpenAI](https://www.anyscale.com/blog/ray-summit-2022-stories-large-language-models){:target="_blank"}, [Spotify](https://engineering.atspotify.com/2023/02/unleashing-ml-innovation-at-spotify-with-ray/){:target="_blank"}, [Netflix](https://netflixtechblog.com/scaling-media-machine-learning-at-netflix-f19b400243){:target="_blank"}, [Instacart](https://tech.instacart.com/distributed-machine-learning-at-instacart-4b11d7569423){:target="_blank"}, [Doordash](https://doordash.engineering/2023/06/20/how-doordash-built-an-ensemble-learning-model-for-time-series-forecasting/){:target="_blank"} + many [more](https://www.anyscale.com/user-stories){:target="_blank"} using it to develop their ML applications. We're going to start by initializing Ray inside our notebooks:

```python linenums="1"
import ray
```

```python linenums="1"
# Initialize Ray
if ray.is_initialized():
    ray.shutdown()
ray.init()
```

We can also view our cluster resources to view the available compute resources:

```python linenums="1"
ray.cluster_resources()
```

!!! quote "💻 Local"
    If you are running this on a local laptop (no GPU), use the CPU count from `ray.cluster_resources()` to set your resources. For example if your machine has 10 CPUs:

    <pre class="output">
    {'CPU': 10.0,
     'object_store_memory': 2147483648.0,
     'node:127.0.0.1': 1.0}
     </pre>

    ```python
    num_workers = 6  # prefer to do a few less than total available CPU (1 for head node + 1 for background tasks)
    resources_per_worker={"CPU": 1, "GPU": 0}
    ```

!!! example "🚀 Anyscale"
    On our Anyscale Workspace, the `ray.cluster_resources()` command will produce:

    <pre class="output">
    {'CPU': 8.0,
    'node:__internal_head__': 1.0,
    'node:10.0.56.150': 1.0,
    'memory': 34359738368.0,
    'object_store_memory': 9492578304.0}
    </pre>

    These cluster resources only reflect our head node (1 [m5.2xlarge](https://instances.vantage.sh/aws/ec2/m5.2xlarge){:target="_blank"}). But recall earlier in our [compute configuration](#compute) that we also added [g4dn.xlarge](https://instances.vantage.sh/aws/ec2/g4dn.xlarge){:target="_blank"} worker nodes (each has 1 GPU and 4 CPU) to our cluster. But because we set `min_workers=0`, our worker nodes will autoscale ( up to `max_workers`) as they're needed for specific workloads (ex. training). So we can set the # of workers and resources by worker based on this insight:

    ```python
    # Workers (1 g4dn.xlarge)
    num_workers = 1
    resources_per_worker={"CPU": 3, "GPU": 1}
    ```

Head on over to the next lesson, where we'll motivate the specific application that we're trying to build from a product and systems design perspective. And after that, we're ready to start developing!

<!-- Course signup -->
{% include "templates/signup.md" %}

<!-- Citation -->
{% include "templates/cite.md" %}