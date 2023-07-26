---
template: lesson.html
title: APIs for Model Serving
description: Designing and deploying an API to serve machine learning models.
keywords: serving, deployment,api, fastapi, mlops, machine learning
image: https://madewithml.com/static/images/mlops.png
repository: https://github.com/GokuMohandas/Made-With-ML
---

{% include "styles/lesson.md" %}

## Intuition

Our [CLI application](cli.md){:target="_blank"} made it much easier to interact with our models, especially for fellow team members who may not want to delve into the codebase. But there are several limitations to serving our models with a CLI:

- users need access to the terminal, codebase, virtual environment, etc.
- CLI outputs on the terminal are not exportable

To address these issues, we're going to develop an application programming interface (API) that will *anyone* to interact with our application with a simple request.

> The end user may not directly interact with our API but may use UI/UX components that send requests to our it.

## Serving

APIs allow different applications to communicate with each other in real-time. But when it comes to serving predictions, we need to first decide if we'll do that in batches or real-time, which is entirely based on the feature space (finite vs. unbound).

### Batch serving

We can make batch predictions on a finite set of inputs which are then written to a database for low latency inference. When a user or downstream process sends an inference request in real-time, cached results from the database are returned.

<div class="ai-center-all">
    <img width="600" src="/static/images/mlops/systems-design/batch_serving.png" alt="batch serving">
</div>

- ✅&nbsp; generate and cache predictions for very fast inference for users.
- ✅&nbsp; the model doesn't need to be spun up as it's own service since it's never used in real-time.
- ❌&nbsp; predictions can become stale if user develops new interests that aren’t captured by the old data that the current predictions are based on.
- ❌&nbsp; input feature space must be finite because we need to generate all the predictions before they're needed for real-time.

!!! question "Batch serving tasks"
    What are some tasks where batch serving is ideal?

    ??? quote "Show answer"
        Recommend content that *existing* users will like based on their viewing history. However, *new* users may just receive some generic recommendations based on their explicit interests until we process their history the next day. And even if we're not doing batch serving, it might still be useful to cache very popular sets of input features (ex. combination of explicit interests leads to certain recommended content) so that we can serve those predictions faster.

### Real-time serving

We can also serve live predictions, typically through a request to an API with the appropriate input data.

<div class="ai-center-all">
    <img width="400" src="/static/images/mlops/systems-design/real_time_serving.png" alt="real-time serving">
</div>

- ✅&nbsp; can yield more up-to-date predictions which may yield a more meaningful user experience, etc.
- ❌&nbsp; requires managed microservices to handle request traffic.
- ❌&nbsp; requires real-time monitoring since input space in unbounded, which could yield erroneous predictions.

In this lesson, we'll create the API required to enable real-time serving. The interactions in our situation involve the client (users, other applications, etc.) sending a *request* (ex. prediction request) with the appropriate inputs to the server (our application with a trained model) and receiving a *response* (ex. prediction) in return.

<div class="ai-center-all">
    <img width="550" src="/static/images/mlops/api/interactions.png" alt="client api interactions">
</div>

## Request

Users will interact with our API in the form of a request. Let's take a look at the different components of a request:

### URI

A uniform resource identifier (URI) is an identifier for a specific resource.

<pre class="output ai-center-all" style="padding-left: 0rem; padding-right: 0rem; font-weight: 600;">
<span style="color:#d63939">https://</span><span style="color:#206bc4">localhost:</span><span style="color: #4299e1">8000</span><span style="color:#2fb344">/models/{modelId}/</span><span style="color:#ae3ec9">?filter=passed</span><span style="color:#f76707">#details</span>
</pre>

<div class="row">
  <div class="col-md-6">
    <div class="md-typeset__table">
      <table>
        <thead>
          <tr>
            <th align="left">Parts of the URI</th>
            <th align="left">Description</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td align="left"><span style="color: #d63939;">scheme</span></td>
            <td align="left">protocol definition</td>
          </tr>
          <tr>
            <td align="left"><span style="color: #206bc4;">domain</span></td>
            <td align="left">address of the website</td>
          </tr>
          <tr>
            <td align="left"><span style="color: #4299e1;">port</span></td>
            <td align="left">endpoint</td>
          </tr>
          <tr>
            <td align="left"><span style="color: #2fb344;">path</span></td>
            <td align="left">location of the resource</td>
          </tr>
          <tr>
            <td align="left"><span style="color: #ae3ec9;">query string</span></td>
            <td align="left">parameters to identify resources</td>
          </tr>
          <tr>
            <td align="left"><span style="color: #f76707;">anchor</span></td>
            <td align="left">location on webpage</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
  <div class="col-md-6">
    <div class="md-typeset__table">
      <table>
        <thead>
          <tr>
            <th align="left">Parts of the path</th>
            <th align="left">Description</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td align="left"><code>/models</code></td>
            <td align="left">collection resource of all <code>models</code></td>
          </tr>
          <tr>
            <td align="left"><code>/models/{modelID}</code></td>
            <td align="left">single resource from the <code>models</code> collection</td>
          </tr>
          <tr>
            <td align="left"><code>modelId</code></td>
            <td align="left">path parameters</td>
          </tr>
          <tr>
            <td align="left"><code>filter</code></td>
            <td align="left">query parameter</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</div>

### Method

The method is the operation to execute on the specific resource defined by the URI. There are many possible [methods](https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods){:target="_blank"} to choose from, but the four below are the most popular, which are often referred to as [CRUD](https://en.wikipedia.org/wiki/Create,_read,_update_and_delete){:target="_blank"} because they allow you to **C**reate, **R**ead, **U**pdate and **D**elete.

- `#!js GET`: get a resource.
- `#!js POST`: create or update a resource.
- `#!js PUT/PATCH`: create or update a resource.
- `#!js DELETE`: delete a resource.

!!! note
    You could use either the `POST` or `PUT` request method to create and modify resources but the main difference is that `PUT` is idempotent which means you can call the method repeatedly and it'll produce the same state every time. Whereas, calling `POST` multiple times can result in creating multiple instance and so changes the overall state each time.

    ```bash
    POST /models/<new_model> -d {}       # error since we haven't created the `new_model` resource yet
    POST /models -d {}                   # creates a new model based on information provided in data
    POST /models/<existing_model> -d {}  # updates an existing model based on information provided in data

    PUT /models/<new_model> -d {}        # creates a new model based on information provided in data
    PUT /models/<existing_model> -d {}   # updates an existing model based on information provided in data
    ```

We can use [cURL](https://linuxize.com/post/curl-rest-api/){:target="_blank"} to execute our API calls with the following options:

```bash
curl --help
```

<pre class="output">
Usage: curl [options...] <url>
-X, --request  HTTP method (ie. GET)
-H, --header   headers to be sent to the request (ex. authentication)
-d, --data     data to POST, PUT/PATCH, DELETE (usually JSON)
...
</pre>

For example, if we want to GET all `models`, our cURL command would look like this:
```bash
curl -X GET "http://localhost:8000/models"
```

<br>

### Headers

Headers contain information about a certain event and are usually found in both the client's request as well as the server's response. It can range from what type of format they'll send and receive, authentication and caching info, etc.
```bash
curl -X GET "http://localhost:8000/" \          # method and URI
    -H  "accept: application/json"  \           # client accepts JSON
    -H  "Content-Type: application/json" \      # client sends JSON
```

<br>

### Body

The body contains information that may be necessary for the request to be processed. It's usually a JSON object sent during `POST`, `PUT`/`PATCH`, `DELETE` request methods.

```bash
curl -X POST "http://localhost:8000/models" \   # method and URI
    -H  "accept: application/json" \            # client accepts JSON
    -H  "Content-Type: application/json" \      # client sends JSON
    -d "{'name': 'RoBERTa', ...}"               # request body
```

<br>

## Response

The response we receive from our server is the result of the request we sent. The response also includes headers and a body which should include the proper HTTP status code as well as explicit messages, data, etc.

```bash
{
  "message": "OK",
  "method": "GET",
  "status-code": 200,
  "url": "http://localhost:8000/",
  "data": {}
}
```

> We may also want to include other metadata in the response such as model version, datasets used, etc. Anything that the downstream consumer may be interested in or metadata that might be useful for inspection.

There are many [HTTP status codes](https://en.wikipedia.org/wiki/List_of_HTTP_status_codes){:target="_blank"} to choose from depending on the situation but here are the most common options:

<center>

| Code                        | Description                                                               |
| :-------------------------- | :------------------------------------------------------------------------ |
| `200 OK`                    | method operation was successful.                                          |
| `201 CREATED`               | `POST` or `PUT` method successfully created a resource.                   |
| `202 ACCEPTED`              | the request was accepted for processing (but processing may not be done). |
| `400 BAD REQUEST`           | server cannot process the request because of a client side error.         |
| `401 UNAUTHORIZED`          | you're missing required authentication.                                   |
| `403 FORBIDDEN`             | you're not allowed to do this operation.                                  |
| `404 NOT FOUND`             | the resource you're looking for was not found.                            |
| `500 INTERNAL SERVER ERROR` | there was a failure somewhere in the system process.                      |
| `501 NOT IMPLEMENTED`       | this operation on the resource doesn't exist yet.                         |

</center>

## Best practices

When designing our API, there are some best practices to follow:

- URI paths, messages, etc. should be as explicit as possible. Avoid using cryptic resource names, etc.
- Use nouns, instead of verbs, for naming resources. The request method already accounts for the verb (✅&nbsp; `GET /users` not ❌&nbsp; `GET /get_users`).
- Plural nouns (✅&nbsp; `GET /users/{userId}` not ❌&nbsp; `GET /user/{userID}`).
- Use dashes in URIs for resources and path parameters but use underscores for query parameters (`GET /nlp-models/?find_desc=bert`).
- Return appropriate HTTP and informative messages to the user.

## Implementation

We're going to define our API in a separate `app` directory because, in the future, we may have additional packages like `tagifai` and we don't want our app to be attached to any one package. Inside our `app` directory, we'll create the follow scripts:

```bash
mkdir app
cd app
touch api.py gunicorn.py schemas.py
cd ../
```

```bash
app/
├── api.py          - FastAPI app
├── gunicorn.py     - WSGI script
└── schemas.py      - API model schemas
```

- [`api.py`](https://github.com/GokuMohandas/Made-With-ML/tree/main/app/api.py){:target="_blank"}: the main script that will include our API initialization and endpoints.
- [`gunicorn.py`](https://github.com/GokuMohandas/Made-With-ML/tree/main/app/gunicorn.py){:target="_blank"}: script for defining API worker configurations.
- [`schemas.py`](https://github.com/GokuMohandas/Made-With-ML/tree/main/app/schemas.py){:target="_blank"}: definitions for the different objects we'll use in our resource endpoints.

## FastAPI

We're going to use [FastAPI](https://fastapi.tiangolo.com/){:target="_blank"} as our framework to build our API service. There are plenty of other framework options out there such as [Flask](https://flask.palletsprojects.com/){:target="_blank"}, [Django](https://www.djangoproject.com/){:target="_blank"} and even non-Python based options like [Node](https://nodejs.org/en/){:target="_blank"}, [Angular](https://angular.io/){:target="_blank"}, etc. FastAPI combines many of the advantages across these frameworks and is maturing quickly and becoming more widely adopted. It's notable advantages include:

- development in Python
- highly [performant](https://fastapi.tiangolo.com/benchmarks/){:target="_blank"}
- data validation via [pydantic](https://pydantic-docs.helpmanual.io/){:target="_blank"}
- autogenerated documentation
- dependency injection
- security via OAuth2

```bash
pip install fastapi==0.78.0
```

```bash
# Add to requirements.txt
fastapi==0.78.0
```

> Your choice of framework also depends on your team's existing systems and processes. However, with the wide adoption of microservices, we can wrap our specific application in any framework we choose and expose the appropriate resources so all other systems can easily communicate with it.

### Initialization

The first step is to initialize our API in our `api.py` script` by defining metadata like the title, description and version:

```python linenums="1"
# app/api.py
from fastapi import FastAPI

# Define application
app = FastAPI(
    title="TagIfAI - Made With ML",
    description="Classify machine learning projects.",
    version="0.1",
)
```

Our first endpoint is going to be a simple one where we want to show that everything is working as intended. The path for the endpoint will just be `/` (when a user visit our base URI) and it'll be a `GET` request. This simple endpoint is often used as a health check to ensure that our application is indeed up and running properly.

```python linenums="1" hl_lines="4"
# app/api.py
from http import HTTPStatus
from typing import Dict

@app.get("/")
def _index() -> Dict:
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {},
    }
    return response
```

We let our application know that the endpoint is at `/` through the path operation decorator in line 4 and we return a JSON response with the `200 OK` HTTP status code.

> In our actual [`api.py`](https://github.com/GokuMohandas/Made-With-ML/tree/main/app/api.py){:target="_blank"} script, you'll notice that even our index function looks different. Don't worry, we're slowly adding components to our endpoints and justifying them along the way.


### Launching

We're using [Uvicorn](https://www.uvicorn.org/){:target="_blank"}, a fast [ASGI](https://en.wikipedia.org/wiki/Asynchronous_Server_Gateway_Interface){:target="_blank"} server that can run asynchronous code in a single process to launch our application.

```bash
pip install uvicorn==0.17.6
```

```bash
# Add to requirements.txt
uvicorn==0.17.6
```

We can launch our application with the following command:

```bash
uvicorn app.api:app \       # location of app (`app` directory > `api.py` script > `app` object)
    --host 0.0.0.0 \        # localhost
    --port 8000 \           # port 8000
    --reload \              # reload every time we update
    --reload-dir tagifai \  # only reload on updates to `tagifai` directory
    --reload-dir app        # and the `app` directory
```

<pre class="output">
INFO:     Will watch for changes in these directories: ['/Users/goku/Documents/madewithml/mlops/app', '/Users/goku/Documents/madewithml/mlops/tagifai']
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [57609] using statreload
INFO:     Started server process [57611]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
</pre>

> Notice that we only reload on changes to specific directories, as this is to avoid reloading on files that won't impact our application such as log files, etc.

If we want to manage multiple uvicorn workers to enable parallelism in our application, we can use [Gunicorn](https://gunicorn.org/){:target="_blank"} in conjunction with Uvicorn. This will usually be done in a production environment where we'll be dealing with meaningful traffic. We've included a [`app/gunicorn.py`](https://github.com/GokuMohandas/Made-With-ML/tree/main/app/gunicorn.py){:target="_blank"} script with the customizable configuration and we can launch all the workers with the follow command:
```bash
gunicorn -c config/gunicorn.py -k uvicorn.workers.UvicornWorker app.api:app
```

We'll add both of these commands to our `README.md` file as well:
```md
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload --reload-dir tagifai --reload-dir app  # dev
gunicorn -c app/gunicorn.py -k uvicorn.workers.UvicornWorker app.api:app  # prod
```

### Requests

Now that we have our application running, we can submit our `GET` request using several different methods:

- Visit the endpoint on a browser at [http://localhost:8000/](http://localhost:8000/){:target="_blank"}
- cURL
```bash
curl -X GET http://localhost:8000/
```
- Access endpoints via code. Here we show how to do it with the [requests](https://requests.readthedocs.io/en/master/){:target="_blank"} library in Python but it can be done with most popular languages. You can even use an [online tool](https://curl.trillworks.com/){:target="_blank"} to convert your cURL commands into code!
```python linenums="1"
import json
import requests

response = requests.get("http://localhost:8000/")
print (json.loads(response.text))
```
- Using external tools like [Postman](https://www.postman.com/use-cases/application-development/){:target="_blank"}, which is great for managed tests that you can save and share with other, etc.

For all of these, we'll see the exact same response from our API:
<pre class="output">
{
  "message": "OK",
  "status-code": 200,
  "data": {}
}
</pre>


### Decorators
In our `GET \` request's response above, there was not a whole lot of information about the actual request, but it's useful to have details such as URL, timestamp, etc. But we don't want to do this individually for each endpoint, so let's use [decorators](../foundations/python.md#decorators){:target="_blank"} to automatically add relevant metadata to our responses

```python linenums="1" hl_lines="10"
# app/api.py
from datetime import datetime
from functools import wraps
from fastapi import FastAPI, Request

def construct_response(f):
    """Construct a JSON response for an endpoint."""

    @wraps(f)
    def wrap(request: Request, *args, **kwargs) -> Dict:
        results = f(request, *args, **kwargs)
        response = {
            "message": results["message"],
            "method": request.method,
            "status-code": results["status-code"],
            "timestamp": datetime.now().isoformat(),
            "url": request.url._url,
        }
        if "data" in results:
            response["data"] = results["data"]
        return response

    return wrap
```

We're passing in a [Request](https://fastapi.tiangolo.com/advanced/using-request-directly/){:target="_blank"} instance in line 10 so we can access information like the request method and URL. Therefore, our endpoint functions also need to have this Request object as an input argument. Once we receive the results from our endpoint function `f`, we can append the extra details and return a more informative response. To use this decorator, we just have to wrap our functions accordingly.

```python linenums="1" hl_lines="2"
@app.get("/")
@construct_response
def _index(request: Request) -> Dict:
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {},
    }
    return response
```
<pre class="output">
{
    message: "OK",
    method: "GET",
    status-code: 200,
    timestamp: "2021-02-08T13:19:11.343801",
    url: "http://localhost:8000/",
    data: { }
}
</pre>

There are also some built-in decorators we should be aware of. We've already seen the path operation decorator (ex. `@app.get("/")`) which defines the path for the endpoint as well as [other attributes](https://fastapi.tiangolo.com/tutorial/path-operation-configuration/){:target="_blank"}. There is also the [events decorator](https://fastapi.tiangolo.com/advanced/events/){:target="_blank"} (`@app.on_event()`) which we can use to startup and shutdown our application. For example, we use the (`@app.on_event("startup")`) event to load the artifacts for the model to use for inference. The advantage of doing this as an event is that our service won't start until this is complete and so no requests will be prematurely processed and cause errors. Similarly, we can perform shutdown events with (`@app.on_event("shutdown")`), such as saving logs, cleaning, etc.

```python linenums="1" hl_lines="6"
from pathlib import Path
from config import logger
from tagifai import main

@app.on_event("startup")
def load_artifacts():
    global artifacts
    run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
    artifacts = main.load_artifacts(model_dir=config.MODEL_DIR)
    logger.info("Ready for inference!")
```

### Documentation

When we define an endpoint, FastAPI automatically generates some documentation (adhering to [OpenAPI](https://swagger.io/specification/){:target="_blank"} standards) based on the it's inputs, typing, outputs, etc. We can access the [Swagger UI](https://swagger.io/tools/swagger-ui/){:target="_blank"} for our documentation by going to `/docs` endpoints on any browser while the api is running.

<div class="ai-center-all">
    <img width="500" src="/static/images/mlops/api/documentation.png" alt="API documentation">
</div>

Click on an endpoint > `Try it out` > `Execute` to see what the server's response will look like. Since this was a `GET` request without any inputs, our request body was empty but for other method's we'll need to provide some information (we'll illustrate this when we do a `POST` request).

<div class="ai-center-all">
    <img width="600" src="/static/images/mlops/api/execute.png" alt="executing API calls">
</div>

Notice that our endpoint is organized under sections in the UI. We can use `tags` when defining our endpoints in the script:
```python linenums="1" hl_lines="1"
@app.get("/", tags=["General"])
@construct_response
def _index(request: Request) -> Dict:
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {},
    }
    return response
```

> You can also use `/redoc` endpoint to view the [ReDoc](https://redocly.github.io/redoc/){:target="_blank"} documentation or [Postman](https://www.postman.com/use-cases/application-development/){:target="_blank"} to execute and manage tests that you can save and share with others.

## Resources

When designing the resources for our API , we need to think about the following questions:

- `#!js [USERS]`: Who are the end users? This will define what resources need to be exposed.

    - developers who want to interact with the API.
    - product team who wants to test and inspect the model and it's performance.
    - backend service that wants to classify incoming projects.

- `#!js [ACTIONS]`: What actions do our users want to be able to perform?

    - prediction for a given set of inputs
    - inspection of performance
    - inspection of training arguments

### Query parameters

```python linenums="1" hl_lines="3"
@app.get("/performance", tags=["Performance"])
@construct_response
def _performance(request: Request, filter: str = None) -> Dict:
    """Get the performance metrics."""
    performance = artifacts["performance"]
    data = {"performance":performance.get(filter, performance)}
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": data,
    }
    return response
```

Notice that we're passing an optional query parameter `filter` here to indicate the subset of performance we care about. We can include this parameter in our `GET` request like so:

```bash
curl -X "GET" \
  "http://localhost:8000/performance?filter=overall" \
  -H "accept: application/json"
```

And this will only produce the subset of the performance we indicated through the query parameter:

```json
{
  "message": "OK",
  "method": "GET",
  "status-code": 200,
  "timestamp": "2021-03-21T13:12:01.297630",
  "url": "http://localhost:8000/performance?filter=overall",
  "data": {
    "performance": {
      "precision": 0.8941372402587212,
      "recall": 0.8333333333333334,
      "f1": 0.8491658224308651,
      "num_samples": 144
    }
  }
}
```

### Path parameters
Our next endpoint will be to `GET` the arguments used to train the model. This time, we're using a path parameter `args`, which is a **required** field in the URI.

```python linenums="1" hl_lines="1 3"
@app.get("/args/{arg}", tags=["Arguments"])
@construct_response
def _arg(request: Request, arg: str) -> Dict:
    """Get a specific parameter's value used for the run."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {
            arg: vars(artifacts["args"]).get(arg, ""),
        },
    }
    return response
```

We can perform our `GET` request like so, where the `param` is part of the request URI's path as opposed to being part of it's query string.
```bash
curl -X "GET" \
  "http://localhost:8000/args/learning_rate" \
  -H "accept: application/json"
```

And we'd receive a response like this:

```json
{
  "message": "OK",
  "method": "GET",
  "status-code": 200,
  "timestamp": "2021-03-21T13:13:46.696429",
  "url": "http://localhost:8000/params/hidden_dim",
  "data": {
    "learning_rate": 0.14688087680118794
  }
}
```

We can also create an endpoint to produce all the arguments that were used:

??? quote "View `GET /args`"

    ```python linenums="1"
    @app.get("/args", tags=["Arguments"])
    @construct_response
    def _args(request: Request) -> Dict:
        """Get all arguments used for the run."""
        response = {
            "message": HTTPStatus.OK.phrase,
            "status-code": HTTPStatus.OK,
            "data": {
                "args": vars(artifacts["args"]),
            },
        }
        return response
    ```

    We can perform our `GET` request like so, where the `param` is part of the request URI's path as opposed to being part of it's query string.

    ```bash
    curl -X "GET" \
    "http://localhost:8000/args" \
    -H "accept: application/json"
    ```

    And we'd receive a response like this:

    ```json
    {
    "message":"OK",
    "method":"GET",
    "status-code":200,
    "timestamp":"2022-05-25T11:56:37.344762",
    "url":"http://localhost:8001/args",
    "data":{
        "args":{
        "shuffle":true,
        "subset":null,
        "min_freq":75,
        "lower":true,
        "stem":false,
        "analyzer":"char_wb",
        "ngram_max_range":8,
        "alpha":0.0001,
        "learning_rate":0.14688087680118794,
        "power_t":0.158985493618746
        }
      }
    }
    ```

### Schemas

Now it's time to define our endpoint for prediction. We need to consume the inputs that we want to classify and so we need to define the schema that needs to be followed when defining those inputs.

```python
# app/schemas.py
from typing import List
from fastapi import Query
from pydantic import BaseModel

class Text(BaseModel):
    text: str = Query(None, min_length=1)

class PredictPayload(BaseModel):
    texts: List[Text]
```

Here we're defining a `PredictPayload` object as a List of `Text` objects called `texts`. Each `Text` object is a string that defaults to `#!python None` and must have a minimum length of 1 character.

!!! note
    We could've just defined our `PredictPayload` like so:
    ```python linenums="1"
    class PredictPayload(BaseModel):
        texts: List[str] = Query(None, min_length=1)
    ```
    But we wanted to create very explicit schemas in case we want to incorporate more [validation](#validation) or add additional parameters in the future.

We can now use this payload in our predict endpoint:

```python linenums="1" hl_lines="6"
from app.schemas import PredictPayload
from tagifai import predict

@app.post("/predict", tags=["Prediction"])
@construct_response
def _predict(request: Request, payload: PredictPayload) -> Dict:
    """Predict tags for a list of texts."""
    texts = [item.text for item in payload.texts]
    predictions = predict.predict(texts=texts, artifacts=artifacts)
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"predictions": predictions},
    }
    return response
```

We need to adhere to the `PredictPayload` schema when we want to user our `/predict` endpoint:

```bash
curl -X 'POST' 'http://0.0.0.0:8000/predict' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "texts": [
        {"text": "Transfer learning with transformers for text classification."},
        {"text": "Generative adversarial networks for image generation."}
      ]
    }'
```

<pre class="output">
{
  "message":"OK",
  "method":"POST",
  "status-code":200,
  "timestamp":"2022-05-25T12:23:34.381614",
  "url":"http://0.0.0.0:8001/predict",
  "data":{
    "predictions":[
      {
        "input_text":"Transfer learning with transformers for text classification.",
        "predicted_tag":"natural-language-processing"
      },
      {
        "input_text":"Generative adversarial networks for image generation.",
        "predicted_tag":"computer-vision"
      }
    ]
  }
}
</pre>

### Validation

#### Built-in

We're using pydantic's [`BaseModel`](https://pydantic-docs.helpmanual.io/usage/models/){:target="_blank"} object here because it offers built-in validation for all of our schemas. In our case, if a `Text` instance is less than 1 character, then our service will return the appropriate error message and code:

```bash
curl -X POST "http://localhost:8000/predict" -H  "accept: application/json" -H  "Content-Type: application/json" -d "{\"texts\":[{\"text\":\"\"}]}"
```
<pre class="output">
{
  "detail": [
    {
      "loc": [
        "body",
        "texts",
        0,
        "text"
      ],
      "msg": "ensure this value has at least 1 characters",
      "type": "value_error.any_str.min_length",
      "ctx": {
        "limit_value": 1
      }
    }
  ]
}
</pre>

#### Custom

We can also add custom validation on a specific entity by using the `@validator` decorator, like we do to ensure that list of `texts` is not empty.

```python linenums="1" hl_lines="4-8"
class PredictPayload(BaseModel):
    texts: List[Text]

    @validator("texts")
    def list_must_not_be_empty(cls, value):
        if not len(value):
            raise ValueError("List of texts to classify cannot be empty.")
        return value
```

```bash
curl -X POST "http://localhost:8000/predict" -H  "accept: application/json" -H  "Content-Type: application/json" -d "{\"texts\":[]}"
```
<pre class="output">
{
  "detail":[
    {
      "loc":[
        "body",
        "texts"
      ],
      "msg": "List of texts to classify cannot be empty.",
      "type": "value_error"
    }
  ]
}
</pre>

### Extras

Lastly, we can add a [`schema_extra`](https://fastapi.tiangolo.com/tutorial/schema-extra-example/){:target="_blank"} object under a `Config` class to depict what an example `PredictPayload` should look like. When we do this, it automatically appears in our endpoint's documentation (click `Try it out`).

```python linenums="1" hl_lines="10-18"
class PredictPayload(BaseModel):
    texts: List[Text]

    @validator("texts")
    def list_must_not_be_empty(cls, value):
        if not len(value):
            raise ValueError("List of texts to classify cannot be empty.")
        return value

    class Config:
        schema_extra = {
            "example": {
                "texts": [
                    {"text": "Transfer learning with transformers for text classification."},
                    {"text": "Generative adversarial networks in both PyTorch and TensorFlow."},
                ]
            }
        }
```

<div class="ai-center-all">
    <img width="1000" src="/static/images/mlops/api/predict.png" alt="inference with APIs">
</div>


## Product

To make our API a standalone product, we'll need to create and manage a database for our users and resources. These users will have credentials which they will use for authentication and use their privileges to be able to communicate with our service. And of course, we can display a rendered frontend to make all of this seamless with HTML forms, buttons, etc. This is exactly how the [old MWML platform](https://twitter.com/madewithml/status/1284503478685978625){:target="_blank"} was built and we leveraged FastAPI to deliver high performance for 500K+ daily service requests.

If you are building a product, then I highly recommending forking this [generation template](https://fastapi.tiangolo.com/project-generation/){:target="_blank"} to get started. It includes the backbone architecture you need for your product:

- Databases (models, migrations, etc.)
- Authentication via JWT
- Asynchronous task queue with Celery
- Customizable frontend via Vue JS
- Docker integration
- so much more!

However, for the majority of ML developers, thanks to the wide adoption of microservices, we don't need to do all of this. A well designed API service that can seamlessly communicate with all other services (framework agnostic) will fit into any process and add value to the overall product. Our main focus should be to ensure that our service is working as it should and constantly improve, which is exactly what the next cluster of lessons will focus on ([testing](testing.md){:target="_blank"} and [monitoring](monitoring.md){:target="_blank"})

## Model server

Besides wrapping our models as separate, scalable microservices, we can also have a purpose-built model server to host our models. Model servers provide a registry with an API layer to seamlessly inspect, update, serve, rollback, etc. multiple versions of models. They also offer automatic scaling to meet throughput and latency needs. Popular options include [BentoML](https://www.bentoml.com/){:target="_blank"}, [MLFlow](https://docs.databricks.com/applications/mlflow/model-serving.html){:target="_blank"}, [TorchServe](https://pytorch.org/serve/){:target="_blank"}, [RedisAI](https://oss.redislabs.com/redisai/){:target="_blank"}, [Nvidia Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server){:target="_blank"}, etc.

> Model servers are experiencing a lot of adoption for their ability to standardize the model deployment and serving processes across the team -- enabling seamless upgrades, validation and integration.

<!-- Course signup -->
{% include "templates/signup.md" %}

<!-- Citation -->
{% include "templates/cite.md" %}