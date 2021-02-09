---
description: Using first principles to designing and implement a RESTful API to wrap ML functionality.
image: https://madewithml.com/static/images/applied_ml.png
---

:octicons-mark-github-16: [Repository](https://github.com/GokuMohandas/applied-ml){:target="_blank"}

Using first principles to designing and implement a RESTful API to wrap ML functionality.

## Intuition

So far our workflows have involved directly running functions from our Python scripts and more recently, using the [CLI application](cli.md){:target="_blank"} to quickly execute commands. But not all of our users will want to work at the code level or even download the package as we would need to for the CLI app. Instead, many users will simply want to use the functionality of our model and inspect the relevant details around it. To address this, we can develop an application programming interface (API) that provides the appropriate level of abstraction that enables our users to interact with the underlying data in our application.

### Interactions

The interactions in our situation involve the client (users, other applications, etc.) sending a *request* to the server (our application) and receiving a *response* in return.

<div class="ai-center-all">
    <img width="500" src="https://raw.githubusercontent.com/GokuMohandas/madewithml/main/images/applied-ml/api/interactions.png">
</div>

#### Request

We'll first take a look at the different components of a request:

##### URI

A uniform resource identifier (URI) is an identifier for a specific resource.

<pre class="output" style="padding-left: 0rem; padding-right: 0rem; font-weight: 600;">
<span style="color:#d63939">https://</span><span style="color:#206bc4">localhost:</span><span style="color: #4299e1">5000</span><span style="color:#2fb344">/users/{userId}/models/{modelId}/</span><span style="color:#ae3ec9">?filter=completed</span><span style="color:#f76707">#details</span>
</pre>

<!-- <center>

| Parts of the URI                                    | Description  |
| :-------------------------------------------------- | :----------- |
| <span style="color: #d63939;">scheme</span>         | defines which protocol to use.                                                     |
| <span style="color: #206bc4;">domain</span>         | address of your website.                                                           |
| <span style="color: #4299e1;">port</span>           | communication endpoint. If not define, it's usually 80 for HTTP and 443 for HTTPS. |
| <span style="color: #2fb344;">path</span>           | location of the resource of interest.                                              |
| <span style="color: #ae3ec9;">query string</span>   | data sent to endpoint to identify specific resources.                              |
| <span style="color: #f76707;">anchor</span>         | specific location inside an HTML page.                                             |

| Parts of the path       | Description                                                       |
| :---------------------- | :---------------------------------------------------------------- |
| `/users`                | collection resource of all `users`                                |
| `/users/{userID}`       | single resource for a specific user `userId`                      |
| `/models`               | sub-collection resource `models` for the specific user `userID`   |
| `/models/{modelID}`     | single resource for the `userID`'s `models` sub-collection        |
| `userID` and `modelId`  | path parameters                                                   |
| `filter` and `lang`     | query parameters                                                  |

</center> -->

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
            <td align="left">defines which protocol to use.</td>
          </tr>
          <tr>
            <td align="left"><span style="color: #206bc4;">domain</span></td>
            <td align="left">address of the website.</td>
          </tr>
          <tr>
            <td align="left"><span style="color: #4299e1;">port</span></td>
            <td align="left">communication endpoint. If not defined, it's usually 80 for HTTP and 443 for HTTPS.</td>
          </tr>
          <tr>
            <td align="left"><span style="color: #2fb344;">path</span></td>
            <td align="left">location of the resource of interest.</td>
          </tr>
          <tr>
            <td align="left"><span style="color: #ae3ec9;">query string</span></td>
            <td align="left">parameters sent to endpoint to identify specific resources.</td>
          </tr>
          <tr>
            <td align="left"><span style="color: #f76707;">anchor</span></td>
            <td align="left">specific location inside an HTML page.</td>
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
            <td align="left"><code>/users</code></td>
            <td align="left">collection resource of all <code>users</code></td>
          </tr>
          <tr>
            <td align="left"><code>/users/{userID}</code></td>
            <td align="left">single resource for a specific user <code>userId</code></td>
          </tr>
          <tr>
            <td align="left"><code>/models</code></td>
            <td align="left">sub-collection resource <code>models</code> for the specific user <code>userID</code></td>
          </tr>
          <tr>
            <td align="left"><code>/models/{modelID}</code></td>
            <td align="left">single resource for the <code>userID</code>'s <code>models</code> sub-collection</td>
          </tr>
          <tr>
            <td align="left"><code>userID</code> and <code>modelId</code></td>
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

##### Method

The method is the operation to execute on the specific resource defined by the URI. There are many possible [methods](https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods){:target="_blank"} to choose from, but here are the four most popular, which are often referred to as [CRUD](https://en.wikipedia.org/wiki/Create,_read,_update_and_delete){:target="_blank"} because they allow you to Create, Read, Update and Delete.

- `GET`: get a resource
- `POST`: create or update a resource
- `PUT/PATCH`: create or update a resource
- `DELETE`: delete a resource

!!! note
    You could use either the `POST` or `PUT` request method to create and modify resources but the main difference is that `PUT` is idempotent which means you can call the method repeatedly and it'll produce the same state every time. Whereas, calling `POST` multiple times can result in creating multiple instance and so changing the overall state each time.

    ```bash
    POST /models/<new_model> -d {}       # error since we haven't created the `new_model` resource yet
    POST /models -d {}                   # creates a new model based on information provided in data
    POST /models/<existing_model> -d {}  # updates an existing model based on information provided in data

    PUT /models/<new_model> -d {}        # creates a new model based on information provided in data
    PUT /models/<existing_model> -d {}   # updates an existing model based on information provided in data
    ```

We can use [cURL](https://linuxize.com/post/curl-rest-api/){:target="_blank"} to execute our API calls with the following options:

<div class="animated-code">

    ```console
    # cURL options
    $ curl --help
    ...
    -X, --request  HTTP method (ie. GET)
    -H, --header   headers to be sent to the request (ex. authentication)
    -d, --data     data to POST, PUT/PATCH, DELETE (usually JSON)
    ...
    ```

</div>
<script src="../../../static/js/termynal.js"></script>

For example, if we want to GET all `users`:
```bash
curl -X GET "http://localhost:5000/users"
```

<br>

##### Headers

Headers contain information about a certain event and are usually found in both the client's request as well as the server's response. It can range from what type of format they'll send and receive, authentication and caching info, etc.
```bash
curl -X GET "http://localhost:5000/" \          # method and URI
    -H  "accept: application/json"  \           # client accepts JSON
    -H  "Content-Type: application/json" \      # client sends JSON
```

<br>

##### Body

The body contains information that may be necessary for the request to be processed. It's usually a JSON object sent during `POST`, `PUT`/`PATCH`, `DELETE` request methods.

```bash
curl -X POST "http://localhost:5000/models" \   # method and URI
    -H  "accept: application/json" \            # client accepts JSON
    -H  "Content-Type: application/json" \      # client sends JSON
    -d "{'name': 'RoBERTa', ...}"               # request body
```

<br>

#### Response

The response we receive from our server is the result of the request we sent. The response also includes headers and a body which should include the proper HTTP status code as well as explicit messages, data, etc.

```bash
{
  "message": "OK",
  "method": "GET",
  "status-code": 200,
  "url": "http://localhost:5000/",
  "data": {}
}
```

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

### Best practices

When designing our API, there are some best practices to follow:

- URI paths, messages, etc. should be as explicit as possible. Avoid using cryptic resource names, etc.
- Nouns not verbs when naming. The request method already takes care of the verb (`GET /users not` `GET /get_users`).
- Plural nouns (`GET /users/{userId}` not `GET /user/{userID}`).
- Dashes in URIs for resources and path parameters but use underscores for query parameters (`GET /admin-users/?find_desc=super`).
- Return appropriate HTTP and informative messages to the user.

## Application

We're going to organize our API under the [`app`](https://github.com/GokuMohandas/applied-ml/tree/main/app){:target="_blank"} directory because in the future we may have additional packages like `tagifai` so we don't want our app to be attached to any one package. Our API will be defined in the following scripts:

- [`api.py`](https://github.com/GokuMohandas/applied-ml/tree/main/app/api.py){:target="_blank"}: the main script that will include our API initialization and endpoints.
- [`schemas.py`](https://github.com/GokuMohandas/applied-ml/tree/main/app/schemas.py){:target="_blank"}: definitions for the different objects we'll use in our resource endpoints.

We'll step through the components in these scripts to show how we'll design our API.

### FastAPI

We're going to use [FastAPI](https://fastapi.tiangolo.com/){:target="_blank"} as our framework to build our API service. There are plenty of other framework options out there such as Flask, Django and even non-Python based options like Node, Angular, etc. FastAPI is a relative newcomer that combines many of the advantages across these frameworks and is maturing quickly and becoming more widely adopted. It's notable advantages include:

- highly [performant](https://fastapi.tiangolo.com/benchmarks/){:target="_blank"}
- data validation via [pydantic](https://pydantic-docs.helpmanual.io/){:target="_blank"}
- autogenerated documentation
- dependency injection
- security via OAuth2

!!! note
    Your choice of framework also depends on your team's existing systems and processes. However, with the wide adoption of microservices, we can wrap our specific application is any framework we choose and expose the appropriate resources so all other systems can easily communicate with it.

To show how intuitive and powerful FastAPI is, we could laboriously go through the [documentation](https://fastapi.tiangolo.com/){:target="_blank"} but instead we'll walk through everything as we cover the components of our own application.

### Initialization

The first step is to initialize our API in our [`app/api.py`](https://github.com/GokuMohandas/applied-ml/tree/main/app/api.py){:target="_blank"} file by defining metadata like the title, description and version.
```python linenums="1"
from fastapi import FastAPI

# Define application
app = FastAPI(
    title="TagIfAI - Made With ML",
    description="Predict relevant tags given a text input.",
    version="0.1",
)
```

Our first endpoint is going to be a simple one where we want to show that everything is working as intended. The path for the endpoint will just be `/` (when a user visit our base URI) and it'll be a `GET` request. This simple endpoint definition is often used as a health check to ensure that our application is indeed up and running properly.
```python linenums="1" hl_lines="3"
from http import HTTPStatus

@app.get("/")
def _index():
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {},
    }
    return response
```

We let our application know that the endpoint is at `/` through the path operation decorator in line 3 and we simply return a JSON response with the `200 OK` HTTP status code. Let's go ahead and start our application and see what this response looks like!

!!! note
    In our actual [`api.py`](https://github.com/GokuMohandas/applied-ml/tree/main/app/api.py){:target="_blank"} script, you'll notice that even our index function looks different. Don't worry, we're slowly adding components to our endpoints and justifying them along the way.


### Launching

We can launch our application with the following command (also saved as a Makefile target as `make app`):

```bash
uvicorn app.api:app \       # location of app (`app` directory > `api.py` script > `app` object)
    --host 0.0.0.0 \        # localhost
    --port 5000 \           # port 5000
    --reload \              # reload every time we update
    --reload-dir tagifai \  # only reload on updates to `tagifai` directory
    --reload-dir app        # and the `app` directory
```

We're using [Uvicorn](https://www.uvicorn.org/){:target="_blank"}, a fast ASGI server (it can run asynchronous code in a single process) to launch our application. Notice that we only reload on changes to specific directories, as this is to avoid reloading on files that won't impact our application such as log files, etc.

!!! note
    If we want to manage multiple uvicorn workers to enable parallelism in our application, we can use [Gunicorn](https://gunicorn.org/){:target="_blank"} in conjunction with Uvicorn. This will usually be done in a production environment where we'll be dealing with meaningful traffic. I've included a [`config/gunicorn.py`](https://github.com/GokuMohandas/applied-ml/tree/main/config/gunicorn.py){:target="_blank"} script with the customizable configuration and we can launch all the workers with the follow command (or `make app-prod`):
    ```bash
    gunicorn -c config/gunicorn.py -k uvicorn.workers.UvicornWorker app.api:app
    ```

### Requests

Now that we have our application running, we can submit our `GET` request using several different methods:

- Visit the endpoint on your browser at [http://localhost:5000/](http://localhost:5000/){:target="_blank"}
- cURL
```bash
curl -X GET http://localhost:5000/
```
- Access endpoints via code. Here we show how to do it with the [requests](https://requests.readthedocs.io/en/master/){:target="_blank"} library in Python but it can be done with most popular languages. You can even use an [online tool](https://curl.trillworks.com/){:target="_blank"} to convert your cURL commands into code!
```python linenums="1"
import json
import requests

response = requests.get('http://localhost:5000/')
print (json.loads(response.text))
```
- Directly in the API's autogenerated documentation (which we'll see later).
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
We're going to use [decorators](../ml-foundations/python.md#decorators){:target="_blank"} to wrap some of our endpoints so we can customize our function's inputs and outputs. In our `GET \` request's response above, there was not a whole lot of information about the actual request, so we should append details such as URL, timestamp, etc. But we don't want to do this individually for each endpoint so let's use a decorator to append the request information for every response.

```python linenums="1" hl_lines="6"
def construct_response(f):
    """Construct a JSON response for an endpoint's results."""

    @wraps(f)
    def wrap(request: Request, *args, **kwargs):
        results = f(request, *args, **kwargs)

        # Construct response
        response = {
            "message": results["message"],
            "method": request.method,
            "status-code": results["status-code"],
            "timestamp": datetime.now().isoformat(),
            "url": request.url._url,
        }

        # Add data
        if "data" in results:
            response["data"] = results["data"]

        return response

    return wrap
```

We're passing in a [Request](https://fastapi.tiangolo.com/advanced/using-request-directly/){:target="_blank"} instance in line 6 so we can access information like the request method and URL. Therefore, our endpoint functions also need to have this Request object as an input argument. Once we receive the results from our endpoint function `f`, we can append the extra details and return more informative `response`. To use this decorator, we just have to wrap our functions accordingly.

```python linenums="1"
@app.get("/")
@construct_response
def _index(request: Request):
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
    url: "http://localhost:5000/",
    data: { }
}
</pre>

We also have another decorator in our api script called `validate_run_id` which wraps around relevant functions where a `run_id` argument is passed in. This decorator validates that the given `run_id` in the client's request is indeed valid before proceeding with it's operations. We'll see this decorator in action soon.

There are also some built-in decorators we should be aware of. We've already seen the path operation decorator (ex. `@app.get("/")`) which defines the path for the endpoint as well as [other attributes](https://fastapi.tiangolo.com/tutorial/path-operation-configuration/){:target="_blank"}. There is also the [events decorator](https://fastapi.tiangolo.com/advanced/events/){:target="_blank"} (`@app.on_event()`) which we can use to startup and shutdown our application. For example, we use the `startup` event to load all the previous best runs and identify the best run (and model) to use for inference. The advantage of doing this as an Event is that our service won't start until this is ready so no requests will be prematurely processed and error out.

```python linenums="1" hl_lines="1"
@app.on_event("startup")
def load_best_artifacts():
    global runs, run_ids, best_artifacts, best_run_id
    runs = utils.get_sorted_runs(
        experiment_name="best", order_by=["metrics.f1 DESC"]
    )
    run_ids = [run["run_id"] for run in runs]
    best_run_id = run_ids[0]
    best_artifacts = predict.load_artifacts(run_id=best_run_id)
```


### Documentation

When we define an endpoint, FastAPI automatically generates some documentation, adhering to [OpenAPI](https://swagger.io/specification/){:target="_blank"} standards, based on the function's inputs, typing, outputs, etc. We can access the [Swagger UI](https://swagger.io/tools/swagger-ui/){:target="_blank"} for our documentation by going to `/docs` endpoints on any browser.

<div class="ai-center-all">
    <img width="400" src="https://raw.githubusercontent.com/GokuMohandas/madewithml/main/images/applied-ml/api/documentation.png">
</div>

Click on an endpoint > `Try it out` > `Execute` to see what the server's response will look like. Since this was a `GET` request without any inputs, our request body was empty but for other method's we'll need to provide some information (we'll illustrate this when we do a `POST` request).

<div class="ai-center-all">
    <img width="450" src="https://raw.githubusercontent.com/GokuMohandas/madewithml/main/images/applied-ml/api/execute.png">
</div>

You'll notice that our endpoints are organized under sections in the UI. This is because we used `tags` when defining our endpoints in the script.
```python linenums="1" hl_lines="1"
@app.get("/", tags=["General"])
@construct_response
def _index(request: Request):
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {},
    }
    return response
```

!!! note
    You can also use `/redoc` endpoint to view the [ReDoc](https://redocly.github.io/redoc/){:target="_blank"} documentation or [Postman](https://www.postman.com/use-cases/application-development/){:target="_blank"} to execute and manage tests that you can save and share with others.

### Resources

When designing the resources for our API service, we need to think about the following questions:

- Who are the users? This will define what resources need to be exposed.
> Our users include anyone will want to receive the relevant tags for a given input. They may not necessarily be technical or aware of how machine learning works.
- What functionality do we want to enable our users with?
> Though there are many different processes we *could* enable for our users (optimize, train, delete, update models, etc.), we're going to scope our service by only enabling prediction at this time. However, our Python scripts and the CLI application if anyone wants to be able to do more (ie. train a model).
- What are the objects (or entities) that we'll need to build and expose resources around?
> As our data grows, we'll be optimizing and saving the best runs. Therefore, it's logical to expose all the best previous runs (in descending order by performance) as well as enabling actions on each run resource (ie. prediction).

Our first resource endpoint will be to `GET` all the runs. Recall that we've already loaded all the `runs` (sorted) using the `load_best_artifacts` function wrapped by the startup Event decorator (`@app.on_event("startup")`):

```python linenums="1" hl_lines="4-6"
@app.on_event("startup")
def load_best_artifacts():
    global runs, run_ids, best_artifacts, best_run_id
    runs = utils.get_sorted_runs(
        experiment_name="best", order_by=["metrics.f1 DESC"]
    )
    ...
```

#### Query parameters

```python linenums="1" hl_lines="3"
@app.get("/runs", tags=["Runs"])
@construct_response
def _runs(request: Request, top: Optional[int] = None) -> Dict:
    """Get all runs sorted by f1 score."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"runs": runs[:top]},
    }
    return response
```
You'll notice that we're passing an optional query parameter `top` here to indicate the top N runs to retrieve (None will return all runs). We'd include this parameter in our `GET` request like so:

```bash
curl -X GET "http://localhost:5000/runs?top=10" -H  "accept: application/json"
```

#### Path parameters
Our next endpoint will be to `GET` information about a specific run (ex. granular performance) based on it's `run_id`. This time, we're using a path parameter which is a required field in the URI. This `run_id` will be passed to the function after it's validated through the `validate_run_id` function.

```python linenums="1" hl_lines="1 4"
@app.get("/runs/{run_id}", tags=["Runs"])
@construct_response
@validate_run_id
def _run(request: Request, run_id: str) -> Dict:
    """Get details about a specific run."""
    artifacts = predict.load_artifacts(run_id=run_id)
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"run_id": run_id, "performance": artifacts["performance"]},
    }
    return response
```

We'd perform our `GET` request like so, where the `run_id` is part of the request URI's path as opposed to being part of it's query string.
```bash
curl -X GET "http://localhost:5000/runs/264ac530b78c42608e5dea1086bc2c73" -H  "accept: application/json"
```

#### Schemas

Users can list all runs, find out more information about a specific run and now we want to enable them to get predictions on some input text from any of these runs.
```python linenums="1" hl_lines="1 4"
@app.post("/runs/{run_id}/predict", tags=["Runs"])
@construct_response
@validate_run_id
def _predict(request: Request, run_id: str, payload: PredictPayload) -> Dict:
    """Predict tags for a list of texts using artifacts from run `run_id`."""
    artifacts = predict.load_artifacts(run_id=run_id)
    texts = [item.text for item in payload.texts]
    predictions = predict.predict(texts=texts, artifacts=artifacts)
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"run_id": run_id, "predictions": predictions},
    }
    return response
```

Like before, we'll consume the `run_id` as a path parameter and validate it. But this time, we're also receiving a payload from the request's body which contains information as to what to predict on. The definition of this `PredictionPayload` is defined in our [`app/schemas.py`](https://github.com/GokuMohandas/applied-ml/tree/main/app/schemas.py){:target="_blank"} script:
```python linenums="1" hl_lines="12"
from typing import List

from fastapi import Query
from pydantic import BaseModel, validator


class Text(BaseModel):
    text: str = Query(None, min_length=1)


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
                    {"text": "Transfer learning with transformers for self-supervised learning."},
                    {"text": "Generative adversarial networks in both PyTorch and TensorFlow."},
                ]
            }
        }
```

In line 12, we're defining the `PredictPayload` object as a list of `Text` objects called `texts`. Each `Text` object is a string that defaults to `None` and must have a minimum length of 1 character.

!!! note
    We could've just defined our `PredictPayload` like so:
    ```python linenums="1" hl_lines="2"
    class PredictPayload(BaseModel):
        texts: List[str] = Query(None, min_length=1)
    ```
    But we wanted to create very explicit schemas in case we add different parameters in the future and to show how to apply granular validation to them.


#### Validation

##### Built-in

We're using pydantic's [`BaseModel`](pydantic BaseModel: https://pydantic-docs.helpmanual.io/usage/models/){:target="_blank"} object here because it offers built-in validation for all of our schemas. In our case, if a `Text` instance is less than 1 character, then our service will return the appropriate error message and code.

```bash
curl -X POST "http://localhost:5000/predict" -H  "accept: application/json" -H  "Content-Type: application/json" -d "{\"texts\":[{\"text\":\"\"}]}"
```
<pre class="output">
# 422 Error: Unprocessable Entity
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

##### Custom

We can also add custom validation on a specific entity by using the `@validator` decorator, like we do to ensure that list of `texts` is not empty.

```python linenums="1" hl_lines="4"
class PredictPayload(BaseModel):
    texts: List[Text]

    @validator("texts")
    def list_must_not_be_empty(cls, value):
        if not len(value):
            raise ValueError("List of texts to classify cannot be empty.")
        return value

    ...
```

```bash
curl -X POST "http://localhost:5000/predict" -H  "accept: application/json" -H  "Content-Type: application/json" -d "{\"texts\":[]}"
```
<pre class="output">
{
  "detail": [
    {
      "loc": [
        "body",
        "texts"
      ],
      "msg": "List of texts to classify cannot be empty.",
      "type": "value_error"
    }
  ]
}
</pre>

#### Extras

Lastly, we have a [`schema_extra`](https://fastapi.tiangolo.com/tutorial/schema-extra-example/){:target="_blank"} object under the `Config` class to depict what an example `PredictPayload` should look like. When we do this, it automatically appears in our endpoint's documentation when we want to "Try it out".

<div class="ai-center-all">
    <img width="1000" src="https://raw.githubusercontent.com/GokuMohandas/madewithml/main/images/applied-ml/api/predict.png">
</div>


!!! note
    We've also implemented an additional endpoint to allow our users to simply `POST /predict` with the appropriate payload without specifying any `run_id` so they don't have to worry about `runs` or other resources.
    ```python linenums="1"
    @app.post("/predict", tags=["General"])
    @construct_response
    def _best_predict(request: Request, payload: PredictPayload) -> Dict:
        """Predict tags for a list of texts using the best run. """
        # Predict
        texts = [item.text for item in payload.texts]
        predictions = predict.predict(texts=texts, artifacts=best_artifacts)
        response = {
            "message": HTTPStatus.OK.phrase,
            "status-code": HTTPStatus.OK,
            "data": {"run_id": best_run_id, "predictions": predictions},
        }
        return response
    ```


### Projects

To make our API a standalone product, we'll need to create and manage a database for our users and resources. These users will have credentials which they will use for authentication and use their privileges to be able to communicate with our service. And of course, we can display a rendered frontend to make all of this seamless with HTML forms, buttons, etc. This is exactly how the [old MWML platform](https://twitter.com/madewithml/status/1284503478685978625){:target="_blank"} was built and we leveraged FastAPI to deliver high performance for 500K+ daily service requests.

If you are building a product, then I highly recommending forking this [generation template](https://fastapi.tiangolo.com/project-generation/){:target="_blank"} to get started. It includes the backbone architecture you need for your product:

- Databases (models, migrations, etc.)
- Authentication via JWT
- Asynchronous task queue with Celery
- Customizable frontend via Vue JS
- Docker integration
- so much more!

However, for the majority of ML developers, thanks to the wide adoption of microservices, we don't need to do all of this. A well designed API service that can seamlessly communicate with all other services (framework agnostic) will fit into any process and add value to the overall product. Our main focus should be to ensure that our service is working as it should and constantly improve, which is exactly what the next cluster of lessons will focus on (testing, monitoring, serving, etc.)


!!! note
    We've only covered the foundations of using FastAPI but there's so much more we can do. Be sure to check out their [advanced documentation](https://fastapi.tiangolo.com/advanced/){:target="_blank"} to see everything we can leverage.
