---
template: lesson.html
title: Logging for ML Systems
description: Keep records of the important events in our application.
keywords: logging, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/mlops.png
repository: https://github.com/GokuMohandas/mlops-course
---

{% include "styles/lesson.md" %}

## Intuition

Logging is the process of tracking and recording key events that occur in our applications for the purpose of inspection, debugging, etc. They're a whole lot more powerful than `print` statements because they allow us to send specific pieces of information to specific locations with custom formatting, shared interfaces, etc. This makes logging a key proponent in being able to surface insightful information from the internal processes of our application.

## Components

There are a few overarching concepts to be aware of:

- `#!js Logger`: emits the log messages from our application.
- `#!js Handler`: sends log records to a specific location.
- `#!js Formatter`: formats and styles the log records.

There is so much [more](https://docs.python.org/3/library/logging.html){:target="_blank"} to logging such as filters, exception logging, etc. but these basics will allows us to do everything we need for our application.

## Levels

Before we create our specialized, configured logger, let's look at what logged messages look like by using the basic configuration.
```python linenums="1"
import logging
import sys

# Create super basic logger
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# Logging levels (from lowest to highest priority)
logging.debug("Used for debugging your code.")
logging.info("Informative messages from your code.")
logging.warning("Everything works but there is something to be aware of.")
logging.error("There's been a mistake with the process.")
logging.critical("There is something terribly wrong and process may terminate.")
```
<pre class="output">
DEBUG:root:Used for debugging your code.
INFO:root:Informative messages from your code.
WARNING:root:Everything works but there is something to be aware of.
ERROR:root:There's been a mistake with the process.
CRITICAL:root:There is something terribly wrong and process may terminate.
</pre>

These are the basic [levels](https://docs.python.org/3/library/logging.html#logging-levels){:target="_blank"} of logging, where `DEBUG` is the lowest priority and `CRITICAL` is the highest. We defined our logger using [`basicConfig`](https://docs.python.org/3/library/logging.html#logging.basicConfig){:target="_blank"} to emit log messages to stdout (ie. our terminal console), but we also could've written to any other stream or even a file. We also defined our logging to be sensitive to log messages starting from level `DEBUG`. This means that all of our logged messages will be displayed since `DEBUG` is the lowest level. Had we made the level `ERROR`, then only `ERROR` and `CRITICAL` log message would be displayed.

```python linenums="1" hl_lines="5"
import logging
import sys

# Create super basic logger
logging.basicConfig(stream=sys.stdout, level=logging.ERROR)

# Logging levels (from lowest to highest priority)
logging.debug("Used for debugging your code.")
logging.info("Informative messages from your code.")
logging.warning("Everything works but there is something to be aware of.")
logging.error("There's been a mistake with the process.")
logging.critical("There is something terribly wrong and process may terminate.")
```
<pre class="output">
ERROR:root:There's been a mistake with the process.
CRITICAL:root:There is something terribly wrong and process may terminate.
</pre>

## Configuration

First we'll set the location of our logs in our `config.py` script:

``` python linenums="1"
# config/config.py
LOGS_DIR = Path(BASE_DIR, "logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)
```

Next, we'll configure the logger for our application:

```python linenums="1"
# config/config.py
import logging
import sys
logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "minimal": {"format": "%(message)s"},
        "detailed": {
            "format": "%(levelname)s %(asctime)s [%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
            "formatter": "minimal",
            "level": logging.DEBUG,
        },
        "info": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "info.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.INFO,
        },
        "error": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "error.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.ERROR,
        },
    },
    "root": {
        "handlers": ["console", "info", "error"],
        "level": logging.INFO,
        "propagate": True,
    },
}
```

1. `[Lines 6-11]`: define two different [Formatters](https://docs.python.org/3/library/logging.html#formatter-objects){:target="_blank"} (determine format and style of log messages), minimal and detailed, which use various [LogRecord attributes](https://docs.python.org/3/library/logging.html#logrecord-attributes){:target="_blank"} to create a formatting template for log messages.
2. `[Lines 12-35]`: define the different [Handlers](https://docs.python.org/3/library/logging.html#handler-objects){:target="_blank"} (details about location of where to send log messages):
    - `#!js console`: sends log messages (using the `minimal` formatter) to the `stdout` stream for messages above level `DEBUG` (ie. all logged messages).
    - `#!js info`: send log messages (using the `detailed` formatter) to `logs/info.log` (a file that can be up to `1 MB` and we'll backup the last `10` versions of it) for messages above level `INFO`.
    - `#!js error`: send log messages (using the `detailed` formatter) to `logs/error.log` (a file that can be up to `1 MB` and we'll backup the last `10` versions of it) for messages above level `ERROR`.
3. `[Lines 36-40]`: attach our different handlers to our root [Logger](https://docs.python.org/3/library/logging.html#logger-objects){:target="_blank"}.



We chose to use a dictionary to configure our logger but there are other ways such as Python script, configuration file, etc. Click on the different options below to expand and view the respective implementation.

??? quote "Python script"

    ```python linenums="1"
    import logging
    from rich.logging import RichHandler

    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Create handlers
    console_handler = RichHandler(markup=True)
    console_handler.setLevel(logging.DEBUG)
    info_handler = logging.handlers.RotatingFileHandler(
        filename=Path(LOGS_DIR, "info.log"),
        maxBytes=10485760,  # 1 MB
        backupCount=10,
    )
    info_handler.setLevel(logging.INFO)
    error_handler = logging.handlers.RotatingFileHandler(
        filename=Path(LOGS_DIR, "error.log"),
        maxBytes=10485760,  # 1 MB
        backupCount=10,
    )
    error_handler.setLevel(logging.ERROR)

    # Create formatters
    minimal_formatter = logging.Formatter(fmt="%(message)s")
    detailed_formatter = logging.Formatter(
        fmt="%(levelname)s %(asctime)s [%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
    )

    # Hook it all up
    console_handler.setFormatter(fmt=minimal_formatter)
    info_handler.setFormatter(fmt=detailed_formatter)
    error_handler.setFormatter(fmt=detailed_formatter)
    logger.addHandler(hdlr=console_handler)
    logger.addHandler(hdlr=info_handler)
    logger.addHandler(hdlr=error_handler)
    ```

??? quote "Configuration file"

    1. Place this inside a `logging.config` file:
    ```
    [formatters]
    keys=minimal,detailed

    [formatter_minimal]
    format=%(message)s

    [formatter_detailed]
    format=
        %(levelname)s %(asctime)s [%(name)s:%(filename)s:%(funcName)s:%(lineno)d]
        %(message)s

    [handlers]
    keys=console,info,error

    [handler_console]
    class=StreamHandler
    level=DEBUG
    formatter=minimal
    args=(sys.stdout,)

    [handler_info]
    class=handlers.RotatingFileHandler
    level=INFO
    formatter=detailed
    backupCount=10
    maxBytes=10485760
    args=("logs/info.log",)

    [handler_error]
    class=handlers.RotatingFileHandler
    level=ERROR
    formatter=detailed
    backupCount=10
    maxBytes=10485760
    args=("logs/error.log",)

    [loggers]
    keys=root

    [logger_root]
    level=INFO
    handlers=console,info,error
    ```

    2. Place this inside your Python script:
    ```python linenums="1"
    import logging
    import logging.config
    from rich.logging import RichHandler

    # Use config file to initialize logger
    logging.config.fileConfig(Path(CONFIG_DIR, "logging.config"))
    logger = logging.getLogger()
    logger.handlers[0] = RichHandler(markup=True)  # set rich handler
    ```

We can load our logger configuration dict like so:

```python linenums="1"
# config/config.py
from rich.logging import RichHandler
logging.config.dictConfig(logging_config)
logger = logging.getLogger()
logger.handlers[0] = RichHandler(markup=True)  # pretty formatting

# Sample messages (note that we use configured `logger` now)
logger.debug("Used for debugging your code.")
logger.info("Informative messages from your code.")
logger.warning("Everything works but there is something to be aware of.")
logger.error("There's been a mistake with the process.")
logger.critical("There is something terribly wrong and process may terminate.")
```
<pre style="font-size: 0.7rem;">
<span style="color: #39BC70;">DEBUG</span>    Used for debugging your code.                                 <span style="color: #A2A2A2;">config.py:71</span>
<span style="color: #2871CF;">INFO</span>     Informative messages from your code.                          <span style="color: #A2A2A2;">config.py:72</span>
<span style="color: #BF1825;">WARNING</span>  Everything works but there is something to be aware of.       <span style="color: #A2A2A2;">config.py:73</span>
<span style="color: #F53745;">ERROR</span>    There's been a mistake with the process.                      <span style="color: #A2A2A2;">config.py:74</span>
<span style="color: #1E1E1E; background-color: #DF1426;">CRITICAL</span> There is something terribly wrong and process may terminate.  <span style="color: #A2A2A2;">config.py:75</span>
</pre>

We use [RichHandler](https://rich.readthedocs.io/en/stable/logging.html){:target="_blank"} for our `console` handler to get pretty formatting for the log messages. This is not a preinstalled library so we'll need to install and add to `requirements.txt`:

```bash
pip install rich==12.4.4
```

```bash
# Add to requirements.txt
rich==12.4.4
```

Our logged messages become stored inside the respective files in our logs directory:

```bash
logs/
    ├── info.log
    └── error.log
```

And since we defined a detailed formatter, we would see informative log messages like these:
<pre>
<span style="color: #2871CF;">INFO</span> <span style="color: #5A9C4B;">2020-10-21</span> 11:18:42,102 [<span style="color: #3985B9;">config.py</span>:module:<span style="color: #3D9AD9;">72</span>]
Informative messages from your code.
</pre>

## Application

In our project, we can replace all of our print statements into logging statements:

```python linenums="1"
print("✅ Saved data!")
```

<div class="ai-center-all">
    ──── &nbsp; becomes: &nbsp; ────
</div>

```python linenums="1"
from config.config import logger
logger.info("✅ Saved data!")
```

All of our log messages are at the `INFO` level but while developing we may have had to use `DEBUG` levels and we also add some `ERROR` or `CRITICAL` log messages if our system behaves in an unintended manner.

- **what**: log all the necessary details you want to surface from our application that will be useful *during* development and *afterwards* for retrospective inspection.

- **where**: a best practice is to not clutter our modular functions with log statements. Instead we should log messages outside of small functions and inside larger workflows. For example, there are no log messages inside any of our scripts except the `main.py` and `train.py` files. This is because these scripts use the smaller functions defined in the other scripts (data.py, evaluate.py, etc.). If we ever feel that we the need to log within our other functions, then it usually indicates that the function needs to be broken down further.

> The [Elastic stack](https://www.elastic.co/what-is/elk-stack){:target="_blank"} (formerly ELK stack) is a common option for production level logging. It combines the features of [Elasticsearch](https://www.elastic.co/elasticsearch/){:target="_blank"} (distributed search engine), [Logstash](https://www.elastic.co/logstash){:target="_blank"} (ingestion pipeline) and [Kibana](https://www.elastic.co/kibana){:target="_blank"} (customizable visualization). We could also simply [upload](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3-uploading-files.html){:target="_blank"} our logs to a cloud blog storage (ex. S3, Google Cloud Storage, etc.).

<!-- Course signup -->
{% include "templates/course-signup.md" %}

<!-- Citation -->
{% include "templates/cite.md" %}