---
description: Keep records of the important events in our application.
image: https://madewithml.com/static/images/applied_ml.png
---

:octicons-mark-github-16: [Repository](https://github.com/GokuMohandas/applied-ml){:target="_blank"}

Keep records of the important events in your application.

## Intuition

Logging the process of tracking and recording key events that occur in our applications. We want to log events so we can use them to inspect processes, fix issues, etc. They're a whole lot more powerful than `print` statements because they allow us to send specific pieces of information to specific locations, not to mention custom formatting, shared interface with other Python packages, etc. We should use logging to provide insight into the internal processes of our application to notify our users of the important events that are occurring.

## Application

Let's look at what logging looks like in our [application](https://github.com/GokuMohandas/applied-ml){:target="_blank"}. There are a few overarching concepts to be aware of first before we can create and use our loggers.

- `#!js Logger`: the main object that emits the log messages from our application.
- `#!js Handler`: used for sending log records to a specific location and specifications for that location (name, size, etc.).
- `#!js Formatter`: used for style and layout of the log records.

There is so much [more](https://docs.python.org/3/library/logging.html){:target="_blank"} to logging such as filters, exception logging, etc. but these basics will allows us to do everything we need for our application.

Before we create our specialized, configured logger, let's look at what logged messages even look like by using a very basic configuration.
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

These are the basic [levels](https://docs.python.org/3/library/logging.html#logging-levels){:target="_blank"} of logging where `DEBUG` is the lowest priority and `CRITICAL` is the highest. We defined out logger using [`basicConfig`](https://docs.python.org/3/library/logging.html#logging.basicConfig){:target="_blank"} to emit log messages to our stdout console (we also could've written to any other stream or even a file) and to be sensitive to log messages starting from level `DEBUG`. This means that all of our logged messages will be displayed since `DEBUG` is the lowest level. Had we made the level `ERROR`, then only `ERROR` and `CRITICAL` log message would be displayed.

Now let's go ahead and create more configured loggers that will be useful for our application (our code is inside [`tagifai/config.py`](https://github.com/GokuMohandas/applied-ml/blob/main/tagifai/config.py){:target="_blank"}. First, we'll define a configuration dictionary object:

```python linenums="1"
# Logger
logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "minimal": {"format": "%(message)s"},
        "detailed": {
            "format": "%(levelname)s %(asctime)s [%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
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
    "loggers": {
        "root": {
            "handlers": ["console", "info", "error"],
            "level": logging.INFO,
            "propagate": True,
        },
    },
}
```

1. `[Lines 4-9]`: define two different [Formatters](https://docs.python.org/3/library/logging.html#formatter-objects){:target="_blank"} (determine format and style of log messages), minimal and detailed, which use various [LogRecord attributes](https://docs.python.org/3/library/logging.html#logrecord-attributes){:target="_blank"} to create a formatting template for log messages.
2. `[Lines 10-33]`: define the different [Handlers](https://docs.python.org/3/library/logging.html#handler-objects){:target="_blank"} (details about location of where to send log messages):
    - `#!js console`: sends log messages (using the `minimal` formatter) to the `stdout` stream for messages above level `DEBUG`.
    - `#!js info`: send log messages (using the `detailed` formatter) to `logs/info.log` (a file that can be up to `1 MB` and we'll backup the last `10` versions of it) for messages above level `INFO`.
    - `#!js error`: send log messages (using the `detailed` formatter) to `logs/error.log` (a file that can be up to `1 MB` and we'll backup the last `10` versions of it) for messages above level `ERROR`.
3. `[Lines 34-37]`: attach our different handlers to our [Logger](https://docs.python.org/3/library/logging.html#logger-objects){:target="_blank"}.

We can load our configuration dict like so:
```python linenums="1"
logging.config.dictConfig(logging_config)
logger = logging.getLogger("root")
logger.handlers[0] = RichHandler(markup=True)

# Sample messages (not we use configured `logger` now)
logger.debug("Used for debugging your code.")
logger.info("Informative messages from your code.")
logger.warning("Everything works but there is something to be aware of.")
logger.error("There's been a mistake with the process.")
logger.critical("There is something terribly wrong and process may terminate.")
```
<pre style="font-size: 0.7rem;">
<span style="font-weight: 600; color: #39BC70;">DEBUG</span>    Used for debugging your code.                                 <span style="font-weight: 600; color: #A2A2A2;">config.py:71</span>
<span style="font-weight: 600; color: #2871CF;">INFO</span>     Informative messages from your code.                          <span style="font-weight: 600; color: #A2A2A2;">config.py:72</span>
<span style="font-weight: 600; color: #BF1825;">WARNING</span>  Everything works but there is something to be aware of.       <span style="font-weight: 600; color: #A2A2A2;">config.py:73</span>
<span style="font-weight: 600; color: #F53745;">ERROR</span>    There's been a mistake with the process.                      <span style="font-weight: 600; color: #A2A2A2;">config.py:74</span>
<span style="font-weight: 600; color: #1E1E1E; background-color: #DF1426;">CRITICAL</span> There is something terribly wrong and process may terminate.  <span style="font-weight: 600; color: #A2A2A2;">config.py:75</span>
</pre>

!!! note
    We use [RichHandler](https://rich.readthedocs.io/en/stable/logging.html){:target="_blank"} for our `console` handler to get pretty formatting for the log messages.

We can also check our `logs/info.log` and `logs/error.log` files to see the log messages that should go to each of those files based on the levels we set for their handlers. Because we used the `detailed` formatter, we should be seeing more informative log messages there:
<pre>
<span style="font-weight: 600; color: #2871CF;">INFO</span> <span style="font-weight: 600; color: #5A9C4B;">2020-10-21</span> 11:18:42,102 [<span style="font-weight: 600; color: #3985B9;">config.py</span>:module:<span style="font-weight: 600; color: #3D9AD9;">72</span>]
Informative messages from your code.
</pre>

We chose to define a dictionary configuration for our logger but there are other ways too such as coding directly in scripts, using config file, etc. Click on the different options below to expand and view the respective implementation.

??? note "Coding directly in scripts (click to expand)"

    ```python linenums="1"
    import logging
    from rich.logging import RichHandler

    # Create logger
    logger = logging.getLogger("root")
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
        fmt="%(levelname)s %(asctime)s [%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
    )

    # Hook it all up
    console_handler.setFormatter(fmt=minimal_formatter)
    info_handler.setFormatter(fmt=detailed_formatter)
    error_handler.setFormatter(fmt=detailed_formatter)
    logger.addHandler(hdlr=console_handler)
    logger.addHandler(hdlr=info_handler)
    logger.addHandler(hdlr=error_handler)
    ```

??? note "Using a config file (click to expand)"

    1. Place this inside a `logging.config` file:
    ```
    [formatters]
    keys=minimal,detailed

    [formatter_minimal]
    format=%(message)s

    [formatter_detailed]
    format=
        %(levelname)s %(asctime)s [%(filename)s:%(funcName)s:%(lineno)d]
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
    logger = logging.getLogger("root")
    logger.handlers[0] = RichHandler(markup=True)  # set rich handler
    ```