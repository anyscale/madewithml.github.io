---
description: An automation tool that organizes commands for our application's processes.
image: https://madewithml.com/static/images/applied_ml.png
---

:octicons-mark-github-16: [Repository](https://github.com/GokuMohandas/applied-ml){:target="_blank"}

An automation tool that organizes commands for our application's processes.

## Intuition

We have just started and there are already so many different commands to keep track of. To help with this, we're going to use a [`Makefile`](https://opensource.com/article/18/8/what-how-makefile){:target="_blank"} which is a automation tool that organizes our commands. This makes it very easy for us to organize relevant commands as well as organize it for others who may be new to our application.

## Application

Inside our Makefile, we can see a list of functions (help, install, etc.). These functions (also known as `targets`) can sometimes have `prerequisites` that need to be met (can be other targets) and on the next line a ++tab++ followed by a `recipe`.

```bash
target: prerequisites
<TAB> recipe
```

We can execute any of our targets by typing `make <target>`:

<div class="animated-code">

    ```console
    # View all targets
    $ make help
    Usage: tagifai [OPTIONS] COMMAND [ARGS]
    👉  Commands:
        install         : installs required packages.
        install-dev     : installs development requirements.
        install-test    : installs test requirements.

    # Make a target
    $ make install-dev
    python -m pip install -e ".[dev]"
    ...
    ```

</div>
<script src="../../../static/js/termynal.js"></script>

We'll be adding more targets to our Makefile in subsequent lessons (testing, styling, etc.) but there's one more concept to illustrate. A Makefile is called as such because traditionally the `targets` are supposed to be files we can make. However, Makefiles are also commonly used as command shortcuts which can lead to confusion when a file with a certain name exists and a command with the same name exists! For example if you a directory called `docs` and a `target` in your Makefile called `docs`, when you run `make docs` you'll get this message:

<div class="animated-code">

    ```console
    $ make docs
    make: `docs` is up to date.
    ```

</div>

We can fix this by defining a `PHONY` target in our makefile by adding this line:
```bash
# Inside your Makefile
.PHONY: docs
```

Putting this all together, we can now install our package for different situations like so:
```bash
make install         # installs required packages only
make install-dev     # installs required + dev packages
make install-test    # installs required + test packages
```

!!! note
    There's a whole lot [more](https://www.gnu.org/software/make/manual/make.html){:target="_blank"} to Makefiles but this is plenty for our application most applied ML projects.