# How to contribute

The project is hosted on [GitHub](https://github.com/yhtang/FunFact). For
questions, suggestions and bug reports, please take advantage of the
[issue tracking system](https://github.com/yhtang/FunFact/issues). In
addition, contributions are very welcomed and could be submitted as
[pull requests](https://github.com/yhtang/FunFact/pulls).

## Submitting a bug report or a feature request

Please feel free to open an issue should you run into a bug or wish a
feature could be implemented.

When submitting an issue, please try to follow the guidelines below:

-   Include a minimal reproducible example of the issue for bug reports.
-   Provide a mock code snippt for feature suggestions.
-   Provide a full traceback when an exception is raised.
-   Please include your operating system type and version number, as
    well as your Python and `funfact` versions. This
    information can be found by running:

```python
import platform; print(platform.platform())
import sys; print('Python', sys.version)
import funfact; print('FunFact', funfact.__version__)
```

# Contributing Code

The most recommended way to contribute is to fork the [main
repository](https://github.com/yhtang/FunFact), then submit a \"pull
request\" following the procedure below:

1.  [Fork](https://docs.github.com/en/github/getting-started-with-github/fork-a-repo)
    the project repository.
2.  Clone your own fork to local disk via `git clone`
3.  [Setting up the development
    environment](#setting-up-the-development-environment)
4.  Create a branch for development via
    `git checkout -b feature/<feature-name> master` (replace
    `feature-name` with the actual name of the feature).
5.  Make changes on the feature branch
6.  Test the changes with [Quality assurance
    measures](#quality-assurance-measures).
7.  Push the completed feature to your own fork, then [create a pull
    request](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request).

Development Guide
-----------------

### Setting up the development environment

First create a Python virtual environment containing all dependencies
for development:

```bash
virtualenv venv
source venv/bin/activate
pip install flake8  # for lint a.k.a. code style check
python setup.py develop
```

### Python code style

Variable, function, file, module, and package names should use all lower
case letters with underscores being word separators. For example, use
`module_name.py` instead of `Module-Name.py`, and `some_function()`
instead of `SomeFunction()`. Class names should use the Pascal case. For
example, use `ClassName` instead of `class_name`.

In addition, the [PEP8](https://www.python.org/dev/peps/pep-0008/) style
guide should be followed. An (incomplete) summary of the style guide is:

-   4 spaces per indentation level
-   79 characters at most per line
-   2 blank lines around top-level functions and class definitions
-   1 blank line around method definitions inside a class
-   1 module per import line, imports always at top of file
-   UTF-8 source file encoding
-   0 space on the inner side of parentheses, e.g. use `(x + y)` instead
    of `( x + y )`.
-   1 space on both sides of binary operators (including assignments),
    e.g. use `x = y * 3 + z` instead of `x=y*3+z`.
-   0 space around keyword argument assignments, e.g. use `f(x=1)`
    instead of `f(x = 1)`.

Comformance to the style guide can be checked via [Code style
check](#code-style-check).

### Quality assurance measures

#### Unit tests

```bash
tox -e py38
```

#### Code style check

```bash
make lint
```

#### Coverage test

```bash
tox -e coverage
```

Coverage reports are stored in the `htmlcov` directory.
