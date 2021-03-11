
# Contributing to VTL

To ensure the library is stable, and to ensure best practices related to
versioning and deployment are followed, contributors must follow some general
guidelines.

## Tooling

The package [poetry](https://python-poetry.org/) as its dependency management and build backend.
This means that adding new dependencies is done by calling 
``` 
poetry add {dependency} 
``` 
and the package is installed locally by calling 
``` 
poetry install 
```
inside the package directory.

To set up a local development environment, from the root of the package, call:
```
python3.8 -m venv venv --prompt "vtl" && . venv/bin/activate && poetry install
```

Deployment to pypi is done with [twine](https://pypi.org/project/twine/) in a
Github Actions workflow that triggers whenever a tagged commit is pushed to the
master branch.

## Branching guide

We use a simple feature-branch pattern where each new development is written in
a new branch (termed a feature branch). Each feature branch will most likely
revolve around a single kind of transform. When the transform is deemed stable,
and it passes a standard battery of tests, a pull request is submitted
describing the change.
