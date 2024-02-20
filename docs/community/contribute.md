---
title: Contribute
---

## What contributions?

- **New features**. Please start a new [discussion][discussions], or [come chat with us][discord] beforehand!
- **Bug reports** with a minimum working examples in the [issue tracker][issues]
- **Bug fixes** are wonderful.
- **Documentation** are very valuable to us! The community will be forever grateful.

Note that the [issue tracker][issues] is only intended for actionable items. In doubt, open a [discussion][discussions] or [come talk to us][discord].

## How to contribute?

### Setup

First, [fork the repository on GitHub](https://github.com/outlines-dev/outlines/fork) and clone the fork locally:

```bash
git clone git@github.com/YourUserName/outlines.git
cd outlines
```

Create a new virtual environment. *If you are using conda*:

```bash
conda env create -f environment.yml
```

*If you are using venv*:

```python
python -m venv .venv
source .venv/bin/activate
```

Then install the dependencies in editable mode, and install the pre-commit hooks:

```python
pip install -e .[test]
pre-commit install
```

#### Developing Serve Endpoint Via Docker

```bash
docker build -t outlines-serve .
docker run -p 8000:8000 outlines-serve --model="mistralai/Mistral-7B-Instruct-v0.2"
```

This builds `outlines-serve` and runs on `localhost:8000` with the model `Mistral-7B-Instruct-v0.2`

### Before pushing your code

Run the tests:

```python
pytest
```

And run the code style checks:

```python
pre-commit run --all-files
```

#### Performance testing

Run benchmark tests:

```python
pytest --benchmark-only
```

([other pytest-benchmark command line options](https://pytest-benchmark.readthedocs.io/en/latest/usage.html#commandline-options))

### Open a Pull Request

Create a new branch on your fork, commit and push the changes:

```bash
git checkout -b new-branch
git add .
git commit -m "Changes I made"
git push origin new-branch
```

Then you can [open a pull request][pull-requests] on GitHub. It should prompt you to do so. Every subsequent change that you make on your branch will update the pull request.

Do not hesitate to open a draft PR before your contribution is ready, especially if you have questions and/or need feedback. If you need help, come tell us on [Discord][discord].

[discord]: https://discord.gg/R9DSu34mGd
[discussions]: https://github.com/outlines-dev/outlines/discussions
[issues]: https://github.com/outlines-dev/outlines/issues
[pull-requests]: https://github.com/outlines-dev/outlines/pulls
