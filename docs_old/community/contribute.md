---
title: Contribute
---

## What contributions?

- **Documentation** contributions are very valuable to us!
- **Examples.** Show us what you did with Outlines :)
- **Bug reports** with a minimum working examples in the [issue tracker][issues]
- **Bug fixes** are always a pleasure to review.
- **New features**. Please start a new [discussion][discussions], or [come chat with us][discord] beforehand!

Note that the [issue tracker][issues] is only intended for actionable items. In doubt, open a [discussion][discussions] or [come talk to us][discord].

## How to contribute?

### Setup

First, [fork the repository on GitHub](https://github.com/dottxt-ai/outlines/fork) and clone the fork locally:

```shell
git clone git@github.com/YourUserName/outlines.git
cd outlines
```

Create a new virtual environment:

*If you are using `uv`*:

```shell
uv venv
source .venv/bin/activate
alias pip="uv pip" # ... or just remember to prepend any pip command with uv in the rest of this guide
```

*If you are using `venv`*:

```shell
python -m venv .venv
source .venv/bin/activate
```

*If you are using `conda`*:

```shell
conda env create -f environment.yml
```

Then install the dependencies in editable mode, and install the `pre-commit` hooks:

```shell
pip install -e ".[test]"
pre-commit install
```
If you own a GPU and want to run the vLLM tests you will have to run:

```shell
pip install -e ".[test-gpu]"
```

instead.

Outlines provides optional dependencies for different supported backends, which you can install with

```shell
pip install ".[vllm]"
```

A list of supported optional dependencies can be found in the [installation guide](/installation).

### Using VSCode DevContainer / GitHub Codespaces

If you want a fully pre-configured development environment, you can use VSCode DevContainers or GitHub Codespaces.

#### VSCode DevContainer

1. Ensure that the [Docker](https://www.docker.com/get-started/) daemon is running on your machine.
2. Install the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension in VSCode.
3. Open the Outlines repository in VSCode. When prompted, **Reopen in Container** (or press `F1` and select "Remote-Containers: Reopen in Container").
4. Run the normal setup steps. Your environment will not complain about missing system dependencies!

#### GitHub Codespaces

1. Navigate to the Outlines repository on GitHub.
2. Click on the **Code** button and select the **Codespaces** tab.
3. Click **Create codespace on main** (or another branch you are working on).
4. GitHub will launch a pre-configured cloud development environment.

You will not have access to a GPU, but you'll be able to make basic contributions to the project on the go while using a fully featured web-based IDE.

### Before pushing your code

Run the tests:

```shell
pytest
```

And run the code style checks:

```shell
pre-commit run --all-files
```

### Benchmarking

Outlines uses [asv](https://asv.readthedocs.io) for automated benchmark testing. Benchmarks are run automatically before pull requests are merged to prevent performance degradation.

You can run the benchmark test suite locally with the following command:

```shell
asv run --config benchmarks/asv.conf.json
```

Caveats:

- If you're on a device with CUDA, you must add the argument `--launch-method spawn`
- Uncommitted code will not be benchmarked, you must first commit your changes.

#### Run a specific test:

```shell
asv run --config benchmarks/asv.conf.json -b bench_json_schema.JsonSchemaBenchmark.time_json_schema_to_fsm
```

#### Profile a specific test:

```shell
asv run --config benchmarks/asv.conf.json --profile -b bench_json_schema.JsonSchemaBenchmark.time_json_schema_to_fsm
```

#### Compare to `origin/main`

```shell
get fetch origin
asv continuous origin/main HEAD --config benchmarks/asv.conf.json
```

#### ASV PR Behavior

- **View ASV Benchmark Results:** Open the workflow, view `BENCHMARK RESULTS` section.
- Merging is blocked unless benchmarks are run for the latest commit.
- Benchmarks fail if performance degrades by more than 10% for any individual benchmark.
- The "Benchmark PR" workflow runs when it is manually dispatched, or if the `run_benchmarks` label is added to the PR they run for every commit.

### Contribute to the documentation

To work on the *documentation* you will need to install the related dependencies:

```shell
pip install -r requirements-doc.txt
```

To build the documentation and serve it locally, run the following command in the repository's root folder:

```shell
mkdocs serve
```

By following the instruction you will be able to view the documentation locally.
It will be updated every time you make a change.

## Open a Pull Request

Create a new branch on your fork, commit and push the changes:

```shell
git checkout -b new-branch
git add .
git commit -m "Changes I made"
git push origin new-branch
```

Then you can [open a pull request][pull-requests] on GitHub. It should prompt you to do so. Every subsequent change that you make on your branch will update the pull request.

Do not hesitate to open a draft PR before your contribution is ready, especially if you have questions and/or need feedback. If you need help, come tell us on [Discord][discord].

[discord]: https://discord.gg/R9DSu34mGd
[discussions]: https://github.com/dottxt-ai/outlines/discussions
[issues]: https://github.com/dottxt-ai/outlines/issues
[pull-requests]: https://github.com/dottxt-ai/outlines/pulls
