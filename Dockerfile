FROM python:3.10

WORKDIR /outlines

RUN pip install --upgrade pip

# Copy necessary build components
COPY pyproject.toml .
COPY outlines ./outlines

# Install outlines and outlines[serve]
# .git required by setuptools-scm
RUN --mount=source=.git,target=.git,type=bind \
    pip install --no-cache-dir .[serve]

# https://outlines-dev.github.io/outlines/reference/vllm/
ENTRYPOINT python3 -m outlines.serve.serve
