# CPU-only image: runs the reference test suite and benchmarks.
# GPU-gated tests skip automatically inside the container.
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml setup.py README.md LICENSE ./
COPY src/ src/
COPY tests/ tests/
COPY benchmarks/ benchmarks/
COPY scripts/ scripts/

RUN pip install --no-cache-dir numpy pytest \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -e .

CMD ["pytest", "tests/", "-q"]
