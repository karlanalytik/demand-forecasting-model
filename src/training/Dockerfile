FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /opt/ml/code

COPY src/training/requirements.txt /opt/ml/code/requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY src /opt/ml/code/src

RUN mkdir -p /opt/ml/input/data/train && \
    mkdir -p /opt/ml/model && \
    mkdir -p /opt/ml/output/data

ENV PYTHONPATH=/opt/ml/code

ENTRYPOINT ["python", "-m", "src.training"]
