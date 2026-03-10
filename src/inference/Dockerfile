FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /opt/ml/code

COPY src/inference/requirements.txt /opt/ml/code/requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY src /opt/ml/code/src

ENV PYTHONPATH=/opt/ml/code

EXPOSE 8080

ENTRYPOINT ["python", "src/inference/serve.py"]
