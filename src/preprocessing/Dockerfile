FROM python:3.13-slim
WORKDIR /app

# Install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src ./src

# Default entrypoint
ENTRYPOINT ["python", "-m", "src.preprocessing"]

# Default arguments (can be overwritten)
CMD ["--raw-path", "data/raw", "--output-path", "data/prep"]
