FROM python:3.13.12
WORKDIR /app

# Install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY src ./src

# Define entrypoint
ENTRYPOINT ["python", "script_v3.py"]

# Defect arguments (can be overwritten)
CMD []
