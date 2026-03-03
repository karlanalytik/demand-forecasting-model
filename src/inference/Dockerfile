FROM python:3.13-slim
WORKDIR /app

# Install dependencies
COPY src/inference/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src ./src

# Default entrypoint
ENTRYPOINT ["python", "-m", "src.inference"]

# Default arguments (can be overwritten)
CMD [ \
    "--input_path", "data/inference/test.csv", \
    "--raw_data_dir", "data/raw", \
    "--prep_data_dir", "data/prep", \
    "--model_path", "artifacts/xgboost_model.joblib", \
    "--output_path", "data/predictions/predictions.csv", \
    "--month", "11", \
    "--date_block_num", "34" \
    ]
