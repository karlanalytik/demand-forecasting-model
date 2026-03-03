FROM python:3.13-slim
WORKDIR /app

# Install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src ./src

# Default entrypoint
ENTRYPOINT ["python", "-m", "src.inference"]

# Default arguments (can be overwritten)
CMD [ \
    "--data_path", "data/prep/sales_prep.csv", \
    "--target", "item_cnt_month", \
    "--model_path", "artifacts/xgboost_model.joblib", \
    "--pred_path", "predictions/pred.csv" \
    ]
