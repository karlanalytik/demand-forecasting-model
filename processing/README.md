# SageMaker Processing Job (BYOC) – Preprocessing with Scikit-learn

##  Description

This project implements a **SageMaker Processing Job** using a custom container (**BYOC: Bring Your Own Container**) to execute a **data preprocessing pipeline** with `pandas` and `scikit-learn`.

The goal is to build a reproducible and decoupled workflow that transforms raw data stored in S3 into features ready for model training.

---

## Architecture

The implemented flow follows this pattern:

S3 (raw data)
↓ 
/opt/ml/processing/input/
↓ 
preprocess.py (BYOC container)
↓
/opt/ml/processing/output/
↓
S3 (processed data)

> Note: The script does not interact directly with S3. SageMaker automatically manages data transfer.

---

##  Project Structure

demand-forecasting-model/
├── processing/
    ├── container/  
    │   └── Dockerfile  
    ├── code/  
    │   └── preprocess.py  
    └── sm_processing_byoc.ipynb  

---

##  BYOC Container

A Docker image was built based on:

- python:3.11-slim
- pandas - numpy
- scikit-learn

---

##  Implemented Process

The notebook `sm_processing_byoc.ipynb` contains the full workflow:

### 1. Setup
- Initialize SageMaker session
- Retrieve IAM role
- Define S3 bucket and prefixes

### 2. Image Build
- Build container with `docker build --network sagemaker`
- Push image to Amazon ECR

### 3. Data Upload
- Upload raw CSV files to S3 

### 4. Processing Job
- Run job using `ScriptProcessor`
- Use the custom container 

### 5. Data Transformation
The `preprocess.py` script performs:

- Loading and merging multiple datasets
- Data cleaning (dates, duplicates)
- Feature engineering (monthly aggregation)
- Creation of final dataset

### 6. Output
A single output file is generated:

**sales_prep.csv**

Stored in:

s3://<bucket>/demand-forecasting/processing-byoc/processed/

---

##  Results Verification

The following validations were performed:

- The Processing Job completed with status `Completed`
- The file `sales_prep.csv` was successfully generated in S3
- The transformed dataset contains expected columns
- The first rows were inspected in the notebook

---

##  Reproducibility

This pipeline is:

- Reproducible: fully executed from the notebook
- Scalable: runs on managed SageMaker infrastructure
- Decoupled: independent from local environment

---

##  Screenshots

![Repositories](docs/images/ECR_preprocessing.png)
![Success preprocessing](docs/images/Success_preprocessing.png)
![S3 preprocessing](docs/images/S3_preprocessing.png)
