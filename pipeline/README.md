# SageMaker Pipeline — BYOC (End-to-End)

This folder contains an end-to-end Machine Learning pipeline built with Amazon SageMaker Pipelines using a Bring Your Own Container (BYOC) approach. All steps in the workflow rely on custom Docker images instead of AWS managed containers.

## Contents
- `Pipeline_notebook_byoc.ipynb`: Main notebook that defines, builds, and executes the pipeline.
- `sagemaker_pipeline_byoc.ipynb`: Supporting notebook for container setup, image handling, and environment configuration.

## Pipeline Architecture
The pipeline orchestrates the full ML lifecycle for a demand forecasting use case:

1. **ProcessingStep (Preprocessing)**  
   Custom container performs data cleaning and feature engineering, generating:
   - Train dataset
   - Validation dataset
   - Test dataset (with target for evaluation)
   - Inference dataset (without target)

2. **TrainingStep**  
   Custom training container fits an XGBoost model using the prepared datasets.

3. **ProcessingStep (Evaluation)**  
   Evaluates the trained model on the test dataset and computes RMSE.  
   Results are stored in `evaluation.json`.

4. **ConditionStep**  
   Compares RMSE against a configurable threshold:
   - If RMSE ≤ threshold → continue pipeline
   - Else → trigger FailStep

5. **ModelStep (Create Model)**  
   Creates a SageMaker model using the custom inference container.

6. **TransformStep (Batch Transform)**  
   Runs batch inference using the inference dataset.

7. **ModelStep (Register Model)**  
   Registers the model in SageMaker Model Registry for versioning and governance.

8. **FailStep**
   Stops the pipeline if the model does not meet performance requirements.

## Key Features
- Fully BYOC implementation across all steps
- RMSE-based evaluation for regression
- Automated model validation and decision-making
- Proper separation of evaluation and inference datasets
- Consistent dependency management across containers
- End-to-end reproducible and modular workflow

## Execution and Outputs
The pipeline is executed from the notebook and generates:
- Processed datasets in S3
- Trained model artifact (`model.tar.gz`)
- Evaluation report (`evaluation.json`)
- Batch inference outputs
- Registered model in SageMaker Model Registry

## Result
The pipeline completes successfully with status `Succeeded`, demonstrating a production-style ML workflow fully controlled through custom containers.

## Screenshots

![Pipeline Containers](../docs/images/pipeline_containers.png)
![Pipeline Preprocessing Image](../docs/images/pipeline_preprocessing_image.png)
![Pipeline Training Image](../docs/images/pipeline_training_image.png)
![Pipeline Inference Image](../docs/images/pipeline_inference_image.png)
![Pipeline Success](../docs/images/pipeline_succed1.png)
![Pipeline Success - Details](../docs/images/pipeline_succed2.png)
![Pipeline Registered Model](../docs/images/pipeline_registered_model.png)
![Pipeline S3 outputs](../docs/images/pipeline_s3_outputs.png)
