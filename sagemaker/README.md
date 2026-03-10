## SageMaker Training and Inference Deployment

This notebook demonstrates how the demand forecasting model is trained and deployed on **Amazon SageMaker** using a **Bring Your Own Container (BYOC)** approach.

The workflow covers the full lifecycle of a machine learning deployment in SageMaker, including container creation, model training, and real-time inference.

### Workflow

The notebook performs the following steps:

1. **Environment Setup**
   - Initialize SageMaker session
   - Retrieve AWS region, account ID, and execution role
   - Configure default S3 bucket

2. **Build Training Container**
   - Build a custom Docker image containing the training code and dependencies.

3. **Push Image to Amazon ECR**
   - Tag and push the training container to **Amazon Elastic Container Registry (ECR)**.

4. **Launch SageMaker Training Job**
   - Train the model using the custom container.
   - The dataset is downloaded from S3 and the trained model artifact is saved back to S3.

5. **Build Inference Container**
   - Create a separate Docker image for model serving.
   - The container exposes the required SageMaker endpoints (`/ping` and `/invocations`).

6. **Deploy Real-Time Endpoint**
   - Deploy the trained model to a SageMaker endpoint using the inference container.

7. **Invoke the Endpoint**
   - Send a sample payload to the deployed endpoint to obtain predictions.

### Result

The endpoint returns predictions in JSON format: **"predictions": [...]**

### Clean up

After testing, the endpoint should be deleted to avoid unnecessary AWS costs: **predictor.delete_endpoint()**