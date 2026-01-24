# Demand Forecasting – Working Environment

This repository constitutes the **working environment for the Demand Forecasting project**. It contains all the necessary notebooks, scripts, and technical requirements to reproduce the end-to-end Machine Learning workflow used to predict future sales in a retail context.

---

## Project Overview

The main objective of this project is to develop an **end-to-end demand forecasting model**, covering the full data science lifecycle: data exploration, data cleaning, feature engineering, model experimentation, training, evaluation, and prediction.

The work is primarily documented and executed through Jupyter notebooks, with a clear separation between exploratory work and the final, streamlined modeling pipeline.

---

## Main Notebook

### `forecast_predict_model.ipynb`

This notebook is the **core working document** of the project. It includes:

- In-depth exploration and understanding of the dataset  
- Careful data cleaning and preprocessing  
- Extensive feature engineering, including the creation of lags and aggregated variables  
- Detailed reasoning and decision-making behind each transformation applied to the data  
- Exploration and comparison of two modeling approaches:
  - **XGBoost**
  - **Ridge Regression**

This notebook reflects the full analytical process and experimentation carried out during the project.

---

## Modular Notebook Pipeline

In addition to the main working notebook, the repository includes a **cleaned and structured version of the final workflow**, split into three notebooks:

- **`01_eda.ipynb`**  
  → Exploratory Data Analysis (EDA)

- **`02_features.ipynb`**  
  → Feature engineering and lag generation

- **`03_train.ipynb`**  
  → Model training, prediction, and generation of `submission.csv`

These notebooks represent the **final, concise pipeline** used for this delivery.

---

## Technical Requirements

The repository includes the files that specify the **technical requirements** needed to run the notebooks and scripts correctly (e.g., dependencies and environment configuration).

---

## Notes

- The project is designed to be reproducible end-to-end.  
- All modeling and feature engineering decisions are documented either in the main notebook or in the modular pipeline.  
- The focus is on methodological rigor, interpretability, and clear communication of results.
