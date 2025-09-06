# Online Shopping Analysis & Revenue Prediction

This project represents my initial exploration of design patterns in a machine learning context. It predicts user behavior on online shopping platforms, incorporates MLflow and ZenML for model and pipeline management, and is delivered through a Streamlit web application.

## 🗂️Project Structure
```
online-shopping-analysis/
├── app.py                   # streamlit application
├── analysis/
│   ├── EDA.ipynb            # initial exploratory data analysis based on strategies in analysis_src/
│   └── analysis_src/        # supporting scripts for EDA strategies
├── model/
│   └── pipeline.pkl          # pre-trained machine learning model
├── steps/                    # concrete steps for the ML pipeline based on strategies in src/
├── src/                      # core modules for feature engineering, outlier detection, and other utilities
├── pipelines/
│   └── training_pipeline.py  # pipeline definition for model training
└── requirements.txt          # project dependencies
```
## 📊Machine Learning Workflow and Deployment Overview
The machine learning workflow leverages MLflow for experiment tracking & model versioning, while ZenML structures the pipeline to ensure version control of pipeline components. 
The project is deployed as an interactive web application using Streamlit, allowing users to input data and receive predictions.

## 🔗Link to Streamlit App
[Streamlit App](https://online-shopping-analysis.streamlit.app/)

## ✨References 
- [UCI Dataset](https://archive.ics.uci.edu/dataset/468/online%2Bshoppers%2Bpurchasing%2Bintention%2Bdataset?)
- [Reference for Template Structure](https://github.com/vn33/MLOps_House-Price-Prediction-using-ZenML-and-MLflow?utm_source=chatgpt.com)
