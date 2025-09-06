# Online Shopping Analysis & Revenue Prediction

This project represents my initial exploration of design patterns (Strategy, Factory) in a machine learning context. It predicts user behavior on online shopping platforms, incorporates MLflow and ZenML for model and pipeline management, and is delivered through a Streamlit web application.

## 🗂️Project Structure

online-shopping-analysis/
├── app.py # Streamlit application for user interface
├── analysis/
│ └── EDA.ipynb # initial exploratory data analysis for ml models based on strateges defined in analysis_src/
│ └── analysis_src/
├── model/
│ └── pipeline.pkl # pre-trained model
├── steps/ # contains concrete steps for my pipeline based on strategies in scr/ 
├── src/
├── pipelines/ 
│ └── training_pipeline.py # pipeline for model training
└── requirements.txt # Project dependencies

## 📊Machine Learning Workflow and Deployment Overview
The machine learning workflow leverages MLflow for experiment tracking & model versioning, while ZenML structures the pipeline to ensure version control of pipeline components. 
The project is deployed as an interactive web application using Streamlit, allowing users to input data and receive predictions.

## 🔗Link to Streamlit App
[Streamlit App](https://online-shopping-analysis.streamlit.app/)
