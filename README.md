# ANN_1_CustomerChurn
# Customer Churn Prediction

This project uses an **Artificial Neural Network (ANN)** to predict customer churn based on various customer features. The model is built using **TensorFlow 2.17.1** and deployed as a **Streamlit** web application.

### Features:
- **TensorFlow-based model** to predict customer churn.
- User-friendly **Streamlit interface** for real-time predictions.
- **Data preprocessing**: Encoding categorical features, scaling numerical features.
- **Model deployment**: Easy to deploy on **Streamlit Cloud** or local environment.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Model Overview](#model-overview)
- [Streamlit Application](#streamlit-application)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Project Overview

This project focuses on predicting customer churn in a business environment. The model is built using **TensorFlow 2.17.1**, and the **Streamlit app** allows users to input customer data and get a churn probability. The prediction is based on several features including:

- **Geography**
- **Gender**
- **Age**
- **Balance**
- **Credit Score**
- **Estimated Salary**
- **Tenure**
- **Number of Products**
- **Has Credit Card**
- **Is Active Member**

### Key Steps:
1. **Data Preprocessing**: Label encoding for categorical variables and scaling of numerical data.
2. **Model Building**: An ANN model with one input layer, two hidden layers, and one output layer for churn prediction.
3. **Model Training**: Trained on the preprocessed data and saved in HDF5 format.
4. **Model Inference**: A **Streamlit app** that takes user input, processes it, and outputs the churn probability.

---

## Installation

### Step 1: Install Python 3.11 (or below)

Ensure you're using Python **3.11** or lower for compatibility with **TensorFlow 2.17.1**.

- On **Ubuntu/Linux**:
    ```bash
    sudo apt install python3.11 python3.11-venv
    ```

- On **Windows/macOS**, download and install Python 3.11 from [python.org](https://www.python.org/downloads/release/python-3110/).

### Step 2: Create a Virtual Environment

Create and activate a new Python environment:

```bash
python3.11 -m venv churn-pred-env
source churn-pred-env/bin/activate  # On Windows: churn-pred-env\Scripts\activate
