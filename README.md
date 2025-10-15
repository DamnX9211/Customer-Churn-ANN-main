# Customer Churn Prediction Using Artificial Neural Networks

A comprehensive machine learning project that predicts customer churn using artificial neural networks (ANN) with TensorFlow/Keras. The project includes data preprocessing, model training, hyperparameter tuning, and a user-friendly Streamlit web application for real-time predictions.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Web Application](#web-application)
- [Contributing](#contributing)

## 🎯 Overview

Customer churn is a critical business metric that measures the rate at which customers stop doing business with a company. This project uses artificial neural networks to predict whether a bank customer will leave the bank based on various customer attributes and banking behavior patterns.

**Key Objectives:**
- Predict customer churn with high accuracy
- Identify key factors influencing customer retention
- Provide an interactive tool for real-time churn prediction
- Demonstrate end-to-end machine learning workflow

## 📊 Dataset

The project uses a bank customer dataset (`Churn_Modelling.csv`) containing 10,000 customer records with the following features:

**Input Features:**
- `CreditScore`: Customer's credit score
- `Geography`: Customer's location (France, Germany, Spain)
- `Gender`: Customer's gender (Male, Female)
- `Age`: Customer's age
- `Tenure`: Number of years as a bank customer
- `Balance`: Account balance
- `NumOfProducts`: Number of bank products used
- `HasCrCard`: Whether customer has a credit card (1/0)
- `IsActiveMember`: Whether customer is an active member (1/0)
- `EstimatedSalary`: Customer's estimated salary

**Target Variable:**
- `Exited`: Whether customer churned (1) or not (0)

## ✨ Features

- **Data Preprocessing**: Automated handling of categorical variables, feature scaling
- **Neural Network Models**: Deep learning models for both classification and regression
- **Hyperparameter Tuning**: Grid search optimization for best model parameters
- **Model Persistence**: Save and load trained models and preprocessors
- **Interactive Web App**: Streamlit-based interface for real-time predictions
- **Comprehensive Analysis**: Multiple Jupyter notebooks for different aspects
- **Visualization**: TensorBoard integration for training monitoring

## 📁 Project Structure

```
Customer-Churn-ANN-main/
├── Churn_Modelling.csv              # Dataset
├── experiments.ipynb                # Main data preprocessing and model training
├── hyperparametertuningann.ipynb    # Hyperparameter optimization
├── prediction.ipynb                 # Example predictions and model testing
├── salaryregression.ipynb           # Additional regression model
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── model.h5                        # Trained churn prediction model
├── regression_model.h5             # Trained salary regression model
├── label_encoder_gender.pkl        # Gender label encoder
├── onehot_encoder_geo.pkl          # Geography one-hot encoder
├── scaler.pkl                      # Feature scaler
├── .gitignore                      # Git ignore file
└── README.md                       # Project documentation
```

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/DamnX9211/Customer-Churn-ANN-main.git
   cd Customer-Churn-ANN-main/Customer-Churn-ANN-main
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### 1. Training the Model

Run the Jupyter notebooks in the following order:

```bash
# Start Jupyter Lab/Notebook
jupyter lab

# Run notebooks in order:
# 1. experiments.ipynb - Data preprocessing and initial model training
# 2. hyperparametertuningann.ipynb - Hyperparameter optimization
# 3. prediction.ipynb - Test model predictions
```

### 2. Web Application

Launch the Streamlit web application for interactive predictions:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### 3. Making Predictions

**Using the Web App:**
- Fill in customer details in the sidebar
- View real-time churn probability
- Get instant churn prediction

**Using Python Code:**
```python
import tensorflow as tf
import pickle
import pandas as pd

# Load model and preprocessors
model = tf.keras.models.load_model('model.h5')
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Make prediction
# [Your prediction code here]
```

## 🏗️ Model Architecture

### Classification Model (Churn Prediction)
- **Input Layer**: 12 features (after preprocessing)
- **Hidden Layer 1**: 64 neurons, ReLU activation
- **Hidden Layer 2**: 32 neurons, ReLU activation  
- **Output Layer**: 1 neuron, Sigmoid activation
- **Loss Function**: Binary Crossentropy
- **Optimizer**: Adam
- **Metrics**: Accuracy

### Regression Model (Salary Prediction)
- **Input Layer**: 12 features
- **Hidden Layer 1**: 64 neurons, ReLU activation
- **Hidden Layer 2**: 32 neurons, ReLU activation
- **Output Layer**: 1 neuron, Linear activation
- **Loss Function**: Mean Absolute Error
- **Optimizer**: Adam

## 📈 Results

### Model Performance
- **Best Hyperparameters**: 16 neurons, 1 layer, 100 epochs
- **Cross-Validation Score**: ~85.76%
- **Training Accuracy**: ~86%
- **Validation Accuracy**: ~86%

### Key Findings
- Geography and age are significant predictors of churn
- Customers with higher balances are less likely to churn
- Active members show lower churn rates
- Model achieves good performance with relatively simple architecture

## 🌐 Web Application

The Streamlit web application provides:

- **Interactive Input Form**: Easy-to-use sliders and dropdowns
- **Real-time Predictions**: Instant churn probability calculation
- **User-friendly Interface**: Clean and intuitive design
- **Probability Visualization**: Clear display of churn likelihood

**Features:**
- Geography selection (France, Germany, Spain)
- Gender selection
- Age slider (18-92)
- Balance input
- Credit score input
- Estimated salary input
- Tenure slider (0-10 years)
- Number of products (1-4)
- Credit card status
- Active member status

## 🤝 Contributing

Contributions are welcome! Please feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the MIT License.

## 📞 Contact

For questions or suggestions, please open an issue in the GitHub repository.

---

**Built with:** Python • TensorFlow • Keras • Scikit-learn • Streamlit • Pandas • NumPy
