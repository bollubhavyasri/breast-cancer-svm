Breast Cancer Diagnosis System Using Machine Learning

A Machine Learning-based web application that predicts whether a breast tumor is Benign or Malignant using medical diagnostic data.
The project uses Support Vector Machine (SVM) algorithms with different kernels and provides interactive visualizations for better understanding of classification performance.

📌 Project Overview

Breast cancer is one of the most common diseases affecting women worldwide. Early diagnosis is extremely important for improving survival rates and reducing treatment complexity.

This project aims to develop an intelligent healthcare support system that can assist in predicting breast cancer using Machine Learning techniques. The application analyzes medical input parameters and classifies tumors accurately.

The system includes:

Machine Learning prediction models
Multiple SVM kernels
Decision boundary visualization
PCA-based data visualization
Interactive web interface
🚀 Features
Predicts Breast Cancer as:
Benign
Malignant
Uses Multiple SVM Kernels:
Linear Kernel
RBF Kernel
Polynomial Kernel
Interactive Web Interface
Real-time Prediction System
PCA Visualization
Decision Boundary Graphs
Easy-to-use Healthcare Dashboard
🛠️ Technologies Used
Programming Language
Python
Machine Learning Libraries
Scikit-learn
NumPy
Pandas
Visualization
Matplotlib
Seaborn
Web Framework
Streamlit / Django
📂 Project Structure
Breast-Cancer-Prediction/
│
├── dataset/
│   └── breast_cancer.csv
│
├── models/
│   └── svm_model.pkl
│
├── images/
│   ├── homepage.png
│   ├── prediction.png
│   ├── linear_kernel.png
│   ├── rbf_kernel.png
│   ├── polynomial_kernel.png
│   └── visualization.png
│
├── app.py
├── train_model.py
├── requirements.txt
└── README.md
⚙️ Installation
Clone the Repository
git clone https://github.com/your-username/breast-cancer-prediction.git
Navigate to Project Folder
cd breast-cancer-prediction
Install Dependencies
pip install -r requirements.txt
▶️ Run the Application

For Streamlit:

streamlit run app.py

For Django:

python manage.py runserver
📊 Machine Learning Models Used
1. Linear Kernel SVM
Works well for linearly separable data
Faster computation
Simple decision boundary
2. RBF Kernel SVM
Handles non-linear data efficiently
Provides high accuracy
Best performing kernel in this project
3. Polynomial Kernel SVM
Captures complex relationships
Flexible classification boundary
📈 Visualizations Included
PCA Scatter Plot
Decision Boundary Visualization
Feature Distribution Graphs
Correlation Heatmaps
Kernel Comparison Graphs

🎯 Future Scope
Integration with Deep Learning
Mobile Application Development
Real-time Hospital Data Integration
Cloud Deployment
Multi-Cancer Prediction System
AI Chatbot Assistance
📚 Dataset

Dataset used:

Breast Cancer Wisconsin Dataset

Available from:

Kaggle
Scikit-learn datasets
👨‍💻 Author

Bollu Bhavya Sri
Machine Learning Project
