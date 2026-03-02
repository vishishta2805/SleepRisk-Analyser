# Sleep Risk Analyzer

A data-driven application that predicts the risk of common sleep disorders such as **Insomnia** and **Sleep Apnea** based on lifestyle and physiological indicators.

This project demonstrates a complete end-to-end machine learning workflow — from exploratory data analysis to deployment.

---

## Problem Statement

Sleep disorders are influenced by multiple behavioral and cardiovascular factors.  
This system analyzes inputs such as sleep duration, stress level, BMI category, blood pressure, and physical activity to estimate disorder risk.

The goal is to build an interpretable predictive system and deploy it through an interactive web interface.

---

## Project Workflow

1. **Exploratory Data Analysis (EDA)**
   - Feature distribution analysis
   - Correlation analysis
   - Class imbalance inspection

2. **Data Preprocessing**
   - Handling categorical variables
   - Feature scaling
   - Missing value handling
   - Pipeline construction

3. **Model Training**
   - Random Forest Classifier
   - Train-test split validation
   - Cross-validation

4. **Model Evaluation**
   - Accuracy
   - Precision / Recall
   - F1-score
   - Confusion Matrix

5. **Deployment**
   - Streamlit web application
   - Interactive user inputs
   - Real-time predictions
   - Feature importance visualization

---

## Tech Stack

- Python
- Pandas
- Scikit-learn
- NumPy
- Streamlit
- Joblib

---

## Project Structure

```
Sleep-Risk-Analyzer/
│
├── app.py
├── style.css
├── requirements.txt
│
├── src/
│   ├── train.py
│   ├── evaluate.py
│   └── preprocessing.py
│
├── notebooks/
│   └── eda.ipynb
│
├── models/
│   └── pipeline.pkl
```
## How to Run Locally

1. Clone the repository
- git clone https://github.com/vishishta2805/SleepRisk-Analyser.git
- cd Sleep-Risk-Analyzer

2. Create virtual environment
- python -m venv venv
- source venv/bin/activate # Mac/Linux

3. Install dependencies
- pip install -r requirements.txt

4. Run the application
- streamlit run app.py
