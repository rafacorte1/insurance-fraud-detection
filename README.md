# 🛡️ Insurance Fraud Detection

An end-to-end data analysis and machine learning project to detect fraudulent auto insurance claims — featuring exploratory data analysis (EDA), a Random Forest classification model, and an interactive Streamlit dashboard.

---

## 📌 Project Overview

Insurance fraud costs the US industry over **$40 billion per year** (FBI estimate). This project explores a real-world auto insurance claims dataset to identify patterns associated with fraudulent behavior and build a predictive model to flag suspicious claims.

This project demonstrates skills in:
- Exploratory Data Analysis (EDA)
- Feature Engineering & Preprocessing
- Machine Learning (Classification)
- Interactive Dashboard Development

---

## 🚀 Live Demo

> Run locally following the instructions below.

![Dashboard Preview](assets/dashboard_preview.png)

---

## 📊 Dashboard Pages

| Page | Description |
|---|---|
| **Overview** | Total claims, fraud rate KPIs and fraud distribution |
| **Fraud Analysis** | Fraud breakdown by incident type, vehicle make and claim amount |
| **Feature Insights** | Correlation of features with fraud outcome |
| **ML Prediction** | Model performance metrics, confusion matrix and feature importance |

---

## 🤖 ML Model

| | |
|---|---|
| **Algorithm** | Random Forest Classifier |
| **Train/Test Split** | 80% / 20% |
| **Key Metrics** | Precision, Recall, F1-Score |
| **Explainability** | Feature Importance Chart |

---

## 🛠️ Tech Stack

- **Language:** Python 3.11+
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Visualization:** Plotly, Streamlit
- **Version Control:** Git & GitHub

---

## 📁 Project Structure

The project is organized as follows:

- **data/** — Dataset files (not included, see instructions below)
- **notebooks/** — Exploratory Data Analysis (`01_eda.ipynb`)
- **src/** — Data loading and preprocessing utilities (`utils.py`)
- **app/** — Streamlit dashboard (`app.py`)
- `requirements.txt` — Python dependencies
- `README.md` — Project documentation

---

## ⚙️ Getting Started

**1. Clone the repository**

```bash
git clone https://github.com/rafacorte1/insurance-fraud-detection.git
cd insurance-fraud-detection
```

**2. Create and activate a virtual environment**

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

Mac/Linux:
```bash
source .venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Download the dataset**

Download the dataset from Kaggle and place the CSV file inside the `data/` folder:

🔗 [Auto Insurance Claims Dataset — Kaggle](https://www.kaggle.com/datasets/buntyshah/auto-insurance-claims-data)

The file should be placed at: `data/insurance_claims.csv`

**5. Run the dashboard**

```bash
streamlit run app/app.py
```

---

## 📈 Key Findings

- The dataset presents a fraud rate of approximately **24%**
- Claims involving **Major Damage** incidents show significantly higher fraud rates
- **Incident type** and **insured occupation** are among the strongest predictors of fraud
- The Random Forest model achieves strong performance in identifying fraudulent claims

---

## 👤 Author

**Rafael Corte**  
Data Analyst & Automation Specialist

[![LinkedIn](https://img.shields.io/badge/LinkedIn-rafaelcorte-blue?logo=linkedin)](https://linkedin.com/in/rafaelcorte)
[![GitHub](https://img.shields.io/badge/GitHub-rafacorte1-black?logo=github)](https://github.com/rafacorte1)

---

## 📄 License

This project is licensed under the MIT License.
