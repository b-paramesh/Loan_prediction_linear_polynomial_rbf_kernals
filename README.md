# 💳 Smart Loan Approval System (SVM)

An AI-powered **Loan Approval Prediction System** built using **Support Vector Machines (SVM)** and deployed with **Streamlit**.  
The application predicts whether a loan should be **Approved or Rejected** based on applicant details and displays a confidence score with risk classification.

---

## 📌 Project Overview

Financial institutions receive a large number of loan applications every day.  
Manual verification is time-consuming and may lead to inconsistent decisions.

This project demonstrates how **Machine Learning** can assist loan officers by:
- Analyzing applicant financial and personal data
- Predicting loan approval status
- Providing confidence scores and risk indicators

This system acts as a **decision-support tool** and not a replacement for human judgment.

---

## 🚀 Features

- ✅ Loan approval prediction using **Support Vector Machines (SVM)**
- 🔁 Multiple kernel options:
  - Linear SVM
  - Polynomial SVM
  - RBF SVM
- 📊 Confidence score for each prediction
- 🚦 Risk classification:
  - 🟢 Low Risk
  - 🟡 Medium Risk
  - 🔴 High Risk
- 🧠 Real-time predictions using **Streamlit**
- 📋 Applicant summary and easy-to-understand decision explanation

---

## 🧠 Machine Learning Workflow

1. **Dataset Loading**
   - Loan dataset loaded from a CSV file

2. **Data Preprocessing**
   - Missing values handled using median and mode
   - Categorical variables encoded using `LabelEncoder`
   - Feature scaling performed using `StandardScaler`

3. **Model Training**
   - Dataset split into training (80%) and testing (20%)
   - Support Vector Classifier trained with selected kernel

4. **Prediction**
   - User input scaled using trained scaler
   - Model predicts loan approval
   - Probability used to calculate confidence score

---

## 🛠️ Tech Stack

- **Language:** Python  
- **Frontend:** Streamlit  
- **Machine Learning:** Scikit-learn  
- **Data Processing:** Pandas, NumPy  
- **Model:** Support Vector Machine (SVM)

---

## 📂 Project Structure

📁 Smart-Loan-Approval-System
│
├── app.py # Streamlit application
├── train_u6lujuX_CVtuZ9i.csv # Loan dataset
├── requirements.txt # Project dependencies
└── README.md # Project documentation


---

## 📊 Dataset Description

The dataset contains the following features:

- Gender  
- Married  
- Dependents  
- Education  
- Self Employed  
- Applicant Income  
- Coapplicant Income  
- Loan Amount  
- Loan Amount Term  
- Credit History  
- Property Area  
- Loan Status (Target Variable)

---

## ▶️ How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/smart-loan-approval-system.git
cd smart-loan-approval-system
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Run the Streamlit App
streamlit run app.py
4️⃣ Open in Browser
http://localhost:8501
🧪 Sample Output
Loan Decision: Approved / Rejected

Confidence Score: e.g., 85.32%

Risk Level: Low / Medium / High

Kernel Used: Linear / Polynomial / RBF

Decision Explanation: Business-friendly reasoning

📌 Business Use Case
This system can be used by:

Banks

Financial institutions

FinTech companies

To:

Reduce manual loan evaluation

Improve decision consistency

Minimize default risk

⚠️ This application should not be used as the sole authority for loan approval.

🔮 Future Enhancements
📈 Model accuracy comparison dashboard

🧠 Feature importance and explainability

🌐 Cloud deployment

🔐 User authentication
