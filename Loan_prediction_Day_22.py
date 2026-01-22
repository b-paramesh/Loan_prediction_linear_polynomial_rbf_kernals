import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# ======================================
# APP CONFIG
# ======================================
st.set_page_config(
    page_title="Smart Loan Approval System",
    layout="centered"
)

# ======================================
# TITLE & DESCRIPTION
# ======================================
st.title("💳 Smart Loan Approval System")
st.write(
    "This system uses **Support Vector Machines (SVM)** to predict whether a loan "
    "should be **Approved or Rejected** based on applicant details."
)

st.markdown("---")

# ======================================
# LOAD DATASET (FROM SAME FOLDER)
# ======================================
try:
    df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")
except Exception as e:
    st.error("❌ Dataset not found. Please keep CSV in the same folder.")
    st.stop()

# ======================================
# DATA PREPROCESSING
# ======================================
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median(), inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

label_cols = [
    'Gender', 'Married', 'Dependents', 'Education',
    'Self_Employed', 'Property_Area', 'Loan_Status'
]

le = LabelEncoder()
for col in label_cols:
    df[col] = le.fit_transform(df[col])

X = df.drop(['Loan_ID', 'Loan_Status'], axis=1)
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# ======================================
# SIDEBAR INPUT SECTION
# ======================================
st.sidebar.header("📝 Applicant Details")

applicant_income = st.sidebar.number_input(
    "Applicant Income", min_value=0.0, step=500.0
)

loan_amount = st.sidebar.number_input(
    "Loan Amount", min_value=0.0, step=10.0
)

credit_history = st.sidebar.selectbox(
    "Credit History", ["Yes", "No"]
)

employment_status = st.sidebar.selectbox(
    "Employment Status", ["Not Self Employed", "Self Employed"]
)

property_area = st.sidebar.selectbox(
    "Property Area", ["Rural", "Semiurban", "Urban"]
)

credit_history = 1 if credit_history == "Yes" else 0
self_employed = 1 if employment_status == "Self Employed" else 0
property_area = {"Rural": 0, "Semiurban": 1, "Urban": 2}[property_area]

# ======================================
# MODEL SELECTION
# ======================================
st.sidebar.header("⚙️ SVM Kernel Selection")

kernel_choice = st.sidebar.radio(
    "Choose kernel:",
    ["Linear SVM", "Polynomial SVM", "RBF SVM"]
)

# ======================================
# PREDICTION BUTTON
# ======================================
if st.sidebar.button("🔍 Check Loan Eligibility"):
    with st.spinner("Analyzing applicant profile..."):

        if kernel_choice == "Linear SVM":
            model = SVC(kernel="linear", probability=True)
        elif kernel_choice == "Polynomial SVM":
            model = SVC(kernel="poly", degree=3, probability=True)
        else:
            model = SVC(kernel="rbf", probability=True)

        model.fit(X_train, y_train)

        user_input = np.array([[
            1, 1, 0, 1,
            self_employed,
            applicant_income,
            0,
            loan_amount,
            360,
            credit_history,
            property_area
        ]])

        user_input = scaler.transform(user_input)

        prediction = model.predict(user_input)[0]
        confidence = model.predict_proba(user_input)[0].max() * 100

    # ======================================
    # OUTPUT SECTION
    # ======================================
    st.subheader("📊 Loan Decision Result")

    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.write(f"**Confidence Score:** {confidence:.2f}%")
    st.write(f"**Kernel Used:** {kernel_choice}")

    # ======================================
    # RISK INDICATOR
    # ======================================
    if confidence >= 80:
        st.success("🟢 Low Risk Applicant")
    elif confidence >= 50:
        st.warning("🟡 Medium Risk Applicant")
    else:
        st.error("🔴 High Risk Applicant")

    # ======================================
    # APPLICANT SUMMARY
    # ======================================
    with st.expander("📋 Applicant Summary"):
        st.write(f"Applicant Income: {applicant_income}")
        st.write(f"Loan Amount: {loan_amount}")
        st.write(f"Credit History: {'Good' if credit_history==1 else 'Bad'}")
        st.write(f"Employment Status: {'Self Employed' if self_employed==1 else 'Not Self Employed'}")
        st.write(f"Property Area: {property_area}")

    # ======================================
    # BUSINESS EXPLANATION
    # ======================================
    st.subheader("📌 Decision Explanation")

    if prediction == 1:
        st.write(
            "Based on the applicant's **credit history, income level, and employment pattern**, "
            "the model predicts that the applicant is **likely to repay the loan**."
        )
    else:
        st.write(
            "Based on the applicant's **financial profile and credit history**, "
            "the model identifies a **higher risk of loan default**, hence the loan is rejected."
        )

# ======================================
# FOOTER
# ======================================
st.markdown("---")
st.caption(
    "⚠️ This system is an AI-based decision support tool and should not be used "
    "as the sole authority for loan approval."
)
