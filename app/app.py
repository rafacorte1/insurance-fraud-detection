import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import load_data, preprocess


st.set_page_config(page_title="Insurance Fraud Detection", layout="wide")

@st.cache_data
def get_data():
    df = load_data("data/insurance_claims.csv")
    return df

@st.cache_data
def get_model(df):
    df_model = preprocess(df)
    X = df_model.drop(columns=['fraud_reported'])
    y = df_model['fraud_reported']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test, X.columns

df = get_data()
model, X_test, y_test, feature_names = get_model(df)

page = st.sidebar.selectbox("Navigate", ["Overview", "Fraud Analysis", "Feature Insights", "ML Prediction"])

# --- PAGE 1: OVERVIEW ---
if page == "Overview":
    st.title("🛡️ Insurance Fraud Detection Dashboard")
    total = len(df)
    fraud = df['fraud_reported'].value_counts().get('Y', 0)
    fraud_rate = round(fraud / total * 100, 2)
    avg_claim = round(df['total_claim_amount'].mean(), 2)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Claims", total)
    col2.metric("Fraud Rate", f"{fraud_rate}%")
    col3.metric("Avg Claim Amount", f"${avg_claim:,}")

    fig = px.pie(df, names='fraud_reported', title='Fraud vs Non-Fraud Claims',
                 color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig, use_container_width=True)

# --- PAGE 2: FRAUD ANALYSIS ---
elif page == "Fraud Analysis":
    st.title("📊 Fraud Analysis")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x='incident_type', color='fraud_reported',
                           barmode='group', title='Fraud by Incident Type')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.histogram(df, x='auto_make', color='fraud_reported',
                           barmode='group', title='Fraud by Vehicle Category')
        st.plotly_chart(fig, use_container_width=True)

    fig = px.box(df, x='fraud_reported', y='total_claim_amount',
                 title='Claim Amount Distribution by Fraud Status',
                 color='fraud_reported')
    st.plotly_chart(fig, use_container_width=True)

# --- PAGE 3: FEATURE INSIGHTS ---
elif page == "Feature Insights":
    st.title("🔍 Feature Insights")
    df_proc = preprocess(df)
    corr = df_proc.corr()[['fraud_reported']].drop('fraud_reported').sort_values('fraud_reported')
    fig = px.bar(corr, x=corr.index, y='fraud_reported',
                 title='Feature Correlation with Fraud',
                 labels={'fraud_reported': 'Correlation'})
    st.plotly_chart(fig, use_container_width=True)

# --- PAGE 4: ML PREDICTION ---
elif page == "ML Prediction":
    st.title("🤖 ML Fraud Prediction")
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Precision", f"{report['1']['precision']:.2f}")
    col2.metric("Recall", f"{report['1']['recall']:.2f}")
    col3.metric("F1-Score", f"{report['1']['f1-score']:.2f}")

    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(10)
    fig = px.bar(importances, title='Top 10 Feature Importances',
                 labels={'index': 'Feature', 'value': 'Importance'})
    st.plotly_chart(fig, use_container_width=True)

    cm = confusion_matrix(y_test, y_pred)
    fig_cm = px.imshow(cm, text_auto=True, title='Confusion Matrix',
                       labels=dict(x="Predicted", y="Actual"),
                       color_continuous_scale='Blues')
    st.plotly_chart(fig_cm, use_container_width=True)