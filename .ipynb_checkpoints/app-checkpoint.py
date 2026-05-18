import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import base64

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Breast Cancer System", layout="wide")

# ---------------- BACKGROUND FUNCTION ----------------
def add_bg(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """, unsafe_allow_html=True)

# ---------------- SESSION ----------------
if "page" not in st.session_state:
    st.session_state.page = "home"

def go_to_app():
    st.session_state.page = "app"

def go_home():
    st.session_state.page = "home"

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    df = df.drop(["id", "Unnamed: 32"], axis=1)
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
    return df

df = load_data()
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# ---------------- SCALER ----------------
scaler = pickle.load(open("scaler.pkl", "rb"))
X_scaled = scaler.transform(X)

# ---------------- PCA ----------------
@st.cache_data
def get_pca(X_scaled):
    pca = PCA(n_components=2)
    return pca.fit_transform(X_scaled)

X_pca = get_pca(X_scaled)

# ---------------- MODEL ----------------
@st.cache_resource
def train_models(kernel):
    kernel_map = {
        "Linear": "linear",
        "Polynomial": "poly",
        "RBF": "rbf"
    }

    k = kernel_map[kernel]

    model_pred = SVC(kernel=k, probability=True)
    model_pred.fit(X_scaled, y)

    model_vis = SVC(kernel=k)
    model_vis.fit(X_pca, y)

    return model_pred, model_vis

# =====================================================
# 🏠 HOME PAGE
# =====================================================
if st.session_state.page == "home":

    add_bg("bg.jpg")

    st.markdown("""
    <h1 style='text-align:center;'>🏥 AI Breast Cancer Prediction System</h1>
    <h3 style='text-align:center;'>Machine Learning Diagnostic Tool</h3>
    """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.button("🚀 Start Analysis", on_click=go_to_app)

# =====================================================
# 📊 MAIN APP (NO BACKGROUND)
# =====================================================
else:

    st.sidebar.title("⚙ Controls")

    kernel = st.sidebar.radio(
        "Choose Kernel",
        ["Linear", "Polynomial", "RBF"]
    )

    view = st.sidebar.radio(
        "📊 Panel",
        ["Prediction", "Decision Boundary", "Dataset Insights"]
    )

    model_pred, model_vis = train_models(kernel)

    st.title("AI Analysis Dashboard")

    # ---------------- PREDICTION ----------------
    if view == "Prediction":

        st.subheader("🧪 Patient Prediction")

        col1, col2 = st.columns(2)

        with col1:
            radius = st.number_input("Radius Mean", value=14.0)
            texture = st.number_input("Texture Mean", value=20.0)
            perimeter = st.number_input("Perimeter Mean", value=90.0)

        with col2:
            area = st.number_input("Area Mean", value=600.0)
            smoothness = st.number_input("Smoothness Mean", value=0.1)
            compactness = st.number_input("Compactness Mean", value=0.2)

        if st.button("Predict"):

            input_data = np.array(X.mean()).reshape(1, -1)

            input_data[0][0] = radius
            input_data[0][1] = texture
            input_data[0][2] = perimeter
            input_data[0][3] = area
            input_data[0][4] = smoothness
            input_data[0][5] = compactness

            input_scaled = scaler.transform(input_data)

            pred = model_pred.predict(input_scaled)
            prob = model_pred.predict_proba(input_scaled)[0][1]

            if pred[0] == 1:
                st.error(f"🔴 High Risk - CANCER (Malignant) (Risk: {prob*100:.2f}%)")
            else:
                st.success(f"🟢 Low Risk - NON CANCEROUS (Benign) (Risk: {prob*100:.2f}%)")

    # ---------------- DECISION BOUNDARY ----------------
    elif view == "Decision Boundary":

        st.subheader("📊 Decision Boundary")

        x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
        y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 200),
            np.linspace(y_min, y_max, 200)
        )

        Z = model_vis.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        fig, ax = plt.subplots()

        ax.contourf(xx, yy, Z, alpha=0.3)

        scatter = ax.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=y,
            edgecolors='k',
            cmap=plt.cm.coolwarm
        )

        handles = [
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=plt.cm.coolwarm(0.1), markeredgecolor='k',
                       markersize=8, label="Benign"),
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=plt.cm.coolwarm(0.9), markeredgecolor='k',
                       markersize=8, label="Malignant")
        ]

        ax.legend(handles=handles, title="Diagnosis")
        ax.set_title(f"{kernel} Kernel Decision Boundary")

        st.pyplot(fig)

    # ---------------- DATASET INSIGHTS ----------------
    elif view == "Dataset Insights":

        st.subheader("📌 Dataset Insights & Visualization")

        st.markdown("### 📊 Distribution of Benign vs Malignant Cases")

        diagnosis_counts = df["diagnosis"].value_counts().rename({
            0: "Benign",
            1: "Malignant"
        })

        st.bar_chart(diagnosis_counts)

        st.markdown("---")

        st.markdown("### 📈 Average Feature Values")

        st.line_chart(df.drop("diagnosis", axis=1).mean())

        st.markdown("---")

        st.markdown("### ⚙️ Kernel Accuracy Comparison")

        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        kernels = ["linear", "poly", "rbf"]
        accuracies = []

        for k in kernels:
            m = SVC(kernel=k)
            m.fit(X_train, y_train)
            y_pred = m.predict(X_test)
            accuracies.append(accuracy_score(y_test, y_pred))

        acc_df = pd.DataFrame({
            "Kernel": kernels,
            "Accuracy": accuracies
        })

        st.bar_chart(acc_df.set_index("Kernel"))

    st.sidebar.button("⬅ Back to Home", on_click=go_home)
