import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Page Configuration
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🛍️",
    layout="wide"
)

# Simple Styling
st.markdown("""
<style>
.main {
    background-color: #0E1117;
    color: white;
}
h1, h2, h3 {
    color: #4CAF50;
}
.sidebar .sidebar-content {
    background-color: #161A23;
}
</style>
""", unsafe_allow_html=True)

# Load Dataset
df = pd.read_csv("Mall_Customers.csv")

# Load Model and Scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Sidebar
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Home",
        "Dataset",
        "EDA",
        "Elbow Method",
        "Cluster Visualization",
        "Prediction"
    ]
)

# Home Page
if page == "Home":

    st.title("🛍️ Customer Segmentation using K-Means Clustering")

    st.write("""
    This project performs customer segmentation using:
    
    - Annual Income
    - Spending Score
    
    Machine Learning Algorithm Used:
    - K-Means Clustering
    """)

    st.image(
    "https://images.unsplash.com/photo-1520607162513-77705c0f0d4a",
    width=900
)

# Dataset Page
elif page == "Dataset":

    st.title("📂 Mall Customer Dataset")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Shape")
    st.write(df.shape)

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

# EDA Page
elif page == "EDA":

    st.title("📊 Exploratory Data Analysis")

    # Histogram
    st.subheader("Annual Income Distribution")

    fig, ax = plt.subplots(figsize=(8,5))

    sns.histplot(
        df['Annual Income (k$)'],
        kde=True,
        color='skyblue'
    )

    plt.title("Distribution of Annual Income")

    st.pyplot(fig)

    # Spending Score Distribution
    st.subheader("Spending Score Distribution")

    fig, ax = plt.subplots(figsize=(8,5))

    sns.histplot(
        df['Spending Score (1-100)'],
        kde=True,
        color='orange'
    )

    plt.title("Distribution of Spending Score")

    st.pyplot(fig)

    # Heatmap
    st.subheader("Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(6,4))

    sns.heatmap(
        df.corr(numeric_only=True),
        annot=True,
        cmap='coolwarm'
    )

    plt.title("Correlation Matrix")

    st.pyplot(fig)

# Elbow Method
elif page == "Elbow Method":

    st.title("📈 Elbow Method")

    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

    from sklearn.preprocessing import StandardScaler

    scaler_data = StandardScaler()

    X_scaled = scaler_data.fit_transform(X)

    from sklearn.cluster import KMeans

    wcss = []

    for i in range(1,11):

        kmeans = KMeans(
            n_clusters=i,
            random_state=42,
            n_init=10
        )

        kmeans.fit(X_scaled)

        wcss.append(kmeans.inertia_)

    fig, ax = plt.subplots(figsize=(8,5))

    plt.plot(range(1,11), wcss, marker='o')

    plt.title("Elbow Method")
    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS")

    st.pyplot(fig)

# Cluster Visualization
elif page == "Cluster Visualization":

    st.title("🎯 Customer Segments")

    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

    X_scaled = scaler.transform(X)

    clusters = model.predict(X_scaled)

    df['Cluster'] = clusters

    fig, ax = plt.subplots(figsize=(10,6))

    sns.scatterplot(
        x='Annual Income (k$)',
        y='Spending Score (1-100)',
        hue='Cluster',
        palette='Set2',
        data=df,
        s=100
    )

    plt.title("Customer Segmentation")

    # Cluster Labels
    plt.annotate(
        "High Income\nHigh Spending",
        xy=(90,85),
        bbox=dict(boxstyle="round", fc="yellow")
    )

    plt.annotate(
        "Low Income\nLow Spending",
        xy=(20,20),
        bbox=dict(boxstyle="round", fc="lightblue")
    )

    plt.annotate(
        "Average Customers",
        xy=(55,50),
        bbox=dict(boxstyle="round", fc="lightgreen")
    )

    st.pyplot(fig)

# Prediction Page
elif page == "Prediction":

    st.title(" Predict Customer Segment")

    income = st.slider(
        "Annual Income (k$)",
        10,
        150,
        50
    )

    spending = st.slider(
        "Spending Score (1-100)",
        1,
        100,
        50
    )

    if st.button("Predict Cluster"):

        input_data = np.array([[income, spending]])

        scaled_input = scaler.transform(input_data)

        prediction = model.predict(scaled_input)

        st.success(
            f"Customer belongs to Cluster {prediction[0]}"
        )

        if prediction[0] == 0:
            st.write("🛍️ Premium Customer")

        elif prediction[0] == 1:
            st.write("💰 Budget Customer")

        else:
            st.write("👤 Regular Customer")