import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from autoviz.AutoViz_Class import AutoViz_Class
import geopandas as gpd

AV = AutoViz_Class()

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Data Science Practical Lab", layout="wide", page_icon="📊")

st.title("📘 Data Science Practical Experiments Dashboard")
st.markdown("### Interactive Tool for Performing Data Science Lab Modules")

# --- Global Variable ---
if "df" not in st.session_state:
    st.session_state.df = None

# --- Helper: Load Dataset ---
def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith(".txt"):
            return pd.read_csv(uploaded_file, delimiter="\t")
        else:
            st.error("Unsupported file format.")
    except Exception as e:
        st.error(f"Error loading file: {e}")

# --- Upload Section ---
st.sidebar.header("📂 Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a dataset", type=["csv", "xlsx", "xls", "txt"])
if uploaded_file:
    st.session_state.df = load_data(uploaded_file)
    st.success(f"Loaded {uploaded_file.name} successfully!")

# --- Module 1: Exploration of Real-World Datasets ---
st.subheader("🧩 Module 1: Data Exploration & Analysis")

if st.session_state.df is not None:
    df = st.session_state.df

    if st.button("🔍 View Dataset Info"):
        st.write("**First 5 Rows of Dataset:**")
        st.dataframe(df.head())
        st.write("**Shape:**", df.shape)
        st.write("**Columns:**", df.columns.tolist())

    if st.button("📈 Analyze Dataset Statistics"):
        st.write("**Descriptive Statistics:**")
        st.dataframe(df.describe(include="all"))

    if st.button("🔬 Feature Analysis (Sklearn-style Overview)"):
        st.write("**Feature Types:**")
        st.write(df.dtypes)
else:
    st.warning("Please upload a dataset to begin.")

# --- Module 2: Data Processing using NumPy & Pandas ---
st.subheader("🧮 Module 2: Data Processing")

if st.session_state.df is not None:
    df = st.session_state.df

    if st.button("🧹 Clean Missing Values"):
        df.fillna(df.mean(numeric_only=True), inplace=True)
        st.success("Missing values imputed successfully!")

    if st.button("⚖️ Normalize Numeric Columns"):
        numeric_cols = df.select_dtypes(include=np.number).columns
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        st.success("Numeric columns normalized (Min-Max Scaling).")

    st.session_state.df = df

# --- Module 3: Visualization ---
st.subheader("📊 Module 3: Data Visualization")

if st.session_state.df is not None:
    df = st.session_state.df

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🎨 Seaborn Visualization"):
            numeric_cols = df.select_dtypes(include=np.number).columns
            if len(numeric_cols) >= 2:
                st.write("Pairplot using Seaborn:")
                fig = sns.pairplot(df[numeric_cols])
                st.pyplot(fig)
            else:
                st.warning("Need at least 2 numeric columns for visualization.")

    with col2:
        if st.button("🤖 AutoViz Visualization"):
            temp_path = "temp_data.csv"
            df.to_csv(temp_path, index=False)
            st.info("Launching AutoViz... (check logs in terminal)")
            AV.AutoViz(temp_path)
else:
    st.info("Upload a dataset to visualize.")

# --- Module 4: Data Cleaning & Normalization ---
st.subheader("🧼 Module 4: Cleaning & Preprocessing")

if st.session_state.df is not None:
    df = st.session_state.df
    if st.button("🩺 View Missing Values Summary"):
        st.write(df.isnull().sum())
    if st.button("🧾 Drop Duplicates"):
        df.drop_duplicates(inplace=True)
        st.success("Duplicate rows removed successfully!")
    st.session_state.df = df

# --- Module 5: Geospatial & Temporal Visualization ---
st.subheader("🗺️ Module 5: Geospatial and Temporal Data")

if st.session_state.df is not None:
    df = st.session_state.df

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🌍 Geospatial Visualization"):
            try:
                world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
                fig, ax = plt.subplots(figsize=(8, 6))
                world.plot(ax=ax, color='lightblue', edgecolor='black')
                plt.title("Geospatial Map Example")
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Geospatial visualization failed: {e}")

    with col2:
        if st.button("⏳ Temporal Visualization"):
            datetime_cols = df.select_dtypes(include=['datetime64']).columns
            if len(datetime_cols) > 0:
                df['Year'] = df[datetime_cols[0]].dt.year
                year_counts = df['Year'].value_counts().sort_index()
                fig, ax = plt.subplots()
                year_counts.plot(kind='bar', ax=ax)
                plt.title("Temporal Distribution by Year")
                st.pyplot(fig)
            else:
                st.warning("No datetime column found for temporal analysis.")
else:
    st.info("Upload dataset for geospatial/temporal visualization.")

# --- Module 6: Data Transformation & Wrangling ---
st.subheader("🧰 Module 6: Transformation & Wrangling")

if st.session_state.df is not None:
    df = st.session_state.df
    if st.button("🔄 Transform Data (Demo)"):
        df_transformed = df.copy()
        df_transformed = df_transformed.apply(lambda x: x*2 if np.issubdtype(x.dtype, np.number) else x)
        st.success("Numeric columns doubled (demo transformation).")
        st.write(df_transformed.head())

    if st.button("🧩 Wrangle Data"):
        st.write("Dropping duplicates and resetting index...")
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)
        st.success("Data wrangling completed.")
        st.dataframe(df.head())
    st.session_state.df = df

# --- Footer ---
st.markdown("---")
st.markdown("#### 💡 Developed by Ryan’s AI Assistant | Powered by Streamlit")
