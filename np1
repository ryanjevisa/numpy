import streamlit as st
import pandas as pd
import numpy as np
import io

# Optional: For visualization and geospatial
import matplotlib.pyplot as plt
import seaborn as sns
from autoviz.AutoViz_Class import AutoViz_Class
import geopandas as gpd

# Title and Sidebar
st.set_page_config(page_title="Data Science Practical Lab", layout="wide")
st.title("🧪 Data Science Practical Lab")
st.sidebar.title("Modules")

# File uploader
st.header("1️⃣ Upload Your Dataset")
uploaded_file = st.file_uploader("Upload CSV, Excel, or Text file", type=['csv', 'xlsx', 'xls', 'txt'])
df = None

if uploaded_file:
    # Handling different formats
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith('.txt'):
        df = pd.read_csv(uploaded_file, delimiter="\t")
    else:
        st.error("Unsupported file format")
    st.success(f"Loaded file: {uploaded_file.name}")

# MODULE 1: Exploration & Collection
st.sidebar.subheader("Module 1: Exploration")
with st.expander("🔍 Module 1: Exploration of Real-World Datasets"):
    col1, col2, col3 = st.columns(3)
    if col1.button("Show Raw Data"):
        if df is not None:
            st.dataframe(df)
        else:
            st.warning("Please upload a dataset.")

    if col2.button("Analyze Dataset Format"):
        if df is not None:
            st.write(f"Shape: {df.shape}")
            st.write("Columns:", df.columns.tolist())
            st.write("Dtypes:", df.dtypes)
        else:
            st.warning("Please upload a dataset.")

    if col3.button("Collect Data from API"):
        st.info("API data collection demo (customize as needed).")
        # Example: st.code("import requests\nrequests.get('https://api.example.com/data')")

# MODULE 2: Data Processing
st.sidebar.subheader("Module 2: Processing")
with st.expander("🛠️ Module 2: Data Processing"):
    col1, col2 = st.columns(2)
    if col1.button("Numpy & Pandas Operations"):
        if df is not None:
            st.write(df.describe())
            st.write("Null Values:", df.isnull().sum())
        else:
            st.warning("Please upload a dataset.")

    if col2.button("Feature Analysis (scikit-learn)"):
        st.info("Analyze individual features (e.g., using SelectKBest, etc.)")

# MODULE 3: Visualization
st.sidebar.subheader("Module 3: Visualization")
with st.expander("📊 Module 3: Visualization"):
    col1, col2 = st.columns(2)
    if col1.button("Seaborn & Matplotlib"):
        if df is not None:
            st.write("Example Histogram:")
            st.pyplot(plt.hist(df.select_dtypes(include=np.number).iloc[:,0].dropna()))
        else:
            st.warning("Please upload a dataset.")

    if col2.button("AutoViz"):
        st.info("Autoviz will auto-visualize your dataset (install and configure).")

# MODULE 4: Data Cleaning
st.sidebar.subheader("Module 4: Cleaning")
with st.expander("🧹 Module 4: Data Cleaning"):
    col1, col2, col3 = st.columns(3)
    if col1.button("Impute Missing Values"):
        st.info("Use pandas/scikit-learn to impute missing values.")

    if col2.button("Data Normalization"):
        st.info("Normalize data using scikit-learn.")

    if col3.button("Clean Data (Python)"):
        st.info("Show cleaning options (remove duplicates, etc.)")

# MODULE 5: Geospatial & Temporal
st.sidebar.subheader("Module 5: Geospatial & Temporal")
with st.expander("🗺️ Module 5: Geospatial & Temporal Data"):
    col1, col2 = st.columns(2)
    if col1.button("Geospatial Analysis"):
        st.info("Upload shapefile/GeoJSON or use geopandas for spatial analysis.")

    if col2.button("Temporal Visualization"):
        st.info("Visualize time series data.")

# MODULE 6: Data Transformation & Wrangling
st.sidebar.subheader("Module 6: Transformation & Wrangling")
with st.expander("🔄 Module 6: Data Transformation & Wrangling"):
    col1, col2 = st.columns(2)
    if col1.button("Transform Data"):
        st.info("Apply transformations, encode categorical features, etc.")

    if col2.button("Wrangle Data"):
        st.info("Reshape, melt, or pivot your dataset as needed.")

# Style: Make it attractive with icons and colors
st.markdown(
    """
    <style>
        .stButton>button {background-color: #4CAF50; color: white;}
        .st-expanderHeader {font-size: 20px;}
    </style>
    """, unsafe_allow_html=True
)

st.sidebar.markdown("Made with ❤️ using Streamlit")

# Note: Expand each button's code to suit your course/practical needs!
