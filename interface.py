import streamlit as st
import pandas as pd
import sqlite3
import os

st.set_page_config(page_title="Data Analysis Dashboard", layout="wide")

# --- Keep track of active page ---
if "page" not in st.session_state:
    st.session_state.page = "Data Upload"

# --- Sidebar Buttons ---
st.sidebar.title("📊 Dashboard Panel")
if st.sidebar.button("Data Upload", use_container_width=True):
    st.session_state.page = "Data Upload"
if st.sidebar.button("Raw Data", use_container_width=True):
    st.session_state.page = "Raw Data"
if st.sidebar.button("Data Cleaning", use_container_width=True):
    st.session_state.page = "Data Cleaning"
if st.sidebar.button("Data Visualization", use_container_width=True):
    st.session_state.page = "Data Visualization"
if st.sidebar.button("Report Download", use_container_width=True):
    st.session_state.page = "Report Download"

# --- Keep DataFrame and file info in session ---
if "df" not in st.session_state:
    st.session_state.df = None
if "file_ext" not in st.session_state:
    st.session_state.file_ext = None
if "sqlite_file" not in st.session_state:
    st.session_state.sqlite_file = None

# --- Data Upload Page ---
if st.session_state.page == "Data Upload":
    st.title("📂 Upload Your Data File")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["csv", "xls", "xlsx", "json", "txt", "sqlite", "db"]
    )

    if uploaded_file is not None:
        file_name = uploaded_file.name
        st.session_state.file_ext = os.path.splitext(file_name)[1].lower()
        file_size = uploaded_file.size
        size_str = f"{file_size / 1024:.2f} KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.2f} MB"

        # File details box
        st.markdown(
            f"""
            <div style="background-color:#1e1e1e; padding:15px; border-radius:10px; border: 1px solid #444;">
                <h4 style="margin:0;">📑 File Details</h4>
                <p><b>Filename:</b> {file_name}</p>
                <p><b>File Type:</b> {st.session_state.file_ext.upper().replace('.', '')}</p>
                <p><b>File Size:</b> {size_str}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        try:
            if st.session_state.file_ext == ".csv":
                st.session_state.df = pd.read_csv(uploaded_file)
            elif st.session_state.file_ext in [".xls", ".xlsx"]:
                st.session_state.df = pd.read_excel(uploaded_file)
            elif st.session_state.file_ext == ".json":
                st.session_state.df = pd.read_json(uploaded_file)
            elif st.session_state.file_ext == ".txt":
                st.session_state.df = pd.read_csv(uploaded_file, delimiter="\t")
            elif st.session_state.file_ext in [".sqlite", ".db"]:
                # Save SQLite file to disk temporarily
                temp_path = f"temp_{file_name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.session_state.sqlite_file = temp_path

                # Listing the tables...........................................
                conn = sqlite3.connect(temp_path)
                tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
                st.write("**📂 Tables found in SQL file:**", tables)
                conn.close()
            st.success("✅ File uploaded successfully!")
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")

# -----Raw Data Page------
elif st.session_state.page == "Raw Data":
    st.title("📜 Raw Data")
    if st.session_state.df is not None:
        st.dataframe(st.session_state.df)
    elif st.session_state.file_ext in [".sqlite", ".db"] and st.session_state.sqlite_file:
        # Load first table from SQLite
        conn = sqlite3.connect(st.session_state.sqlite_file)
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
        if not tables.empty:
            first_table = tables.iloc[0, 0]
            df_sql = pd.read_sql_query(f"SELECT * FROM {first_table}", conn)
            st.dataframe(df_sql)
        conn.close()
    else:
        st.warning("⚠ No data uploaded yet. Please upload a file first.")

# -----Data Cleaning Page-----
elif st.session_state.page == "Data Cleaning":
    st.title("🧹 Data Cleaning")
    if st.session_state.df is not None:
        st.write("Add cleaning functions here (remove duplicates, handle missing values, etc.)")
    else:
        st.warning("⚠ No data uploaded yet. Please upload a file first.")

# -----Data Visualization Page-----
elif st.session_state.page == "Data Visualization":
    st.title("📊 Data Visualization")
    if st.session_state.df is not None:
        st.write("Add charts/plots here.")
    else:
        st.warning("⚠ No data uploaded yet. Please upload a file first.")

# -----Report Download Page-----
elif st.session_state.page == "Report Download":
    st.title("📥 Report Download")
    if st.session_state.df is not None:
        st.write("Add PDF/Excel export options here.")
    else:
        st.warning("⚠ No data uploaded yet. Please upload a file first.")
