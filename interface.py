import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
import tempfile
import io
from pathlib import Path

# Optional libs (used when available)
try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    import plotly.express as px
except Exception:
    px = None

try:
    import matplotlib.pyplot as plt
    from moviepy.editor import ImageSequenceClip
except Exception:
    plt = None

st.set_page_config(page_title="Smart Data Analysis Dashboard", layout="wide")

# -------------------------
# --- Session state defaults
# -------------------------
if "page" not in st.session_state:
    st.session_state.page = "Data Upload"
if "files" not in st.session_state:
    # each item: {"name":..., "ext":".csv", "path": None or temp path, "df": DataFrame or None, "raw": uploaded object}
    st.session_state.files = []
if "active_file_idx" not in st.session_state:
    st.session_state.active_file_idx = 0
if "temp_files" not in st.session_state:
    st.session_state.temp_files = []
if "orig_dfs" not in st.session_state:
    st.session_state.orig_dfs = []  # list of original dfs (for reset)
if "pages" not in st.session_state:
    st.session_state.pages = ["Data Upload", "Raw Data", "Data Cleaning", "Data Visualization", "Report & Export", "Hypothesis Summary"]

# -------------------------
# --- Helper functions
# -------------------------
def _save_temp_file(uploaded):
    """Save uploaded file to temporary path and return the path"""
    suffix = Path(uploaded.name).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.getbuffer())
    tmp.flush()
    tmp.close()
    st.session_state.temp_files.append(tmp.name)
    return tmp.name

def _read_pdf_to_df(path):
    """Try to extract tables from PDF using pdfplumber if available. Returns a concatenated DataFrame or None."""
    if not pdfplumber:
        raise RuntimeError("pdfplumber not installed. Install with pip install pdfplumber to parse PDFs.")
    tables = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            try:
                tbl = page.extract_table()
                if tbl and len(tbl) > 1:
                    df = pd.DataFrame(tbl[1:], columns=tbl[0])
                    tables.append(df)
            except Exception:
                continue
    if tables:
        return pd.concat(tables, ignore_index=True)
    return None

def _cleanup_temp_files():
    for p in st.session_state.temp_files:
        try:
            os.remove(p)
        except Exception:
            pass
    st.session_state.temp_files = []

def _load_file_to_dict(uploaded):
    """Given a streamlit uploaded file (UploadedFile), return a dict with metadata and loaded df if possible."""
    name = uploaded.name
    ext = Path(name).suffix.lower()
    file_entry = {"name": name, "ext": ext, "path": None, "df": None, "raw": uploaded}

    try:
        if ext == ".csv":
            df = pd.read_csv(uploaded)
            file_entry["df"] = df
        elif ext in [".xls", ".xlsx"]:
            df = pd.read_excel(uploaded)
            file_entry["df"] = df
        elif ext == ".json":
            df = pd.read_json(uploaded)
            file_entry["df"] = df
        elif ext == ".txt":
            # try tab delimited first
            try:
                df = pd.read_csv(uploaded, delimiter="\t")
                file_entry["df"] = df
            except Exception:
                uploaded.seek(0)
                df = pd.read_csv(uploaded)
                file_entry["df"] = df
        elif ext in [".sqlite", ".db"]:
            tmp_path = _save_temp_file(uploaded)
            file_entry["path"] = tmp_path
            # list tables
            conn = sqlite3.connect(tmp_path)
            try:
                tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
                file_entry["tables"] = tables['name'].tolist() if not tables.empty else []
                # try load first table by default
                if file_entry["tables"]:
                    table = file_entry["tables"][0]
                    file_entry["df"] = pd.read_sql_query(f"SELECT * FROM '{table}'", conn)
                    file_entry["loaded_table"] = table
            finally:
                conn.close()
        elif ext == ".pdf":
            tmp_path = _save_temp_file(uploaded)
            file_entry["path"] = tmp_path
            if pdfplumber:
                try:
                    df_pdf = _read_pdf_to_df(tmp_path)
                    if df_pdf is not None:
                        file_entry["df"] = df_pdf
                except Exception:
                    # leave df as None but file saved
                    pass
        else:
            # unsupported extension - keep raw, no df
            pass
    except Exception as e:
        # return the file entry with error info in a field
        file_entry["load_error"] = str(e)

    return file_entry

def _ensure_orig_dfs_length():
    # Keep orig_dfs aligned with files
    n_files = len(st.session_state.files)
    while len(st.session_state.orig_dfs) < n_files:
        entry = None
        st.session_state.orig_dfs.append(entry)
    if len(st.session_state.orig_dfs) > n_files:
        st.session_state.orig_dfs = st.session_state.orig_dfs[:n_files]

def _generate_hypotheses_for_df(df, name="Active File"):
    """Return a markdown string list of hypotheses/insights based on simple EDA rules."""
    if df is None:
        return f"**No DataFrame loaded for {name}.** Uploadable tables may exist (e.g., PDF/SQLite)."

    md = []
    n_rows, n_cols = df.shape
    md.append(f"**Dataset:** {name} — {n_rows} rows × {n_cols} columns")

    # missingness
    missing = df.isnull().mean().sort_values(ascending=False)
    high_missing = missing[missing > 0.3]
    if not high_missing.empty:
        md.append("**Missing data:** Some columns have >30% missing values which may bias analyses:")
        for col, frac in high_missing.items():
            md.append(f"- `{col}`: {frac:.0%} missing")
    else:
        md.append("- Missingness looks low (no column >30% missing).")

    # numeric correlations
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] >= 2:
        corr = num.corr().abs().unstack().sort_values(ascending=False).drop_duplicates()
        # get top non-1 correlations
        top_corr = [(a, b, v) for (a, b), v in corr.items() if a != b][:5]
        if top_corr:
            md.append("**Potential relationships (top correlations):**")
            for a, b, v in top_corr:
                md.append(f"- `{a}` vs `{b}` — correlation {v:.2f} — consider hypothesis: changes in `{a}` may be associated with changes in `{b}`")
    else:
        md.append("- Not enough numeric columns to compute correlations.")

    # categorical suggestions
    cat = df.select_dtypes(include=['object', 'category'])
    if not cat.empty:
        cat_summary = []
        for c in cat.columns[:5]:
            nunq = df[c].nunique(dropna=True)
            cat_summary.append((c, nunq))
        md.append("**Categorical columns (sample):**")
        for c, nunq in cat_summary:
            md.append(f"- `{c}`: {nunq} unique values — consider grouping low-frequency categories or encoding for models")
    # outliers hints (simple)
    if num.shape[1] > 0:
        outlier_notes = []
        for c in num.columns[:5]:
            q1 = num[c].quantile(0.25)
            q3 = num[c].quantile(0.75)
            iqr = q3 - q1
            if pd.isna(iqr) or iqr == 0:
                continue
            outlier_pct = ((num[c] < (q1 - 3*iqr)) | (num[c] > (q3 + 3*iqr))).mean()
            if outlier_pct > 0.01:
                outlier_notes.append((c, outlier_pct))
        if outlier_notes:
            md.append("**Outlier note:**")
            for c, pct in outlier_notes:
                md.append(f"- `{c}`: ~{pct:.1%} potential extreme outliers — check data quality or transformation need")

    # basic modeling hint
    if ('target' in df.columns) or (num.shape[1] >= 1 and len(cat.columns) >= 1):
        md.append("- Consider trying a simple predictive model if you have a target column or a clear KPI.")
    else:
        md.append("- No obvious target column detected. If you have one, rename it to 'target' for quick modeling hints.")

    return "\n".join(md)

# -------------------------
# --- Sidebar
# -------------------------
with st.sidebar:
    st.title("📊 Dashboard Panel")

    # Quick page buttons (keeps original behavior)
    for p in st.session_state.pages:
        if st.button(p, use_container_width=True):
            st.session_state.page = p

    st.markdown("---")
    st.write("**Files uploaded**")
    # file selector
    if st.session_state.files:
        names = [f["name"] for f in st.session_state.files]
        idx = st.selectbox("Active file", options=list(range(len(names))), format_func=lambda i: f"{i+1}. {names[i]}", index=st.session_state.active_file_idx)
        st.session_state.active_file_idx = idx
    else:
        st.info("No files uploaded yet. Go to Data Upload to add files.")

# -------------------------
# --- Page: Data Upload
# -------------------------
if st.session_state.page == "Data Upload":
    st.title("📂 Upload Your Data File(s)")
    st.write("You may upload **multiple** files. The app will keep each file separately. Select an active file on the left to work on it.")
    uploaded_files = st.file_uploader(
        "Choose file(s)",
        type=["csv", "xls", "xlsx", "json", "txt", "sqlite", "db", "pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        added = 0
        for uploaded in uploaded_files:
            # Avoid adding duplicates by name+size check
            already = any((f["name"] == uploaded.name and getattr(f.get("raw"), "size", None) == uploaded.size) for f in st.session_state.files)
            if already:
                continue
            entry = _load_file_to_dict(uploaded)
            st.session_state.files.append(entry)
            added += 1
        _ensure_orig_dfs_length()
        if added:
            st.success(f"Added {added} new file(s). Use the sidebar to select the active file.")
    st.markdown("---")

    if st.session_state.files:
        # show uploaded files table
        files_info = []
        for i, f in enumerate(st.session_state.files):
            files_info.append({
                "Index": i+1,
                "Name": f["name"],
                "Ext": f.get("ext"),
                "Has DF": bool(f.get("df") is not None),
                "Temp Path": bool(f.get("path") is not None)
            })
        st.dataframe(pd.DataFrame(files_info))

        # allow removing a file
        if st.button("Remove active file"):
            idx = st.session_state.active_file_idx
            if 0 <= idx < len(st.session_state.files):
                # if temp path exists, remove it
                p = st.session_state.files[idx].get("path")
                try:
                    if p and os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass
                st.session_state.files.pop(idx)
                # adjust index
                st.session_state.active_file_idx = max(0, st.session_state.active_file_idx - 1)
                _ensure_orig_dfs_length()
                st.rerun()

    if st.button("Next →"):
        st.session_state.page = st.session_state.pages[(st.session_state.pages.index(st.session_state.page) + 1) % len(st.session_state.pages)]
        st.rerun()

# -------------------------
# --- Page: Raw Data Explorer
# -------------------------
elif st.session_state.page == "Raw Data":
    st.title("📜 Raw Data Explorer")
    if not st.session_state.files:
        st.warning("⚠ No files uploaded yet. Please upload a file first.")
    else:
        # Option to compare first two files side-by-side
        compare = st.checkbox("Compare first two files side-by-side", value=False)
        if compare and len(st.session_state.files) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                st.header(f"File 1: {st.session_state.files[0]['name']}")
                df1 = st.session_state.files[0].get("df")
                if df1 is not None:
                    st.write(df1.head())
                    st.dataframe(df1)
                else:
                    st.info("No dataframe loaded for this file.")
            with col2:
                st.header(f"File 2: {st.session_state.files[1]['name']}")
                df2 = st.session_state.files[1].get("df")
                if df2 is not None:
                    st.write(df2.head())
                    st.dataframe(df2)
                else:
                    st.info("No dataframe loaded for this file.")
        else:
            active = st.session_state.files[st.session_state.active_file_idx]
            st.subheader(f"Active file: {active['name']}")
            df = active.get("df")
            if df is not None:
                cols = df.columns.tolist()
                with st.expander("Preview & Controls", expanded=True):
                    st.write(df.head())
                    view_cols = st.multiselect("Columns to display", cols, default=cols)
                    if view_cols:
                        st.dataframe(df[view_cols])

                # quick filters
                if st.checkbox("Enable quick filter"):
                    filter_col = st.selectbox("Filter column", cols)
                    op = st.selectbox("Operator", ["==", "!=", "contains", ">", "<", ">=", "<="]) 
                    val = st.text_input("Value to compare (interpreted as string or number)")
                    if st.button("Apply filter"):
                        try:
                            df2 = df.copy()
                            if op == "contains":
                                df2 = df2[df2[filter_col].astype(str).str.contains(val, na=False, case=False)]
                            else:
                                # numeric?
                                try:
                                    num = float(val)
                                    if op == "==":
                                        df2 = df2[df2[filter_col] == num]
                                    elif op == "!=":
                                        df2 = df2[df2[filter_col] != num]
                                    elif op == ">":
                                        df2 = df2[df2[filter_col] > num]
                                    elif op == "<":
                                        df2 = df2[df2[filter_col] < num]
                                    elif op == ">=":
                                        df2 = df2[df2[filter_col] >= num]
                                    elif op == "<=":
                                        df2 = df2[df2[filter_col] <= num]
                                except ValueError:
                                    if op == "==":
                                        df2 = df2[df2[filter_col].astype(str) == val]
                                    elif op == "!=":
                                        df2 = df2[df2[filter_col].astype(str) != val]
                            st.write(f"Filtered rows: {len(df2)}")
                            st.dataframe(df2)
                            # update in session (user asked that changes reflect)
                            st.session_state.files[st.session_state.active_file_idx]["df"] = df2
                        except Exception as e:
                            st.error(f"Could not apply filter: {e}")

                # sorting
                if st.checkbox("Enable sorting"):
                    sort_cols = st.multiselect("Columns to sort by", cols)
                    asc = st.radio("Order", ("Ascending", "Descending"))
                    if st.button("Apply sort") and sort_cols:
                        df_sorted = df.sort_values(by=sort_cols, ascending=(asc == "Ascending"))
                        st.session_state.files[st.session_state.active_file_idx]["df"] = df_sorted
                        st.success("Sorted")
                        st.rerun()
            else:
                st.info("No DataFrame available for this file. If it's a SQL/PDF file, try loading table from the Data Upload page or re-upload with a table-containing file.")

    if st.button("Next →"):
        st.session_state.page = st.session_state.pages[(st.session_state.pages.index(st.session_state.page) + 1) % len(st.session_state.pages)]
        st.rerun()

# # -------------------------
# # --- Page: Data Cleaning
# # -------------------------
# elif st.session_state.page == "Data Cleaning":
#     st.title("🧹 Data Cleaning")
#     if not st.session_state.files:
#         st.warning("⚠ No files uploaded yet. Please upload a file first.")
#     else:
#         active = st.session_state.files[st.session_state.active_file_idx]
#         st.subheader(f"Active file: {active['name']}")
#         df = active.get("df")

#         if df is None:
#             st.info("No DataFrame loaded for this file.")
#         else:
#             # --- Show data quality summary ---
#             st.write("### 🔎 Data Quality Overview")
#             st.write("**Basic Info:**")
#             st.write(df.describe(include="all").transpose())

#             st.write("**Missing Values per Column:**")
#             st.dataframe(df.isnull().sum().reset_index().rename(columns={"index": "Column", 0: "Missing Values"}))

#             # --- Cleaning options ---
#             st.write("### ⚙️ Cleaning Options")
#             # with st.form("clean_form"):
#             #     drop_dup = st.checkbox("Remove duplicate rows", value=False)

#             #     # Drop columns by missing threshold
#             #     drop_na_thresh = st.slider(
#             #         "Drop columns with more than X% missing values",
#             #         min_value=0, max_value=100, value=50, step=5
#             #     )

#             #     # Fill missing values
#             #     fill_na_cols = st.multiselect("Choose columns to fill missing values", df.columns.tolist())
#             #     fill_method = st.selectbox("Fill method", ["None", "Mean", "Median", "Mode", "Constant"])
#             #     const_val = None
#             #     if fill_method == "Constant":
#             #         const_val = st.text_input("Constant value to use for filling NAs", value="0")

#             #     apply_btn = st.form_submit_button("✅ Apply Cleaning")

#             # # --- Apply cleaning ---
#             # if apply_btn:
#             #     cleaned = df.copy()

#             #     # Remove duplicates
#             #     if drop_dup:
#             #         cleaned = cleaned.drop_duplicates()

#             #     # Drop columns above threshold
#             #     thresh = 1 - (drop_na_thresh / 100.0)
#             #     cleaned = cleaned.dropna(axis=1, thresh=thresh * len(cleaned))

#             #     # Fill missing values
#             #     for c in fill_na_cols:
#             #         if fill_method == "Mean" and pd.api.types.is_numeric_dtype(cleaned[c]):
#             #             cleaned[c] = cleaned[c].fillna(cleaned[c].mean())
#             #         elif fill_method == "Median" and pd.api.types.is_numeric_dtype(cleaned[c]):
#             #             cleaned[c] = cleaned[c].fillna(cleaned[c].median())
#             #         elif fill_method == "Mode":
#             #             cleaned[c] = cleaned[c].fillna(cleaned[c].mode()[0] if not cleaned[c].mode().empty else 0)
#             #         elif fill_method == "Constant":
#             #             cleaned[c] = cleaned[c].fillna(const_val)

#             #     st.session_state.files[st.session_state.active_file_idx]["df"] = cleaned
#             #     st.success("✨ Cleaning applied! Data has been updated.")
#             #     st.dataframe(cleaned.head())

#     if st.button("Next →"):
#         st.session_state.page = st.session_state.pages[
#             (st.session_state.pages.index(st.session_state.page) + 1) % len(st.session_state.pages)
#         ]
#         st.rerun()
# -------------------------
# --- Page: Data Cleaning
# -------------------------
elif st.session_state.page == "Data Cleaning":
    st.title("🧹 Data Cleaning")

    if not st.session_state.files:
        st.warning("⚠ No files uploaded yet. Please upload a file first.")
    else:
        active = st.session_state.files[st.session_state.active_file_idx]
        st.subheader(f"Active file: {active['name']}")
        df = active.get("df")

        if df is None:
            st.info("No DataFrame loaded for this file.")
        else:
            st.write("### 🔎 Data Overview")
            st.write("Shape:", df.shape)
            st.dataframe(df.head())

            # --- Missing Values ---
            with st.expander("📉 Handle Missing Values"):
                st.write(df.isnull().sum().reset_index().rename(columns={"index": "Column", 0: "Missing Values"}))
                missing_option = st.radio(
                    "Choose missing value handling method",
                    ["Do Nothing", "Drop Rows", "Drop Columns", "Fill with Mean", "Fill with Median", "Fill with Mode", "Fill with Constant"]
                )
                if missing_option == "Fill with Constant":
                    const_val = st.text_input("Enter constant value:", "0")

                if st.button("Apply Missing Value Handling"):
                    cleaned = df.copy()
                    if missing_option == "Drop Rows":
                        cleaned = cleaned.dropna()
                    elif missing_option == "Drop Columns":
                        cleaned = cleaned.dropna(axis=1)
                    elif missing_option == "Fill with Mean":
                        cleaned = cleaned.fillna(cleaned.mean(numeric_only=True))
                    elif missing_option == "Fill with Median":
                        cleaned = cleaned.fillna(cleaned.median(numeric_only=True))
                    elif missing_option == "Fill with Mode":
                        for col in cleaned.columns:
                            if cleaned[col].isnull().any():
                                cleaned[col].fillna(cleaned[col].mode()[0], inplace=True)
                    elif missing_option == "Fill with Constant":
                        cleaned = cleaned.fillna(const_val)

                    st.session_state.files[st.session_state.active_file_idx]["df"] = cleaned
                    st.success("✅ Missing values handled!")
                    st.dataframe(cleaned.head())

            # --- Duplicates ---
            with st.expander("📌 Handle Duplicates"):
                dup_count = df.duplicated().sum()
                st.write(f"Found **{dup_count}** duplicate rows.")
                if dup_count > 0:
                    if st.button("Remove Duplicates"):
                        cleaned = df.drop_duplicates()
                        st.session_state.files[st.session_state.active_file_idx]["df"] = cleaned
                        st.success("✅ Duplicates removed!")
                        st.dataframe(cleaned.head())

            # --- Outliers ---
            with st.expander("📊 Handle Outliers (IQR Method)"):
                num = df.select_dtypes(include=[np.number])
                if not num.empty:
                    col = st.selectbox("Choose column", num.columns)
                    q1, q3 = num[col].quantile([0.25, 0.75])
                    iqr = q3 - q1
                    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    outliers = ((num[col] < lower) | (num[col] > upper)).sum()
                    st.write(f"Detected **{outliers}** outliers in column `{col}`.")

                    action = st.radio("Action", ["Do Nothing", "Remove Outliers", "Cap at Bounds"])
                    if st.button("Apply Outlier Handling"):
                        cleaned = df.copy()
                        if action == "Remove Outliers":
                            cleaned = cleaned[(cleaned[col] >= lower) & (cleaned[col] <= upper)]
                        elif action == "Cap at Bounds":
                            cleaned[col] = np.where(cleaned[col] < lower, lower,
                                                    np.where(cleaned[col] > upper, upper, cleaned[col]))
                        st.session_state.files[st.session_state.active_file_idx]["df"] = cleaned
                        st.success("✅ Outlier handling applied!")
                        st.dataframe(cleaned.head())
                else:
                    st.info("No numeric columns available for outlier detection.")

            # --- Column Standardization ---
            with st.expander("📝 Standardize Columns"):
                rename_cols = st.checkbox("Convert column names to lowercase & trim spaces")
                if st.button("Apply Column Standardization") and rename_cols:
                    cleaned = df.copy()
                    cleaned.columns = cleaned.columns.str.strip().str.lower().str.replace(" ", "_")
                    st.session_state.files[st.session_state.active_file_idx]["df"] = cleaned
                    st.success("✅ Column names standardized!")
                    st.dataframe(cleaned.head())

    if st.button("Next →"):
        st.session_state.page = st.session_state.pages[(st.session_state.pages.index(st.session_state.page) + 1) % len(st.session_state.pages)]
        st.rerun()


# -------------------------
# --- Page: Data Visualization
# -------------------------
# elif st.session_state.page == "Data Visualization":
#     st.title("📈 Data Visualization")
#     if not st.session_state.files:
#         st.warning("⚠ No files uploaded yet. Please upload a file first.")
#     else:
#         active = st.session_state.files[st.session_state.active_file_idx]
#         st.subheader(f"Active file: {active['name']}")
#         df = active.get("df")
#         if df is not None:
#             st.write("Preview:", df.head())

#             chart_type = st.selectbox("Chart type", ["Line", "Bar", "Histogram", "Boxplot", "Scatter", "Pie"])
#             cols = df.columns.tolist()
#             if chart_type in ["Line", "Bar", "Scatter"]:
#                 x = st.selectbox("X axis", cols)
#                 y = st.selectbox("Y axis", cols)
#                 if px:
#                     if chart_type == "Line":
#                         fig = px.line(df, x=x, y=y)
#                     elif chart_type == "Bar":
#                         fig = px.bar(df, x=x, y=y)
#                     else:
#                         fig = px.scatter(df, x=x, y=y)
#                     st.plotly_chart(fig, use_container_width=True)
#                 else:
#                     st.error("Plotly not installed.")
#             elif chart_type == "Histogram":
#                 col = st.selectbox("Column", cols)
#                 if px:
#                     fig = px.histogram(df, x=col)
#                     st.plotly_chart(fig, use_container_width=True)
#             elif chart_type == "Boxplot":
#                 col = st.selectbox("Column", cols)
#                 if px:
#                     fig = px.box(df, y=col)
#                     st.plotly_chart(fig, use_container_width=True)
#             elif chart_type == "Pie":
#                 labels = st.selectbox("Labels column", cols)
#                 values = st.selectbox("Values column", cols)
#                 if px:
#                     fig = px.pie(df, names=labels, values=values)
#                     st.plotly_chart(fig, use_container_width=True)

#         else:
#             st.info("No DataFrame available for this file.")

#     if st.button("Next →"):
#         st.session_state.page = st.session_state.pages[(st.session_state.pages.index(st.session_state.page) + 1) % len(st.session_state.pages)]
#         st.rerun()
elif st.session_state.page == "Data Visualization":
    st.title("📈 Data Visualization")
    if not st.session_state.files:
        st.warning("⚠ No files uploaded yet. Please upload a file first.")
    else:
        if len(st.session_state.files) >= 2:
            # If two or more files, show side by side
            col1, col2 = st.columns(2)
            for i, col in enumerate([col1, col2]):
                with col:
                    file = st.session_state.files[i]
                    st.subheader(f"File {i+1}: {file['name']}")
                    df = file.get("df")
                    if df is not None:
                        st.write("Preview:", df.head())

                        # Default chart type as None
                        chart_type = st.selectbox(
                            f"Chart type for {file['name']}", 
                            ["None", "Line", "Bar", "Histogram", "Boxplot", "Scatter", "Pie"], 
                            key=f"chart_type_{i}"
                        )

                        if chart_type != "None":
                            cols = df.columns.tolist()
                            if chart_type in ["Line", "Bar", "Scatter"]:
                                x = st.selectbox("X axis", cols, key=f"x_{i}")
                                y = st.selectbox("Y axis", cols, key=f"y_{i}")
                                if px:
                                    if chart_type == "Line":
                                        fig = px.line(df, x=x, y=y)
                                    elif chart_type == "Bar":
                                        fig = px.bar(df, x=x, y=y)
                                    else:
                                        fig = px.scatter(df, x=x, y=y)
                                    st.plotly_chart(fig, use_container_width=True)
                            elif chart_type == "Histogram":
                                col_sel = st.selectbox("Column", cols, key=f"hist_{i}")
                                if px:
                                    fig = px.histogram(df, x=col_sel)
                                    st.plotly_chart(fig, use_container_width=True)
                            elif chart_type == "Boxplot":
                                col_sel = st.selectbox("Column", cols, key=f"box_{i}")
                                if px:
                                    fig = px.box(df, y=col_sel)
                                    st.plotly_chart(fig, use_container_width=True)
                            elif chart_type == "Pie":
                                labels = st.selectbox("Labels column", cols, key=f"pie_labels_{i}")
                                values = st.selectbox("Values column", cols, key=f"pie_values_{i}")
                                if px:
                                    fig = px.pie(df, names=labels, values=values)
                                    st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No DataFrame available for this file.")
        else:
            # Single file case (same as before)
            active = st.session_state.files[st.session_state.active_file_idx]
            st.subheader(f"Active file: {active['name']}")
            df = active.get("df")
            if df is not None:
                st.write("Preview:", df.head())

                chart_type = st.selectbox(
                    "Chart type", 
                    ["None", "Line", "Bar", "Histogram", "Boxplot", "Scatter", "Pie"]
                )
                if chart_type != "None":
                    cols = df.columns.tolist()
                    if chart_type in ["Line", "Bar", "Scatter"]:
                        x = st.selectbox("X axis", cols)
                        y = st.selectbox("Y axis", cols)
                        if px:
                            if chart_type == "Line":
                                fig = px.line(df, x=x, y=y)
                            elif chart_type == "Bar":
                                fig = px.bar(df, x=x, y=y)
                            else:
                                fig = px.scatter(df, x=x, y=y)
                            st.plotly_chart(fig, use_container_width=True)
                    elif chart_type == "Histogram":
                        col = st.selectbox("Column", cols)
                        if px:
                            fig = px.histogram(df, x=col)
                            st.plotly_chart(fig, use_container_width=True)
                    elif chart_type == "Boxplot":
                        col = st.selectbox("Column", cols)
                        if px:
                            fig = px.box(df, y=col)
                            st.plotly_chart(fig, use_container_width=True)
                    elif chart_type == "Pie":
                        labels = st.selectbox("Labels column", cols)
                        values = st.selectbox("Values column", cols)
                        if px:
                            fig = px.pie(df, names=labels, values=values)
                            st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No DataFrame available for this file.")

    if st.button("Next →"):
        st.session_state.page = st.session_state.pages[
            (st.session_state.pages.index(st.session_state.page) + 1) % len(st.session_state.pages)
        ]
        st.rerun()


# -------------------------
# --- Page: Report & Export
# -------------------------
elif st.session_state.page == "Report & Export":
    st.title("📑 Report & Export")
    if not st.session_state.files:
        st.warning("⚠ No files uploaded yet. Please upload a file first.")
    else:
        active = st.session_state.files[st.session_state.active_file_idx]
        st.subheader(f"Active file: {active['name']}")
        df = active.get("df")
        if df is not None:
            st.write("Shape:", df.shape)
            st.write("Columns:", df.columns.tolist())

            if st.button("Export CSV"):
                csv = df.to_csv(index=False).encode()
                st.download_button("Download CSV", csv, file_name="export.csv", mime="text/csv")

            if st.button("Export Excel"):
                towrite = io.BytesIO()
                df.to_excel(towrite, index=False, engine='openpyxl')
                towrite.seek(0)
                st.download_button("Download Excel", towrite, file_name="export.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            if st.button("Export JSON"):
                js = df.to_json(orient="records").encode()
                st.download_button("Download JSON", js, file_name="export.json", mime="application/json")
        else:
            st.info("No DataFrame available for this file.")

    if st.button("Next →"):
        st.session_state.page = st.session_state.pages[(st.session_state.pages.index(st.session_state.page) + 1) % len(st.session_state.pages)]
        st.rerun()

# -------------------------
# --- Page: Hypothesis Summary
# -------------------------
# elif st.session_state.page == "Hypothesis Summary":
#     st.title("🧠 Hypothesis Summary")
#     if not st.session_state.files:
#         st.warning("⚠ No files uploaded yet.")
#     else:
#         for f in st.session_state.files:
#             st.subheader(f["name"])
#             md = _generate_hypotheses_for_df(f.get("df"), name=f["name"])
#             st.markdown(md)

#     if st.button("Back to Upload"):
#         st.session_state.page = "Data Upload"
#         st.rerun()
elif st.session_state.page == "Hypothesis Summary":
    st.title("🧠 Hypothesis Summary")
    
    if not st.session_state.files:
        st.warning("⚠ No files uploaded yet.")
    else:
        for f in st.session_state.files:
            st.markdown(f"## 📂 {f['name']}")
            df = f.get("df")

            if df is None:
                st.info("No DataFrame available.")
                continue

            # --- Missing Data ---
            with st.expander("📉 Missing Data"):
                missing = df.isnull().mean().sort_values(ascending=False)
                if missing.sum() == 0:
                    st.success("✅ No missing values detected.")
                else:
                    st.bar_chart(missing)
                    high_missing = missing[missing > 0.3]
                    if not high_missing.empty:
                        st.error(f"❌ Columns with >30% missing: {', '.join(high_missing.index)}")
                    else:
                        st.warning("⚠ Some columns have missing values (but <30%).")

            # --- Correlations ---
            num = df.select_dtypes(include=[np.number])
            if num.shape[1] >= 2:
                with st.expander("🔗 Potential Relationships"):
                    corr = num.corr()
                    st.dataframe(corr.round(2))
                    top_corr = corr.abs().unstack().sort_values(ascending=False)
                    top_corr = top_corr[top_corr < 1].head(5)
                    st.info("📌 Top correlations worth checking:")
                    for idx, val in top_corr.items():
                        st.write(f"- {idx[0]} vs {idx[1]} → correlation {val:.2f}")

            # --- Categorical Insights ---
            cat = df.select_dtypes(exclude=[np.number])
            if not cat.empty:
                with st.expander("🗂 Categorical Insights"):
                    st.write("🔎 Unique value counts:")
                    for col in cat.columns[:5]:
                        st.write(f"- {col}: {df[col].nunique()} unique values")
                        st.bar_chart(df[col].value_counts().head(10))

            # --- Outliers ---
            if not num.empty:
                with st.expander("📌 Outliers"):
                    for col in num.columns[:3]:  # show first 3 numeric columns
                        q1, q3 = num[col].quantile([0.25, 0.75])
                        iqr = q3 - q1
                        outliers = ((num[col] < (q1 - 1.5 * iqr)) | (num[col] > (q3 + 1.5 * iqr))).sum()
                        if outliers > 0:
                            st.error(f"❌ {col}: {outliers} outliers detected")
                        else:
                            st.success(f"✅ {col}: no major outliers")
                        st.box_chart = st.line_chart(num[col].sample(min(200, len(num[col]))))

            # --- Modeling Hint ---
            with st.expander("💡 Modeling Hint"):
                st.info("This dataset could be useful for predictive modeling. Consider selecting a target variable.")
    
    if st.button("Back to Upload"):
        st.session_state.page = "Data Upload"
