import streamlit as st
import pandas as pd
import plotly.express as px
import os
import re

# --- Configuration ---
OPENFOODFACTS_CSV = "openfoodfacts_breakfast_products.csv" 
EUROSTAT_RETAIL_CSV = "eurostat_retail_sales.csv" 
ONS_RETAIL_SALES_CSV = "ons_retail_sales.csv" 
EUROSTAT_CONSUMPTION_CSV = "eurostat_consumption_expenditure.csv" 
EUROSTAT_ECOMMERCE_CSV = "eurostat_ecommerce_buyers.csv" 
EUROSTAT_POPULATION_CSV = "eurostat_population.csv" 

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="European Breakfast Product Tracker", 
    layout="wide", 
    initial_sidebar_state="expanded"
    # Theme applied via .streamlit/config.toml
)

st.title("🥣 European Breakfast Product Market Dashboard")
st.markdown("---") 

# --- Data Loading Functions ---

@st.cache_data
def load_eurostat_retail_sales_specific(filepath):
    """
    Loads eurostat_retail_sales.csv which is assumed to be already flat with 'date', 'geo', 'value' columns.
    """
    if not os.path.exists(filepath):
        st.error(f"Error: Data file '{filepath}' not found. Please ensure it's in the same folder as the app.")
        return pd.DataFrame()
    try:
        df = pd.read_csv(filepath)
        
        # Standardize column names if they are slightly different
        if 'Date' in df.columns: df = df.rename(columns={'Date': 'date'})
        if 'Geo' in df.columns: df = df.rename(columns={'Geo': 'geo'})
        if 'Value' in df.columns: df = df.rename(columns={'Value': 'value'})
        
        # Ensure required columns exist
        if 'date' not in df.columns or 'geo' not in df.columns or 'value' not in df.columns:
            st.warning(f"Retail sales CSV '{filepath}' missing expected columns (date, geo, value). Found: {df.columns.tolist()}")
            return pd.DataFrame()
            
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df.dropna(subset=['date'], inplace=True)
        
        return df[['date', 'geo', 'value']].copy()

    except Exception as e:
        st.error(f"Failed to load or parse Eurostat retail sales data from '{filepath}': {e}")
        return pd.DataFrame()


@st.cache_data
def load_eurostat_long_format_data(filepath, specific_filters=None):
    """
    Loads and preprocesses Eurostat CSVs that are in 'long' format (have TIME_PERIOD and OBS_VALUE).
    """
    if not os.path.exists(filepath):
        st.error(f"Error: Data file '{filepath}' not found. Please ensure it's in the same folder as the app.")
        return pd.DataFrame()
    try:
        df = pd.read_csv(filepath)
        
        # Standardize all column names to lowercase immediately after loading
        df.columns = [col.lower().strip() for col in df.columns]
        
        time_period_col = None
        obs_value_col = None

        if 'time_period' in df.columns: time_period_col = 'time_period'
        if 'obs_value' in df.columns: obs_value_col = 'obs_value'

        if not time_period_col or not obs_value_col:
            st.warning(f"Warning: Expected 'time_period' and 'obs_value' in '{filepath}'. Found: {df.columns.tolist()}")
            return pd.DataFrame()
        
        df_processed = df.copy()
        
        df_processed = df_processed.rename(columns={
            time_period_col: 'date',
            obs_value_col: 'value',
            'geo': 'geo' # 'geo' should already be lowercase
        })

        # Standardize other common Eurostat dimension columns (case-insensitive)
        for original_col in df_processed.columns.tolist():
            if original_col.upper() == 'UNIT': df_processed = df_processed.rename(columns={original_col: 'unit'})
            elif original_col.upper() == 'FREQ': df_processed = df_processed.rename(columns={original_col: 'freq'})
            elif original_col.upper() == 'COFOG_L2': df_processed = df_processed.rename(columns={original_col: 'cofog_l2'})
            elif original_col.upper() == 'IND_TYPE': df_processed = df_processed.rename(columns={original_col: 'ind_type'})
            elif original_col.upper() == 'AGE': df_processed = df_processed.rename(columns={original_col: 'age'})
            elif original_col.upper() == 'SEX': df_processed = df_processed.rename(columns={original_col: 'sex'})
            elif original_col.upper() == 'NACE_R2': df_processed = df_processed.rename(columns={original_col: 'nace_r2'}) 
            elif original_col.upper() == 'INDIC_BT': df_processed = df_processed.rename(columns={original_col: 'indic_bt'}) 


        df_processed['value'] = pd.to_numeric(
            df_processed['value'].astype(str).str.replace(r'[a-zA-Z\s:]', '', regex=True), 
            errors='coerce'
        )
        df_processed.dropna(subset=['value'], inplace=True)

        def parse_eurostat_date(date_str):
            if pd.isna(date_str): return pd.NaT
            date_str = str(date_str).strip()
            if re.match(r'^\d{4}$', date_str): return pd.to_datetime(date_str, format='%Y')
            elif re.match(r'^\d{4}Q[1-4]$', date_str): return pd.to_datetime(f'{date_str[:4]}-{int(date_str[5])*3-2}-01')
            elif re.match(r'^\d{4}M\d{2}$', date_str): return pd.to_datetime(date_str, format='%YM%m')
            return pd.NaT

        df_processed['date'] = df_processed['date'].apply(parse_eurostat_date)
        df_processed.dropna(subset=['date'], inplace=True)

        if specific_filters:
            for col_filter_name, filter_value in specific_filters.items():
                found_filter_col = None
                for col in df_processed.columns:
                    if col_filter_name.lower() == col.lower(): # Match case-insensitively
                        found_filter_col = col
                        break

                if found_filter_col:
                    df_processed = df_processed[df_processed[found_filter_col].astype(str).str.strip() == str(filter_value).strip()]
                else:
                    st.warning(f"Filter column '{col_filter_name}' not found in DataFrame for filtering '{filepath}'. Available columns: {df_processed.columns.tolist()}")
            
            if df_processed.empty:
                st.info(f"No data remaining in '{filepath}' after applying filters: {specific_filters}")
                return pd.DataFrame()

        required_final_cols = ['date', 'geo', 'value']
        if 'geo' not in df_processed.columns:
            st.warning(f"Final 'geo' column missing after processing {filepath}. Defaulting to 'Unknown'.")
            df_processed['geo'] = 'Unknown' 
        if 'value' not in df_processed.columns:
            st.warning(f"Final 'value' column missing after processing {filepath}. Defaulting to 0.")
            df_processed['value'] = 0 

        return df_processed[required_final_cols].copy()

    except Exception as e:
        st.error(f"Failed to load or parse Eurostat data from '{filepath}': {e}")
        return pd.DataFrame()

@st.cache_data
def load_ons_retail_sales_data():
    """Loads ONS Retail Sales data from a CSV file."""
    if not os.path.exists(ONS_RETAIL_SALES_CSV):
        st.error(f"Error: ONS data file '{ONS_RETAIL_SALES_CSV}' not found. Please download it manually from ONS and place it in the folder.")
        return pd.DataFrame()
    try:
        df = pd.read_csv(ONS_RETAIL_SALES_CSV)
        
        date_col_name = None
        for col in df.columns:
            if 'date' in col.lower() or 'month' in col.lower() or 'time' in col.lower() or 'year' in col.lower():
                date_col_name = col
                break
        if not date_col_name:
            if len(df.columns) >= 4:
                date_col_name = df.columns[3]
            else:
                st.warning(f"Could not identify date column in ONS data: {df.columns.tolist()}")
                return pd.DataFrame()

        value_col_name = None
        for col in df.columns:
            if 'value' in col.lower() or 'obs' in col.lower():
                value_col_name = col
                break
        if not value_col_name:
            if len(df.columns) >= 5:
                value_col_name = df.columns[4]
            else:
                st.warning(f"Could not identify value column in ONS data: {df.columns.tolist()}")
                return pd.DataFrame()
        
        ons_df = df[[date_col_name, value_col_name]].copy()
        ons_df.columns = ['date', 'value']

        ons_df['date'] = pd.to_datetime(ons_df['date'], errors='coerce')
        ons_df.dropna(subset=['date', 'value'], inplace=True)
        ons_df['geo'] = 'UK'

        return ons_df[['date', 'geo', 'value']].copy()
    except Exception as e:
        st.error(f"Failed to load or parse ONS retail sales data from '{ONS_RETAIL_SALES_CSV}': {e}")
        return pd.DataFrame()

@st.cache_data
def load_openfoodfacts_data():
    """Loads breakfast product data from the openfoodfacts_breakfast_products.csv file."""
    if not os.path.exists(OPENFOODFACTS_CSV):
        st.warning(f"Product data file '{OPENFOODFACTS_CSV}' not found. Please run `openfoodfacts_scraper.py` first to generate it.")
        return pd.DataFrame()
    try:
        df = pd.read_csv(OPENFOODFACTS_CSV, encoding='utf-8')
        return df
    except Exception as e:
        st.error(f"Failed to load or parse product data from '{OPENFOODFACTS_CSV}': {e}")
        return pd.DataFrame()

# --- Data Processing for Insights ---
def calculate_yoy_growth(df, value_col='value', date_col='date', geo_col='geo'):
    """Calculates Year-over-Year growth for a DataFrame."""
    if df.empty or value_col not in df.columns or date_col not in df.columns or geo_col not in df.columns:
        return pd.DataFrame()
    
    df_sorted = df.sort_values(by=[geo_col, date_col]).copy()
    df_sorted['year'] = df_sorted[date_col].dt.year
    
    if df_sorted[date_col].dt.month.nunique() == 1 and df_sorted[date_col].dt.day.nunique() == 1:
        shift_period = 1 # For yearly data
    elif df_sorted[date_col].dt.month.nunique() <= 4 and df_sorted[date_col].dt.day.nunique() == 1:
        shift_period = 4 # For quarterly data
    else:
        shift_period = 12 # For monthly data
    
    df_sorted['prev_period_value'] = df_sorted.groupby(geo_col)[value_col].shift(shift_period)
    df_sorted['YoY_Growth'] = ((df_sorted[value_col] - df_sorted['prev_period_value']) / df_sorted['prev_period_value']) * 100
    
    return df_sorted.dropna(subset=['YoY_Growth'])

def calculate_per_capita(value_df, population