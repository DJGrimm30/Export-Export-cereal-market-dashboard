import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import os

# --- Configuration ---
# Output CSV filename from Open Food Facts scraper
OPENFOODFACTS_CSV = "openfoodfacts_cereals.csv"
EUROSTAT_CSV = "eurostat_retail_sales.csv" # Ensure this file is in your project folder

# --- Load Eurostat Retail Sales CSV ---
@st.cache_data
def load_eurostat_data():
    """
    Loads Eurostat retail sales data from a CSV file.
    Assumes the CSV has 'date', 'geo', and 'value' columns.
    """
    if not os.path.exists(EUROSTAT_CSV):
        st.error(f"Error: Eurostat data file '{EUROSTAT_CSV}' not found. Please ensure it's in the same folder as the app.")
        return pd.DataFrame()
    try:
        df = pd.read_csv(EUROSTAT_CSV)
        # Rename the first column to 'date' for consistency, assuming it's the date column
        # This is a common pattern for Eurostat TSV/CSV files
        if df.columns[0] == 'DATAFLOW': # A common Eurostat header that needs to be ignored or remapped
             # Attempt to find actual data columns based on Eurostat format
             # Often, the first column after metadata is the time period
            if 'TIME_PERIOD' in df.columns:
                df = df.rename(columns={'TIME_PERIOD': 'date'})
            elif 'time' in df.columns:
                df = df.rename(columns={'time': 'date'})
            else:
                df = df.rename(columns={df.columns[0]: 'date'}) # Fallback to first column
        else:
            df = df.rename(columns={df.columns[0]: 'date'})

        # Convert 'date' column to datetime objects
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        # Drop rows where date conversion failed
        df.dropna(subset=['date'], inplace=True)

        # Ensure 'geo' and 'value' columns exist for plotting
        if 'geo' not in df.columns:
            st.warning(f"Warning: 'geo' column not found in {EUROSTAT_CSV}. Please ensure your Eurostat CSV has a 'geo' column for countries.")
            # Attempt to infer country column if 'geo' is missing, e.g., 'UNIT' or 'na_item'
            # Or handle by dropping this section of the UI gracefully
            df['geo'] = 'Unknown' # Placeholder

        if 'value' not in df.columns:
            st.warning(f"Warning: 'value' column not found in {EUROSTAT_CSV}. Please ensure your Eurostat CSV has a 'value' column for the index.")
            df['value'] = 0 # Placeholder

        return df
    except Exception as e:
        st.error(f"Failed to load or parse Eurostat data from '{EUROSTAT_CSV}': {e}")
        return pd.DataFrame()

# --- Load Open Food Facts Product Data ---
@st.cache_data
def load_openfoodfacts_data():
    """
    Loads cereal product data from the openfoodfacts_cereals.csv file.
    """
    if not os.path.exists(OPENFOODFACTS_CSV):
        st.warning(f"Product data file '{OPENFOODFACTS_CSV}' not found. Please run openfoodfacts_scraper.py first to generate it.")
        return pd.DataFrame()
    try:
        df = pd.read_csv(OPENFOODFACTS_CSV, encoding='utf-8')
        return df
    except Exception as e:
        st.error(f"Failed to load or parse product data from '{OPENFOODFACTS_CSV}': {e}")
        return pd.DataFrame()

# --- Streamlit UI ---
st.set_page_config(page_title="European Cereal Tracker", layout="wide", initial_sidebar_state="expanded")
st.title("🥣 European Retail Cereal Market Dashboard")

# Create tabs for navigation
# No reference to 'carrefour_cereals.csv' anymore
tab1, tab2, tab3 = st.tabs(["Open Food Facts Product Data", "Eurostat Retail Sales Trends", "Product Breakdown by Store/Brand"])

with tab1:
    st.header("Open Food Facts Cereal Product Data")
    st.write("This tab displays product information for cereals collected from Open Food Facts across various European countries.")

    df_off = load_openfoodfacts_data()

    if not df_off.empty:
        st.subheader("Raw Product Data Sample")
        st.dataframe(df_off.head(20))

        st.subheader("Product Count by Country")
        country_count = df_off['Country'].value_counts().reset_index()
        country_count.columns = ['Country', 'Count']
        fig_country = px.bar(country_count, x='Country', y='Count', title='Number of Products Listed by Country')
        st.plotly_chart(fig_country)

        st.subheader("Top 10 Brands by Product Count")
        brand_count = df_off['Brand'].value_counts().head(10).reset_index()
        brand_count.columns = ['Brand', 'Count']
        fig_brand = px.bar(brand_count, x='Brand', y='Count', title='Top 10 Brands in Database')
        st.plotly_chart(fig_brand)

        st.download_button(
            "Download Open Food Facts Data CSV",
            data=df_off.to_csv(index=False).encode('utf-8'),
            file_name=OPENFOODFACTS_CSV,
            mime="text/csv",
        )
    else:
        st.info("No Open Food Facts product data available. Please ensure `openfoodfacts_cereals.csv` exists and is accessible.")


with tab2:
    st.header("Eurostat Retail Sales Trends")
    st.write("This tab visualizes the retail trade index for food products from Eurostat, offering insights into market performance over time.")

    eurostat_df = load_eurostat_data()

    if not eurostat_df.empty:
        # Filter for relevant product group if necessary, assuming 'FOOD' or a similar category exists
        # In actual Eurostat data, you might need to filter by 'nace_r2' or other dimensions
        # For simplicity, we'll assume the loaded CSV is already filtered or general enough
        
        available_countries = eurostat_df['geo'].unique()
        if 'UK' in available_countries: # Ensure UK is an option if relevant
            available_countries = ['UK'] + [c for c in available_countries if c != 'UK'] # Prioritize UK if present

        selected_country = st.selectbox(
            "Select country for sales trend (Eurostat)", 
            options=available_countries,
            key="eurostat_country_select" # Unique key for selectbox
        )
        
        filtered_eurostat = eurostat_df[eurostat_df['geo'] == selected_country].copy()
        
        if not filtered_eurostat.empty:
            # Sort by date for proper line chart
            filtered_eurostat = filtered_eurostat.sort_values(by='date')
            fig_sales = px.line(
                filtered_eurostat, 
                x="date", 
                y="value", 
                title=f"Retail Sales Index in {selected_country} (Eurostat)", 
                labels={"value": "Retail Trade Index (Volume)", "date": "Date"},
                hover_data={"value": ":.2f"} # Format hover for index value
            )
            st.plotly_chart(fig_sales)
        else:
            st.info(f"No Eurostat sales data found for {selected_country}. Please check the data file.")
    else:
        st.info("No Eurostat retail sales data available. Please ensure `eurostat_retail_sales.csv` is correctly loaded.")

with tab3:
    st.header("Product Breakdown by Store and Brand")
    st.write("Explore the distribution of products by the stores they are sold in and analyze brands present in the dataset.")

    df_off_tab3 = load_openfoodfacts_data() # Reload data for this tab

    if not df_off_tab3.empty:
        # Clean 'Store' data for better visualization (e.g., split by comma)
        df_off_tab3['Store_List'] = df_off_tab3['Store'].astype(str).apply(lambda x: [s.strip() for s in x.split(',') if s.strip()])
        exploded_stores = df_off_tab3.explode('Store_List')

        if not exploded_stores.empty:
            st.subheader("Product Listings by Store (from Open Food Facts)")
            store_count = exploded_stores['Store_List'].value_counts().reset_index()
            store_count.columns = ['Store', 'Count']
            # Filter out generic or 'N/A' stores
            store_count = store_count[~store_count['Store'].isin(['N/A', ''])]
            if not store_count.empty:
                fig_store = px.bar(
                    store_count.head(20), # Show top 20 stores
                    x='Store', 
                    y='Count', 
                    title='Top 20 Product Listings by Store (Open Food Facts)',
                    labels={'Count': 'Number of Products'}
                )
                st.plotly_chart(fig_store)
            else:
                st.info("No meaningful store data found in Open Food Facts.")
        else:
            st.info("No store data to display.")

        st.subheader("Brand Distribution (from Open Food Facts)")
        brand_selector = st.selectbox(
            "Select a Brand to view its products", 
            options=['All Brands'] + sorted(df_off_tab3['Brand'].dropna().unique().tolist()),
            key="brand_select"
        )
        
        if brand_selector != 'All Brands':
            filtered_by_brand = df_off_tab3[df_off_tab3['Brand'] == brand_selector]
            st.dataframe(filtered_by_brand[['Product', 'Quantity', 'Store', 'Country', 'URL']])
        else:
            st.info("Select a brand from the dropdown to see its products.")

    else:
        st.info("No product data available from Open Food Facts to analyze store and brand breakdown.")
