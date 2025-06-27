import streamlit as st
import pandas as pd
import plotly.express as px
import os

# --- Configuration ---
# Output CSV filename from Open Food Facts scraper (now expanded to breakfast products)
OPENFOODFACTS_CSV = "openfoodfacts_breakfast_products.csv" 
EUROSTAT_CSV = "eurostat_retail_sales.csv" # Ensure this file is in your project folder

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="European Breakfast Product Tracker", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.title("🥣 European Breakfast Product Market Dashboard")
st.markdown("---") 

# --- Data Loading Functions ---
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
        if df.columns[0] == 'DATAFLOW':
            if 'TIME_PERIOD' in df.columns:
                df = df.rename(columns={'TIME_PERIOD': 'date'})
            elif 'time' in df.columns:
                df = df.rename(columns={'time': 'date'})
            else:
                df = df.rename(columns={df.columns[0]: 'date'})
        else:
            df = df.rename(columns={df.columns[0]: 'date'})

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df.dropna(subset=['date'], inplace=True)

        if 'geo' not in df.columns:
            st.warning(f"Warning: 'geo' column not found in {EUROSTAT_CSV}. Defaulting to 'Unknown'.")
            df['geo'] = 'Unknown'

        if 'value' not in df.columns:
            st.warning(f"Warning: 'value' column not found in {EUROSTAT_CSV}. Defaulting to 0.")
            df['value'] = 0

        return df
    except Exception as e:
        st.error(f"Failed to load or parse Eurostat data from '{EUROSTAT_CSV}': {e}")
        return pd.DataFrame()

@st.cache_data
def load_openfoodfacts_data():
    """
    Loads breakfast product data from the openfoodfacts_breakfast_products.csv file.
    """
    if not os.path.exists(OPENFOODFACTS_CSV):
        st.warning(f"Product data file '{OPENFOODFACTS_CSV}' not found. Please run `openfoodfacts_scraper.py` first to generate it.")
        return pd.DataFrame()
    try:
        df = pd.read_csv(OPENFOODFACTS_CSV, encoding='utf-8')
        return df
    except Exception as e:
        st.error(f"Failed to load or parse product data from '{OPENFOODFACTS_CSV}': {e}")
        return pd.DataFrame()

# --- Load Data at App Start ---
df_openfoodfacts = load_openfoodfacts_data()
df_eurostat = load_eurostat_data()

# --- Sidebar Filters ---
st.sidebar.header("Global Filters")

# Country Multi-select Filter
all_countries = sorted(df_openfoodfacts['Country'].dropna().unique().tolist()) if not df_openfoodfacts.empty else []
selected_countries = st.sidebar.multiselect(
    "Select Countries for Product Data",
    options=all_countries,
    default=all_countries # Default to all selected
)

# Product Type (Search Term) Multi-select Filter
all_search_terms = sorted(df_openfoodfacts['Search_Term'].dropna().unique().tolist()) if not df_openfoodfacts.empty else []
selected_search_terms = st.sidebar.multiselect(
    "Select Breakfast Product Types",
    options=all_search_terms,
    default=all_search_terms # Default to all selected
)

# Apply global filters to Open Food Facts data
filtered_df_off = df_openfoodfacts[
    (df_openfoodfacts['Country'].isin(selected_countries)) &
    (df_openfoodfacts['Search_Term'].isin(selected_search_terms))
] if not df_openfoodfacts.empty else pd.DataFrame()


# --- Main Content Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Overview & Product List", "📈 Sales Trends (Eurostat)", "📦 Product Breakdown & Search"])

with tab1:
    st.header("Overview & Filtered Product List")
    st.write("Explore a comprehensive list of breakfast products based on your selections.")

    # Search Bar
    search_query = st.text_input("Search Product Name or Brand", "").strip().lower()
    
    display_df = filtered_df_off.copy()
    if search_query:
        display_df = display_df[
            display_df['Product'].astype(str).str.lower().str.contains(search_query) |
            display_df['Brand'].astype(str).str.lower().str.contains(search_query)
        ]

    st.subheader(f"Filtered Products ({len(display_df)} items)")
    if not display_df.empty:
        # Display as cards or a more interactive table
        st.dataframe(display_df[['Product', 'Brand', 'Quantity', 'Store', 'Country', 'Search_Term', 'URL']], use_container_width=True)
    else:
        st.info("No products found matching the selected filters and search query.")

    # Overview charts based on filtered data
    if not display_df.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Product Count by Country")
            country_count_filtered = display_df['Country'].value_counts().reset_index()
            country_count_filtered.columns = ['Country', 'Count']
            fig_country_filtered = px.bar(
                country_count_filtered, 
                x='Country', 
                y='Count', 
                title='Number of Products Listed by Country (Filtered)'
            )
            st.plotly_chart(fig_country_filtered, use_container_width=True)
        with col2:
            st.subheader("Top Brands (Filtered)")
            brand_count_filtered = display_df['Brand'].value_counts().head(10).reset_index()
            brand_count_filtered.columns = ['Brand', 'Count']
            fig_brand_filtered = px.bar(
                brand_count_filtered, 
                x='Brand', 
                y='Count', 
                title='Top 10 Brands (Filtered)'
            )
            st.plotly_chart(fig_brand_filtered, use_container_width=True)

        st.download_button(
            "Download Filtered Data CSV",
            data=display_df.to_csv(index=False).encode('utf-8'),
            file_name="filtered_breakfast_products.csv",
            mime="text/csv",
        )


with tab2:
    st.header("Eurostat Retail Sales Trends")
    st.write("Visualize overall retail trade index for food products in European countries.")

    if not df_eurostat.empty:
        # Comprehensive list of European countries (ISO 2-letter codes)
        # This list can be expanded or refined as needed to include more EU/EEA/EFTA countries.
        # Ensure these codes match the 'geo' column in your Eurostat CSV.
        all_european_countries_for_eurostat = sorted([
            "AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "EL", "ES", "FI", "FR", "HR", "HU",
            "IE", "IS", "IT", "LT", "LU", "LV", "MT", "NL", "NO", "PL", "PT", "RO", "SE", "SI",
            "SK", "UK" # Including UK as it's often relevant for European data analysis
        ])
        
        # Prioritize 'UK' if it's in the list
        if 'UK' in all_european_countries_for_eurostat:
            all_european_countries_for_eurostat.insert(0, all_european_countries_for_eurostat.pop(all_european_countries_for_eurostat.index('UK')))
        
        # --- Updated: Use the comprehensive list for selection options ---
        selected_eurostat_country = st.selectbox(
            "Select Country for Sales Trend (Eurostat)", 
            options=all_european_countries_for_eurostat, # Use the comprehensive list
            key="eurostat_country_select_main"
        )
        
        # Filter the loaded dataframe based on selection
        filtered_eurostat_data = df_eurostat[df_eurostat['geo'] == selected_eurostat_country].copy()
        
        if not filtered_eurostat_data.empty:
            # Sort by date for proper line chart
            filtered_eurostat_data = filtered_eurostat_data.sort_values(by='date')
            fig_sales = px.line(
                filtered_eurostat_data, 
                x="date", 
                y="value", 
                title=f"Retail Sales Index in {selected_eurostat_country} (Eurostat)", 
                labels={"value": "Retail Trade Index (Volume)", "date": "Date"},
                hover_data={"value": ":.2f"},
                markers=True # Add markers to line chart
            )
            st.plotly_chart(fig_sales, use_container_width=True)
        else:
            # --- Updated: Provide clearer feedback if data is missing for selected country ---
            st.info(f"No Eurostat sales data found in the loaded CSV for {selected_eurostat_country}. Please ensure this country's data is present in '{EUROSTAT_CSV}'.")
    else:
        st.info("Eurostat retail sales data not available.")

with tab3:
    st.header("Detailed Product Breakdown & Analysis")
    st.write("Deep dive into product distribution by type, store, and brand based on your global filters.")

    if not filtered_df_off.empty:
        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Products by Breakfast Type")
            # Using filtered data for this chart
            search_term_count_filtered = filtered_df_off['Search_Term'].value_counts().reset_index()
            search_term_count_filtered.columns = ['Breakfast Type', 'Count']
            fig_search_term = px.pie(
                search_term_count_filtered, 
                values='Count', 
                names='Breakfast Type', 
                title='Distribution of Products by Breakfast Type'
            )
            st.plotly_chart(fig_search_term, use_container_width=True)

        with col4:
            st.subheader("Products by Store")
            # Clean 'Store' data for better visualization (e.g., split by comma)
            filtered_df_off['Store_List'] = filtered_df_off['Store'].astype(str).apply(lambda x: [s.strip() for s in x.split(',') if s.strip()])
            exploded_stores = filtered_df_off.explode('Store_List')
            
            store_count = exploded_stores['Store_List'].value_counts().reset_index()
            store_count.columns = ['Store', 'Count']
            store_count = store_count[~store_count['Store'].isin(['N/A', ''])] # Filter out generic/NA stores
            
            if not store_count.empty:
                fig_store = px.bar(
                    store_count.head(15), # Show top 15 stores
                    x='Store', 
                    y='Count', 
                    title='Top Stores by Product Listings',
                    labels={'Count': 'Number of Products'}
                )
                st.plotly_chart(fig_store, use_container_width=True)
            else:
                st.info("No meaningful store data found in Open Food Facts for selected filters.")
        
        st.subheader("Explore Brands and Products")
        # Multi-select Brand filter
        available_brands = sorted(filtered_df_off['Brand'].dropna().unique().tolist())
        selected_brands = st.multiselect(
            "Select Brands to view products", 
            options=available_brands,
            key="brand_select_tab3"
        )

        if selected_brands:
            filtered_by_brands = filtered_df_off[filtered_df_off['Brand'].isin(selected_brands)]
            st.dataframe(filtered_by_brands[['Product', 'Brand', 'Quantity', 'Store', 'Country', 'Search_Term', 'URL']], use_container_width=True)
        else:
            st.info("Select one or more brands to view their products.")

    else:
        st.info("No product data available for analysis. Adjust filters or check data source.")
