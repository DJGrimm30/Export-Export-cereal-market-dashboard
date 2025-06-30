import streamlit as st
import pandas as pd
import plotly.express as px
import os

# --- Configuration ---
# Output CSV filename from Open Food Facts scraper (now expanded to breakfast products)
OPENFOODFACTS_CSV = "openfoodfacts_breakfast_products.csv" 
EUROSTAT_RETAIL_CSV = "eurostat_retail_sales.csv" # Original Eurostat retail sales
# New: Manual download consumer insight files
ONS_RETAIL_SALES_CSV = "ons_retail_sales.csv" # New: ONS Retail Sales Data (manual download)
EUROSTAT_CONSUMPTION_CSV = "eurostat_consumption_expenditure.csv" # Manual download: Household Consumption
EUROSTAT_ECOMMERCE_CSV = "eurostat_ecommerce_buyers.csv" # Manual download: E-commerce buyers
EUROSTAT_POPULATION_CSV = "eurostat_population.csv" # Manual download: Population data

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
def load_eurostat_data(filepath, geo_col='geo', value_col='value', date_col='date', specific_filters=None):
    """
    Generic function to load and preprocess Eurostat CSVs (now used for manually downloaded files).
    Assumes first column is time period or DATAFLOW if not specified.
    """
    if not os.path.exists(filepath):
        st.error(f"Error: Data file '{filepath}' not found. Please ensure it's in the same folder as the app.")
        return pd.DataFrame()
    try:
        df = pd.read_csv(filepath)
        
        # --- Robust column detection for manually downloaded Eurostat files ---
        # First, try to identify time period column using common names
        detected_date_col = None
        for col_name in ['TIME_PERIOD', 'time', 'date', 'DATE']: # Common names for date/time in Eurostat
            if col_name in df.columns:
                detected_date_col = col_name
                break
        
        # If not found directly, check if the first column is complex (like 'DATAFLOW,.../TIME_PERIOD')
        if not detected_date_col and '\\' in df.columns[0]:
            detected_date_col = df.columns[0].split('\\')[-1].strip() # Get header after backslash
            # Split the first column into individual dimensions
            complex_header_dims = df.columns[0].split('\\')[0].split(',')
            for i, dim_name in enumerate(complex_header_dims):
                df[dim_name.strip()] = df[df.columns[0]].apply(lambda x: x.split(',')[i].strip() if len(x.split(',')) > i else None)
            df = df.drop(columns=[df.columns[0]]) # Drop the original complex column
        elif not detected_date_col: # Fallback to first column if no other date column found
            detected_date_col = df.columns[0]
            
        # Rename the detected date column to 'date'
        if detected_date_col and detected_date_col != 'date':
            df = df.rename(columns={detected_date_col: 'date'})
        
        # Ensure 'date' column is datetime
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df.dropna(subset=['date'], inplace=True)

        # Identify GEO column
        detected_geo_col = None
        for col_name in ['GEO', 'geo', 'REGION', 'country']: # Common names for geographical info
            if col_name in df.columns:
                detected_geo_col = col_name
                break
        if detected_geo_col and detected_geo_col != 'geo':
            df = df.rename(columns={detected_geo_col: 'geo'})
        elif 'geo' not in df.columns: # If no geo column found, and not specified in map
            st.warning(f"Warning: 'geo' column not found in {filepath}. Defaulting to 'Unknown'.")
            df['geo'] = 'Unknown'

        # Identify value column
        detected_value_col = None
        for col_name in ['VALUE', 'value', 'OBS_VALUE']: # Common names for value
            if col_name in df.columns:
                detected_value_col = col_name
                break
        if detected_value_col and detected_value_col != 'value':
            df = df.rename(columns={detected_value_col: 'value'})
        elif 'value' not in df.columns: # If no value column found, and not specified in map
            st.warning(f"Warning: 'value' column not found in {filepath}. Defaulting to 0.")
            df['value'] = 0 # Placeholder

        # Apply specific filters if provided (e.g., NACE_R2, COFOG_L2, IND_TYPE, AGE, SEX, UNIT, INDIC_BT)
        if specific_filters:
            for col, val in specific_filters.items():
                if col in df.columns:
                    df = df[df[col].astype(str).str.strip() == str(val).strip()] # Strip whitespace for robust comparison
                else:
                    st.warning(f"Filter column '{col}' not found in {filepath}. Skipping filter.")

        return df[['date', 'geo', 'value']].copy() # Return only the essential columns
    except Exception as e:
        st.error(f"Failed to load or parse Eurostat data from '{filepath}': {e}")
        return pd.DataFrame()

@st.cache_data
def load_ons_retail_sales_data():
    """
    Loads ONS Retail Sales data from a CSV file.
    Assumes the CSV structure typical for ONS time series downloads.
    """
    if not os.path.exists(ONS_RETAIL_SALES_CSV):
        st.error(f"Error: ONS data file '{ONS_RETAIL_SALES_CSV}' not found. Please download it manually from ONS and place it in the folder.")
        return pd.DataFrame()
    try:
        df = pd.read_csv(ONS_RETAIL_SALES_CSV)
        # ONS CSVs usually have a header row and then the data.
        # Common ONS structure:
        # 1st col: CDID (unique identifier)
        # 2nd col: TSMV (Name of the series)
        # 3rd col: Units (e.g., percentage, millions)
        # 4th col: Date (e.g., Jan 2023)
        # 5th col: Value
        
        # Look for a column that contains dates (e.g., 'Date', 'Month', 'Year')
        date_col_name = None
        for col in df.columns:
            if 'date' in col.lower() or 'month' in col.lower() or 'time' in col.lower():
                date_col_name = col
                break
        if not date_col_name:
            # Fallback: assume the 4th column is often the date in ONS downloads
            if len(df.columns) >= 4:
                date_col_name = df.columns[3]
            else:
                st.warning(f"Could not identify date column in ONS data: {df.columns.tolist()}")
                return pd.DataFrame()

        # Look for a column that contains values (e.g., 'Value', 'Observation')
        value_col_name = None
        for col in df.columns:
            if 'value' in col.lower() or 'obs' in col.lower(): # 'obs' for OBS_VALUE
                value_col_name = col
                break
        if not value_col_name:
            # Fallback: assume the 5th column is often the value in ONS downloads
            if len(df.columns) >= 5:
                value_col_name = df.columns[4]
            else:
                st.warning(f"Could not identify value column in ONS data: {df.columns.tolist()}")
                return pd.DataFrame()
        
        # Select and rename columns
        ons_df = df[[date_col_name, value_col_name]].copy()
        ons_df.columns = ['date', 'value'] # Standardize names

        # Convert date column
        ons_df['date'] = pd.to_datetime(ons_df['date'], errors='coerce')
        ons_df.dropna(subset=['date', 'value'], inplace=True)
        ons_df['geo'] = 'UK' # ONS data is for UK/GB

        return ons_df[['date', 'geo', 'value']].copy()
    except Exception as e:
        st.error(f"Failed to load or parse ONS retail sales data from '{ONS_RETAIL_SALES_CSV}': {e}")
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
df_eurostat_retail = load_eurostat_data(EUROSTAT_RETAIL_CSV, specific_filters={'NACE_R2': 'G47', 'INDIC_BT': 'VOL_IDX_RT'})
df_eurostat_consumption = load_eurostat_data(EUROSTAT_CONSUMPTION_CSV, specific_filters={'COFOG_L2': 'CP01', 'UNIT': 'PC_CP'})
df_eurostat_ecommerce = load_eurostat_data(EUROSTAT_ECOMMERCE_CSV, specific_filters={'IND_TYPE': 'I_IBSPS', 'UNIT': 'PC_IND'})
df_eurostat_population = load_eurostat_data(EUROSTAT_POPULATION_CSV, specific_filters={'AGE': 'TOTAL', 'SEX': 'T'})
df_ons_retail_sales = load_ons_retail_sales_data() # New: Load ONS data

# --- Sidebar Filters ---
st.sidebar.header("Global Filters")

# Country Multi-select Filter
all_countries_off = sorted(df_openfoodfacts['Country'].dropna().unique().tolist()) if not df_openfoodfacts.empty else []
selected_countries_off = st.sidebar.multiselect(
    "Select Countries for Product Data",
    options=all_countries_off,
    default=all_countries_off
)

# Product Type (Search Term) Multi-select Filter
all_search_terms = sorted(df_openfoodfacts['Search_Term'].dropna().unique().tolist()) if not df_openfoodfacts.empty else []
selected_search_terms = st.sidebar.multiselect(
    "Select Breakfast Product Types",
    options=all_search_terms,
    default=all_search_terms
)

# Apply global filters to Open Food Facts data
filtered_df_off = df_openfoodfacts[
    (df_openfoodfacts['Country'].isin(selected_countries_off)) &
    (df_openfoodfacts['Search_Term'].isin(selected_search_terms))
] if not df_openfoodfacts.empty else pd.DataFrame()


# --- Main Content Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview & Product List", "📈 Sales Trends (Eurostat)", "📦 Product Breakdown & Search", "🧠 Consumer Insights & Macro Trends"])

with tab1:
    st.header("Overview & Filtered Product List")
    st.write("Explore a comprehensive list of breakfast products based on your selections.")

    search_query = st.text_input("Search Product Name or Brand", "").strip().lower()
    
    display_df = filtered_df_off.copy()
    if search_query:
        display_df = display_df[
            display_df['Product'].astype(str).str.lower().str.contains(search_query) |
            display_df['Brand'].astype(str).str.lower().str.contains(search_query)
        ]

    st.subheader(f"Filtered Products ({len(display_df)} items)")
    if not display_df.empty:
        st.dataframe(display_df[['Product', 'Brand', 'Quantity', 'Store', 'Country', 'Search_Term', 'URL']], use_container_width=True)
    else:
        st.info("No products found matching the selected filters and search query.")

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

    if not df_eurostat_retail.empty:
        all_eurostat_retail_countries = sorted(df_eurostat_retail['geo'].dropna().unique().tolist())
        
        if 'UK' in all_eurostat_retail_countries:
            all_eurostat_retail_countries.insert(0, all_eurostat_retail_countries.pop(all_eurostat_retail_countries.index('UK')))
        
        selected_eurostat_country = st.selectbox(
            "Select Country for Sales Trend (Eurostat Retail)", 
            options=all_eurostat_retail_countries,
            key="eurostat_retail_country_select" 
        )
        
        filtered_eurostat_retail_data = df_eurostat_retail[df_eurostat_retail['geo'] == selected_eurostat_country].copy()
        
        if not filtered_eurostat_retail_data.empty:
            filtered_eurostat_retail_data = filtered_eurostat_retail_data.sort_values(by='date')
            fig_sales = px.line(
                filtered_eurostat_retail_data, 
                x="date", 
                y="value", 
                title=f"Retail Sales Index in {selected_eurostat_country} (Eurostat)", 
                labels={"value": "Retail Trade Index (Volume)", "date": "Date"},
                hover_data={"value": ":.2f"},
                markers=True
            )
            st.plotly_chart(fig_sales, use_container_width=True)
        else:
            st.info(f"No Eurostat retail sales data found in the loaded CSV for {selected_eurostat_country}.")
    else:
        st.info("Eurostat retail sales data not available.")

with tab3:
    st.header("Detailed Product Breakdown & Analysis")
    st.write("Deep dive into product distribution by type, store, and brand based on your global filters.")

    if not filtered_df_off.empty:
        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Products by Breakfast Type")
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
            filtered_df_off['Store_List'] = filtered_df_off['Store'].astype(str).apply(lambda x: [s.strip() for s in x.split(',') if s.strip()])
            exploded_stores = filtered_df_off.explode('Store_List')
            
            store_count = exploded_stores['Store_List'].value_counts().reset_index()
            store_count.columns = ['Store', 'Count']
            store_count = store_count[~store_count['Store'].isin(['N/A', ''])] 
            
            if not store_count.empty:
                fig_store = px.bar(
                    store_count.head(15), 
                    x='Store', 
                    y='Count', 
                    title='Top Stores by Product Listings',
                    labels={'Count': 'Number of Products'}
                )
                st.plotly_chart(fig_store, use_container_width=True)
            else:
                st.info("No meaningful store data found in Open Food Facts for selected filters.")
        
        st.subheader("Explore Brands and Products")
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


with tab4: # Consumer Insights tab
    st.header("🧠 Consumer Insights & Macro Trends")
    st.write("Explore broader economic and demographic trends influencing consumer behavior in Europe and the UK.")

    # Selectbox for which insight to view
    insight_type = st.selectbox(
        "Select Consumer Insight Type",
        options=["ONS Retail Sales (UK)", "Household Consumption (Eurostat)", "E-commerce Penetration (Eurostat)", "Population Trends (Eurostat)"],
        key="insight_type_select"
    )

    col_insight1, col_insight2 = st.columns(2) # Two columns for controls and chart

    # --- ONS Retail Sales ---
    if insight_type == "ONS Retail Sales (UK)":
        if not df_ons_retail_sales.empty:
            # ONS data is typically only for UK, so no country filter needed here directly
            st.subheader("UK Retail Sales Index (ONS)")
            fig_ons_retail = px.line(
                df_ons_retail_sales,
                x="date",
                y="value",
                title="UK Retail Sales Index (ONS)",
                labels={"value": "Retail Sales Index", "date": "Date"},
                hover_data={"value": ":.2f"},
                markers=True
            )
            col_insight2.plotly_chart(fig_ons_retail, use_container_width=True)
        else:
            col_insight2.info(f"ONS Retail Sales data not available. Please ensure '{ONS_RETAIL_SALES_CSV}' is loaded.")

    # --- Household Consumption Expenditure (Eurostat) ---
    elif insight_type == "Household Consumption (Eurostat)":
        if not df_eurostat_consumption.empty:
            countries_consumption = sorted(df_eurostat_consumption['geo'].dropna().unique().tolist())
            selected_country_consumption = col_insight1.selectbox(
                "Select Country for Consumption Trends",
                options=countries_consumption,
                key="consumption_country_select"
            )
            filtered_consumption = df_eurostat_consumption[
                (df_eurostat_consumption['geo'] == selected_country_consumption)
            ].sort_values(by='date')
            
            if not filtered_consumption.empty:
                fig_consumption = px.line(
                    filtered_consumption,
                    x="date",
                    y="value",
                    title=f"Household Consumption on Food in {selected_country_consumption}",
                    labels={"value": "Consumption Expenditure (%)", "date": "Year"},
                    hover_data={"value": ":.2f"},
                    markers=True
                )
                col_insight2.plotly_chart(fig_consumption, use_container_width=True)
            else:
                col_insight2.info(f"No consumption data for {selected_country_consumption}. Check data in '{EUROSTAT_CONSUMPTION_CSV}'.")
        else:
            col_insight2.info(f"Household consumption data not available. Please ensure '{EUROSTAT_CONSUMPTION_CSV}' is loaded.")

    # --- E-commerce Penetration (Eurostat) ---
    elif insight_type == "E-commerce Penetration (Eurostat)":
        if not df_eurostat_ecommerce.empty:
            countries_ecommerce = sorted(df_eurostat_ecommerce['geo'].dropna().unique().tolist())
            selected_country_ecommerce = col_insight1.selectbox(
                "Select Country for E-commerce Trends",
                options=countries_ecommerce,
                key="ecommerce_country_select"
            )
            filtered_ecommerce = df_eurostat_ecommerce[
                (df_eurostat_ecommerce['geo'] == selected_country_ecommerce)
            ].sort_values(by='date')

            if not filtered_ecommerce.empty:
                fig_ecommerce = px.line(
                    filtered_ecommerce,
                    x="date",
                    y="value",
                    title=f"Individuals Buying Online in {selected_country_ecommerce}",
                    labels={"value": "Percentage of Individuals (%)", "date": "Year"},
                    hover_data={"value": ":.2f"},
                    markers=True
                )
                col_insight2.plotly_chart(fig_ecommerce, use_container_width=True)
            else:
                col_insight2.info(f"No e-commerce data for {selected_country_ecommerce}. Check data in '{EUROSTAT_ECOMMERCE_CSV}'.")
        else:
            col_insight2.info(f"E-commerce data not available. Please ensure '{EUROSTAT_ECOMMERCE_CSV}' is loaded.")

    # --- Population Trends (Eurostat) ---
    elif insight_type == "Population Trends (Eurostat)":
        if not df_eurostat_population.empty:
            countries_population = sorted(df_eurostat_population['geo'].dropna().unique().tolist())
            selected_country_population = col_insight1.selectbox(
                "Select Country for Population Trends",
                options=countries_population,
                key="population_country_select"
            )
            filtered_population = df_eurostat_population[
                (df_eurostat_population['geo'] == selected_country_population)
            ].sort_values(by='date')

            if not filtered_population.empty:
                fig_population = px.line(
                    filtered_population,
                    x="date",
                    y="value",
                    title=f"Total Population in {selected_country_population}",
                    labels={"value": "Population", "date": "Year"},
                    hover_data={"value": ":.0f"},
                    markers=True
                )
                col_insight2.plotly_chart(fig_population, use_container_width=True)
            else:
                col_insight2.info(f"No population data for {selected_country_population}. Check data in '{EUROSTAT_POPULATION_CSV}'.")
        else:
            col_insight2.info(f"Population data not available. Please ensure '{EUROSTAT_POPULATION_CSV}' is loaded.")
