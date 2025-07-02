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
def load_eurostat_data(filepath, geo_col='geo', value_col='value', date_col='date', specific_filters=None):
    """
    Generic function to load and preprocess Eurostat CSVs.
    Handles complex first column headers common in Eurostat downloads by melting.
    """
    st.write(f"Attempting to load: {filepath}") # Debug print
    if not os.path.exists(filepath):
        st.error(f"Error: Data file '{filepath}' not found. Please ensure it's in the same folder as the app.")
        return pd.DataFrame()
    try:
        df = pd.read_csv(filepath)
        st.write(f"Raw columns for {filepath}: {df.columns.tolist()}") # Debug print
        
        raw_first_col_header = df.columns[0]
        
        if '\\' in raw_first_col_header:
            dimension_names_str = raw_first_col_header.split('\\')[0]
            time_period_header_name_raw = raw_first_col_header.split('\\')[1].strip()
        else:
            dimension_names_str = raw_first_col_header
            time_period_header_name_raw = None 
        
        dimension_names = [d.strip() for d in dimension_names_str.split(',')]
        st.write(f"Detected dimension names from first header: {dimension_names}") # Debug print

        temp_dim_split_col = '__temp_dim_split__'
        df[temp_dim_split_col] = df[raw_first_col_header]

        for i, dim_name in enumerate(dimension_names):
            df[dim_name] = df[temp_dim_split_col].apply(lambda x: x.split(',')[i].strip() if len(x.split(',')) > i else None)
        
        df = df.drop(columns=[raw_first_col_header, temp_dim_split_col])
        st.write(f"Columns after splitting first header: {df.columns.tolist()}") # Debug print

        time_period_cols_to_melt = [col for col in df.columns if re.match(r'^\d{4}(Q[1-4]|M\d{2})?$', col.strip())]
        
        value_col_in_raw_data = 'OBS_VALUE' # Standard Eurostat value column
        
        if not time_period_cols_to_melt or value_col_in_raw_data not in df.columns:
            st.warning(f"Warning: Could not identify time periods or OBS_VALUE column in {filepath}. Columns: {df.columns.tolist()}")
            return pd.DataFrame()

        id_vars_for_melt = [col for col in df.columns if col not in time_period_cols_to_melt and col != value_col_in_raw_data]
        
        df_melted = df.melt(id_vars=id_vars_for_melt, value_vars=time_period_cols_to_melt, var_name='date_raw', value_name='value_raw')

        df_melted['value'] = pd.to_numeric(
            df_melted['value_raw'].astype(str).str.replace(r'[a-zA-Z\s:]', '', regex=True), 
            errors='coerce'
        )
        df_melted.dropna(subset=['value'], inplace=True)

        def parse_eurostat_date(date_str):
            if pd.isna(date_str): return pd.NaT
            date_str = str(date_str).strip()
            if re.match(r'^\d{4}$', date_str): return pd.to_datetime(date_str, format='%Y')
            elif re.match(r'^\d{4}Q[1-4]$', date_str): return pd.to_datetime(f'{date_str[:4]}-{int(date_str[5])*3-2}-01')
            elif re.match(r'^\d{4}M\d{2}$', date_str): return pd.to_datetime(date_str, format='%YM%m')
            return pd.NaT

        df_melted['date'] = df_melted['date_raw'].apply(parse_eurostat_date)
        df_melted.dropna(subset=['date'], inplace=True)

        if 'GEO' in df_melted.columns:
            df_melted = df_melted.rename(columns={'GEO': 'geo'})
        
        # --- NEW: Standardize other common Eurostat dimension columns ---
        if 'UNIT' in df_melted.columns:
            df_melted = df_melted.rename(columns={'UNIT': 'unit'})
        if 'FREQ' in df_melted.columns:
            df_melted = df_melted.rename(columns={'FREQ': 'freq'})

        # Apply specific filters (e.g., NACE_R2='G47', COFOG_L2='CP01')
        if specific_filters:
            st.write(f"Applying filters for {filepath}: {specific_filters}") # Debug print
            for col_filter_name, filter_value in specific_filters.items():
                if col_filter_name in df_melted.columns:
                    # Filter and also print how many rows remain
                    initial_rows = len(df_melted)
                    df_melted = df_melted[df_melted[col_filter_name].astype(str).str.strip() == str(filter_value).strip()]
                    st.write(f"Filtered by {col_filter_name}='{filter_value}'. Rows: {initial_rows} -> {len(df_melted)}") # Debug print
                else:
                    st.warning(f"Filter column '{col_filter_name}' not found in DataFrame for filtering '{filepath}'. Available columns: {df_melted.columns.tolist()}")

        required_final_cols = ['date', 'geo', 'value']
        if 'geo' not in df_melted.columns:
            st.warning(f"Final 'geo' column missing after processing {filepath}. Defaulting to 'Unknown'.")
            df_melted['geo'] = 'Unknown' 
        if 'value' not in df_melted.columns:
            st.warning(f"Final 'value' column missing after processing {filepath}. Defaulting to 0.")
            df_melted['value'] = 0 

        final_df = df_melted[required_final_cols].copy()
        st.write(f"Final DataFrame shape for {filepath}: {final_df.shape}") # Debug print
        st.write(f"Final DataFrame head for {filepath}:\n{final_df.head()}") # Debug print
        return final_df

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

def calculate_per_capita(value_df, population_df, value_col='value', pop_col='value', date_col='date', geo_col='geo'):
    """
    Calculates per capita figures by merging value data with population data.
    Assumes both dataframes have 'date', 'geo', and 'value' columns.
    Population values should be in the same units (e.g., thousands, millions) as consumption values for meaningful per capita.
    """
    if value_df.empty or population_df.empty:
        return pd.DataFrame()

    value_df['year'] = value_df[date_col].dt.year
    population_df['year'] = population_df[date_col].dt.year

    merged_df = pd.merge(
        value_df,
        population_df[[geo_col, 'year', pop_col]].rename(columns={pop_col: 'population'}),
        on=[geo_col, 'year'],
        how='left'
    )
    
    merged_df['Per_Capita_Value'] = merged_df[value_col] / merged_df['population']
    
    return merged_df.dropna(subset=['Per_Capita_Value'])

# --- Load Data at App Start ---
df_openfoodfacts = load_openfoodfacts_data()
df_eurostat_retail = load_eurostat_data(EUROSTAT_RETAIL_CSV, specific_filters={'NACE_R2': 'G47', 'INDIC_BT': 'VOL_IDX_RT'})
df_eurostat_consumption = load_eurostat_data(EUROSTAT_CONSUMPTION_CSV, specific_filters={'COFOG_L2': 'CP01', 'UNIT': 'PC_CP'})
df_eurostat_ecommerce = load_eurostat_data(EUROSTAT_ECOMMERCE_CSV, specific_filters={'IND_TYPE': 'I_IBSPS', 'UNIT': 'PC_IND'})
df_eurostat_population = load_eurostat_data(EUROSTAT_POPULATION_CSV, specific_filters={'AGE': 'TOTAL', 'SEX': 'T'})
df_ons_retail_sales = load_ons_retail_sales_data()

# --- Sidebar Filters ---
st.sidebar.header("Global Filters")

all_countries_off = sorted(df_openfoodfacts['Country'].dropna().unique().tolist()) if not df_openfoodfacts.empty else []
selected_countries_off = st.sidebar.multiselect(
    "Select Countries for Product Data",
    options=all_countries_off,
    default=all_countries_off
)

all_search_terms = sorted(df_openfoodfacts['Search_Term'].dropna().unique().tolist()) if not df_openfoodfacts.empty else []
selected_search_terms = st.sidebar.multiselect(
    "Select Breakfast Product Types",
    options=all_search_terms,
    default=all_search_terms
)

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
            brand_count_filtered = display_df['Brand'].value_counts().reset_index() 
            brand_count_filtered.columns = ['Brand', 'Count']
            brand_count_filtered['Percentage'] = (brand_count_filtered['Count'] / brand_count_filtered['Count'].sum()) * 100

            fig_brand_filtered = px.pie( 
                brand_count_filtered.head(10), 
                values='Count', 
                names='Brand', 
                title='Top 10 Brands Distribution (Filtered)',
                hover_data=['Percentage'],
                labels={'Percentage': '%'}
            )
            fig_brand_filtered.update_traces(textinfo='percent+label')
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
            retail_yoy_growth = calculate_yoy_growth(filtered_eurostat_retail_data, geo_col='geo')
            
            if not retail_yoy_growth.empty:
                st.subheader(f"Retail Sales Index YoY Growth in {selected_eurostat_country}")
                fig_retail_yoy = px.line(
                    retail_yoy_growth,
                    x="date",
                    y="YoY_Growth",
                    title=f"Retail Sales Index YoY Growth in {selected_eurostat_country}",
                    labels={"YoY_Growth": "YoY Growth (%)", "date": "Date"},
                    hover_data={"YoY_Growth": ":.2f%"},
                    markers=True
                )
                st.plotly_chart(fig_retail_yoy, use_container_width=True)
            else:
                st.info(f"Not enough data to calculate YoY growth for Retail Sales in {selected_eurostat_country}.")

            st.subheader(f"Retail Sales Index in {selected_eurostat_country} (Eurostat)")
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
            st.subheader("Products by Breakfast Type (Distribution)")
            search_term_count_filtered = filtered_df_off['Search_Term'].value_counts().reset_index()
            search_term_count_filtered.columns = ['Breakfast Type', 'Count']
            
            search_term_count_filtered['Percentage'] = (search_term_count_filtered['Count'] / search_term_count_filtered['Count'].sum()) * 100

            fig_search_term = px.pie(
                search_term_count_filtered, 
                values='Count', 
                names='Breakfast Type', 
                title='Distribution of Products by Breakfast Type',
                hover_data=['Percentage'], 
                labels={'Percentage': '%'}
            )
            fig_search_term.update_traces(textinfo='percent+label')
            st.plotly_chart(fig_search_term, use_container_width=True)

        with col4:
            st.subheader("Products by Store (Distribution)")
            filtered_df_off['Store_List'] = filtered_df_off['Store'].astype(str).apply(lambda x: [s.strip() for s in x.split(',') if s.strip()])
            exploded_stores = filtered_df_off.explode('Store_List')
            
            store_count = exploded_stores['Store_List'].value_counts().reset_index()
            store_count.columns = ['Store', 'Count']
            store_count = store_count[~store_count['Store'].isin(['N/A', ''])] 
            
            if not store_count.empty:
                store_count['Percentage'] = (store_count['Count'] / store_count['Count'].sum()) * 100
                fig_store = px.pie( 
                    store_count.head(15), 
                    values='Count', 
                    names='Store', 
                    title='Top 15 Stores by Product Listings Distribution',
                    hover_data=['Percentage'],
                    labels={'Percentage': '%'}
                )
                fig_store.update_traces(textinfo='percent+label')
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

    insight_type = st.selectbox(
        "Select Consumer Insight Type",
        options=["ONS Retail Sales (UK)", "Household Consumption (Eurostat)", "E-commerce Penetration (Eurostat)", "Population Trends (Eurostat)"],
        key="insight_type_select"
    )

    col_insight1, col_insight2 = st.columns(2)

    # --- ONS Retail Sales ---
    if insight_type == "ONS Retail Sales (UK)":
        if not df_ons_retail_sales.empty:
            ons_yoy_growth = calculate_yoy_growth(df_ons_retail_sales, geo_col='geo')
            if not ons_yoy_growth.empty:
                st.subheader("UK Retail Sales Index (ONS) - YoY Growth")
                fig_ons_yoy = px.line(
                    ons_yoy_growth,
                    x="date",
                    y="YoY_Growth",
                    title="UK Retail Sales Index YoY Growth (ONS)",
                    labels={"YoY_Growth": "YoY Growth (%)", "date": "Date"},
                    hover_data={"YoY_Growth": ":.2f%"},
                    markers=True
                )
                col_insight2.plotly_chart(fig_ons_yoy, use_container_width=True)
            else:
                col_insight2.info(f"Not enough data to calculate YoY growth for ONS Retail Sales.")

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
                consumption_yoy_growth = calculate_yoy_growth(filtered_consumption, geo_col='geo')
                if not consumption_yoy_growth.empty:
                    st.subheader(f"Household Consumption on Food YoY Growth in {selected_country_consumption}")
                    fig_consumption_yoy = px.line(
                        consumption_yoy_growth,
                        x="date",
                        y="YoY_Growth",
                        title=f"Household Consumption on Food YoY Growth in {selected_country_consumption}",
                        labels={"YoY_Growth": "YoY Growth (%)", "date": "Year"},
                        hover_data={"YoY_Growth": ":.2f%"},
                        markers=True
                    )
                    col_insight2.plotly_chart(fig_consumption_yoy, use_container_width=True)
                else:
                    col_insight2.info(f"Not enough data to calculate YoY growth for consumption in {selected_country_consumption}.")

                if not df_eurostat_population.empty:
                    consumption_per_capita = calculate_per_capita(
                        filtered_consumption, df_eurostat_population,
                        value_col='value', pop_col='value', date_col='date', geo_col='geo'
                    )
                    if not consumption_per_capita.empty:
                        st.subheader(f"Household Food Consumption Per Capita in {selected_country_consumption}")
                        fig_consumption_pc = px.line(
                            consumption_per_capita,
                            x="date",
                            y="Per_Capita_Value",
                            title=f"Household Food Consumption Per Capita in {selected_country_consumption}",
                            labels={"Per_Capita_Value": "Consumption Per Capita", "date": "Year"},
                            hover_data={"Per_Capita_Value": ":.2f"},
                            markers=True
                        )
                        col_insight2.plotly_chart(fig_consumption_pc, use_container_width=True)
                    else:
                        col_insight2.info(f"Not enough population data for per capita consumption in {selected_country_consumption}.")


                st.subheader(f"Household Consumption on Food in {selected_country_consumption}")
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
                ecommerce_yoy_growth = calculate_yoy_growth(filtered_ecommerce, geo_col='geo')
                if not ecommerce_yoy_growth.empty:
                    st.subheader(f"Individuals Buying Online YoY Growth in {selected_country_ecommerce}")
                    fig_ecommerce_yoy = px.line(
                        ecommerce_yoy_growth,
                        x="date",
                        y="YoY_Growth",
                        title=f"Individuals Buying Online YoY Growth in {selected_country_ecommerce}",
                        labels={"YoY_Growth": "YoY Growth (%)", "date": "Year"},
                        hover_data={"YoY_Growth": ":.2f%"},
                        markers=True
                    )
                    col_insight2.plotly_chart(fig_ecommerce_yoy, use_container_width=True)
                else:
                    col_insight2.info(f"Not enough data to calculate YoY growth for e-commerce in {selected_country_ecommerce}.")

                st.subheader(f"Individuals Buying Online in {selected_country_ecommerce}")
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
                population_yoy_growth = calculate_yoy_growth(filtered_population, geo_col='geo')
                if not population_yoy_growth.empty:
                    st.subheader(f"Total Population YoY Growth in {selected_country_population}")
                    fig_population_yoy = px.line(
                        population_yoy_growth,
                        x="date",
                        y="YoY_Growth",
                        title=f"Total Population YoY Growth in {selected_country_population}",
                        labels={"YoY_Growth": "YoY Growth (%)", "date": "Year"},
                        hover_data={"YoY_Growth": ":.2f%"},
                        markers=True
                    )
                    col_insight2.plotly_chart(fig_population_yoy, use_container_width=True)
                else:
                    col_insight2.info(f"Not enough data to calculate YoY growth for population in {selected_country_population}.")

                st.subheader(f"Total Population in {selected_country_population}")
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
