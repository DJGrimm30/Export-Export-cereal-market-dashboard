# Streamlit dashboard app (simplified placeholder)
import streamlit as st
import pandas as pd

df = pd.read_csv('carrefour_cereals.csv')
st.title('European Cereal Market Dashboard')
st.dataframe(df)