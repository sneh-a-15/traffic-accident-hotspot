import pandas as pd
import streamlit as st

@st.cache_data
def load_data(path, limit=10000):
    """Load and preprocess accident data from CSV"""
    df = pd.read_csv(path, nrows=limit)
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')
    df['Hour'] = df['Start_Time'].dt.hour
    df['DayOfWeek'] = df['Start_Time'].dt.day_name()
    df['Month'] = df['Start_Time'].dt.month_name()
    df['Year'] = df['Start_Time'].dt.year
    df['Date'] = df['Start_Time'].dt.date
    df = df.dropna(subset=['Severity', 'City', 'Weather_Condition'])
    return df